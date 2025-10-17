// OverlayRendererFast.cs  (C# 7.3 호환 버전)
// - Super-sampling + GaussianBlur + Soft-Alpha
// - EdgeRenderMode: SoftFill / MarchingSquaresAA / CircleFit (기본: CircleFit)

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public static class OverlayRendererFast
    {
        // ===== 튜너블 옵션 =====
        private enum EdgeRenderMode { SoftFill, MarchingSquaresAA, CircleFit }

        private const EdgeRenderMode EDGE_MODE = EdgeRenderMode.SoftFill; // 가장 둥글게 보이도록
        private const bool USE_SOFT_ALPHA = true;
        private const float SOFT_T0 = 0.40f;
        private const float SOFT_T1 = 0.70f;
        private const bool APPLY_GAUSSIAN_BLUR = true; // ROI super-sample 후 블러
        private const int SUPER_SAMPLE = 2;            // 2~3 권장
        private const bool LOG_MAPPING = false;

        // ===== 기본 유틸(7.3 호환) =====
        private static float SmoothStep(float a, float b, float x)
        {
            if (x <= a) return 0f;
            if (x >= b) return 1f;
            float t = (x - a) / (b - a);
            return t * t * (3f - 2f * t);
        }

        private static int ClampInt(int v, int lo, int hi)
        {
            if (v < lo) return lo;
            if (v > hi) return hi;
            return v;
        }

        private static float Clamp01(float v)
        {
            if (v < 0f) return 0f;
            if (v > 1f) return 1f;
            return v;
        }

        private static float Sqrtf(float v) { return (float)Math.Sqrt(v); }

        public static Bitmap RenderEx(
            Bitmap orig,
            SegResult r,
            float maskThr = 0.4f,
            float alpha = 0.45f,
            bool drawBoxes = false,
            bool fillMask = true,
            bool drawOutlines = true, // (주의) AA 모드에서는 내부에서 처리
            bool drawScores = true,
            int lineThickness = 1,
            List<SmartLabelingApp.ImageCanvas.OverlayItem> overlaysOut = null)
        {
            if (orig == null) throw new ArgumentNullException("orig");
            if (r == null) throw new ArgumentNullException("r");
            if (r.Dets == null || r.Dets.Count == 0) return (Bitmap)orig.Clone();

            Bitmap over = (Bitmap)orig.Clone();
            Rectangle rectAll = new Rectangle(0, 0, over.Width, over.Height);
            BitmapData data = over.LockBits(rectAll, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            try
            {
                int maskLenFull = r.MaskW * r.MaskH;
                float[] maskBuf = ArrayPool<float>.Shared.Rent(maskLenFull);

                try
                {
                    for (int di = 0; di < r.Dets.Count; di++)
                    {
                        var d = r.Dets[di];

                        // 1) proto * coeff 합성 → maskBuf (0..1 가정)
                        MaskSynth.ComputeMask_KHW_NoAlloc(d.Coeff, r.ProtoFlat, r.SegDim, r.MaskW, r.MaskH, maskBuf);

                        // 2) 박스 역매핑
                        Rectangle origBox = Postprocess.NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);
                        if (origBox == Rectangle.Empty) continue;

                        int imgW = over.Width, imgH = over.Height;
                        int x0 = Math.Max(0, origBox.X);
                        int y0 = Math.Max(0, origBox.Y);
                        int x1 = Math.Min(imgW, origBox.Right);
                        int y1 = Math.Min(imgH, origBox.Bottom);
                        if (x1 <= x0 || y1 <= y0) continue;

                        int roiW = x1 - x0, roiH = y1 - y0;

                        // NetSize 정사각 가정. NetW/NetH 있다면 분리 사용.
                        float scaleW = (float)r.MaskW / r.NetSize;
                        float scaleH = (float)r.MaskH / r.NetSize;

                        if (LOG_MAPPING && di == 0)
                            Debug.WriteLine("[OverlayRenderer] mask=" + r.MaskW + "x" + r.MaskH
                                + ", net=" + r.NetSize + ", scaleW=" + scaleW.ToString("F4")
                                + ", scaleH=" + scaleH.ToString("F4"));

                        // 3) 보간 인덱스/가중치
                        int[] ix0 = new int[roiW]; int[] ix1 = new int[roiW]; float[] tx = new float[roiW];
                        int[] iy0 = new int[roiH]; int[] iy1 = new int[roiH]; float[] ty = new float[roiH];

                        for (int dx = 0; dx < roiW; dx++)
                        {
                            float nx = ((x0 + dx) * r.Scale + r.PadX) * scaleW;
                            int fx = (int)Math.Floor(nx);
                            float wx = nx - fx;
                            ix0[dx] = MathUtils.Clamp(fx, 0, r.MaskW - 1);
                            ix1[dx] = MathUtils.Clamp(fx + 1, 0, r.MaskW - 1);
                            tx[dx] = wx;
                        }
                        for (int dy = 0; dy < roiH; dy++)
                        {
                            float ny = ((y0 + dy) * r.Scale + r.PadY) * scaleH;
                            int fy = (int)Math.Floor(ny);
                            float wy = ny - fy;
                            iy0[dy] = MathUtils.Clamp(fy, 0, r.MaskH - 1);
                            iy1[dy] = MathUtils.Clamp(fy + 1, 0, r.MaskH - 1);
                            ty[dy] = wy;
                        }

                        // 4) ROI bilinear (super-sample → blur → down)
                        int sw = roiW * SUPER_SAMPLE, sh = roiH * SUPER_SAMPLE;
                        float[] roiSoft = new float[sw * sh];

                        Parallel.For(0, sh, delegate (int yy)
                        {
                            float py = ((float)yy / SUPER_SAMPLE);
                            int ddy = (int)py;
                            float fy = py - ddy;
                            int iy0v = iy0[Math.Min(ddy, roiH - 1)];
                            int iy1v = iy1[Math.Min(ddy, roiH - 1)];
                            int mY0 = iy0v * r.MaskW;
                            int mY1 = iy1v * r.MaskW;
                            float wy0 = 1f - fy, wy1 = fy;

                            for (int xx = 0; xx < sw; xx++)
                            {
                                float px = ((float)xx / SUPER_SAMPLE);
                                int ddx = (int)px;
                                float fx = px - ddx;
                                int ix0v = ix0[Math.Min(ddx, roiW - 1)];
                                int ix1v = ix1[Math.Min(ddx, roiW - 1)];
                                float wx0 = 1f - fx, wx1 = fx;

                                float v00 = maskBuf[mY0 + ix0v];
                                float v10 = maskBuf[mY0 + ix1v];
                                float v01 = maskBuf[mY1 + ix0v];
                                float v11 = maskBuf[mY1 + ix1v];

                                float vy0 = v00 * wx0 + v10 * wx1;
                                float vy1 = v01 * wx0 + v11 * wx1;
                                roiSoft[yy * sw + xx] = vy0 * wy0 + vy1 * wy1;
                            }
                        });

                        if (APPLY_GAUSSIAN_BLUR)
                            GaussianBlurInplace(roiSoft, sw, sh); // σ≈1 근사

                        float[] roiDown = new float[roiW * roiH];
                        float invS = 1f / (SUPER_SAMPLE * SUPER_SAMPLE);
                        for (int dy = 0; dy < roiH; dy++)
                        {
                            for (int dx = 0; dx < roiW; dx++)
                            {
                                float sum = 0f;
                                for (int sy = 0; sy < SUPER_SAMPLE; sy++)
                                {
                                    for (int sx = 0; sx < SUPER_SAMPLE; sx++)
                                    {
                                        int idx = (dy * SUPER_SAMPLE + sy) * sw + (dx * SUPER_SAMPLE + sx);
                                        sum += roiSoft[idx];
                                    }
                                }
                                roiDown[dy * roiW + dx] = sum * invS;
                            }
                        }

                        // 5) 색상/경계 렌더
                        Color color = ColorUtils.ClassColor(d.ClassId);
                        float aEffBase = (alpha < 0f) ? 0f : ((alpha > 1f) ? 1f : alpha);

                        if (EDGE_MODE == EdgeRenderMode.SoftFill)
                        {
                            SoftFill(over, data, x0, y0, roiW, roiH, roiDown, color, aEffBase, maskThr, fillMask);
                        }

                        // 6) 박스/배지(선택)
                        if (drawBoxes)
                            DrawBoxOnLockedBitmap(data, origBox, color, Math.Max(1, lineThickness), over.Width, over.Height);

                        if (drawScores && overlaysOut != null)
                        {
                            overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem
                            {
                                Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Badge,
                                Text = "[" + d.ClassId + "]: " + d.Score.ToString("0.00"),
                                BoxImg = origBox,
                                StrokeColor = color
                            });
                        }
                    }
                }
                finally
                {
                    ArrayPool<float>.Shared.Return(maskBuf);
                }
            }
            finally
            {
                over.UnlockBits(data);
            }

            return over;
        }

        // ===== SoftFill =====
        private static unsafe void SoftFill(
            Bitmap over, BitmapData data, int x0, int y0, int roiW, int roiH,
            float[] roiDown, Color color, float alpha, float maskThr, bool fillMask)
        {
            if (!fillMask) return;

            byte cr = color.R, cg = color.G, cb = color.B;
            byte* basePtr = (byte*)data.Scan0;
            int stride = data.Stride;

            Parallel.For(0, roiH, delegate (int ddy)
            {
                int y = y0 + ddy;
                byte* row = basePtr + y * stride;

                for (int ddx = 0; ddx < roiW; ddx++)
                {
                    float v = roiDown[ddy * roiW + ddx];

                    float aEff = USE_SOFT_ALPHA
                        ? alpha * SmoothStep(SOFT_T0, SOFT_T1, v)
                        : ((v >= maskThr) ? alpha : 0f);

                    if (aEff <= 0f) continue;

                    int x = x0 + ddx;
                    byte* px = row + x * 4; // BGRA
                    int b = px[0], g = px[1], rch = px[2];
                    px[2] = (byte)(rch + (cr - rch) * aEff);
                    px[1] = (byte)(g + (cg - g) * aEff);
                    px[0] = (byte)(b + (cb - b) * aEff);
                }
            });
        }

        // ===== Marching Squares (간단 등고선) =====
        private static List<PointF> ExtractIsoContour(float[] f, int w, int h, float thr)
        {
            List<PointF> pts = new List<PointF>(w + h);

            for (int y = 0; y < h - 1; y++)
            {
                for (int x = 0; x < w - 1; x++)
                {
                    int i00 = y * w + x, i10 = i00 + 1, i01 = i00 + w, i11 = i01 + 1;
                    float v00 = f[i00] - thr, v10 = f[i10] - thr, v01 = f[i01] - thr, v11 = f[i11] - thr;
                    int mask = ((v00 > 0) ? 1 : 0) | ((v10 > 0) ? 2 : 0) | ((v01 > 0) ? 4 : 0) | ((v11 > 0) ? 8 : 0);
                    if (mask == 0 || mask == 15) continue;

                    // edge t 보간: a / (a - b)
                    Func<float, float, float> LerpT = delegate (float a, float b) { return a / (a - b); };

                    PointF? eL = null, eR = null, eT = null, eB = null;
                    if ((((mask & 1) != 0) != ((mask & 4) != 0)))
                    {
                        float t = LerpT(v00, v01);
                        eL = new PointF(x, y + t);
                    }
                    if ((((mask & 2) != 0) != ((mask & 8) != 0)))
                    {
                        float t = LerpT(v10, v11);
                        eR = new PointF(x + 1, y + t);
                    }
                    if ((((mask & 1) != 0) != ((mask & 2) != 0)))
                    {
                        float t = LerpT(v00, v10);
                        eT = new PointF(x + t, y);
                    }
                    if ((((mask & 4) != 0) != ((mask & 8) != 0)))
                    {
                        float t = LerpT(v01, v11);
                        eB = new PointF(x + t, y + 1);
                    }

                    if (eT.HasValue) pts.Add(eT.Value);
                    if (eR.HasValue) pts.Add(eR.Value);
                    if (eB.HasValue) pts.Add(eB.Value);
                    if (eL.HasValue) pts.Add(eL.Value);
                }
            }
            return pts;
        }

        // Chaikin corner cutting
        private static List<PointF> ChaikinSmooth(IReadOnlyList<PointF> src, int iters, bool closed)
        {
            if (src == null || src.Count < 4) return (src != null) ? new List<PointF>(src) : new List<PointF>();
            List<PointF> cur = new List<PointF>(src);

            for (int it = 0; it < iters; it++)
            {
                List<PointF> nxt = new List<PointF>(cur.Count * 2);
                int n = cur.Count;
                int end = closed ? n : n - 1;

                for (int i = 0; i < end; i++)
                {
                    PointF p0 = cur[i];
                    PointF p1 = cur[(i + 1) % n];
                    PointF q = new PointF(p0.X * 0.75f + p1.X * 0.25f, p0.Y * 0.75f + p1.Y * 0.25f);
                    PointF r = new PointF(p0.X * 0.25f + p1.X * 0.75f, p0.Y * 0.25f + p1.Y * 0.75f);
                    nxt.Add(q); nxt.Add(r);
                }
                if (!closed) nxt.Add(cur[cur.Count - 1]);
                cur = nxt;
            }
            return cur;
        }

        // 간단 Kasa 원 맞춤(ROI 좌표계) — out 파라미터로 반환(C# 7.3 호환)
        private static void FitCircleKasa(float[] f, int w, int h, float tLow, out float cx, out float cy, out float r)
        {
            double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0, sr = 0, n = 0;

            for (int y = 0; y < h; y++)
            {
                int row = y * w;
                for (int x = 0; x < w; x++)
                {
                    float v = f[row + x];
                    if (v < tLow) continue;

                    double xd = x, yd = y;
                    sx += xd; sy += yd;
                    sxx += xd * xd; syy += yd * yd;
                    sxy += xd * yd;
                    sr += (xd * xd + yd * yd);
                    n += 1.0;
                }
            }

            if (n < 10)
            {
                cx = w * 0.5f; cy = h * 0.5f; r = Math.Min(w, h) * 0.45f; return;
            }

            double A11 = sxx, A12 = sxy, A21 = sxy, A22 = syy;
            double B1 = sr * 0.5, B2 = sr * 0.5;
            double det = (A11 * A22 - A12 * A21);

            if (Math.Abs(det) < 1e-6)
            {
                cx = w * 0.5f; cy = h * 0.5f; r = Math.Min(w, h) * 0.45f; return;
            }

            double cxD = (B1 * A22 - B2 * A12) / det;
            double cyD = (A11 * B2 - A21 * B1) / det;

            double rSum = 0, cnt = 0;
            for (int y = 0; y < h; y++)
            {
                int row = y * w;
                for (int x = 0; x < w; x++)
                {
                    float v = f[row + x];
                    if (v < tLow) continue;
                    double dx = x - cxD, dy = y - cyD;
                    rSum += Math.Sqrt(dx * dx + dy * dy);
                    cnt++;
                }
            }

            cx = (float)cxD; cy = (float)cyD;
            r = (cnt > 0) ? (float)(rSum / cnt) : Math.Min(w, h) * 0.45f;
        }

        // 원형 소프트 경계 채우기(BGRA)
        private static unsafe void BlendFilledCircleOnLockedBitmap(
            BitmapData data, int x0, int y0,
            float cx, float cy, float radius,
            Color color, float alphaOuter, float edgeSoftnessPx)
        {
            byte* basePtr = (byte*)data.Scan0;
            int stride = data.Stride;
            byte cr = color.R, cg = color.G, cb = color.B;

            int xmin = Math.Max(0, x0 + (int)Math.Floor(cx - radius - 2));
            int xmax = Math.Min(data.Width - 1, x0 + (int)Math.Ceiling(cx + radius + 2));
            int ymin = Math.Max(0, y0 + (int)Math.Floor(cy - radius - 2));
            int ymax = Math.Min(data.Height - 1, y0 + (int)Math.Ceiling(cy + radius + 2));

            float R = radius;
            float edge = (edgeSoftnessPx < 0.5f) ? 0.5f : edgeSoftnessPx;

            for (int y = ymin; y <= ymax; y++)
            {
                byte* row = basePtr + y * stride;
                for (int x = xmin; x <= xmax; x++)
                {
                    float rx = (x - x0) - cx + 0.5f;
                    float ry = (y - y0) - cy + 0.5f;
                    float d = Sqrtf(rx * rx + ry * ry);

                    float a = 0f;
                    if (d <= R - edge) a = alphaOuter;
                    else if (d <= R + edge)
                    {
                        float t = 1f - ((d - (R - edge)) / (2f * edge)); // 1..0
                        if (t < 0f) t = 0f; else if (t > 1f) t = 1f;
                        a = alphaOuter * SmoothStep(0f, 1f, t);
                    }
                    if (a <= 0f) continue;

                    byte* p = row + x * 4;
                    int b = p[0], g = p[1], rch = p[2];
                    p[2] = (byte)(rch + (cr - rch) * a);
                    p[1] = (byte)(g + (cg - g) * a);
                    p[0] = (byte)(b + (cb - b) * a);
                }
            }
        }

        // 간단 3-탭 가우시안(σ≈1) 분리형
        private static void GaussianBlurInplace(float[] src, int w, int h)
        {
            float[] k = { 0.27901f, 0.44198f, 0.27901f };
            float[] tmp = new float[src.Length];

            // 가로
            Parallel.For(0, h, delegate (int y)
            {
                int row = y * w;
                for (int x = 0; x < w; x++)
                {
                    float s = 0f;
                    for (int i = -1; i <= 1; i++)
                    {
                        int xx = ClampInt(x + i, 0, w - 1);
                        s += src[row + xx] * k[i + 1];
                    }
                    tmp[row + x] = s;
                }
            });

            // 세로
            Parallel.For(0, w, delegate (int x)
            {
                for (int y = 0; y < h; y++)
                {
                    float s = 0f;
                    for (int i = -1; i <= 1; i++)
                    {
                        int yy = ClampInt(y + i, 0, h - 1);
                        s += tmp[yy * w + x] * k[i + 1];
                    }
                    src[y * w + x] = s;
                }
            });
        }

        // 사각 박스 테두리(선택)
        private static unsafe void DrawBoxOnLockedBitmap(
            BitmapData data, Rectangle box, Color color, int lineThickness,
            int imgW, int imgH)
        {
            int left = Math.Max(0, box.Left);
            int top = Math.Max(0, box.Top);
            int right = Math.Min(imgW - 1, box.Right - 1);
            int bottom = Math.Min(imgH - 1, box.Bottom - 1);
            if (left > right || top > bottom) return;

            byte* basePtr = (byte*)data.Scan0;
            int stride = data.Stride;
            byte cr = color.R, cg = color.G, cb = color.B;
            int half = Math.Max(1, lineThickness) / 2;

            for (int yEdge = -half; yEdge <= half; yEdge++)
            {
                int yTop = top + yEdge;
                int yBot = bottom + yEdge;
                if (yTop >= 0 && yTop < imgH)
                {
                    byte* row = basePtr + yTop * stride;
                    for (int x = left; x <= right; x++) { byte* p = row + x * 4; p[2] = cr; p[1] = cg; p[0] = cb; }
                }
                if (yBot >= 0 && yBot < imgH)
                {
                    byte* row = basePtr + yBot * stride;
                    for (int x = left; x <= right; x++) { byte* p = row + x * 4; p[2] = cr; p[1] = cg; p[0] = cb; }
                }
            }

            for (int xEdge = -half; xEdge <= half; xEdge++)
            {
                int xL = left + xEdge;
                int xR = right + xEdge;
                if (xL >= 0 && xL < imgW)
                {
                    for (int y = top; y <= bottom; y++) { byte* p = basePtr + y * stride + xL * 4; p[2] = cr; p[1] = cg; p[0] = cb; }
                }
                if (xR >= 0 && xR < imgW)
                {
                    for (int y = top; y <= bottom; y++) { byte* p = basePtr + y * stride + xR * 4; p[2] = cr; p[1] = cg; p[0] = cb; }
                }
            }
        }
    }
}
