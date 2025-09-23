using System;
using System.Buffers;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public static class OverlayRendererFast
    {
        /// <summary>
        /// 공통 오버레이 렌더러
        /// - fillMask: 비트맵에 직접 마스크 채움(알파 블렌딩)
        /// - drawOutlines: 외곽선 픽셀을 비트맵에 직접 그림
        /// - drawBoxes: 박스 테두리를 비트맵에 직접 그림
        /// - drawScores: 배지(텍스트)는 비트맵에 그리지 않고, overlaysOut(이미지캔버스)로만 생성
        /// - lineThickness: 박스 테두리와 외곽선 두께(픽셀)
        /// - scoreScale: (비사용; ImageCanvas가 배지 스타일을 담당) 유지만
        /// - overlaysOut: ImageCanvas로 전달할 배지 오버레이를 담을 리스트 (null이면 배지 생성 생략)
        /// - labelProvider: clsId -> 라벨 문자열 (null이면 "cls:{id}")
        /// </summary>
        public static Bitmap RenderEx(Bitmap orig, SegResult r, float maskThr = 0.8f, float alpha = 0.45f, bool drawBoxes = false, bool fillMask = true, bool drawOutlines = true,
            bool drawScores = true, int lineThickness = 1, List<SmartLabelingApp.ImageCanvas.OverlayItem> overlaysOut = null)
        {
            if (orig == null) throw new ArgumentNullException(nameof(orig));
            if (r == null) throw new ArgumentNullException(nameof(r));
            if (r.Dets == null || r.Dets.Count == 0) return (Bitmap)orig.Clone();

            var over = (Bitmap)orig.Clone();
            var rectAll = new Rectangle(0, 0, over.Width, over.Height);
            var data = over.LockBits(rectAll, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            try
            {
                int maskLenFull = r.MaskW * r.MaskH;
                float[] maskBuf = ArrayPool<float>.Shared.Rent(maskLenFull);
                try
                {
                    // 최소 두께 보정
                    int thick = Math.Max(1, lineThickness);

                    for (int di = 0; di < r.Dets.Count; di++)
                    {
                        var d = r.Dets[di];

                        // 1) 마스크 합성(KHW) → maskBuf에 기록
                        MaskSynth.ComputeMask_KHW_NoAlloc(d.Coeff, r.ProtoFlat, r.SegDim, r.MaskW, r.MaskH, maskBuf);

                        // 2) 박스 원본 좌표
                        var origBox = Postprocess.NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);
                        if (origBox == Rectangle.Empty) continue;

                        // 3) ROI 보간 준비
                        int imgW = over.Width, imgH = over.Height;
                        int x0 = Math.Max(0, origBox.X);
                        int y0 = Math.Max(0, origBox.Y);
                        int x1 = Math.Min(imgW, origBox.Right);
                        int y1 = Math.Min(imgH, origBox.Bottom);
                        if (x1 <= x0 || y1 <= y0) continue;

                        int roiW = x1 - x0, roiH = y1 - y0;
                        var ix0 = new int[roiW]; var ix1 = new int[roiW]; var tx = new float[roiW];
                        var iy0 = new int[roiH]; var iy1 = new int[roiH]; var ty = new float[roiH];

                        float scaleW = (float)r.MaskW / r.NetSize;
                        float scaleH = (float)r.MaskH / r.NetSize;

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

                        // 4) 채움/윤곽선
                        bool[] bin = (drawOutlines ? new bool[roiW * roiH] : null);
                        Color color = ColorUtils.ClassColor(d.ClassId);
                        byte cr = color.R, cg = color.G, cb = color.B;
                        float a = MathUtils.Clamp(alpha, 0f, 1f);

                        unsafe
                        {
                            byte* basePtr = (byte*)data.Scan0;
                            int stride = data.Stride;

                            Parallel.For(0, roiH, ddy =>
                            {
                                int y = y0 + ddy;
                                byte* row = basePtr + y * stride;
                                int mY0 = iy0[ddy] * r.MaskW;
                                int mY1 = iy1[ddy] * r.MaskW;
                                float wy = ty[ddy];
                                float wy0 = 1f - wy, wy1 = wy;

                                for (int ddx = 0; ddx < roiW; ddx++)
                                {
                                    int x = x0 + ddx;
                                    int mX0 = ix0[ddx];
                                    int mX1 = ix1[ddx];
                                    float wx = tx[ddx];
                                    float wx0 = 1f - wx, wx1 = wx;

                                    float v00 = maskBuf[mY0 + mX0];
                                    float v10 = maskBuf[mY0 + mX1];
                                    float v01 = maskBuf[mY1 + mX0];
                                    float v11 = maskBuf[mY1 + mX1];
                                    float vy0 = v00 * wx0 + v10 * wx1;
                                    float vy1 = v01 * wx0 + v11 * wx1;
                                    float v = vy0 * wy0 + vy1 * wy1;

                                    bool pass = (v >= maskThr);
                                    if (bin != null) bin[ddy * roiW + ddx] = pass;

                                    if (!fillMask || !pass) continue;

                                    byte* px = row + x * 4; // BGRA
                                    int b = px[0], g = px[1], rch = px[2];
                                    px[2] = (byte)(rch + (cr - rch) * a);
                                    px[1] = (byte)(g + (cg - g) * a);
                                    px[0] = (byte)(b + (cb - b) * a);
                                }
                            });
                        }

                        if (drawOutlines && bin != null)
                            DrawOutlineOnLockedBitmap(data, x0, y0, roiW, roiH, bin, color, thick);

                        // 5) 박스 테두리(픽셀) 그리기
                        if (drawBoxes)
                            DrawBoxOnLockedBitmap(data, origBox, color, thick, over.Width, over.Height);

                        // 6) 배지(ImageCanvas용) 오버레이 생성만 (비트맵에 글씨 그리지 않음)
                        if (drawScores && overlaysOut != null)
                        {
                            overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem
                            {
                                Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Badge,
                                Text = $"[{d.ClassId}]: {d.Score:0.00}",
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

        // ==== 저수준 픽셀 유틸 ====

        // 외곽선(윤곽) 픽셀 칠하기 — lineThickness(>=1) 적용
        private static unsafe void DrawOutlineOnLockedBitmap(
            BitmapData data, int x0, int y0, int roiW, int roiH,
            bool[] bin, Color color, int lineThickness)
        {
            byte* basePtr = (byte*)data.Scan0;
            int stride = data.Stride;
            byte cr = color.R, cg = color.G, cb = color.B;

            int half = lineThickness / 2;

            for (int dy = 0; dy < roiH; dy++)
            {
                for (int dx = 0; dx < roiW; dx++)
                {
                    int idx = dy * roiW + dx;
                    if (!bin[idx]) continue;

                    bool edge = false;
                    // 4-neighbor
                    if (dx == 0 || !bin[idx - 1]) edge = true;
                    else if (dx == roiW - 1 || !bin[idx + 1]) edge = true;
                    else if (dy == 0 || !bin[idx - roiW]) edge = true;
                    else if (dy == roiH - 1 || !bin[idx + roiW]) edge = true;

                    if (!edge) continue;

                    // 두께 만큼 주변 픽셀도 채우기 (정사각 커널)
                    int pxX = x0 + dx;
                    int pxY = y0 + dy;

                    for (int oy = -half; oy <= half; oy++)
                    {
                        int yy = pxY + oy;
                        if (yy < 0 || yy >= data.Height) continue;

                        byte* row = basePtr + yy * stride;

                        for (int ox = -half; ox <= half; ox++)
                        {
                            int xx = pxX + ox;
                            if (xx < 0 || xx >= data.Width) continue;

                            byte* p = row + xx * 4; // BGRA
                            p[2] = cr; p[1] = cg; p[0] = cb;
                        }
                    }
                }
            }
        }

        // 직사각형 박스 테두리를 픽셀로 그리기 — lineThickness(>=1) 적용
        private static unsafe void DrawBoxOnLockedBitmap(
            BitmapData data, Rectangle box, Color color, int lineThickness,
            int imgW, int imgH)
        {
            // 경계 클램프
            int left = Math.Max(0, box.Left);
            int top = Math.Max(0, box.Top);
            int right = Math.Min(imgW - 1, box.Right - 1);
            int bottom = Math.Min(imgH - 1, box.Bottom - 1);

            if (left > right || top > bottom) return;

            byte* basePtr = (byte*)data.Scan0;
            int stride = data.Stride;
            byte cr = color.R, cg = color.G, cb = color.B;

            int half = lineThickness / 2;

            // 수평선(상/하)
            for (int yEdge = -half; yEdge <= half; yEdge++)
            {
                int yTop = top + yEdge;
                int yBot = bottom + yEdge;
                if (yTop >= 0 && yTop < imgH)
                {
                    byte* row = basePtr + yTop * stride;
                    for (int x = left; x <= right; x++)
                    {
                        byte* p = row + x * 4; p[2] = cr; p[1] = cg; p[0] = cb;
                    }
                }
                if (yBot >= 0 && yBot < imgH)
                {
                    byte* row = basePtr + yBot * stride;
                    for (int x = left; x <= right; x++)
                    {
                        byte* p = row + x * 4; p[2] = cr; p[1] = cg; p[0] = cb;
                    }
                }
            }

            // 수직선(좌/우)
            for (int xEdge = -half; xEdge <= half; xEdge++)
            {
                int xL = left + xEdge;
                int xR = right + xEdge;
                if (xL >= 0 && xL < imgW)
                {
                    for (int y = top; y <= bottom; y++)
                    {
                        byte* p = basePtr + y * stride + xL * 4; p[2] = cr; p[1] = cg; p[0] = cb;
                    }
                }
                if (xR >= 0 && xR < imgW)
                {
                    for (int y = top; y <= bottom; y++)
                    {
                        byte* p = basePtr + y * stride + xR * 4; p[2] = cr; p[1] = cg; p[0] = cb;
                    }
                }
            }
        }
    }
}
