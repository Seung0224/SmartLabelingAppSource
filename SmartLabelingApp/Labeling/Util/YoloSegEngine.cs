// YoloSegEngine.cs
// - TensorRT C API (trt_*)를 P/Invoke로 호출하여 Yolov11-seg .engine 추론 + 오버레이
// - .NET 6+ / x64 / Windows (System.Drawing.Common)
// - 필수: tensorrt_runner.dll (trt_create_engine 등 6개 함수)

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Runtime.InteropServices;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public sealed class YoloSegEngine : IDisposable
    {
        #region P/Invoke (TensorRT DLL)

        const string DllName = "TensorRTRunner"; // tensorrt_runner.dll

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern IntPtr trt_create_engine(string enginePath, int deviceId);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trt_destroy_engine(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_get_input_size(IntPtr handle); // 정사각 고정이면 반환, 동적이면 -1

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_get_mask_meta(IntPtr handle, out int segDim, out int maskH, out int maskW);

        // int trt_infer(handle, float* nchw, int len, float** detOut, int* nDet, int* detC, float** protoOut, int* segDim, int* maskH, int* maskW)
        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_infer(
            IntPtr handle,
            float[] nchw, int nchwLength,
            out IntPtr detOut, out int nDet, out int detC,
            out IntPtr protoOut, out int segDim, out int maskH, out int maskW);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trt_free(IntPtr p);

        #endregion

        #region Types (결과 형식은 ONNX 버전과 호환)

        public sealed class SegResult
        {
            public int NetSize { get; set; }
            public float Scale { get; set; }
            public int PadX { get; set; }
            public int PadY { get; set; }
            public Size Resized { get; set; }
            public Size OrigSize { get; set; }

            public int SegDim { get; set; }
            public int MaskH { get; set; }
            public int MaskW { get; set; }
            public float[] ProtoFlat { get; set; }

            public List<Det> Dets { get; set; }

            public double PreMs { get; set; }
            public double InferMs { get; set; }
            public double PostMs { get; set; }
            public double TotalMs { get; set; }
        }

        public sealed class Det
        {
            public RectangleF Box;   // net 좌표 (ltrb)
            public float Score;
            public int ClassId;
            public float[] Coeff;    // 길이 = SegDim
        }

        #endregion

        private IntPtr _h = IntPtr.Zero;
        private readonly int _deviceId;
        private int _cachedInputNet = -1;        // 엔진 고정 입력(정사각) 추론용 힌트
        private float[] _inBuf = null;           // [1,3,net,net]
        [ThreadStatic] private static float[] _maskBufTLS;

        public YoloSegEngine(string enginePath, int deviceId = 0)
        {
            _deviceId = deviceId;
            _h = trt_create_engine(enginePath, deviceId);
            if (_h == IntPtr.Zero) throw new InvalidOperationException("TensorRT 엔진 생성 실패");

            _cachedInputNet = trt_get_input_size(_h); // -1이면 동적
        }

        public void Dispose()
        {
            if (_h != IntPtr.Zero)
            {
                trt_destroy_engine(_h);
                _h = IntPtr.Zero;
            }
        }

        #region Public API

        /// <summary>
        /// Bitmap 한 장을 넣어 SegResult(박스/점수/클래스/coeff & proto) 반환
        /// </summary>
        public SegResult Infer(Bitmap orig, float conf = 0.9f, float iou = 0.45f)
        {
            if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloSegEngine));
            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0, tPost = 0;

            // ---- 1) 입력 net 크기 결정 (엔진이 고정이면 그 값, 아니면 보통 640)
            int net = _cachedInputNet > 0 ? _cachedInputNet : 640;

            // ---- 2) 레터박스 전처리: R,G,B -> [0..1] float, NCHW
            EnsureInputBuffer(net);
            FillTensorFromBitmap(orig, net, _inBuf, out float scale, out int padX, out int padY, out Size resized);
            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

            // ---- 3) 추론 호출(네이티브)
            IntPtr detPtr = IntPtr.Zero, protoPtr = IntPtr.Zero;
            int nDet = 0, detC = 0, segDim = 0, mw = 0, mh = 0;
            int ok = trt_infer(_h, _inBuf, _inBuf.Length, out detPtr, out nDet, out detC, out protoPtr, out segDim, out mh, out mw);
            if (ok == 0) throw new InvalidOperationException("trt_infer 실패");
            tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

            // ---- 4) 네이티브 버퍼를 관리되는 배열로 복사
            float[] proto = new float[segDim * mw * mh];
            Marshal.Copy(protoPtr, proto, 0, proto.Length);
            trt_free(protoPtr);

            float[] detFlat = Array.Empty<float>();
            if (nDet > 0 && detC > 0)
            {
                detFlat = new float[nDet * detC];
                Marshal.Copy(detPtr, detFlat, 0, detFlat.Length);
            }
            trt_free(detPtr);

            // ---- 5) det 파싱 (형식: [x,y,w,h] + [numClasses] + [segDim])
            //       (TensorRT 엔진은 보통 [N,M,C] 형태로 나오며 여기서는 [M,C]로 펴서 받음)
            var dets = new List<Det>(nDet);
            int numClasses = detC - 4 - segDim;
            if (numClasses <= 0) numClasses = 1; // 안전망

            for (int i = 0; i < nDet; i++)
            {
                int baseIdx = i * detC;
                float x = detFlat[baseIdx + 0];
                float y = detFlat[baseIdx + 1];
                float w = detFlat[baseIdx + 2];
                float h = detFlat[baseIdx + 3];

                // best class
                int bestC = 0; float bestS = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float raw = detFlat[baseIdx + 4 + c];
                    float s = (raw < 0f || raw > 1f) ? Sigmoid(raw) : raw;
                    if (s > bestS) { bestS = s; bestC = c; }
                }
                if (bestS < conf) continue;

                var coeff = new float[segDim];
                int coeffBase = baseIdx + 4 + numClasses;
                Array.Copy(detFlat, coeffBase, coeff, 0, segDim);

                // center → ltrb
                float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, b = y + h / 2f;
                dets.Add(new Det { Box = new RectangleF(l, t, r - l, b - t), Score = bestS, ClassId = bestC, Coeff = coeff });
            }

            // ---- 6) NMS
            dets = Nms(dets, iou);

            tPost = sw.Elapsed.TotalMilliseconds - tPrev;

            return new SegResult
            {
                NetSize = net,
                Scale = scale,
                PadX = padX,
                PadY = padY,
                Resized = resized,
                OrigSize = orig.Size,
                SegDim = segDim,
                MaskH = mh,
                MaskW = mw,
                ProtoFlat = proto,
                Dets = dets,
                PreMs = tPre,
                InferMs = tInfer,
                PostMs = tPost,
                TotalMs = sw.Elapsed.TotalMilliseconds
            };
        }

        /// <summary>빠른 오버레이(ROI에서 바로 보간·블렌딩)</summary>
        public Bitmap OverlayFast(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false)
        {
            var over = (Bitmap)orig.Clone();
            if (r.Dets == null || r.Dets.Count == 0) return over;

            var rectAll = new Rectangle(0, 0, over.Width, over.Height);
            var data = over.LockBits(rectAll, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                foreach (var d in r.Dets)
                {
                    int maskLen = r.MaskW * r.MaskH;
                    if (_maskBufTLS == null || _maskBufTLS.Length < maskLen)
                        _maskBufTLS = new float[maskLen];

                    ComputeMask_NoAlloc(d.Coeff, r.ProtoFlat, r.SegDim, r.MaskW, r.MaskH, _maskBufTLS);

                    var box = NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);
                    BlendMaskIntoOrigROI(data, over.Width, over.Height, box, _maskBufTLS, r.MaskW, r.MaskH, r.NetSize, r.Scale, r.PadX, r.PadY, maskThr, alpha, ClassColor(d.ClassId));

                    if (drawBoxes)
                    {
                        var g = Graphics.FromImage(over);
                        var p = new Pen(ClassColor(d.ClassId), 2f);
                        g.DrawRectangle(p, box);
                    }
                }
            }
            finally { over.UnlockBits(data); }

            return over;
        }

        #endregion

        #region Preprocess / Math helpers (ONNX 흐름과 동일)

        private void EnsureInputBuffer(int net)
        {
            int need = 1 * 3 * net * net;
            if (_inBuf == null || _inBuf.Length != need) _inBuf = new float[need];
        }

        private static void FillTensorFromBitmap(Bitmap src, int net, float[] outNCHW, out float scale, out int padX, out int padY, out Size resized)
        {
            int W = src.Width, H = src.Height;
            scale = Math.Min((float)net / W, (float)net / H);
            int rw = (int)Math.Round(W * scale);
            int rh = (int)Math.Round(H * scale);
            padX = (net - rw) / 2;
            padY = (net - rh) / 2;
            resized = new Size(rw, rh);

            var tmp = new Bitmap(net, net, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(tmp))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                g.DrawImage(src, padX, padY, rw, rh);
            }

            var rect = new Rectangle(0, 0, net, net);
            var bd = tmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* p0 = (byte*)bd.Scan0;
                    int stride = bd.Stride;
                    float inv255 = 1f / 255f;
                    int plane = net * net;
                    for (int y = 0; y < net; y++)
                    {
                        byte* row = p0 + y * stride;
                        for (int x = 0; x < net; x++)
                        {
                            byte b = row[x * 3 + 0];
                            byte g = row[x * 3 + 1];
                            byte r = row[x * 3 + 2];
                            int idx = y * net + x;
                            outNCHW[0 * plane + idx] = r * inv255;
                            outNCHW[1 * plane + idx] = g * inv255;
                            outNCHW[2 * plane + idx] = b * inv255;
                        }
                    }
                }
            }
            finally { tmp.UnlockBits(bd); }
        }

        private static Rectangle NetBoxToOriginal(RectangleF netBox, float scale, int padX, int padY, Size resized, Size orig)
        {
            float l = (netBox.Left - padX) / scale;
            float t = (netBox.Top - padY) / scale;
            float r = (netBox.Right - padX) / scale;
            float b = (netBox.Bottom - padY) / scale;
            int x0 = Clamp((int)Math.Round(l), 0, orig.Width);
            int y0 = Clamp((int)Math.Round(t), 0, orig.Height);
            int x1 = Clamp((int)Math.Round(r), 0, orig.Width);
            int y1 = Clamp((int)Math.Round(b), 0, orig.Height);
            return Rectangle.FromLTRB(x0, y0, x1, y1);
        }

        private static int Clamp(int v, int lo, int hi) => v < lo ? lo : (v > hi ? hi : v);
        private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));

        private static List<Det> Nms(List<Det> dets, float iouThr)
        {
            if (dets.Count <= 1) return dets;
            var order = dets
                .Select((d, i) => (i, d.Score))
                .OrderByDescending(t => t.Score)
                .Select(t => t.i).ToList();

            var keep = new List<Det>();
            var suppressed = new bool[dets.Count];

            for (int _i = 0; _i < order.Count; _i++)
            {
                int i = order[_i];
                if (suppressed[i]) continue;
                keep.Add(dets[i]);

                var a = dets[i].Box;
                float aArea = a.Width * a.Height;

                for (int _j = _i + 1; _j < order.Count; _j++)
                {
                    int j = order[_j];
                    if (suppressed[j]) continue;
                    var b = dets[j].Box;

                    float xx0 = Math.Max(a.Left, b.Left);
                    float yy0 = Math.Max(a.Top, b.Top);
                    float xx1 = Math.Min(a.Right, b.Right);
                    float yy1 = Math.Min(a.Bottom, b.Bottom);
                    float w = Math.Max(0, xx1 - xx0);
                    float h = Math.Max(0, yy1 - yy0);
                    float inter = w * h;
                    float ovr = inter / (aArea + b.Width * b.Height - inter + 1e-6f);
                    if (ovr > iouThr) suppressed[j] = true;
                }
            }
            return keep;
            // 참고: ONNX 버전 로직과 동일한 흐름. :contentReference[oaicite:1]{index=1}
        }

        #endregion

        #region Mask & Overlay (빠른 경로)

        // mask = sigmoid( sum_k proto[k]*coeff[k] ), protoFlat: [segDim, mh, mw]
        private static void ComputeMask_NoAlloc(float[] coeff, float[] protoFlat, int segDim, int mw, int mh, float[] maskOut)
        {
            int vec = Vector<float>.Count;
            Parallel.For(0, mh, y =>
            {
                int rowOff = y * mw;
                int x = 0;
                for (; x <= mw - vec; x += vec)
                {
                    var sumV = new Vector<float>(0f);
                    for (int k = 0; k < segDim; k++)
                    {
                        int off = ((k * mh + y) * mw) + x;
                        var pV = new Vector<float>(protoFlat, off);
                        var cV = new Vector<float>(coeff[k]);
                        sumV += pV * cV;
                    }
                    for (int t = 0; t < vec; t++)
                    {
                        float s = sumV[t];
                        maskOut[rowOff + x + t] = 1f / (1f + (float)Math.Exp(-s));
                    }
                }
                for (; x < mw; x++)
                {
                    float sum = 0f;
                    for (int k = 0; k < segDim; k++) sum += coeff[k] * protoFlat[(k * mh + y) * mw + x];
                    maskOut[rowOff + x] = 1f / (1f + (float)Math.Exp(-sum));
                }
            });
            // 원본 ONNX 파일의 SIMD 경로를 동일 컨셉으로 이식. :contentReference[oaicite:2]{index=2}
        }

        private static void BlendMaskIntoOrigROI(
            BitmapData data, int imgW, int imgH,
            Rectangle boxOrig, float[] mask, int mw, int mh, int netSize,
            float scale, int padX, int padY,
            float thr, float alpha, Color color)
        {
            float sx = (float)mw / netSize;
            float sy = (float)mh / netSize;

            int x0 = Math.Max(0, boxOrig.Left);
            int y0 = Math.Max(0, boxOrig.Top);
            int x1 = Math.Min(imgW, boxOrig.Right);
            int y1 = Math.Min(imgH, boxOrig.Bottom);
            if (x0 >= x1 || y0 >= y1) return;

            byte rr = color.R, gg = color.G, bb = color.B;
            float a = Math.Max(0f, Math.Min(1f, alpha));
            float ia = 1f - a;
            float thrF = Math.Max(0f, Math.Min(1f, thr));

            unsafe
            {
                byte* basePtr = (byte*)data.Scan0;
                int stride = data.Stride;

                int roiW = x1 - x0, roiH = y1 - y0;
                var u0a = new int[roiW]; var fu0a = new float[roiW]; var fu1a = new float[roiW];
                for (int i = 0; i < roiW; i++)
                {
                    float u = (((x0 + i) * scale) + padX) * sx;
                    int u0 = (int)Math.Floor(u);
                    float fu = u - u0;
                    u0a[i] = u0; fu0a[i] = 1 - fu; fu1a[i] = fu;
                }
                var v0a = new int[roiH]; var fv0a = new float[roiH]; var fv1a = new float[roiH];
                for (int j = 0; j < roiH; j++)
                {
                    float v = (((y0 + j) * scale) + padY) * sy;
                    int v0 = (int)Math.Floor(v);
                    float fv = v - v0;
                    v0a[j] = v0; fv0a[j] = 1 - fv; fv1a[j] = fv;
                }

                for (int j = 0; j < roiH; j++)
                {
                    int y = y0 + j;
                    int v0 = v0a[j]; if ((uint)v0 >= (uint)mh - 1) continue;
                    float wv0 = fv0a[j], wv1 = fv1a[j];

                    byte* row = basePtr + y * stride + (x0 * 4);
                    int base00 = v0 * mw;
                    int base01 = base00 + mw;

                    for (int i = 0; i < roiW; i++)
                    {
                        int u0 = u0a[i]; if ((uint)u0 >= (uint)mw - 1) { row += 4; continue; }
                        float wu0 = fu0a[i], wu1 = fu1a[i];

                        float m00 = mask[base00 + u0];
                        float m10 = mask[base00 + u0 + 1];
                        float m01 = mask[base01 + u0];
                        float m11 = mask[base01 + u0 + 1];

                        float mx0 = m00 * wu0 + m10 * wu1;
                        float mx1 = m01 * wu0 + m11 * wu1;
                        float m = mx0 * wv0 + mx1 * wv1;

                        if (m < thrF) { row += 4; continue; }

                        row[0] = (byte)(row[0] * ia + bb * a);
                        row[1] = (byte)(row[1] * ia + gg * a);
                        row[2] = (byte)(row[2] * ia + rr * a);
                        row[3] = 255;
                        row += 4;
                    }
                }
            }
        }

        private static Color ClassColor(int id)
        {
            // 간단한 고정 팔레트
            Color[] tbl = { Color.Lime, Color.DeepSkyBlue, Color.Orange, Color.HotPink, Color.Gold, Color.MediumOrchid };
            return tbl[id % tbl.Length];
        }

        #endregion
    }
}
