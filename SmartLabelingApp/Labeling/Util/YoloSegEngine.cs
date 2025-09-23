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
        public SegResult Infer(Bitmap orig, float conf = 0.5f, float iou = 0.45f)
        {
            if (_h == IntPtr.Zero) throw new ObjectDisposedException(nameof(YoloSegEngine));
            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0, tPost = 0;

            int net = _cachedInputNet > 0 ? _cachedInputNet : 640;
            Trace.WriteLine($"[TRT] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            // ---- 1) 전처리
            EnsureInputBuffer(net);
            FillTensorFromBitmap(orig, net, _inBuf, out float scale, out int padX, out int padY, out Size resized);
            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            // ---- 2) 네이티브 추론
            IntPtr detPtr = IntPtr.Zero, protoPtr = IntPtr.Zero;
            int nDet = 0, detC = 0, segDim = 0, mw = 0, mh = 0;
            Trace.WriteLine("[TRT] Calling trt_infer...");
            int ok = trt_infer(_h, _inBuf, _inBuf.Length,
                              out detPtr, out nDet, out detC,
                              out protoPtr, out segDim, out mh, out mw);
            if (ok == 0) throw new InvalidOperationException("trt_infer 실패");
            tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] trt_infer ok | det shape=({nDet},{detC}), proto=({segDim},{mh},{mw}), inferMs={tInfer:F1}");

            // ---- 3) 배열 복사
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

            // ---- 4) 전치 (C,M → M,C 로 변환)
            if (detC > 512 && nDet <= 512)
            {
                int M = detC, C = nDet;
                var transposed = new float[M * C];
                for (int c = 0; c < C; c++)
                    for (int m = 0; m < M; m++)
                        transposed[m * C + c] = detFlat[c * M + m];
                detFlat = transposed;
                nDet = M;
                detC = C;
                Trace.WriteLine($"[TRT] DET transposed -> shape {nDet}x{detC}");
            }

            // ---- 5) ONNX 스타일 헤더 해석
            int channels = detC;
            int nPred = nDet;
            int numClasses = channels - 4 - segDim;
            if (numClasses <= 0)
                throw new InvalidOperationException($"Invalid head layout: channels={channels}, segDim={segDim}");

            Trace.WriteLine($"[TRT] Parse det header | channelsFirst=True, numClasses={numClasses}, segDim={segDim}, nPred={nPred}");

            // ---- 6) coordScale 감지
            float coordScale = 1f;
            float maxWH = 0f;
            int sample = Math.Min(nPred, 128);
            for (int i = 0; i < sample; i++)
            {
                float ww = detFlat[i * detC + 2];
                float hh = detFlat[i * detC + 3];
                if (ww > maxWH) maxWH = ww;
                if (hh > maxWH) maxWH = hh;
            }
            if (maxWH <= 3.5f) coordScale = net;
            Trace.WriteLine($"[TRT] coordScale={coordScale}, sampleMaxWH={maxWH:F2}");

            // ---- 7) 박스 파싱
            var dets = new List<Det>(256);
            int kept = 0;
            for (int i = 0; i < nPred; i++)
            {
                int baseIdx = i * detC;
                if (baseIdx + 4 + numClasses + segDim > detFlat.Length)
                    break;

                float x = detFlat[baseIdx + 0] * coordScale;
                float y = detFlat[baseIdx + 1] * coordScale;
                float w = detFlat[baseIdx + 2] * coordScale;
                float h = detFlat[baseIdx + 3] * coordScale;

                // ---- cls 점수만 (obj 곱 제거)
                int bestC = 0; float bestS = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float raw = detFlat[baseIdx + 4 + c];
                    float s = (raw < 0f || raw > 1f) ? 1f / (1f + (float)Math.Exp(-raw)) : raw;
                    if (s > bestS) { bestS = s; bestC = c; }
                }
                if (bestS < conf) continue;

                // ---- 작은 박스 필터만 유지
                float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, b = y + h / 2f;
                float bw = r - l, bh = b - t;
                if (bw < 2 || bh < 2) continue; // 2픽셀 미만만 스킵

                var coeff = new float[segDim];
                Array.Copy(detFlat, baseIdx + 4 + numClasses, coeff, 0, segDim);

                dets.Add(new Det
                {
                    Box = new RectangleF(l, t, bw, bh),
                    Score = bestS,
                    ClassId = bestC,
                    Coeff = coeff
                });
                kept++;

                if (i < 8)
                    Trace.WriteLine($"[TRT] det[{i}] keep | box=({l:F1},{t:F1},{r:F1},{b:F1}), score={bestS:F3}, cls={bestC}");
            }

            if (dets.Count > 0) dets = Nms(dets, iou);
            tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] Parsed det rows | total={nPred}, kept={kept}, afterNms={dets.Count}, postMs={tPost:F1}");

            var ret = new SegResult
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

            Trace.WriteLine($"[TRT] Infer() end | dets={ret.Dets.Count}, times(ms): pre={tPre:F1}, infer={tInfer:F1}, post={tPost:F1}, total={ret.TotalMs:F1}");
            return ret;
        }





        public Bitmap OverlayFast(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false)
        {
            if (orig == null) throw new ArgumentNullException(nameof(orig));
            if (r == null || r.Dets == null || r.Dets.Count == 0)
            {
                Trace.WriteLine("[TRT] OverlayFast() early return: no dets");
                // orig 그대로 넘기면 Dispose 문제 생김 → 안전하게 Clone 반환
                return (Bitmap)orig.Clone();
            }

            Trace.WriteLine($"[TRT] OverlayFast() start | img={orig.Width}x{orig.Height}, net={r.NetSize}, proto: K={r.SegDim}, mask={r.MaskW}x{r.MaskH}, dets={r.Dets.Count}, thr={maskThr}, alpha={alpha}");

            var over = (Bitmap)orig.Clone();
            var rectAll = new Rectangle(0, 0, over.Width, over.Height);
            var data = over.LockBits(rectAll, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            try
            {
                int maskLen = r.MaskW * r.MaskH;
                if (_maskBufTLS == null || _maskBufTLS.Length < maskLen)
                    _maskBufTLS = new float[maskLen];

                float[] maskKHW = new float[maskLen];
                float[] maskHWK = new float[maskLen];

                int di = 0;
                foreach (var d in r.Dets)
                {
                    // 1) 마스크 생성 (KHW / HWK 모두)
                    ComputeMask_NoAlloc(d.Coeff, r.ProtoFlat, r.SegDim, r.MaskW, r.MaskH, maskKHW);
                    ComputeMask_NoAlloc_HWK(d.Coeff, r.ProtoFlat, r.SegDim, r.MaskW, r.MaskH, maskHWK);

                    // 2) 구조성 비교 → 더 안정적인 마스크 선택
                    double meanK = 0, meanH = 0, varK = 0, varH = 0;
                    for (int i = 0; i < maskLen; i++) { meanK += maskKHW[i]; meanH += maskHWK[i]; }
                    meanK /= maskLen; meanH /= maskLen;
                    for (int i = 0; i < maskLen; i++) { varK += (maskKHW[i] - meanK) * (maskKHW[i] - meanK); varH += (maskHWK[i] - meanH) * (maskHWK[i] - meanH); }
                    varK /= maskLen; varH /= maskLen;

                    bool useKHW = varK >= varH;
                    Array.Copy(useKHW ? maskKHW : maskHWK, _maskBufTLS, maskLen);

                    if (di < 4)
                        Trace.WriteLine($"[TRT] mask stats det[{di}] | KHW var={varK:E2}, HWK var={varH:E2}, choose={(useKHW ? "KHW" : "HWK")}");

                    // 3) netBox → origBox 변환
                    var origBox = NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);

                    if (origBox == Rectangle.Empty)
                    {
                        Trace.WriteLine($"[TRT] overlay det[{di}] skipped (empty box)");
                        di++;
                        continue;
                    }

                    Trace.WriteLine($"[TRT] overlay det[{di}] | score={d.Score:F3}, cls={d.ClassId}, netBox=({d.Box.Left:F1},{d.Box.Top:F1},{d.Box.Right:F1},{d.Box.Bottom:F1}) -> origBox={origBox}");

                    // 4) ROI에 마스크 블렌딩
                    BlendMaskIntoOrigROI(
                        data, over.Width, over.Height,
                        origBox, _maskBufTLS, r.MaskW, r.MaskH, r.NetSize,
                        r.Scale, r.PadX, r.PadY,
                        maskThr, alpha, ClassColor(d.ClassId));

                    // 5) 박스 옵션
                    if (drawBoxes)
                    {
                        using (var g = Graphics.FromImage(over))
                        using (var p = new Pen(ClassColor(d.ClassId), 2f))
                            g.DrawRectangle(p, origBox);
                    }

                    di++;
                }
            }
            finally { over.UnlockBits(data); }

            Trace.WriteLine("[TRT] OverlayFast() end");
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

        private static Rectangle NetBoxToOriginal(
    RectangleF netBox, float scale, int padX, int padY,
    Size resized, Size origSize)
        {
            // net(640) → letterbox 제거 → 원본 좌표
            float invScale = 1f / Math.Max(1e-6f, scale);
            float l = (netBox.Left - padX) * invScale;
            float t = (netBox.Top - padY) * invScale;
            float r = (netBox.Right - padX) * invScale;
            float b = (netBox.Bottom - padY) * invScale;

            // clamp to original image bounds
            l = Math.Max(0, Math.Min(origSize.Width - 1, l));
            r = Math.Max(0, Math.Min(origSize.Width - 1, r));
            t = Math.Max(0, Math.Min(origSize.Height - 1, t));
            b = Math.Max(0, Math.Min(origSize.Height - 1, b));

            // swap if inverted
            if (r < l) { var tmp = l; l = r; r = tmp; }
            if (b < t) { var tmp = t; t = b; b = tmp; }

            int x = (int)Math.Floor(l);
            int y = (int)Math.Floor(t);
            int w = (int)Math.Ceiling(r - l);
            int h = (int)Math.Ceiling(b - t);

            // guard: invalid or too small box
            if (w <= 0 || h <= 0)
            {
                return Rectangle.Empty;
            }

            // 최종 보정: 원본 크기 넘어가면 잘라냄
            if (x + w > origSize.Width) w = origSize.Width - x;
            if (y + h > origSize.Height) h = origSize.Height - y;

            return new Rectangle(x, y, w, h);
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
        // segDim, mh, mw는 trt_infer에서 받은 값
        // 레이아웃 확인: protoFlat.Length == segDim*mh*mw 인 것은 동일.
        // 단, K의 위치가 마지막이면 이 함수로 계산
        private static void ComputeMask_NoAlloc_HWK(float[] coeff, float[] protoFlatHWK, int segDim, int mw, int mh, float[] maskOut)
        {
            Parallel.For(0, mh, y =>
            {
                int row = y * mw;
                for (int x = 0; x < mw; x++)
                {
                    float sum = 0f;
                    int baseIdx = (y * mw + x) * segDim; // [H,W,K]
                    for (int k = 0; k < segDim; k++)
                        sum += coeff[k] * protoFlatHWK[baseIdx + k];
                    maskOut[row + x] = 1f / (1f + (float)Math.Exp(-sum));
                }
            });
        }


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
            // net(640) 좌표→mask 좌표 스케일
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

                // x, y 각각에 대해 bilinear weights + 인덱스 미리 계산 (center alignment)
                var u0a = new int[roiW]; var fu0a = new float[roiW]; var fu1a = new float[roiW];
                for (int i = 0; i < roiW; i++)
                {
                    // orig -> net : (x + 0.5)*scale + padX - 0.5  (center-aligned)
                    float nx = ((x0 + i + 0.5f) * scale + padX) - 0.5f;
                    float u = nx * sx;
                    float uClamped = Math.Max(0, Math.Min(u, mw - 1.001f));
                    int u0 = (int)uClamped;
                    float fu = uClamped - u0;
                    u0a[i] = u0; fu0a[i] = 1 - fu; fu1a[i] = fu;
                }

                var v0a = new int[roiH]; var fv0a = new float[roiH]; var fv1a = new float[roiH];
                for (int j = 0; j < roiH; j++)
                {
                    float ny = ((y0 + j + 0.5f) * scale + padY) - 0.5f;
                    float v = ny * sy;
                    float vClamped = Math.Max(0, Math.Min(v, mh - 1.001f));
                    int v0 = (int)vClamped;
                    float fv = vClamped - v0;
                    v0a[j] = v0; fv0a[j] = 1 - fv; fv1a[j] = fv;
                }

                for (int j = 0; j < roiH; j++)
                {
                    int y = y0 + j;
                    int v0 = v0a[j]; int v1 = Math.Min(v0 + 1, mh - 1);
                    float wv0 = fv0a[j], wv1 = fv1a[j];

                    byte* row = basePtr + y * stride + (x0 * 4);
                    int base00 = v0 * mw;
                    int base01 = v1 * mw;

                    for (int i = 0; i < roiW; i++)
                    {
                        int u0 = u0a[i]; int u1 = Math.Min(u0 + 1, mw - 1);
                        float wu0 = fu0a[i], wu1 = fu1a[i];

                        float m00 = mask[base00 + u0];
                        float m10 = mask[base00 + u1];
                        float m01 = mask[base01 + u0];
                        float m11 = mask[base01 + u1];

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
