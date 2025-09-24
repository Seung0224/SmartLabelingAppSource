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
        private static extern int trt_infer(IntPtr handle,float[] nchw, int nchwLength,out IntPtr detOut, out int nDet, out int detC,out IntPtr protoOut, out int segDim, out int maskH, out int maskW);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trt_free(IntPtr p);

        #endregion

        private ProtoLayout _protoLayout = ProtoLayout.Unknown;

        private IntPtr _h = IntPtr.Zero;
        private readonly int _deviceId;
        private int _cachedInputNet = -1;        // 엔진 고정 입력(정사각) 추론용 힌트
        private float[] _inBuf = null;           // [1,3,net,net]

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

            int net = _cachedInputNet > 0 ? _cachedInputNet : 640;
            Trace.WriteLine($"[TRT] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            // ---- 1) 전처리
            Preprocess.EnsureNchwBuffer(net, ref _inBuf);
            Preprocess.FillTensorFromBitmap(orig, net, _inBuf, out float scale, out int padX, out int padY, out Size resized);
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

            // ---- 4) 전치 (C,M → M,C 로 변환)  [det 헤더 배열용]
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

                // cls 점수만 (obj 곱 제거)
                int bestC = 0; float bestS = 0f;
                for (int c = 0; c < numClasses; c++)
                {
                    float raw = detFlat[baseIdx + 4 + c];
                    float s = (raw < 0f || raw > 1f) ? (1f / (1f + (float)Math.Exp(-raw))) : raw;
                    if (s > bestS) { bestS = s; bestC = c; }
                }
                if (bestS < conf) continue;

                // 작은 박스 필터(2px 미만 스킵)
                float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, b = y + h / 2f;
                float bw = r - l, bh = b - t;
                if (bw < 2 || bh < 2) continue;

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

            if (dets.Count > 0) dets = Postprocess.Nms(dets, d => d.Box, d => d.Score, iou);

            // ---- 7.5) Proto 레이아웃 감지 & 전치(HWK→KHW)  [여기 추가]
            //  - 첫 프레임에만 감지해서 _protoLayout 캐시
            if (_protoLayout == ProtoLayout.Unknown && dets.Count > 0)
            {
                var coeffSample = dets[0].Coeff; // 상위 1개만으로도 충분
                _protoLayout = ProtoUtils.DetectByVariance(coeffSample, proto, segDim, mw, mh);
                Trace.WriteLine($"[TRT] Proto layout detected = {_protoLayout}");
            }
            //  - HWK이면 이번 프레임 proto를 KHW로 전치
            if (_protoLayout == ProtoLayout.HWK)
            {
                var transposedKHW = new float[segDim * mw * mh];
                ProtoUtils.TransposeHWKtoKHW(proto, segDim, mw, mh, transposedKHW);
                proto = transposedKHW;
                Trace.WriteLine("[TRT] Proto HWK → KHW transposed");
            }

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
                ProtoFlat = proto,   // ← 여기서부터는 KHW 보장
                Dets = dets,
                PreMs = tPre,
                InferMs = tInfer,
                PostMs = tPost,
                TotalMs = sw.Elapsed.TotalMilliseconds
            };

            Trace.WriteLine($"[TRT] Infer() end | dets={ret.Dets.Count}, times(ms): pre={tPre:F1}, infer={tInfer:F1}, post={tPost:F1}, total={ret.TotalMs:F1}");
            return ret;
        }


        #endregion
    }
}
