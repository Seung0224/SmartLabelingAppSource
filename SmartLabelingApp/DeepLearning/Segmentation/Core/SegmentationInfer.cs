// SegmentationInfer.cs
// - ONNX + TensorRT 경로를 한 파일에 통합 (전처리→추론→후처리까지 여기서 끝)
// - 호출 예:
//   * ONNX:   var res = SegmentationInfer.InferOnnx(_onnxSession, bitmap);
//   * TRT:    var res = SegmentationInfer.Infer(_engineSession, bitmap);   // 엔진 인스턴스 전달 (핸들 내부에서 사용)
//   * (옵션)  var res = SegmentationInfer.InferTrt(trtHandle, bitmap);     // IntPtr 직접 사용도 가능
//
// 의존(프로젝트 내 기보유):
//   - Preprocess.cs  : EnsureOnnxInput, EnsureNchwBuffer, FillTensorFromBitmap, DefaultNet
//   - Postprocess.cs : Nms(...)
//   - ProtoUtils.cs  : DetectByVariance(...), TransposeHWKtoKHW(...), enum ProtoLayout
//   - MathUtils.cs   : Sigmoid(...)
//   - 타입: SegResult, Det

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp
{
    public static class SegmentationInfer
    {
        // ============================
        // ONNX 경로 (세션별 캐시)
        // ============================

        private sealed class OnnxCache
        {
            public string InputName;
            public int CurNet;
            public float[] InBuf;
            public DenseTensor<float> Tensor;
            public NamedOnnxValue Nov;
        }

        private static readonly ConditionalWeakTable<InferenceSession, OnnxCache> _onnxCaches =
            new ConditionalWeakTable<InferenceSession, OnnxCache>();

        public static SegResult InferOnnx(
            InferenceSession session, Bitmap orig,
            float conf = 0.9f, float iou = 0.45f)
        {
            if (session is null) throw new ArgumentNullException(nameof(session));
            if (orig is null) throw new ArgumentNullException(nameof(orig));

            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0;

            int net = Preprocess.DefaultNet;
            Trace.WriteLine($"[ONNX] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            var cache = _onnxCaches.GetValue(session, _ => new OnnxCache());

            // 전처리
            Preprocess.EnsureOnnxInput(session, net,
                ref cache.InputName, ref cache.CurNet, ref cache.InBuf, ref cache.Tensor, ref cache.Nov);

            Preprocess.FillTensorFromBitmap(orig, net, cache.InBuf,
                out float scale, out int padX, out int padY, out Size resized);

            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[ONNX] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            // 추론
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outs = null;
            try
            {
                outs = session.Run(new[] { cache.Nov });
                tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                var det3 = outs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>(); // [1,*,*]
                var proto4 = outs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>(); // [1,K,H,W]

                var d3 = det3.Dimensions; var d4 = proto4.Dimensions;
                Trace.WriteLine($"[ONNX] Run ok | det=({d3[0]},{d3[1]},{d3[2]}), proto=({d4[0]},{d4[1]},{d4[2]},{d4[3]}), inferMs={tInfer:F1}");

                // 후처리
                var res = BuildFromOnnx(det3, proto4, net, scale, padX, padY, resized, orig.Size, conf, iou);

                res.PreMs = tPre;
                res.InferMs = tInfer;
                res.TotalMs = sw.Elapsed.TotalMilliseconds;

                Trace.WriteLine($"[ONNX] Infer() end | dets={res.Dets.Count}, times(ms): pre={res.PreMs:F1}, infer={res.InferMs:F1}, post={res.PostMs:F1}, total={res.TotalMs:F1}");
                return res;
            }
            finally { outs?.Dispose(); }
        }

        // ============================
        // TensorRT 경로 (엔진 인스턴스)
        // ============================

        // 네이티브 P/Invoke
        private const string TrtDll = "TensorRTRunner";
        [DllImport(TrtDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_get_input_size(IntPtr handle);

        [DllImport(TrtDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_infer(
            IntPtr handle,
            float[] nchw, int nchwLength,
            out IntPtr detOut, out int nDet, out int detC,
            out IntPtr protoOut, out int segDim, out int maskH, out int maskW);

        [DllImport(TrtDll, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trt_free(IntPtr p);

        // 핸들별 캐시(버퍼/레이아웃)
        private sealed class TrtCache
        {
            public int CachedInputNet = -1;
            public float[] InBuf;              // [1,3,net,net]
            public ProtoLayout Layout = ProtoLayout.Unknown;
        }

        private static readonly ConcurrentDictionary<long, TrtCache> _trtCaches = new ConcurrentDictionary<long, TrtCache>();

        private static TrtCache GetTrtCache(IntPtr h)
        {
            long k = h.ToInt64();
            return _trtCaches.GetOrAdd(k, _ =>
            {
                var c = new TrtCache();
                try { c.CachedInputNet = trt_get_input_size(h); } catch { c.CachedInputNet = -1; }
                return c;
            });
        }

        /// <summary>
        /// 기존: _engineSession.Infer(bitmap, …)
        /// 변경: SegmentationInfer.Infer(_engineSession, bitmap, …)
        /// </summary>
        public static SegResult Infer(
            YoloSegEngine engine, Bitmap orig,
            float conf = 0.9f, float iou = 0.45f)
        {
            if (engine is null) throw new ArgumentNullException(nameof(engine));
            return InferTrt(engine.Handle, engine.FixedInputNet, orig, conf, iou);
        }

        /// <summary>
        /// (옵션) 핸들 직접 사용하는 버전
        /// </summary>
        public static SegResult InferTrt(
            IntPtr trtHandle, Bitmap orig,
            float conf = 0.9f, float iou = 0.45f)
        {
            if (trtHandle == IntPtr.Zero) throw new ArgumentException("Invalid TensorRT handle.", nameof(trtHandle));
            return InferTrt(trtHandle, fixedInputNetHint: -1, orig, conf, iou);
        }

        // 내부 공통 구현 (엔진의 fixed 입력 힌트를 외부에서 넣어줄 수도 있게)
        private static SegResult InferTrt(
            IntPtr trtHandle, int fixedInputNetHint,
            Bitmap orig, float conf, float iou)
        {
            if (orig is null) throw new ArgumentNullException(nameof(orig));

            var cache = GetTrtCache(trtHandle);

            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0;

            int net = (fixedInputNetHint > 0 ? fixedInputNetHint :
                      (cache.CachedInputNet > 0 ? cache.CachedInputNet : Preprocess.DefaultNet));
            Trace.WriteLine($"[TRT] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            // 전처리
            Preprocess.EnsureNchwBuffer(net, ref cache.InBuf);
            Preprocess.FillTensorFromBitmap(orig, net, cache.InBuf,
                out float scale, out int padX, out int padY, out Size resized);

            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            // 네이티브 추론
            IntPtr detPtr = IntPtr.Zero, protoPtr = IntPtr.Zero;
            int nDet = 0, detC = 0, segDim = 0, mh = 0, mw = 0;
            Trace.WriteLine("[TRT] Calling trt_infer...");
            int ok = trt_infer(trtHandle, cache.InBuf, cache.InBuf.Length,
                               out detPtr, out nDet, out detC,
                               out protoPtr, out segDim, out mh, out mw);
            if (ok == 0) throw new InvalidOperationException("trt_infer 실패");
            tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] trt_infer ok | det shape=({nDet},{detC}), proto=({segDim},{mh},{mw}), inferMs={tInfer:F1}");

            // 결과 복사
            float[] proto = new float[segDim * mw * mh]; // 길이 = segDim*mw*mh
            Marshal.Copy(protoPtr, proto, 0, proto.Length);
            trt_free(protoPtr);

            float[] detFlat = Array.Empty<float>();
            if (nDet > 0 && detC > 0)
            {
                detFlat = new float[nDet * detC];
                Marshal.Copy(detPtr, detFlat, 0, detFlat.Length);
            }
            trt_free(detPtr);

            // 후처리 (원본 엔진과 동일한 순서/규칙)
            var res = BuildFromTrt(detFlat, nDet, detC, proto, segDim, mh, mw,
                                   net, scale, padX, padY, resized, orig.Size,
                                   ref cache.Layout, conf, iou);

            res.PreMs = tPre;
            res.InferMs = tInfer;
            res.TotalMs = sw.Elapsed.TotalMilliseconds;

            Trace.WriteLine($"[TRT] Infer() end | dets={res.Dets.Count}, times(ms): pre={res.PreMs:F1}, infer={res.InferMs:F1}, post={res.PostMs:F1}, total={res.TotalMs:F1}");
            return res;
        }

        // ============================
        // 공통 후처리
        // ============================

        // ONNX: det3=[1,*,*], proto4=[1,K,H,W] → 이미 KHW 보장
        private static SegResult BuildFromOnnx(
            Tensor<float> det3, Tensor<float> proto4,
            int net, float scale, int padX, int padY, Size resized, Size origSize,
            float conf, float iou)
        {
            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPost = 0;

            var d3 = det3.Dimensions; // [1, A, C] or [1, C, A]
            var d4 = proto4.Dimensions; // [1, K, H, W]

            int a = d3[1], c = d3[2];
            int segDim = d4[1], mh = d4[2], mw = d4[3];

            bool channelsFirst = (a <= 512 && c >= 1000) || (a <= segDim + 4 + 256);
            int channels = channelsFirst ? a : c;
            int nPred = channelsFirst ? c : a;
            int numClasses = channels - 4 - segDim;
            if (numClasses <= 0)
                throw new InvalidOperationException($"Invalid head layout. feat={channels}, segDim={segDim}");

            float coordScale = 1f, maxWH = 0f;
            int sample = Math.Min(nPred, 128);
            for (int i = 0; i < sample; i++)
            {
                float ww = channelsFirst ? det3[0, 2, i] : det3[0, i, 2];
                float hh = channelsFirst ? det3[0, 3, i] : det3[0, i, 3];
                if (ww > maxWH) maxWH = ww; if (hh > maxWH) maxWH = hh;
            }
            if (maxWH <= 3.5f) coordScale = net;

            var dets = new List<Det>(Math.Min(nPred, 256));
            for (int i = 0; i < nPred; i++)
            {
                float x = (channelsFirst ? det3[0, 0, i] : det3[0, i, 0]) * coordScale;
                float y = (channelsFirst ? det3[0, 1, i] : det3[0, i, 1]) * coordScale;
                float w = (channelsFirst ? det3[0, 2, i] : det3[0, i, 2]) * coordScale;
                float h = (channelsFirst ? det3[0, 3, i] : det3[0, i, 3]) * coordScale;

                int bestC = 0; float bestS = 0f; int baseIdx = 4;
                for (int cidx = 0; cidx < numClasses; cidx++)
                {
                    float raw = channelsFirst ? det3[0, baseIdx + cidx, i] : det3[0, i, baseIdx + cidx];
                    float s = (raw < 0f || raw > 1f) ? MathUtils.Sigmoid(raw) : raw;
                    if (s > bestS) { bestS = s; bestC = cidx; }
                }
                if (bestS < conf) continue;

                var coeff = new float[segDim];
                int coeffBase = baseIdx + numClasses;
                for (int k = 0; k < segDim; k++)
                    coeff[k] = channelsFirst ? det3[0, coeffBase + k, i] : det3[0, i, coeffBase + k];

                float l = x - w * 0.5f, t = y - h * 0.5f, r = x + w * 0.5f, b = y + h * 0.5f;
                if ((r - l) < 2 || (b - t) < 2) continue;

                dets.Add(new Det { Box = new RectangleF(l, t, r - l, b - t), Score = bestS, ClassId = bestC, Coeff = coeff });
            }

            if (dets.Count > 0)
                dets = Postprocess.Nms(dets, d => d.Box, d => d.Score, iou);

            tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

            var protoFlat = proto4.ToArray(); // 이미 [K,H,W]
            return new SegResult
            {
                NetSize = net,
                Scale = scale,
                PadX = padX,
                PadY = padY,
                Resized = resized,
                OrigSize = origSize,
                Dets = dets,
                SegDim = segDim,
                MaskH = mh,
                MaskW = mw,
                ProtoFlat = protoFlat,
                PostMs = tPost
            };
        }

        // TRT: det 플랫 + proto → (필요 시) 헤더 전치/좌표스케일/Proto 레이아웃 감지 후 KHW 보장
        private static SegResult BuildFromTrt(
            float[] detFlat, int nDet, int detC,
            float[] proto, int segDim, int mh, int mw,
            int net, float scale, int padX, int padY, Size resized, Size origSize,
            ref ProtoLayout layout, float conf, float iou)
        {
            // DET 헤더 전치(C,M→M,C) 필요 시
            if (detC > 512 && nDet <= 512)
            {
                int M = detC, C = nDet;
                var t = new float[M * C];
                for (int c = 0; c < C; c++)
                    for (int m = 0; m < M; m++)
                        t[m * C + c] = detFlat[c * M + m];
                detFlat = t; nDet = M; detC = C;
                Trace.WriteLine($"[TRT] DET transposed -> shape {nDet}x{detC}");
            }

            int channels = detC;
            int nPred = nDet;
            int numClasses = channels - 4 - segDim;
            if (numClasses <= 0)
                throw new InvalidOperationException($"Invalid head layout: channels={channels}, segDim={segDim}");

            // 좌표 스케일 힌트
            float coordScale = 1f, maxWH = 0f;
            int sample = Math.Min(nPred, 128);
            for (int i = 0; i < sample; i++)
            {
                float ww = detFlat[i * detC + 2];
                float hh = detFlat[i * detC + 3];
                if (ww > maxWH) maxWH = ww; if (hh > maxWH) maxWH = hh;
            }
            if (maxWH <= 3.5f) coordScale = net;

            // det 파싱
            var dets = new List<Det>(Math.Min(nPred, 256));
            for (int i = 0; i < nPred; i++)
            {
                int baseIdx = i * detC;
                if (baseIdx + 4 + numClasses + segDim > detFlat.Length) break;

                float x = detFlat[baseIdx + 0] * coordScale;
                float y = detFlat[baseIdx + 1] * coordScale;
                float w = detFlat[baseIdx + 2] * coordScale;
                float h = detFlat[baseIdx + 3] * coordScale;

                int bestC = 0; float bestS = 0f;
                for (int cidx = 0; cidx < numClasses; cidx++)
                {
                    float raw = detFlat[baseIdx + 4 + cidx];
                    float s = (raw < 0f || raw > 1f) ? (1f / (1f + (float)Math.Exp(-raw))) : raw;
                    if (s > bestS) { bestS = s; bestC = cidx; }
                }
                if (bestS < conf) continue;

                float l = x - w * 0.5f, t = y - h * 0.5f, r = x + w * 0.5f, b = y + h * 0.5f;
                if ((r - l) < 2 || (b - t) < 2) continue;

                var coeff = new float[segDim];
                Array.Copy(detFlat, baseIdx + 4 + numClasses, coeff, 0, segDim);

                dets.Add(new Det { Box = new RectangleF(l, t, r - l, b - t), Score = bestS, ClassId = bestC, Coeff = coeff });
            }

            if (dets.Count > 0)
                dets = Postprocess.Nms(dets, d => d.Box, d => d.Score, iou);

            // Proto 레이아웃 감지/전치 (원본 엔진과 동일한 인자 순서 유지: segDim, mw, mh)
            if (layout == ProtoLayout.Unknown && dets.Count > 0)
            {
                layout = ProtoUtils.DetectByVariance(dets[0].Coeff, proto, segDim, mw, mh);
                Trace.WriteLine($"[TRT] Proto layout detected = {layout}");
            }
            if (layout == ProtoLayout.HWK)
            {
                var khw = new float[segDim * mw * mh];
                ProtoUtils.TransposeHWKtoKHW(proto, segDim, mw, mh, khw);
                proto = khw;
                Trace.WriteLine("[TRT] Proto HWK → KHW transposed");
            }

            return new SegResult
            {
                NetSize = net,
                Scale = scale,
                PadX = padX,
                PadY = padY,
                Resized = resized,
                OrigSize = origSize,
                Dets = dets,
                SegDim = segDim,
                MaskH = mh,
                MaskW = mw,
                ProtoFlat = proto
            };
        }
    }
}
