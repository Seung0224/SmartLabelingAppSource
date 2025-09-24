// SegmentationInfer.cs
// - ONNX + TensorRT 두 경로를 '한 파일'에 통합
// - 드롭인 교체용: 기존 YoloSegOnnx.Infer(...) / YoloSegEngine.Infer(...) 한 줄을 아래 API로 대체
//   * ONNX                : SegmentationInfer.InferOnnx(session, bitmap, ...)
//   * TensorRT(IntPtr)    : SegmentationInfer.InferTrt(trtHandle, bitmap, ...)
//   * TensorRT(YoloSegEngine) : SegmentationInfer.Infer(engineInstance, bitmap, ...)   <-- 새로 추가
//
// 의존:
//   - Preprocess.cs  : EnsureOnnxInput, EnsureNchwBuffer, FillTensorFromBitmap, DefaultNet
//   - Postprocess.cs : Nms(...)
//   - ProtoUtils.cs  : DetectByVariance(...), TransposeHWKtoKHW(...), enum ProtoLayout
//   - MaskSynth.cs   : (렌더러가 사용)
//   - MathUtils.cs   : Sigmoid(...)
//   - 타입: SegResult, Det

using System;
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
        // ===========================================================
        // ONNX 경로 (세션별 캐시)
        // ===========================================================



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
            float conf = 0.9f, float iou = 0.45f,
            float minBoxAreaRatio = 0.003f,
            float minMaskAreaRatio = 0.003f,
            bool discardTouchingBorder = true)
        {
            if (session is null) throw new ArgumentNullException(nameof(session));
            if (orig is null) throw new ArgumentNullException(nameof(orig));

            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0;

            int net = Preprocess.DefaultNet;
            Trace.WriteLine($"[ONNX] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            var cache = _onnxCaches.GetValue(session, _ => new OnnxCache());

            Preprocess.EnsureOnnxInput(session, net,
                ref cache.InputName, ref cache.CurNet, ref cache.InBuf, ref cache.Tensor, ref cache.Nov);

            Preprocess.FillTensorFromBitmap(orig, net, cache.InBuf,
                out float scale, out int padX, out int padY, out Size resized);

            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[ONNX] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outs = null;
            try
            {
                outs = session.Run(new[] { cache.Nov });
                tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                var det3 = outs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>(); // [1,*,*]
                var proto4 = outs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>(); // [1,K,H,W]

                var d3 = det3.Dimensions; var d4 = proto4.Dimensions;
                Trace.WriteLine($"[ONNX] Run ok | det=({d3[0]},{d3[1]},{d3[2]}), proto=({d4[0]},{d4[1]},{d4[2]},{d4[3]}), inferMs={tInfer:F1}");

                var res = BuildFromOnnx(det3, proto4, net, scale, padX, padY, resized, orig.Size, conf, iou);

                // 필요 시 공통 정책 적용 훅
                // ApplyOptionalPolicies(ref res, minBoxAreaRatio, minMaskAreaRatio, discardTouchingBorder);

                res.PreMs = tPre;
                res.InferMs = tInfer;
                res.TotalMs = sw.Elapsed.TotalMilliseconds;

                Trace.WriteLine($"[ONNX] Infer() end | dets={res.Dets.Count}, times(ms): pre={res.PreMs:F1}, infer={res.InferMs:F1}, post={res.PostMs:F1}, total={res.TotalMs:F1}");
                return res;
            }
            finally { outs?.Dispose(); }
        }

        // ===========================================================
        // TensorRT 경로 (IntPtr 핸들)
        // ===========================================================

        private const string TrtDll = "TensorRTRunner"; // tensorrt_runner.dll
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

        private sealed class TrtCache
        {
            public int CachedInputNet = -1;
            public float[] InBuf;              // [1,3,net,net]
            public ProtoLayout Layout = ProtoLayout.Unknown;
        }

        private static readonly Dictionary<long, TrtCache> _trtCaches = new Dictionary<long, TrtCache>();
        private static TrtCache GetTrtCache(IntPtr h)
        {
            long k = h.ToInt64();
            if (!_trtCaches.TryGetValue(k, out var c))
            {
                c = new TrtCache();
                try { c.CachedInputNet = trt_get_input_size(h); } catch { c.CachedInputNet = -1; }
                _trtCaches[k] = c;
            }
            return c;
        }

        public static SegResult InferTrt(
            IntPtr trtHandle, Bitmap orig,
            float conf = 0.9f, float iou = 0.45f,
            float minBoxAreaRatio = 0.003f,
            float minMaskAreaRatio = 0.003f,
            bool discardTouchingBorder = true)
        {
            if (trtHandle == IntPtr.Zero) throw new ArgumentException("Invalid TensorRT handle.", nameof(trtHandle));
            if (orig is null) throw new ArgumentNullException(nameof(orig));

            var cache = GetTrtCache(trtHandle);

            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0;

            int net = cache.CachedInputNet > 0 ? cache.CachedInputNet : Preprocess.DefaultNet;
            Trace.WriteLine($"[TRT] Infer() start | net={net}, img={orig.Width}x{orig.Height}");

            Preprocess.EnsureNchwBuffer(net, ref cache.InBuf);
            Preprocess.FillTensorFromBitmap(orig, net, cache.InBuf,
                out float scale, out int padX, out int padY, out Size resized);

            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            IntPtr detPtr = IntPtr.Zero, protoPtr = IntPtr.Zero;
            int nDet = 0, detC = 0, segDim = 0, mh = 0, mw = 0;
            Trace.WriteLine("[TRT] Calling trt_infer...");
            int ok = trt_infer(trtHandle, cache.InBuf, cache.InBuf.Length,
                               out detPtr, out nDet, out detC,
                               out protoPtr, out segDim, out mh, out mw);
            if (ok == 0) throw new InvalidOperationException("trt_infer 실패");
            tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[TRT] trt_infer ok | det shape=({nDet},{detC}), proto=({segDim},{mh},{mw}), inferMs={tInfer:F1}");

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

            var res = BuildFromTrt(detFlat, nDet, detC, proto, segDim, mh, mw,
                                   net, scale, padX, padY, resized, orig.Size,
                                   ref cache.Layout, conf, iou);

            // 필요 시 공통 정책
            // ApplyOptionalPolicies(ref res, minBoxAreaRatio, minMaskAreaRatio, discardTouchingBorder);

            res.PreMs = tPre;
            res.InferMs = tInfer;
            res.TotalMs = sw.Elapsed.TotalMilliseconds;

            Trace.WriteLine($"[TRT] Infer() end | dets={res.Dets.Count}, times(ms): pre={res.PreMs:F1}, infer={res.InferMs:F1}, post={res.PostMs:F1}, total={res.TotalMs:F1}");
            return res;
        }

        // ===========================================================
        // ★ 새로 추가: TensorRT 경로 (엔진 인스턴스 오버로드)
        // ===========================================================
        //
        // 기존 호출:
        //   var res = _engineSession.Infer(srcCopy);
        //
        // 교체 한 줄:
        //   var res = SegmentationInfer.Infer(_engineSession, srcCopy);
        //
        // 내부에서 YoloSegEngine의 공개 API를 그대로 호출하므로,
        // 핸들/버퍼/레이아웃 관리는 엔진이 책임지고 "원래 잘 되던" 결과와 1:1 동일합니다.
        //
        public static SegResult Infer(
            YoloSegEngine engine, Bitmap orig,
            float conf = 0.9f, float iou = 0.45f,
            float minBoxAreaRatio = 0.003f,
            float minMaskAreaRatio = 0.003f,
            bool discardTouchingBorder = true)
        {
            if (engine is null) throw new ArgumentNullException(nameof(engine));
            if (orig is null) throw new ArgumentNullException(nameof(orig));
            // 원본 엔진의 로직을 그대로 사용 (정확히 동일한 오버레이/결과 유지)
            return engine.Infer(orig, conf, iou);
        }

        // ===========================================================
        // 공통 후처리
        // ===========================================================

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

            var protoFlat = proto4.ToArray(); // 이미 KHW
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

        private static SegResult BuildFromTrt(
            float[] detFlat, int nDet, int detC,
            float[] proto, int segDim, int mh, int mw,
            int net, float scale, int padX, int padY, Size resized, Size origSize,
            ref ProtoLayout layout, float conf, float iou)
        {
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

            float coordScale = 1f, maxWH = 0f;
            int sample = Math.Min(nPred, 128);
            for (int i = 0; i < sample; i++)
            {
                float ww = detFlat[i * detC + 2];
                float hh = detFlat[i * detC + 3];
                if (ww > maxWH) maxWH = ww; if (hh > maxWH) maxWH = hh;
            }
            if (maxWH <= 3.5f) coordScale = net;

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

        private static void ApplyOptionalPolicies(ref SegResult res,
            float minBoxAreaRatio, float minMaskAreaRatio, bool discardTouchingBorder)
        {
            // 필요 시 공통 정책을 여기에 추가
        }
    }
}
