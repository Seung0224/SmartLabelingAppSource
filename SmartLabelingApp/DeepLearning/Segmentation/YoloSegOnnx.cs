using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Threading;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    // ------------------------------------------------------------------------------------
    // YoloSegOnnx (Annotated)
    //
    // 이 파일은 YOLO-Seg(세그멘테이션 포함) ONNX 모델을 C#에서 실행하기 위한 유틸입니다.
    // 크게 3단계로 동작합니다.
    //   1) Preprocessing : 입력 Bitmap을 모델이 요구하는 정사각형 크기(예: 640x640)에
    //      "레터박스(letterbox)"로 맞추어 텐서에 채워넣습니다. 모델은 이 형식을 기대하므로
    //      이 과정을 하지 않으면 추론이 제대로 되지 않습니다.
    //   2) Infer         : ONNX Runtime으로 모델을 실행해 박스(Detections) + 세그멘테이션
    //      프로토(Proto) + 각 박스별 마스크 계수(Coeff)를 얻습니다. 박스의 좌표계는
    //      네트워크 입력 좌표이며, 이후 원본 좌표로 역변환합니다.
    //   3) Overlay       : 얻어진 마스크(=Proto와 Coeff의 조합)를 원본 이미지에
    //      재매핑(bilinear sampling)하여 지정 색상과 알파로 합성합니다.
    // ------------------------------------------------------------------------------------
    public static class YoloSegOnnx
    {
        static YoloSegOnnx()
        {
            try { ForceWarmupSimdKernels(); } catch { /* ignore */ }
        }

        #region Fields
        
        public enum LabelName
        {
            Liquid = 0,
            Default = 1
        }
        // ----- 라벨/배지 그리기용 상수 (UI 장식) -----
        const int LABEL_BADGE_GAP_PX = 2;
        const int LABEL_BADGE_PADX = 4;
        const int LABEL_BADGE_PADY = 3;
        const int LABEL_BADGE_BORDER_PX = 2;
        const int LABEL_BADGE_ACCENT_W = 4;
        const int LABEL_BADGE_WIPE_PX = 1;

        // 세그먼트(폴리라인 연결용) 보조 구조체
        private struct Seg { public PointF A, B; public Seg(PointF a, PointF b) { A = a; B = b; } }

        // ----- 입력 텐서 관련 캐시 -----
        // _inputName : ONNX 입력 이름
        // _curNet    : 현재 확보된 텐서/버퍼가 가리키는 네트 입력 한 변 크기(예: 640)
        // _inBuf     : [1,3,net,net] 플랫 버퍼
        // _tensor    : DenseTensor<float> 뷰
        // _nov       : ONNX Runtime에 넣을 NamedOnnxValue
        static string _inputName;
        static int _curNet = 0;
        static float[] _inBuf;
        static DenseTensor<float> _tensor;
        static NamedOnnxValue _nov;

        // ----- 세션 캐시 -----
        // _cachedSession  : 최근에 만든 InferenceSession (같은 모델 경로면 재사용)
        // _cachedModelPath: 캐시된 세션의 모델 경로
        // _sessionLock    : 멀티스레드에서 세션 생성/교체 보호
        private static InferenceSession _cachedSession = null;
        private static string _cachedModelPath = null;
        private static readonly object _sessionLock = new object();

        #endregion

        #region Types
        // ----- 오버레이 결과 -----
        public struct OverlayResult
        {
            public Bitmap Image;     // 합성된 최종 이미지
            public List<SmartLabelingApp.ImageCanvas.OverlayItem> Overlays; // UI 폴리라인/박스/배지
        }
        #endregion

        #region SessionManagement

        private static void ForceWarmupSimdKernels()
        {
            // 1) SIMD 경로 강제 터치 (JIT + Vectors 로드)
            int vc = Vector<float>.Count; // ex) 8 or 16

            // 2) 더미 proto/coeff/mask 한 번 계산해 JIT 비용 선지불
            int segDim = 32, mw = 160, mh = 160;        // 실제와 유사한 크기면 더 좋음
            var proto = new float[segDim * mw * mh];
            var coeff = new float[segDim];
            for (int i = 0; i < coeff.Length; i++) coeff[i] = 0.01f * (i + 1);

            // 할당 없는 버전으로 웜업 (패치 B에서 제공)
            var maskBuf = new float[mw * mh];
            MaskSynth.ComputeMask_KHW_NoAlloc(coeff, proto, segDim, mw, mh, maskBuf);

            // 3) 블렌딩도 한 번(아주 작은 ROI) — LockBits/JIT 따뜻하게
            using (var bmp = new Bitmap(32, 32, PixelFormat.Format32bppArgb))
            {
                var box = new Rectangle(2, 2, 28, 28);
                // 이하 숫자는 의미 없는 더미 값 (실행만 함)
                BlendMaskIntoOrigROI(bmp, box, maskBuf, mw, mh, netSize: 640,
                    scale: 1f, padX: 0, padY: 0, thr: 0.5f, alpha: 0.1f, color: Color.Red);
            }
        }

        // 세션 캐시를 활용해 InferenceSession을 가져오거나 생성합니다.
        private static InferenceSession GetOrCreateSession(string modelPath, out double initMs)
        {
            lock (_sessionLock)
            {
                // 같은 모델 경로면 이전 세션 재사용 → 최초 생성 비용 절약
                if (_cachedSession != null && string.Equals(_cachedModelPath, modelPath, StringComparison.OrdinalIgnoreCase))
                {
                    initMs = 0;
                    return _cachedSession;
                }

                // 새로 생성
                var sw = Stopwatch.StartNew();
                var sess = CreateSessionWithFallback(modelPath);
                initMs = sw.Elapsed.TotalMilliseconds;

                if (_cachedSession != null) { try { _cachedSession.Dispose(); } catch { } }
                _cachedSession = sess;
                _cachedModelPath = modelPath;
                return _cachedSession;
            }
        }

        // 세션을 준비하고 간단한 프로그레스 메시지를 보고합니다.
        public static InferenceSession EnsureSession(string modelPath, IProgress<(int percent, string status)> progress = null)
        {
            int p = 0;
            ReportStep(progress, ref p, 5, "모델 경로 확인");
            double initMs = 0;
            bool createdNew = false;
            ReportStep(progress, ref p, 12, "Execution Provider 확인");
            ReportStep(progress, ref p, 20, "세션 옵션 구성");
            ReportStep(progress, ref p, 35, "세션 생성 준비");

            var session = GetOrCreateSession(modelPath, out initMs);
            createdNew = initMs > 0;

            if (createdNew) ReportStep(progress, ref p, 60, $"세션 생성 완료 ({initMs:F0} ms)");
            else { ReportStep(progress, ref p, 45, "캐시된 세션 재사용"); ReportStep(progress, ref p, 60, "세션 확인 완료"); }

            // 메타데이터 읽기(선택)
            try
            {
                ReportStep(progress, ref p, 70, "모델 IO 메타데이터 읽기");
                var inputs = session.InputMetadata;
                var outputs = session.OutputMetadata;
                ReportStep(progress, ref p, 78, $"입력 {inputs.Count} / 출력 {outputs.Count} 확인");
            }
            catch { ReportStep(progress, ref p, 78, "IO 메타데이터 확인(옵션) 건너뜀"); }

            ReportStep(progress, ref p, 85, "웜업 준비");
            ReportStep(progress, ref p, 92, "리소스 초기화");
            ReportStep(progress, ref p, 100, "완료");
            return session;
        }

        // CUDA EP 옵션(있으면 사용) 추가 시도. 실패 시 일반 Append로 시도.
        private static bool TryAppendCudaWithOptions(SessionOptions so)
        {
            try
            {
                var optType = Type.GetType("Microsoft.ML.OnnxRuntime.Provider.OrtCUDAProviderOptions, Microsoft.ML.OnnxRuntime")
                    ?? Type.GetType("Microsoft.ML.OnnxRuntime.Providers.OrtCUDAProviderOptions, Microsoft.ML.OnnxRuntime");
                if (optType != null)
                {
                    var cuda = Activator.CreateInstance(optType);
                    SetIfExists(optType, cuda, "DeviceId", 0);
                    var enumType = optType.Assembly.GetType("Microsoft.ML.OnnxRuntime.Provider.OrtCudnnConvAlgoSearch")
                        ?? optType.Assembly.GetType("Microsoft.ML.OnnxRuntime.Providers.OrtCudnnConvAlgoSearch");
                    if (enumType != null)
                    {
                        var heuristic = Enum.Parse(enumType, "HEURISTIC", ignoreCase: true);
                        SetIfExists(optType, cuda, "CudnnConvAlgoSearch", heuristic);
                    }
                    SetIfExists(optType, cuda, "EnableCudaGraph", 1);
                    SetIfExists(optType, cuda, "DoCopyInDefaultStream", 1);
                    SetIfExists(optType, cuda, "TunableOpEnable", 1);
                    SetIfExists(optType, cuda, "TunableOpTuningEnable", 1);
                    SetIfExists(optType, cuda, "TunableOpMaxTuningDurationMs", 500);
                    var mi = typeof(SessionOptions).GetMethod("AppendExecutionProvider_CUDA", new[] { optType });
                    if (mi != null) { mi.Invoke(so, new object[] { cuda }); return true; }
                }
                var mi2 = typeof(SessionOptions).GetMethod("AppendExecutionProvider", new[] { typeof(string), typeof(IDictionary<string, string>) });
                if (mi2 != null)
                {
                    var opts = new Dictionary<string, string>
                    {
                        ["device_id"] = "0",
                        ["cudnn_conv_algo_search"] = "HEURISTIC",
                        ["enable_cuda_graph"] = "1",
                        ["do_copy_in_default_stream"] = "1",
                        ["tunable_op_enable"] = "1",
                        ["tunable_op_tuning_enable"] = "1",
                        ["tunable_op_max_tuning_duration_ms"] = "500"
                    };
                    mi2.Invoke(so, new object[] { "CUDAExecutionProvider", opts });
                    return true;
                }
            }
            catch { }
            return false;
        }
        private static void TryWarmup(InferenceSession s, int netSize)
        {
            try
            {
                var inputName = s.InputMetadata.Keys.FirstOrDefault();
                if (string.IsNullOrEmpty(inputName)) return;
                var dummy = new DenseTensor<float>(new[] { 1, 3, netSize, netSize });
                var input = NamedOnnxValue.CreateFromTensor(inputName, dummy);
                using (var _ = s.Run(new[] { input })) { }
            }
            catch { }
        }

        private static void SetIfExists(Type t, object obj, string prop, object value)
        {
            var p = t.GetProperty(prop, BindingFlags.Public | BindingFlags.Instance);
            if (p != null && p.CanWrite) p.SetValue(obj, value, null);
        }

        // 모델을 GPU(CUDA) → DML → CPU 순으로 시도해 세션을 생성합니다.
        // 성공 시 입력 버퍼도 미리 준비하고 간단한 웜업을 실행합니다.
        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            SessionOptions so = null;
            try
            {
                so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                if (!TryAppendCudaWithOptions(so)) so.AppendExecutionProvider_CUDA(0);
                var sess = new InferenceSession(modelPath, so);
                Preprocess.EnsureOnnxInput(sess, 640, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
                TryWarmup(sess, 640);
                SmartLabelingApp.MainForm._currentRunTypeName = "GPU";
                return sess;
            }
            catch
            {
                try { so?.Dispose(); } catch { }
            }
            try
            {
                so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                so.AppendExecutionProvider_DML();
                var sess = new InferenceSession(modelPath, so);
                Preprocess.EnsureOnnxInput(sess, 640, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
                TryWarmup(sess, 640);
                SmartLabelingApp.MainForm._currentRunTypeName = "DML EP";
                return sess;
            }
            catch
            {
                try { so?.Dispose(); } catch { }
            }
            var soCpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            var cpu = new InferenceSession(modelPath, soCpu);
            Preprocess.EnsureOnnxInput(cpu, 640, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
            SmartLabelingApp.MainForm._currentRunTypeName = "CPU";
            return cpu;
        }

        #endregion

        #region INFER
        // --------------------------------------------------------------------------------
        // Infer
        // 1) 입력 메타에서 netSize를 파악(없으면 640 등 기본), EnsureInputBuffers로 버퍼 확보
        // 2) FillTensorFromBitmap으로 _inBuf를 채움(레터박스/정규화)
        // 3) session.Run으로 추론 → 출력 중 3D 텐서(Det head), 4D 텐서(Proto head) 선택
        // 4) Det head 파싱:
        //    [x,y,w,h, 클래스별 점수..., seg 계수(segDim)] 형태(채널 우선/후순 혼용 대비)
        //    - 좌표가 [0~1] 범위처럼 작은 값이면 netSize를 곱해 보정(coordScale)
        //    - conf 임계치로 1차 필터 → NMS(iou) 2차 필터
        // 5) Proto head 파싱:
        //    [segDim, mh, mw] 형태의 특징맵을 1차원 배열(ProtoFlat)로 보관
        // 6) SegResult에 모든 메타와 dets/Proto를 담아 반환
        // --------------------------------------------------------------------------------
        public static SegResult Infer(
     InferenceSession session, Bitmap orig,
     float conf = 0.9f, float iou = 0.45f,
     float minBoxAreaRatio = 0.003f,
     float minMaskAreaRatio = 0.003f,
     bool discardTouchingBorder = true)
        {
            var sw = Stopwatch.StartNew();
            double tPrev = 0, tPre = 0, tInfer = 0, tPost = 0;

            string inputName = session.InputMetadata.Keys.First();
            var inMeta = session.InputMetadata[inputName];

            int netH = 640, netW = 640;
            try
            {
                var dims = inMeta.Dimensions;
                if (dims.Length == 4)
                {
                    if (dims[2] > 0) netH = dims[2];
                    if (dims[3] > 0) netW = dims[3];
                }
            }
            catch { }
            int netSize = Math.Max(netH, netW);

            Trace.WriteLine($"[ONNX] Infer() start | net={netSize}, img={orig.Width}x{orig.Height}");

            Preprocess.EnsureOnnxInput(session, netSize, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
            Preprocess.FillTensorFromBitmap(orig, netSize, _inBuf, out float scale, out int padX, out int padY, out Size resized);
            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
            Trace.WriteLine($"[ONNX] Preprocess done | resized={resized.Width}x{resized.Height}, pad=({padX},{padY}), scale={scale:F6}, preMs={tPre:F1}");

            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = null;
            try
            {
                outputs = session.Run(new[] { _nov });
                tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                var t3 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>();
                var t4 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>();
                var d3 = t3.Dimensions; var d4 = t4.Dimensions;

                int a = d3[1], c = d3[2];
                int segDim = d4[1], mh = d4[2], mw = d4[3];
                Trace.WriteLine($"[ONNX] Run ok | det shape=({d3[0]},{d3[1]},{d3[2]}), proto=({d4[0]},{d4[1]},{d4[2]},{d4[3]}), inferMs={tInfer:F1}");

                bool channelsFirst = (a <= 512 && c >= 1000) || (a <= segDim + 4 + 256);
                int channels = channelsFirst ? a : c;
                int nPred = channelsFirst ? c : a;
                int feat = channels;
                int numClasses = feat - 4 - segDim;

                if (numClasses <= 0)
                    throw new InvalidOperationException($"Invalid head layout. feat={feat}, segDim={segDim}");

                Trace.WriteLine($"[ONNX] Parse det header | channelsFirst={channelsFirst}, numClasses={numClasses}, segDim={segDim}, nPred={nPred}");

                float coordScale = 1f;
                float maxWH = 0f;
                {
                    int sample = Math.Min(nPred, 128);
                    for (int i = 0; i < sample; i++)
                    {
                        float ww = channelsFirst ? t3[0, 2, i] : t3[0, i, 2];
                        float hh = channelsFirst ? t3[0, 3, i] : t3[0, i, 3];
                        if (ww > maxWH) maxWH = ww;
                        if (hh > maxWH) maxWH = hh;
                    }
                    if (maxWH <= 3.5f)
                        coordScale = netSize; // [0~1] 좌표로 판단 → netSize 곱
                }

                // 🔽 여기에 로그 추가
                Trace.WriteLine($"[ONNX] coordScale={coordScale}, sampleMaxWH={maxWH}");

                // 박스 파싱
                var dets = new List<Det>(256);
                int kept = 0;
                for (int i = 0; i < nPred; i++)
                {
                    float x = (channelsFirst ? t3[0, 0, i] : t3[0, i, 0]);
                    float y = (channelsFirst ? t3[0, 1, i] : t3[0, i, 1]);
                    float w = (channelsFirst ? t3[0, 2, i] : t3[0, i, 2]);
                    float h = (channelsFirst ? t3[0, 3, i] : t3[0, i, 3]);

                    int bestC = 0; float bestS = 0f;
                    for (int cidx = 0; cidx < numClasses; cidx++)
                    {
                        float raw = channelsFirst ? t3[0, 4 + cidx, i] : t3[0, i, 4 + cidx];
                        float s = (raw < 0f || raw > 1f) ? MathUtils.Sigmoid(raw) : raw;
                        if (s > bestS) { bestS = s; bestC = cidx; }
                    }
                    if (bestS < conf) continue;

                    var coeff = new float[segDim];
                    int baseIdx = 4 + numClasses;
                    for (int k = 0; k < segDim; k++)
                        coeff[k] = (channelsFirst ? t3[0, baseIdx + k, i] : t3[0, i, baseIdx + k]);

                    float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, btm = y + h / 2f;
                    dets.Add(new Det { Box = new RectangleF(l, t, r - l, btm - t), Score = bestS, ClassId = bestC, Coeff = coeff });
                    kept++;
                    if (i < 8)
                        Trace.WriteLine($"[ONNX] det[{i}] keep | box=({l:F1},{t:F1},{r:F1},{btm:F1}), score={bestS:F3}, cls={bestC}");
                }

                if (dets.Count > 0) dets = Postprocess.Nms(dets, d => d.Box, d => d.Score, iou);
                tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
                Trace.WriteLine($"[ONNX] Parsed det rows | total={nPred}, kept={kept}, afterNms={dets.Count}, postMs={tPost:F1}");

                var protoFlat = t4.ToArray();

                var res = new SegResult
                {
                    NetSize = netSize,
                    Scale = scale,
                    PadX = padX,
                    PadY = padY,
                    Resized = resized,
                    OrigSize = orig.Size,
                    Dets = dets,
                    SegDim = segDim,
                    MaskH = mh,
                    MaskW = mw,
                    ProtoFlat = protoFlat,
                    PreMs = tPre,
                    InferMs = tInfer,
                    PostMs = tPost,
                    TotalMs = sw.Elapsed.TotalMilliseconds
                };

                Trace.WriteLine($"[ONNX] Infer() end | dets={res.Dets.Count}, times(ms): pre={tPre:F1}, infer={tInfer:F1}, post={tPost:F1}, total={res.TotalMs:F1}");
                return res;
            }
            finally { outputs?.Dispose(); }
        }
        #endregion

        #region MaskAndOverlay
        // --------------------------------------------------------------------------------
        // Overlay / OverlayFast / Render
        // - Overlay      : 마스크를 정확한 경로로 재매핑하는 참고 구현(단계가 조금 더 많음)
        // - OverlayFast  : ROI 안에서 바로 bilinear로 샘플링/블렌딩하는 빠른 버전
        // - Render       : OverlayFast + 폴리라인/배지 등 오버레이 목록까지 만들어 반환
        //
        // 오버레이 핵심 절차:
        //   1) 각 탐지 d에 대해: mask = sigmoid( sum_k coeff[k] * proto[k] )
        //      (ComputeMaskSIMD는 이 수식을 SIMD로 빠르게 수행)
        //   2) mask를 네트 입력 크기(net x net) 기준 좌표로 맞추고(pad/scale 보정)
        //   3) 원본 좌표로 다시 보간한 뒤, 박스 ROI 밖은 0으로 만들어 누수 방지
        //   4) 색상+알파로 원본에 합성(스레시홀드 미만 픽셀은 건너뜀)
        //   5) (옵션) 마스크 외곽선 폴리라인을 계산해서 UI 오버레이로 추가
        // --------------------------------------------------------------------------------

        private static void BlendMaskIntoOrigROI(
            Bitmap dstRGBA, Rectangle boxOrig, float[] mask, int mw, int mh, int netSize,
            float scale, int padX, int padY, float thr, float alpha, Color color)
        {
            // ----- 기존과 동일한 파라미터 처리 -----
            float sx = (float)mw / netSize;
            float sy = (float)mh / netSize;

            // ROI 클램프
            int x0 = Math.Max(0, boxOrig.Left);
            int y0 = Math.Max(0, boxOrig.Top);
            int x1 = Math.Min(dstRGBA.Width, boxOrig.Right);
            int y1 = Math.Min(dstRGBA.Height, boxOrig.Bottom);
            if (x0 >= x1 || y0 >= y1) return;

            int roiW = x1 - x0;
            int roiH = y1 - y0;

            // 색/알파 (고정)
            byte rr = color.R, gg = color.G, bb = color.B;
            float a = Math.Max(0f, Math.Min(1f, alpha));
            float ia = 1f - a;

            // 임계값: float로 바로 비교 (시각 동일)
            float thrF = Math.Max(0f, Math.Min(1f, thr));

            // ----- 좌표/가중치 사전 계산 -----
            // u = ((x*scale)+padX)*sx, v = ((y*scale)+padY)*sy
            var u0_arr = new int[roiW];
            var fu0_arr = new float[roiW];
            var fu1_arr = new float[roiW];

            for (int i = 0; i < roiW; i++)
            {
                float u = (((x0 + i) * scale) + padX) * sx;
                int u0 = (int)Math.Floor(u);
                float fu = u - u0;
                u0_arr[i] = u0;
                fu0_arr[i] = 1f - fu;
                fu1_arr[i] = fu;
            }

            var v0_arr = new int[roiH];
            var fv0_arr = new float[roiH];
            var fv1_arr = new float[roiH];

            for (int j = 0; j < roiH; j++)
            {
                float v = (((y0 + j) * scale) + padY) * sy;
                int v0 = (int)Math.Floor(v);
                float fv = v - v0;
                v0_arr[j] = v0;
                fv0_arr[j] = 1f - fv;
                fv1_arr[j] = fv;
            }

            // ----- 비트맵 Lock -----
            var rect = new Rectangle(0, 0, dstRGBA.Width, dstRGBA.Height);
            var data = dstRGBA.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)data.Scan0;
                    int stride = data.Stride;

                    // 1) fixed에서 실제 주소를 IntPtr로 빼둠 (람다에서는 IntPtr만 캡처)
                    IntPtr baseAddr = (IntPtr)basePtr;
                    IntPtr maskAddr, u0Addr, fu0Addr, fu1Addr, v0Addr, fv0Addr, fv1Addr;

                    fixed (float* pmask = mask)
                    fixed (int* pu0 = u0_arr)
                    fixed (float* pfu0 = fu0_arr, pfu1 = fu1_arr)
                    fixed (int* pv0 = v0_arr)
                    fixed (float* pfv0 = fv0_arr, pfv1 = fv1_arr)
                    {
                        maskAddr = (IntPtr)pmask;
                        u0Addr = (IntPtr)pu0;
                        fu0Addr = (IntPtr)pfu0;
                        fu1Addr = (IntPtr)pfu1;
                        v0Addr = (IntPtr)pv0;
                        fv0Addr = (IntPtr)pfv0;
                        fv1Addr = (IntPtr)pfv1;
                    }

                    int proc = Environment.ProcessorCount;

                    Parallel.For(0, roiH, new ParallelOptions { MaxDegreeOfParallelism = proc }, j =>
                    {
                        // 2) 람다 내부에서 IntPtr → 포인터 캐스팅 (unsafe 블록)
                        unsafe
                        {
                            byte* baseP = (byte*)baseAddr;
                            float* pmask = (float*)maskAddr;
                            int* pu0 = (int*)u0Addr;
                            float* pfu0 = (float*)fu0Addr;
                            float* pfu1 = (float*)fu1Addr;
                            int* pv0 = (int*)v0Addr;
                            float* pfv0 = (float*)fv0Addr;
                            float* pfv1 = (float*)fv1Addr;

                            int y = y0 + j;
                            int v0 = pv0[j];
                            if ((uint)v0 >= (uint)mh - 1) return;

                            float wv0 = pfv0[j];
                            float wv1 = pfv1[j];

                            byte* p = baseP + y * stride + (x0 * 4);

                            int base00 = v0 * mw;
                            int base01 = base00 + mw;

                            for (int i = 0; i < roiW; i++)
                            {
                                int u0 = pu0[i];
                                if ((uint)u0 >= (uint)mw - 1)
                                {
                                    p += 4;
                                    continue;
                                }

                                float wu0 = pfu0[i];
                                float wu1 = pfu1[i];

                                float m00 = pmask[base00 + u0];
                                float m10 = pmask[base00 + u0 + 1];
                                float m01 = pmask[base01 + u0];
                                float m11 = pmask[base01 + u0 + 1];

                                float mx0 = m00 * wu0 + m10 * wu1;
                                float mx1 = m01 * wu0 + m11 * wu1;
                                float m = mx0 * wv0 + mx1 * wv1;

                                if (m < thrF)
                                {
                                    p += 4;
                                    continue;
                                }

                                // dst = dst*(1-a) + color*a
                                p[0] = (byte)(p[0] * ia + bb * a);
                                p[1] = (byte)(p[1] * ia + gg * a);
                                p[2] = (byte)(p[2] * ia + rr * a);
                                p[3] = 255;

                                p += 4;
                            }
                        }
                    });
                }
            }
            finally
            {
                dstRGBA.UnlockBits(data);
            }
        }

        #endregion
        #region Utils
        

        // 진행률 보고(옵션)
        private static void ReportStep(IProgress<(int percent, string status)> progress, ref int cur, int next, string msg)
        {
            if (progress == null) return;
            if (next < cur) next = cur;
            if (next > 100) next = 100;
            cur = next;
            Thread.Sleep(50);
            progress.Report((cur, msg));
        }
        #endregion
    }
}
