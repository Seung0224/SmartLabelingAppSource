using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Numerics;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;
using static OpenCvSharp.Stitcher;
using static SmartLabelingApp.ImageCanvas;

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
        
        [ThreadStatic] private static float[] _maskBufTLS;

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
        // ----- 추론 결과 컨테이너 -----
        public sealed class SegResult
        {
            // 네트 입력 한 변(정사각형) 크기. 보통 640.
            public int NetSize { get; set; }

            // 레터박스에서 사용된 스케일과 패딩. 원본↔네트 좌표 변환에 필수.
            public float Scale { get; set; }
            public int PadX { get; set; }
            public int PadY { get; set; }
            public Size Resized { get; set; }   // 패딩 전, 스케일 적용 후 크기
            public Size OrigSize { get; set; }  // 원본 이미지 크기

            // 탐지 결과(박스/점수/클래스/마스크 계수)
            public List<Det> Dets { get; set; }

            // 세그멘테이션 프로토(Proto) 관련 파라미터
            // SegDim : 채널 수(각 마스크를 구성하는 기저 개수)
            // MaskH/W: Proto 해상도(보통 작은 값, 예: 160x160)
            // ProtoFlat: [SegDim, MaskH, MaskW] 를 일렬로 가진 버퍼
            public int SegDim { get; set; }
            public int MaskH { get; set; }
            public int MaskW { get; set; }
            public float[] ProtoFlat { get; set; }

            // 단계별 지연 시간(ms)
            public double SessionMs { get; set; }
            public double PreMs { get; set; }
            public double InferMs { get; set; }
            public double PostMs { get; set; }
            public double TotalMs { get; set; }
        }

        // ----- 개별 탐지 -----
        public class Det
        {
            public RectangleF Box;   // 네트 좌표계 박스(중심→좌상/우하로 변환된 값)
            public float Score;      // 신뢰도
            public int ClassId;      // 클래스 인덱스
            public float[] Coeff;    // 이 박스 마스크를 만들기 위한 가중치(길이 = SegDim)
        }

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
            ComputeMaskSIMD_NoAlloc(coeff, proto, segDim, mw, mh, maskBuf);

            // 3) 블렌딩도 한 번(아주 작은 ROI) — LockBits/JIT 따뜻하게
            using (var bmp = new Bitmap(32, 32, PixelFormat.Format32bppArgb))
            {
                var box = new Rectangle(2, 2, 28, 28);
                // 이하 숫자는 의미 없는 더미 값 (실행만 함)
                BlendMaskIntoOrigROI3(bmp, box, maskBuf, mw, mh, netSize: 640,
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

        private static void ComputeMaskSIMD_NoAlloc(float[] coeff, float[] protoFlat, int segDim, int mw, int mh, float[] maskOut)
        {
            int vec = Vector<float>.Count;
            System.Threading.Tasks.Parallel.For(0, mh, y =>
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
                EnsureInputBuffers(sess, 640);
                TryWarmup(sess, 640);
                SmartLabelingApp.MainForm._currentRunTypeName = "CUDA EP";
                return sess;
            }
            catch (Exception ex)
            {
                try { so?.Dispose(); } catch { }
            }
            try
            {
                so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                so.AppendExecutionProvider_DML();
                var sess = new InferenceSession(modelPath, so);
                EnsureInputBuffers(sess, 640);
                TryWarmup(sess, 640);
                SmartLabelingApp.MainForm._currentRunTypeName = "DML EP";
                return sess;
            }
            catch (Exception ex)
            {
                try { so?.Dispose(); } catch { }
            }
            var soCpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            var cpu = new InferenceSession(modelPath, soCpu);
            EnsureInputBuffers(cpu, 640);
            SmartLabelingApp.MainForm._currentRunTypeName = "CPU";
            return cpu;
        }

        // 입력 한 변 크기(net)가 바뀌면 내부 버퍼를 다시 할당합니다.
        private static void EnsureInputBuffers(InferenceSession s, int net)
        {
            if (_tensor != null && _curNet == net) return; // 동일 크기면 재사용
            _inputName = s.InputMetadata.Keys.First();
            _inBuf = new float[1 * 3 * net * net];
            _tensor = new DenseTensor<float>(_inBuf, new[] { 1, 3, net, net });
            _nov = NamedOnnxValue.CreateFromTensor(_inputName, _tensor);
            _curNet = net;
        }
        #endregion

        #region Preprocessing
        // --------------------------------------------------------------------------------
        // FillTensorFromBitmap
        // 1) 원본 크기(WxH)에서 네트 입력(net x net)에 맞게 비율 유지(scale)로 줄입니다.
        // 2) 줄인 결과를 검은 바탕의 정사각형 캔버스(net x net) 중앙에 그립니다(padX/padY).
        // 3) 픽셀을 읽어 [R,G,B] 채널 순서로 0~1로 정규화하여 _inBuf에 채웁니다.
        //    (왜? ONNX 모델이 [1,3,net,net] float 텐서를 입력으로 기대하기 때문)
        // 4) 이후 박스/마스크를 원본 좌표로 되돌리기 위해 scale/padX/padY/resized를 반환합니다.
        // --------------------------------------------------------------------------------
        static void FillTensorFromBitmap(Bitmap src, int net, out float scale, out int padX, out int padY, out Size resized)
        {
            int W = src.Width, H = src.Height;
            scale = Math.Min((float)net / W, (float)net / H);          // 비율 유지 축소
            int rw = (int)Math.Round(W * scale);
            int rh = (int)Math.Round(H * scale);
            padX = (net - rw) / 2;                                    // 중앙 정렬 패딩
            padY = (net - rh) / 2;
            resized = new Size(rw, rh);

            // 24bpp RGB 백버퍼에 레터박싱된 이미지를 만든 뒤, 거기서 채널 분리
            var tmp = new Bitmap(net, net, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(tmp))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                g.DrawImage(src, padX, padY, rw, rh);
            }

            // tmp를 잠그고 바이트를 읽어 [R,G,B] 순서로 float(0~1)로 저장
            var rect = new Rectangle(0, 0, net, net);
            var bd = tmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* p0 = (byte*)bd.Scan0;
                    int stride = bd.Stride;
                    float inv255 = 1f / 255f;
                    int plane = net * net; // 채널 평면 크기
                    for (int y = 0; y < net; y++)
                    {
                        byte* row = p0 + y * stride;
                        for (int x = 0; x < net; x++)
                        {
                            byte b = row[x * 3 + 0];
                            byte g = row[x * 3 + 1];
                            byte r = row[x * 3 + 2];
                            int idx = y * net + x;
                            _inBuf[0 * plane + idx] = r * inv255; // R
                            _inBuf[1 * plane + idx] = g * inv255; // G
                            _inBuf[2 * plane + idx] = b * inv255; // B
                        }
                    }
                }
            }
            finally { tmp.UnlockBits(bd); }
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
        public static SegResult Infer(InferenceSession session, Bitmap orig, float conf = 0.9f, float iou = 0.45f, float minBoxAreaRatio = 0.003f, float minMaskAreaRatio = 0.003f, bool discardTouchingBorder = true)
        {
            var sw = Stopwatch.StartNew();
            double tSession = 0, tPrev = 0, tPre = 0, tInfer = 0, tPost = 0;

            // 입력 이름/크기 확인
            string inputName = session.InputMetadata.Keys.First();
            var inMeta = session.InputMetadata[inputName];

            // netSize 결정(메타에 음수가 있으면 기본 640 사용)
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

            // 입력 텐서 채우기
            EnsureInputBuffers(session, netSize);
            FillTensorFromBitmap(orig, netSize, out float scale, out int padX, out int padY, out Size resized);
            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

            // 추론 실행
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = null;
            try
            {
                outputs = session.Run(new[] { _nov });
                tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                // 출력 중 3D(Det), 4D(Proto) 텐서 선택 및 메타 읽기
                for (int oi = 0; oi < outputs.Count; oi++)
                {
                    try { var t = outputs.ElementAt(oi).AsTensor<float>();}
                    catch { }
                }
                var t3 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>(); // Det head
                var t4 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>(); // Proto
                var d3 = t3.Dimensions;
                var d4 = t4.Dimensions;

                int a = d3[1];
                int c = d3[2];
                int segDim = d4[1];
                int mh = d4[2];
                int mw = d4[3];

                // 채널우선/후순(형식 혼동) 대비
                bool channelsFirst = (a <= 512 && c >= 1000) || (a <= segDim + 4 + 256);
                int channels = channelsFirst ? a : c;   // 피쳐 수(=4 + numClasses + segDim)
                int nPred = channelsFirst ? c : a;      // anchor 수(또는 그리드 포인트 수)
                int feat = channels;
                int numClasses = feat - 4 - segDim;
                if (numClasses < 0)
                {
                    channelsFirst = !channelsFirst;
                    channels = channelsFirst ? a : c;
                    nPred = channelsFirst ? c : a;
                    feat = channels;
                    numClasses = feat - 4 - segDim;
                }

                if (numClasses <= 0) throw new InvalidOperationException($"Invalid head layout. feat={feat}, segDim={segDim}");

                // 좌표 스케일 보정: 모델이 [0~1] 스케일 좌표를 내는 케이스 고려
                netSize = Math.Max(netSize, Math.Max(mh, mw) * 4);
                float coordScale = 1f;
                {
                    int sample = Math.Min(nPred, 128);
                    float maxWH = 0f;
                    for (int i = 0; i < sample; i++)
                    {
                        float ww = channelsFirst ? t3[0, 2, i] : t3[0, i, 2];
                        float hh = channelsFirst ? t3[0, 3, i] : t3[0, i, 3];
                        if (ww > maxWH) maxWH = ww;
                        if (hh > maxWH) maxWH = hh;
                    }
                    if (maxWH <= 3.5f) coordScale = netSize; // 너무 작으면 [0~1]로 보고 netSize 곱
                }

                // 박스/점수/클래스/마스크계수 파싱 + conf 필터
                var dets = new List<Det>(256);
                for (int i = 0; i < nPred; i++)
                {
                    float x = (channelsFirst ? t3[0, 0, i] : t3[0, i, 0]) * coordScale;
                    float y = (channelsFirst ? t3[0, 1, i] : t3[0, i, 1]) * coordScale;
                    float w = (channelsFirst ? t3[0, 2, i] : t3[0, i, 2]) * coordScale;
                    float h = (channelsFirst ? t3[0, 3, i] : t3[0, i, 3]) * coordScale;

                    int bestC = 0; float bestS = 0f;
                    for (int cidx = 0; cidx < numClasses; cidx++)
                    {
                        float raw = channelsFirst ? t3[0, 4 + cidx, i] : t3[0, i, 4 + cidx];
                        float s = (raw < 0f || raw > 1f) ? Sigmoid(raw) : raw; // 로짓이면 Sigmoid
                        if (s > bestS) { bestS = s; bestC = cidx; }
                    }
                    if (bestS < conf) continue; // 1차 필터

                    var coeff = new float[segDim];
                    int baseIdx = 4 + numClasses;
                    for (int k = 0; k < segDim; k++) coeff[k] = (channelsFirst ? t3[0, baseIdx + k, i] : t3[0, i, baseIdx + k]);

                    // center(x,y,w,h) → (l,t,r,b)
                    float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, btm = y + h / 2f;
                    dets.Add(new Det { Box = new RectangleF(l, t, r - l, btm - t), Score = bestS, ClassId = bestC, Coeff = coeff });
                }

                if (dets.Count > 0) dets = Nms(dets, iou); // 2차 필터: NMS
                tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                // Proto(4D)는 1차원 배열로 저장
                var protoFlat = t4.ToArray();

                // 결과 패키징
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
                    SessionMs = tSession,
                    PreMs = tPre,
                    InferMs = tInfer,
                    PostMs = tPost,
                    TotalMs = sw.Elapsed.TotalMilliseconds
                };
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

        // 마스크 = sigmoid( proto[k]·coeff[k] 의 합 )
        private static float[] ComputeMaskSIMD(float[] coeff, float[] protoFlat, int segDim, int mw, int mh)
        {
            var mask = new float[mw * mh];
            int vec = Vector<float>.Count;
            System.Threading.Tasks.Parallel.For(0, mh, y =>
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
                        mask[rowOff + x + t] = 1f / (1f + (float)Math.Exp(-s));
                    }
                }
                for (; x < mw; x++)
                {
                    float sum = 0f;
                    for (int k = 0; k < segDim; k++) sum += coeff[k] * protoFlat[(k * mh + y) * mw + x];
                    mask[rowOff + x] = 1f / (1f + (float)Math.Exp(-sum));
                }
            });
            return mask;
        }

        // 원본 ROI 내부에 마스크를 바로 블렌딩(빠른 길)
        private static void BlendMaskIntoOrigROI(Bitmap dstRGBA, Rectangle boxOrig, float[] mask, int mw, int mh, int netSize, float scale, int padX, int padY, float thr, float alpha, Color color)
        {
            // mw/mh(Proto 해상도) → netSize 로 정규화하여 좌표 매핑
            float sx = (float)mw / netSize;
            float sy = (float)mh / netSize;
            byte rr = color.R, gg = color.G, bb = color.B;
            byte thresh = (byte)Math.Max(0, Math.Min(255, (int)(thr * 255f + 0.5f))); // 스레시홀드
            float a = Math.Max(0f, Math.Min(1f, alpha));

            var rect = new Rectangle(0, 0, dstRGBA.Width, dstRGBA.Height);
            var d = dstRGBA.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* p = (byte*)d.Scan0;
                    int stride = d.Stride;
                    int x0 = Math.Max(0, Math.Min(dstRGBA.Width, boxOrig.Left));
                    int y0 = Math.Max(0, Math.Min(dstRGBA.Height, boxOrig.Top));
                    int x1 = Math.Max(0, Math.Min(dstRGBA.Width, boxOrig.Right));
                    int y1 = Math.Max(0, Math.Min(dstRGBA.Height, boxOrig.Bottom));

                    for (int y = y0; y < y1; y++)
                    {
                        byte* row = p + y * stride;
                        for (int x = x0; x < x1; x++)
                        {
                            // 원본 좌표(x,y) → 네트 좌표(패딩 포함)로 이동
                            float xn = x * scale + padX;
                            float yn = y * scale + padY;

                            // 네트 좌표 → Proto 좌표로 비율 변경
                            float u = xn * sx;
                            float v = yn * sy;

                            // bilinear 보간
                            int u0 = (int)Math.Floor(u);
                            int v0 = (int)Math.Floor(v);
                            float fu = u - u0;
                            float fv = v - v0;
                            if (u0 < 0 || v0 < 0 || u0 + 1 >= mw || v0 + 1 >= mh) continue;
                            int idx00 = v0 * mw + u0;
                            int idx10 = v0 * mw + (u0 + 1);
                            int idx01 = (v0 + 1) * mw + u0;
                            int idx11 = (v0 + 1) * mw + (u0 + 1);
                            float m = mask[idx00] * (1 - fu) * (1 - fv) + mask[idx10] * (fu) * (1 - fv) + mask[idx01] * (1 - fu) * (fv) + mask[idx11] * (fu) * (fv);

                            // 스레시홀드 미만은 투명
                            byte mByte = (byte)(m * 255f + 0.5f);
                            if (mByte < thresh) continue;

                            // 단순 알파 블렌딩
                            int di = x * 4;
                            float db = row[di + 0];
                            float dg = row[di + 1];
                            float dr = row[di + 2];
                            row[di + 0] = (byte)(db * (1 - a) + bb * a);
                            row[di + 1] = (byte)(dg * (1 - a) + gg * a);
                            row[di + 2] = (byte)(dr * (1 - a) + rr * a);
                            row[di + 3] = 255;
                        }
                    }
                }
            }
            finally { dstRGBA.UnlockBits(d); }
        }


        // 필요한 using: using System.Threading.Tasks; using System.Drawing.Imaging; (이미 있을 가능성 큼)
        private static void BlendMaskIntoOrigROI2(
            Bitmap dstRGBA, Rectangle boxOrig, float[] mask, int mw, int mh, int netSize,
            float scale, int padX, int padY, float thr, float alpha, Color color)
        {
            // ===== 기존과 동일한 파라미터 처리 =====
            float sx = (float)mw / netSize;
            float sy = (float)mh / netSize;
            byte rr = color.R, gg = color.G, bb = color.B;
            byte thresh = (byte)Math.Max(0, Math.Min(255, (int)(thr * 255f + 0.5f))); // 스레시홀드
            float a = Math.Max(0f, Math.Min(1f, alpha));

            // ROI 클램프 (기존과 동일)
            int x0 = Math.Max(0, boxOrig.Left);
            int y0 = Math.Max(0, boxOrig.Top);
            int x1 = Math.Min(dstRGBA.Width, boxOrig.Right);
            int y1 = Math.Min(dstRGBA.Height, boxOrig.Bottom);
            if (x0 >= x1 || y0 >= y1) return;

            // ===== 좌표 사전 계산 (한 번만) =====
            // u = (x*scale + padX) * sx  →  x만의 함수
            // v = (y*scale + padY) * sy  →  y만의 함수
            int roiW = x1 - x0;
            int roiH = y1 - y0;

            var u0_arr = new int[roiW];
            var fu_arr = new float[roiW];
            for (int i = 0; i < roiW; i++)
            {
                float u = (((x0 + i) * scale) + padX) * sx;
                int u0 = (int)Math.Floor(u);
                u0_arr[i] = u0;
                fu_arr[i] = u - u0;
            }

            var v0_arr = new int[roiH];
            var fv_arr = new float[roiH];
            for (int j = 0; j < roiH; j++)
            {
                float v = (((y0 + j) * scale) + padY) * sy;
                int v0 = (int)Math.Floor(v);
                v0_arr[j] = v0;
                fv_arr[j] = v - v0;
            }

            // ===== 비트맵 잠금 (기존과 동일: 32bppArgb, 알파는 최종 255로 유지) =====
            var rect = new Rectangle(0, 0, dstRGBA.Width, dstRGBA.Height);
            var d = dstRGBA.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);

            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)d.Scan0;
                    int stride = d.Stride;

                    // ===== 병렬 행 처리 =====
                    Parallel.For(0, roiH, j =>
                    {
                        int y = y0 + j;
                        int v0 = v0_arr[j];
                        float fv = fv_arr[j];

                        // 유효한 v 범위를 벗어나면 전체 행 스킵 (원 코드의 continue 분기와 동치)
                        if (v0 < 0 || v0 + 1 >= mh) return;

                        byte* row = basePtr + y * stride;

                        // (1 - fv), (fv) 사전계산
                        float fv0 = 1f - fv;
                        float fv1 = fv;

                        for (int i = 0; i < roiW; i++)
                        {
                            int x = x0 + i;
                            int u0 = u0_arr[i];
                            float fu = fu_arr[i];

                            if (u0 < 0 || u0 + 1 >= mw) continue;

                            // ===== bilinear 보간 (원식 그대로) =====
                            int idx00 = v0 * mw + u0;
                            int idx10 = idx00 + 1;
                            int idx01 = idx00 + mw;
                            int idx11 = idx01 + 1;

                            float fu0 = 1f - fu;
                            float fu1 = fu;

                            float m =
                                mask[idx00] * fu0 * fv0 +
                                mask[idx10] * fu1 * fv0 +
                                mask[idx01] * fu0 * fv1 +
                                mask[idx11] * fu1 * fv1;

                            // 스레시홀드 체크 (원식 동일)
                            byte mByte = (byte)(m * 255f + 0.5f);
                            if (mByte < thresh) continue;

                            // ===== 단순 알파 블렌딩 (원식 그대로, float → byte 캐스트) =====
                            int di = x * 4;
                            float db = row[di + 0];
                            float dg = row[di + 1];
                            float dr = row[di + 2];

                            row[di + 0] = (byte)(db * (1f - a) + bb * a);
                            row[di + 1] = (byte)(dg * (1f - a) + gg * a);
                            row[di + 2] = (byte)(dr * (1f - a) + rr * a);
                            row[di + 3] = 255; // 기존과 동일
                        }
                    });
                }
            }
            finally
            {
                dstRGBA.UnlockBits(d);
            }
        }

        // 필요한 using:
        // using System.Drawing.Imaging;
        // using System.Threading.Tasks;

        private static void BlendMaskIntoOrigROI3(
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

        // 마스크를 원본 크기의 그레이스케일 이미지에 써서(샘플링 포함) 외곽선 추출에 사용
        private static void WriteMaskGrayToOrigWithinROI(Bitmap dstGray, Rectangle boxOrig, float[] mask, int mw, int mh, int netSize, float scale, int padX, int padY)
        {
            float sx = (float)mw / netSize;
            float sy = (float)mh / netSize;
            var rect = new Rectangle(0, 0, dstGray.Width, dstGray.Height);
            var d = dstGray.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* p = (byte*)d.Scan0;
                    int stride = d.Stride;
                    int x0 = Math.Max(0, Math.Min(dstGray.Width, boxOrig.Left));
                    int y0 = Math.Max(0, Math.Min(dstGray.Height, boxOrig.Top));
                    int x1 = Math.Max(0, Math.Min(dstGray.Width, boxOrig.Right));
                    int y1 = Math.Max(0, Math.Min(dstGray.Height, boxOrig.Bottom));
                    for (int y = y0; y < y1; y++)
                    {
                        byte* row = p + y * stride;
                        for (int x = x0; x < x1; x++)
                        {
                            float xn = x * scale + padX;
                            float yn = y * scale + padY;
                            float u = xn * sx;
                            float v = yn * sy;
                            int u0 = (int)Math.Floor(u);
                            int v0 = (int)Math.Floor(v);
                            float fu = u - u0;
                            float fv = v - v0;
                            int di = x * 4;
                            if (u0 < 0 || v0 < 0 || u0 + 1 >= mw || v0 + 1 >= mh)
                            {
                                row[di + 0] = row[di + 1] = row[di + 2] = 0;
                                row[di + 3] = 0;
                                continue;
                            }
                            int idx00 = v0 * mw + u0;
                            int idx10 = v0 * mw + (u0 + 1);
                            int idx01 = (v0 + 1) * mw + u0;
                            int idx11 = (v0 + 1) * mw + (u0 + 1);
                            float m = mask[idx00] * (1 - fu) * (1 - fv) + mask[idx10] * (fu) * (1 - fv) + mask[idx01] * (1 - fu) * (fv) + mask[idx11] * (fu) * (fv);
                            byte mb = (byte)(m * 255f + 0.5f);
                            row[di + 0] = mb;
                            row[di + 1] = mb;
                            row[di + 2] = mb;
                            row[di + 3] = 255;
                        }
                    }
                }
            }
            finally { dstGray.UnlockBits(d); }
        }

        // 스레시홀드 이상 픽셀의 경계만 폴리라인으로 추적해서 오버레이 도형으로 추가
        private static void CollectMaskOutlineOverlaysExact(Bitmap maskGray, byte thrByte, Color color, float widthPx, List<OverlayItem> overlaysOut)
        {
            if (overlaysOut == null) return;
            int w = maskGray.Width, h = maskGray.Height;
            var rect = new Rectangle(0, 0, w, h);
            var data = maskGray.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    bool[,] B = new bool[w, h];
                    byte* basePtr = (byte*)data.Scan0;
                    int stride = data.Stride;

                    // 이웃 픽셀과 다르면 경계 segment 생성
                    for (int y = 0; y < h; y++)
                    {
                        byte* row = basePtr + y * stride;
                        for (int x = 0; x < w; x++) B[x, y] = row[x * 4] >= thrByte;
                    }
                    var segs = new List<(PointF A, PointF B)>();
                    for (int y = 0; y < h; y++) for (int x = 1; x < w; x++) if (B[x - 1, y] != B[x, y]) segs.Add((new PointF(x, y), new PointF(x, y + 1)));
                    for (int y = 1; y < h; y++) for (int x = 0; x < w; x++) if (B[x, y - 1] != B[x, y]) segs.Add((new PointF(x, y), new PointF(x + 1, y)));

                    // segment들을 폴리라인으로 체인
                    var dict = new Dictionary<(int, int), List<int>>();
                    for (int i = 0; i < segs.Count; i++)
                    {
                        var a = ((int)segs[i].A.X, (int)segs[i].A.Y);
                        var b = ((int)segs[i].B.X, (int)segs[i].B.Y);
                        if (!dict.TryGetValue(a, out var la)) dict[a] = la = new List<int>(); la.Add(i);
                        if (!dict.TryGetValue(b, out var lb)) dict[b] = lb = new List<int>(); lb.Add(i);
                    }

                    var used = new bool[segs.Count];
                    var polylines = new List<PointF[]>();
                    for (int i = 0; i < segs.Count; i++)
                    {
                        if (used[i]) continue;
                        used[i] = true;
                        var path = new List<PointF> { segs[i].A, segs[i].B };
                        bool extended = true;
                        while (extended)
                        {
                            extended = false;
                            var end = path[path.Count - 1];
                            var key = ((int)end.X, (int)end.Y);
                            if (dict.TryGetValue(key, out var lst))
                            {
                                for (int k = lst.Count - 1; k >= 0; k--)
                                {
                                    int si = lst[k]; if (used[si]) continue;
                                    var s = segs[si];
                                    if ((int)s.A.X == (int)end.X && (int)s.A.Y == (int)end.Y) { path.Add(s.B); used[si] = true; extended = true; break; }
                                    if ((int)s.B.X == (int)end.X && (int)s.B.Y == (int)end.Y) { path.Add(s.A); used[si] = true; extended = true; break; }
                                }
                            }
                            if (!extended)
                            {
                                var beg = path[0];
                                var key2 = ((int)beg.X, (int)beg.Y);
                                if (dict.TryGetValue(key2, out var lst2))
                                {
                                    for (int k = lst2.Count - 1; k >= 0; k--)
                                    {
                                        int si = lst2[k]; if (used[si]) continue;
                                        var s = segs[si];
                                        if ((int)s.A.X == (int)beg.X && (int)s.A.Y == (int)beg.Y) { path.Insert(0, s.B); used[si] = true; extended = true; break; }
                                        if ((int)s.B.X == (int)beg.X && (int)s.B.Y == (int)beg.Y) { path.Insert(0, s.A); used[si] = true; extended = true; break; }
                                    }
                                }
                            }
                        }
                        polylines.Add(path.ToArray());
                    }

                    foreach (var poly in polylines)
                    {
                        overlaysOut.Add(new OverlayItem
                        {
                            Kind = OverlayKind.Polyline,
                            PointsImg = poly,
                            Closed = true,
                            StrokeColor = color,
                            StrokeWidthPx = widthPx
                        });
                    }
                }
            }
            finally { maskGray.UnlockBits(data); }
        }

        // Overlay: 참고 구현(정확하지만 단계가 많음)
        public static Bitmap Overlay(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false, bool drawScores = true, List<SmartLabelingApp.ImageCanvas.OverlayItem> overlaysOut = null)
        {
            if (orig == null) throw new ArgumentNullException(nameof(orig));
            if (r == null) throw new ArgumentNullException(nameof(r));
            var over = (Bitmap)orig.Clone();
            using (var g = Graphics.FromImage(over))
            {
                g.DrawImage(orig, 0, 0, orig.Width, orig.Height);
                if (r.Dets == null || r.Dets.Count == 0) return over;

                int segDim = r.SegDim;
                int mh = r.MaskH, mw = r.MaskW;
                var proto = r.ProtoFlat;
                Func<int, int, int, float> ProtoAt = (k, y, x) => proto[(k * mh + y) * mw + x];

                foreach (var d in r.Dets)
                {
                    // 1) Proto·Coeff 합성 → 시그모이드로 마스크 생성
                    var mask = new float[mh * mw];
                    for (int yy = 0; yy < mh; yy++)
                    {
                        int rowOff = yy * mw;
                        for (int xx = 0; xx < mw; xx++)
                        {
                            float v = 0f;
                            for (int k = 0; k < segDim; k++) v += d.Coeff[k] * ProtoAt(k, yy, xx);
                            mask[rowOff + xx] = Sigmoid(v);
                        }
                    }

                    // 2) 마스크를 net 크기로 업샘플 → 패딩 영역 제거 → 원본 크기로 리사이즈
                    using (var maskBmp = FloatMaskToBitmap(mask, mw, mh))
                    using (var up = ResizeBitmap(maskBmp, r.NetSize, r.NetSize))
                    {
                        int cx = Clamp(r.PadX, 0, up.Width);
                        int cy = Clamp(r.PadY, 0, up.Height);
                        int cw = Math.Min(r.Resized.Width, up.Width - cx);
                        int ch = Math.Min(r.Resized.Height, up.Height - cy);
                        if (cw <= 0 || ch <= 0) continue;

                        using (var cropped = CropBitmap(up, new Rectangle(cx, cy, cw, ch)))
                        using (var toOrig = ResizeBitmap(cropped, orig.Width, orig.Height))
                        {
                            // 3) 박스 ROI 밖은 0으로 (마스크 누수 방지)
                            var boxOrig = NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);
                            ZeroOutsideRect(toOrig, boxOrig);

                            // 4) 색상+알파로 합성 + 외곽선/박스/배지 오버레이
                            var color = ClassColor(d.ClassId);
                            AlphaBlendMask(over, toOrig, color, maskThr, alpha);
                            byte thrByte = (byte)(maskThr * 255f + 0.5f);
                            if (overlaysOut != null) CollectMaskOutlineOverlaysExact(toOrig, thrByte, color, 2f, overlaysOut);
                            if (drawScores && overlaysOut != null)
                                overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem { Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Badge, Text = $"[{d.ClassId}]: {d.Score:0.00}", BoxImg = boxOrig, StrokeColor = color });
                            if (drawBoxes && overlaysOut != null)
                                overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem { Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Box, BoxImg = boxOrig, StrokeColor = color, StrokeWidthPx = 3f });
                        }
                    }
                }
            }
            return over;
        }

        // OverlayFast: ROI에서 바로 샘플링/블렌딩(더 빠름)


        public static Bitmap OverlayFast(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false, bool drawScores = true, bool fillMask = true, bool drawoutlines = true, List < SmartLabelingApp.ImageCanvas.OverlayItem> overlaysOut = null)
        {
            if (orig == null) throw new ArgumentNullException(nameof(orig));
            if (r == null) throw new ArgumentNullException(nameof(r));
            var over = (Bitmap)orig.Clone();
            using (var g = Graphics.FromImage(over))
            {
                g.DrawImage(orig, 0, 0, orig.Width, orig.Height);
                if (r.Dets == null || r.Dets.Count == 0) return over;

                int segDim = r.SegDim;
                int mh = r.MaskH, mw = r.MaskW;
                var proto = r.ProtoFlat;
                int maskLen = mw * mh;
                if (_maskBufTLS == null || _maskBufTLS.Length < maskLen) _maskBufTLS = new float[maskLen];

                foreach (var d in r.Dets)
                {
                    // 1) SIMD로 마스크 생성
                    var mask = _maskBufTLS;               // 재사용 버퍼
                    ComputeMaskSIMD_NoAlloc(d.Coeff, proto, segDim, mw, mh, mask);

                    // 2) 박스 원본 좌표로 변환
                    var boxOrig = typeof(YoloSegOnnx).GetMethod("NetBoxToOriginal", BindingFlags.NonPublic | BindingFlags.Static).Invoke(null, new object[] { d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize });
                    var box = (Rectangle)boxOrig;
                    var color = ClassColor(d.ClassId);

                    if (fillMask)
                    {
                        // 3) ROI 내부에 바로 블렌딩
                        BlendMaskIntoOrigROI3(over, box, mask, mw, mh, r.NetSize, r.Scale, r.PadX, r.PadY, maskThr, alpha, color);
                    }
                    if (drawoutlines)
                    {
                        if (overlaysOut != null)
                        {
                            using (var toOrig = new Bitmap(orig.Width, orig.Height, PixelFormat.Format32bppArgb))
                            {
                                WriteMaskGrayToOrigWithinROI(toOrig, box, mask, mw, mh, r.NetSize, r.Scale, r.PadX, r.PadY);
                                byte thrByte = (byte)(maskThr * 255f + 0.5f);
                                CollectMaskOutlineOverlaysExact(toOrig, thrByte, color, 2f, overlaysOut);
                            }
                        }
                    }
                    if (overlaysOut != null)
                    {
                        if (drawScores)
                            overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem { Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Badge, Text = $"[{GetModeName(d.ClassId)}]: {d.Score:0.00}", BoxImg = box, StrokeColor = color });
                        if (drawBoxes)
                            overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem { Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Box, BoxImg = box, StrokeColor = color, StrokeWidthPx = 3f });
                    }
                }
            }
            return over;
        }

        // Render: OverlayFast 실행 + 오버레이 목록을 같이 반환
        public static OverlayResult Render(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false, bool drawScores = true)
        {
            var list = new List<SmartLabelingApp.ImageCanvas.OverlayItem>();
            var bmp = OverlayFast(orig, r, maskThr, alpha, drawBoxes, drawScores, overlaysOut: list);
            return new OverlayResult { Image = bmp, Overlays = list };
        }

        public static OverlayResult RenderInputPreview(Bitmap src, InferenceSession session = null, int? forceNet = null, System.Drawing.Color? padColor = null)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));

            int net = 640;
            if (forceNet.HasValue) net = forceNet.Value;
            else if (session != null)
            {
                try
                {
                    var inputName = session.InputMetadata.Keys.First();
                    var dims = session.InputMetadata[inputName].Dimensions;
                    if (dims.Length == 4)
                    {
                        int h = dims[2] > 0 ? dims[2] : 640;
                        int w = dims[3] > 0 ? dims[3] : 640;
                        net = Math.Max(h, w);
                    }
                }
                catch { /* 기본 640 유지 */ }
            }

            // YoloSegOnnx.FillTensorFromBitmap 과 동일한 스케일/패딩 로직으로 캔버스 생성
            int W = src.Width, H = src.Height;
            float scale = Math.Min((float)net / W, (float)net / H);
            int rw = (int)Math.Round(W * scale);
            int rh = (int)Math.Round(H * scale);
            int padX = (net - rw) / 2;
            int padY = (net - rh) / 2;

            var canvas = new Bitmap(net, net, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(canvas))
            {
                g.Clear(padColor ?? Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.Bilinear;
                g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;
                g.DrawImage(src, padX, padY, rw, rh);
            }

            return new OverlayResult
            {
                Image = canvas,
                Overlays = new List<SmartLabelingApp.ImageCanvas.OverlayItem>() // 비움
            };
        }


        #endregion

        #region Utils
        public static string GetModeName(int value)
        {
            return Enum.IsDefined(typeof(LabelName), value)
                ? ((LabelName)value).ToString()
                : $"Unknown({value})";
        }

        // 네트 좌표 박스를 원본 좌표 박스로 변환(레터박스 역변환)
        private static Rectangle NetBoxToOriginal(RectangleF boxNet, float scale, int padX, int padY, Size resized, Size orig)
        {
            float l = (boxNet.Left - padX) / scale;
            float t = (boxNet.Top - padY) / scale;
            float r = (boxNet.Right - padX) / scale;
            float b = (boxNet.Bottom - padY) / scale;
            int x = (int)Math.Round(Clamp(l, 0f, orig.Width - 1));
            int y = (int)Math.Round(Clamp(t, 0f, orig.Height - 1));
            int xr = (int)Math.Round(Clamp(r, 0f, orig.Width - 1));
            int yb = (int)Math.Round(Clamp(b, 0f, orig.Height - 1));
            int w = xr - x;
            int h = yb - y;
            return new Rectangle(x, y, Math.Max(1, w), Math.Max(1, h));
        }

        // ROI 밖은 마스크를 0으로(누수 방지)
        private static void ZeroOutsideRect(Bitmap maskGray, Rectangle rect)
        {
            rect.Intersect(new Rectangle(0, 0, maskGray.Width, maskGray.Height));
            var full = new Rectangle(0, 0, maskGray.Width, maskGray.Height);
            var d = maskGray.LockBits(full, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)d.Scan0;
                    int stride = d.Stride;
                    for (int y = 0; y < maskGray.Height; y++)
                    {
                        byte* row = basePtr + y * stride;
                        for (int x = 0; x < maskGray.Width; x++)
                        {
                            if (x < rect.Left || x >= rect.Right || y < rect.Top || y >= rect.Bottom)
                            {
                                int idx = x * 4;
                                row[idx + 0] = 0;
                                row[idx + 1] = 0;
                                row[idx + 2] = 0;
                            }
                        }
                    }
                }
            }
            finally { maskGray.UnlockBits(d); }
        }

        // 알파 블렌딩(스레시홀드 이상만 색을 입힘)
        private static void AlphaBlendMask(Bitmap dstRGBA, Bitmap maskGray, Color color, float thr, float alpha)
        {
            var rect = new Rectangle(0, 0, dstRGBA.Width, dstRGBA.Height);
            var dDst = dstRGBA.LockBits(rect, ImageLockMode.ReadWrite, PixelFormat.Format32bppArgb);
            var dMsk = maskGray.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* pDst = (byte*)dDst.Scan0;
                    byte* pMsk = (byte*)dMsk.Scan0;
                    int strideDst = dDst.Stride;
                    int strideMsk = dMsk.Stride;
                    byte rr = color.R, gg = color.G, bb = color.B;
                    byte thresh = (byte)Clamp(thr * 255f, 0f, 255f);
                    float a = Clamp(alpha, 0f, 1f);
                    for (int y = 0; y < rect.Height; y++)
                    {
                        byte* rowDst = pDst + y * strideDst;
                        byte* rowMsk = pMsk + y * strideMsk;
                        for (int x = 0; x < rect.Width; x++)
                        {
                            byte m = rowMsk[x * 4 + 0];
                            if (m >= thresh)
                            {
                                int idx = x * 4;
                                float db = rowDst[idx + 0];
                                float dg = rowDst[idx + 1];
                                float dr = rowDst[idx + 2];
                                rowDst[idx + 0] = (byte)(db * (1 - a) + bb * a);
                                rowDst[idx + 1] = (byte)(dg * (1 - a) + gg * a);
                                rowDst[idx + 2] = (byte)(dr * (1 - a) + rr * a);
                                rowDst[idx + 3] = 255;
                            }
                        }
                    }
                }
            }
            finally { dstRGBA.UnlockBits(dDst); maskGray.UnlockBits(dMsk); }
        }

        // float 마스크 → 8비트 그레이 비트맵
        private static Bitmap FloatMaskToBitmap(float[] m, int w, int h)
        {
            var bmp = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            var rect = new Rectangle(0, 0, w, h);
            var d = bmp.LockBits(rect, ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            try
            {
                unsafe
                {
                    byte* p = (byte*)d.Scan0;
                    int stride = d.Stride;
                    for (int y = 0; y < h; y++)
                    {
                        byte* row = p + y * stride;
                        int off = y * w;
                        for (int x = 0; x < w; x++)
                        {
                            byte v = ClampByte((int)(m[off + x] * 255f + 0.5f));
                            int idx = x * 4;
                            row[idx + 0] = v;
                            row[idx + 1] = v;
                            row[idx + 2] = v;
                            row[idx + 3] = 255;
                        }
                    }
                }
            }
            finally { bmp.UnlockBits(d); }
            return bmp;
        }

        // 간단 리사이즈
        private static Bitmap ResizeBitmap(Bitmap src, int w, int h)
        {
            var dst = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(dst)) { g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic; g.DrawImage(src, 0, 0, w, h); }
            return dst;
        }

        // ROI 크롭
        private static Bitmap CropBitmap(Bitmap src, Rectangle roi)
        {
            var dst = new Bitmap(roi.Width, roi.Height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(dst)) { g.DrawImage(src, new Rectangle(0, 0, roi.Width, roi.Height), roi, GraphicsUnit.Pixel); }
            return dst;
        }

        // 클래스별 색 (간단 난수 기반)
        private static Color ClassColor(int cls)
        {
            var rng = new Random(cls * 123457);
            return Color.FromArgb(255, rng.Next(64, 255), rng.Next(64, 255), rng.Next(64, 255));
        }

        // 수학/도움 함수
        private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));
        private static float Clamp(float v, float min, float max) => v < min ? min : (v > max ? max : v);
        private static int Clamp(int v, int min, int max) => v < min ? min : (v > max ? max : v);
        private static byte ClampByte(int v) => (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));

        // NMS
        private static List<Det> Nms(List<Det> dets, float iouThr)
        {
            var keep = new List<Det>();
            var sorted = dets.OrderByDescending(d => d.Score).ToList();
            while (sorted.Count > 0)
            {
                var a = sorted[0];
                keep.Add(a);
                sorted.RemoveAt(0);
                for (int i = sorted.Count - 1; i >= 0; i--) if (IoU(a.Box, sorted[i].Box) > iouThr) sorted.RemoveAt(i);
            }
            return keep;
        }

        private static float IoU(RectangleF a, RectangleF b)
        {
            float x1 = Math.Max(a.Left, b.Left);
            float y1 = Math.Max(a.Top, b.Top);
            float x2 = Math.Min(a.Right, b.Right);
            float y2 = Math.Min(a.Bottom, b.Bottom);
            float inter = Math.Max(0, x2 - x1) * Math.Max(0, y2 - y1);
            float ua = a.Width * a.Height + b.Width * b.Height - inter + 1e-6f;
            return inter / ua;
        }

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
