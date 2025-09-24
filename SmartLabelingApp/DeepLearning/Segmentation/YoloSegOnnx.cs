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
                BlendMaskIntoOrigROI(bmp, box, maskBuf, mw, mh, netSize: Preprocess.DefaultNet,
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
            double initMs = 0;
            var session = GetOrCreateSession(modelPath, out initMs);
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
                Preprocess.EnsureOnnxInput(sess, Preprocess.DefaultNet, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
                TryWarmup(sess, Preprocess.DefaultNet);
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
                Preprocess.EnsureOnnxInput(sess, Preprocess.DefaultNet, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
                TryWarmup(sess, Preprocess.DefaultNet);
                SmartLabelingApp.MainForm._currentRunTypeName = "CPU";
                return sess;
            }
            catch
            {
                try { so?.Dispose(); } catch { }
            }
            var soCpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            var cpu = new InferenceSession(modelPath, soCpu);
            Preprocess.EnsureOnnxInput(cpu, Preprocess.DefaultNet, ref _inputName, ref _curNet, ref _inBuf, ref _tensor, ref _nov);
            SmartLabelingApp.MainForm._currentRunTypeName = "CPU";
            return cpu;
        }

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

    }
}
