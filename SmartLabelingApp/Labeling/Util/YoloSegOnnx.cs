// File: YoloSegOnnx.cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using System.Threading;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    public static class YoloSegOnnx
    {
        const int LABEL_BADGE_GAP_PX = 2;
        const int LABEL_BADGE_PADX = 4;
        const int LABEL_BADGE_PADY = 3;
        const int LABEL_BADGE_BORDER_PX = 2;
        const int LABEL_BADGE_ACCENT_W = 4;
        const int LABEL_BADGE_WIPE_PX = 1;

        // ------- Logging -------
        private static void Log(string msg)
        {
            var line = $"[YoloSegOnnx {DateTime.Now:HH:mm:ss.fff}] {msg}";
            try { Debug.WriteLine(line); } catch { }
        }
        private static string JoinDims(ReadOnlySpan<int> dims)
        {
            if (dims.Length == 0) return "";
            var sb = new StringBuilder();
            for (int i = 0; i < dims.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(dims[i].ToString());
            }
            return sb.ToString();
        }

        // ===== 세션 캐싱 =====
        private static InferenceSession _cachedSession = null;
        private static string _cachedModelPath = null;
        private static readonly object _sessionLock = new object();

        private static InferenceSession GetOrCreateSession(string modelPath, out double initMs)
        {
            lock (_sessionLock)
            {
                if (_cachedSession != null &&
                    string.Equals(_cachedModelPath, modelPath, StringComparison.OrdinalIgnoreCase))
                {
                    initMs = 0; // 이미 로딩된 세션 재사용
                    return _cachedSession;
                }

                var sw = Stopwatch.StartNew();
                var sess = CreateSessionWithFallback(modelPath);
                initMs = sw.Elapsed.TotalMilliseconds;

                if (_cachedSession != null)
                {
                    try { _cachedSession.Dispose(); } catch { }
                }
                _cachedSession = sess;
                _cachedModelPath = modelPath;
                return _cachedSession;
            }
        }

        public sealed class SegResult
        {
            // 네트워크/리사이즈 관련(좌표 복원용)
            public int NetSize { get; set; }           // ex) 640
            public float Scale { get; set; }           // Letterbox scale
            public int PadX { get; set; }              // Letterbox padX
            public int PadY { get; set; }              // Letterbox padY
            public Size Resized { get; set; }          // Letterbox resized size (w,h)
            public Size OrigSize { get; set; }         // 원본 이미지 크기

            // 출력(후처리) 결과
            public List<Det> Dets { get; set; }        // 박스/클래스/점수/계수
            public int SegDim { get; set; }            // t4 채널 수
            public int MaskH { get; set; }             // proto H
            public int MaskW { get; set; }             // proto W

            // 마스크 합성을 위한 중간데이터(필수)
            // Proto는 [segDim, mh, mw]로 평탄화한 float[] (t4.ToArray())
            public float[] ProtoFlat { get; set; }

            // 타이밍
            public double SessionMs { get; set; }
            public double PreMs { get; set; }
            public double InferMs { get; set; }
            public double PostMs { get; set; }
            public double TotalMs { get; set; }
        }

        public class Det
        {
            public RectangleF Box;   // net 입력 좌표계(xyxy)
            public float Score;
            public int ClassId;
            public float[] Coeff;    // seg_dim
        }
        public static SegResult Infer(InferenceSession session, Bitmap orig, float conf = 0.9f, float iou = 0.45f, float minBoxAreaRatio = 0.003f, float minMaskAreaRatio = 0.003f, bool discardTouchingBorder = true)
        {
            Log("Infer(session, bitmap) called");

            var sw = Stopwatch.StartNew();
            double tSession = 0, tPrev = 0, tPre = 0, tInfer = 0, tPost = 0;

            // 0) 세션/입력 메타
            string inputName = session.InputMetadata.Keys.First();
            var inMeta = session.InputMetadata[inputName];
            Log($"Session OK. Input name: {inputName}");
            try { Log($"Input dims: [{JoinDims(inMeta.Dimensions)}]"); } catch { }

            int netH = 640, netW = 640;
            try
            {
                var dims = inMeta.Dimensions; // [N,C,H,W] or [-1,3,-1,-1]
                if (dims.Length == 4)
                {
                    if (dims[2] > 0) netH = dims[2];
                    if (dims[3] > 0) netW = dims[3];
                }
            }
            catch { }
            int netSize = Math.Max(netH, netW);
            Log($"Chosen netSize: {netSize}");

            // 1) 전처리
            using (var boxed = Letterbox(orig, netSize, out float scale, out int padX, out int padY, out Size resized))
            {
                Log($"Letterbox: scale={scale:F6}, padX={padX}, padY={padY}, resized={resized.Width}x{resized.Height}");
                var input = ToCHWTensor(boxed);
                Log($"Input tensor ready: [1,3,{boxed.Height},{boxed.Width}]");

                tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                // 2) 추론
                IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = null;
                try
                {
                    var inputsList = new List<NamedOnnxValue>(1)
            {
                NamedOnnxValue.CreateFromTensor<float>(inputName, input)
            };

                    outputs = session.Run(inputsList);
                    Log($"session.Run() returned {outputs.Count} outputs");

                    tInfer = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                    for (int oi = 0; oi < outputs.Count; oi++)
                    {
                        try
                        {
                            var t = outputs.ElementAt(oi).AsTensor<float>();
                            Log($"  out[{oi}] dims: [{JoinDims(t.Dimensions)}]");
                        }
                        catch { Log($"  out[{oi}] not a float tensor?"); }
                    }

                    // 3) 후처리(Det + Proto 확보) — 합성은 하지 않음
                    var t3 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>();
                    var t4 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>();

                    var d3 = t3.Dimensions;    // [1,37,8400] or [1,8400,37]
                    var d4 = t4.Dimensions;    // [1, segDim, mh, mw]
                    int a = d3[1];
                    int c = d3[2];

                    int segDim = d4[1];
                    int mh = d4[2];
                    int mw = d4[3];

                    bool channelsFirst = (a <= 512 && c >= 1000) || (a <= segDim + 4 + 256);
                    int channels = channelsFirst ? a : c;
                    int nPred = channelsFirst ? c : a;
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

                    Log($"Parsed heads:");
                    Log($"  det: channelsFirst={channelsFirst}, channels={channels}, nPred={nPred}, feat={feat}");
                    Log($"  proto: segDim={segDim}, mh={mh}, mw={mw}");
                    Log($"  numClasses={numClasses}");

                    if (numClasses <= 0)
                        throw new InvalidOperationException($"Invalid head layout. feat={feat}, segDim={segDim}");

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
                        if (maxWH <= 3.5f) coordScale = netSize;
                    }
                    Log($"coordScale={coordScale:F6}");

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
                            float raw = channelsFirst
                                ? t3[0, 4 + cidx, i]
                                : t3[0, i, 4 + cidx];

                            float s = (raw < 0f || raw > 1f) ? Sigmoid(raw) : raw;
                            if (s > bestS) { bestS = s; bestC = cidx; }
                        }
                        if (bestS < conf) continue;

                        var coeff = new float[segDim];
                        int baseIdx = 4 + numClasses;
                        for (int k = 0; k < segDim; k++)
                        {
                            coeff[k] = channelsFirst
                                ? t3[0, baseIdx + k, i]
                                : t3[0, i, baseIdx + k];
                        }

                        float l = x - w / 2f, t = y - h / 2f, r = x + w / 2f, btm = y + h / 2f;

                        dets.Add(new Det
                        {
                            Box = new RectangleF(l, t, r - l, btm - t),
                            Score = bestS,
                            ClassId = bestC,
                            Coeff = coeff
                        });
                    }
                    Log($"Detections after conf filter: {dets.Count}");

                    if (dets.Count > 0)
                        dets = Nms(dets, iou);
                    Log($"Detections after NMS (iou={iou}): {dets.Count}");

                    tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                    // Proto 보관 (합성 단계에서 사용)
                    var protoFlat = t4.ToArray(); // [1, segDim, mh, mw] → 평탄화

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
                finally
                {
                    if (outputs != null) outputs.Dispose();
                }
            }
        }

        // === 완전 분리: 합성만 수행 (원본 + SegResult → 새 Bitmap 반환) ===
        public static Bitmap Overlay(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false, bool drawScores = true, List<ImageCanvas.InferenceBadge> badgesOut = null)
        {
            if (orig == null) throw new ArgumentNullException(nameof(orig));
            if (r == null) throw new ArgumentNullException(nameof(r));

            var over = new Bitmap(orig.Width, orig.Height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(over))
            {
                g.DrawImage(orig, 0, 0, orig.Width, orig.Height);

                if (r.Dets == null || r.Dets.Count == 0)
                    return over;

                int segDim = r.SegDim;
                int mh = r.MaskH, mw = r.MaskW;
                var proto = r.ProtoFlat;
                Func<int, int, int, float> ProtoAt = (k, y, x) => proto[(k * mh + y) * mw + x];

                int di = 0;
                foreach (var d in r.Dets)
                {
                    // === (1) 마스크 처리 기존 그대로 ===
                    var mask = new float[mh * mw];
                    for (int yy = 0; yy < mh; yy++)
                    {
                        int rowOff = yy * mw;
                        for (int xx = 0; xx < mw; xx++)
                        {
                            float v = 0f;
                            for (int k = 0; k < segDim; k++)
                                v += d.Coeff[k] * ProtoAt(k, yy, xx);
                            mask[rowOff + xx] = Sigmoid(v);
                        }
                    }

                    using (var maskBmp = FloatMaskToBitmap(mask, mw, mh))
                    using (var up = ResizeBitmap(maskBmp, r.NetSize, r.NetSize))
                    {
                        int cx = Clamp(r.PadX, 0, up.Width);
                        int cy = Clamp(r.PadY, 0, up.Height);
                        int cw = Math.Min(r.Resized.Width, up.Width - cx);
                        int ch = Math.Min(r.Resized.Height, up.Height - cy);
                        if (cw <= 0 || ch <= 0) { di++; continue; }

                        using (var cropped = CropBitmap(up, new Rectangle(cx, cy, cw, ch)))
                        using (var toOrig = ResizeBitmap(cropped, orig.Width, orig.Height))
                        {
                            var boxOrig = NetBoxToOriginal(d.Box, r.Scale, r.PadX, r.PadY, r.Resized, r.OrigSize);
                            ZeroOutsideRect(toOrig, boxOrig);

                            var color = ClassColor(d.ClassId);
                            AlphaBlendMask(over, toOrig, color, maskThr, alpha);

                            byte thrByte = (byte)(maskThr * 255f + 0.5f);
                            DrawMaskOutline(toOrig, g, color, 2, thrByte);

                            if (drawBoxes)
                            {
                                using (var pen = new Pen(color, 2))
                                    g.DrawRectangle(pen, boxOrig);
                            }

                            // === (2) 점수/라벨은 "화면 렌더링" 단계로 넘긴다 ===
                            if (drawScores && badgesOut != null)
                            {
                                // 표기 문자열: 필요에 따라 클래스명/점수 포맷 변경
                                string labelText = $"[{d.ClassId}], {d.Score:0.00}";
                                // 이미지 좌표의 박스 그대로 넘김 (ImageCanvas에서 화면좌표로 변환해서 그림)
                                badgesOut.Add(new ImageCanvas.InferenceBadge
                                {
                                    BoxImg = (RectangleF)boxOrig,
                                    Text = labelText,
                                    Accent = color
                                });
                            }

                            // ※ 더 이상 여기서 g.DrawString/DrawLabel 하지 않는다 (화질 저하 방지)
                        }
                    }
                    di++;
                }
            }
            return over;
        }

        // 진행률 헬퍼: 퍼센트 상승만 허용
        private static void ReportStep(IProgress<(int percent, string status)> progress, ref int cur, int next, string msg)
        {
            if (progress == null) return;
            if (next < cur) next = cur;
            if (next > 100) next = 100;
            cur = next;
            Thread.Sleep(50);
            progress.Report((cur, msg));
        }

        /// <summary>
        /// Open에서 호출: 세션을 미리 만들고(또는 재사용) 여러 단계로 진행률 보고
        /// </summary>
        public static InferenceSession EnsureSession(string modelPath, IProgress<(int percent, string status)> progress = null)
        {
            int p = 0;
            ReportStep(progress, ref p, 5, "모델 경로 확인");

            // 캐시 히트면 거의 즉시 끝나므로 그 사이도 단계적으로 표시
            double initMs = 0;
            bool createdNew = false;

            // 1) EP 선택/옵션 구성 (GetOrCreateSession 내부에서 하더라도, 사용자 체감용 단계 분리)
            ReportStep(progress, ref p, 12, "Execution Provider 확인");
            ReportStep(progress, ref p, 20, "세션 옵션 구성");

            // 2) 세션 생성/재사용
            ReportStep(progress, ref p, 35, "세션 생성 준비");
            var session = GetOrCreateSession(modelPath, out initMs); // 내부 캐시/EP 폴백 사용
            createdNew = initMs > 0;

            if (createdNew)
            {
                // 실제 세션 생성이 시간이 걸리면 여기서 크게 점프가 보임
                ReportStep(progress, ref p, 60, $"세션 생성 완료 ({initMs:F0} ms)");
            }
            else
            {
                // 재사용이면 단계는 빠르게 지나가되, 구간은 동일하게 보여줌
                ReportStep(progress, ref p, 45, "캐시된 세션 재사용");
                ReportStep(progress, ref p, 60, "세션 확인 완료");
            }

            // 3) 메타데이터/입출력 검사 (보통 빠르지만 사용자에게 단계감 제공)
            try
            {
                ReportStep(progress, ref p, 70, "모델 IO 메타데이터 읽기");
                var inputs = session.InputMetadata;   // touch
                var outputs = session.OutputMetadata; // touch
                ReportStep(progress, ref p, 78, $"입력 {inputs.Count} / 출력 {outputs.Count} 확인");
            }
            catch
            {
                // 메타 조회 실패해도 세션 자체는 유효할 수 있음 — 진행 계속
                ReportStep(progress, ref p, 78, "IO 메타데이터 확인(옵션) 건너뜀");
            }

            // 4) (선택) 웜업 준비 — 실제 추론은 하지 않되, 사용자 단계감 제공
            ReportStep(progress, ref p, 85, "웜업 준비");
            // 실제 웜업 추론을 여기서 하려면 입력 텐서 준비가 필요함(모델별 상이).
            // 현재는 세션 로드만 목표라서 단계 표시만 하고 넘어감.
            ReportStep(progress, ref p, 92, "리소스 초기화");

            // 5) 완료
            ReportStep(progress, ref p, 100, "완료");

            return session;
        }


        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            // 항상 x64 빌드 전제
            SessionOptions so = null;

            // 1) CUDA EP
            try
            {
                so = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };
                Log("Trying CUDA EP...");
                so.AppendExecutionProvider_CUDA(deviceId: 0); // requires OnnxRuntime.Gpu
                var s = new InferenceSession(modelPath, so);
                Log("Session created with CUDA EP.");
                return s;
            }
            catch (Exception e)
            {
                Log("CUDA EP failed, will try DML EP.");
                Log(e.ToString());
                try { so?.Dispose(); } catch { }
            }

            // 2) DirectML EP
            try
            {
                so = new SessionOptions
                {
                    GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
                };
                Log("Trying DML EP...");
                so.AppendExecutionProvider_DML(); // Windows GPU 드라이버만 있으면 동작
                var s = new InferenceSession(modelPath, so);
                Log("Session created with DML EP.");
                return s;
            }
            catch (Exception e)
            {
                Log("DML EP failed, will fall back to CPU.");
                Log(e.ToString());
                try { so?.Dispose(); } catch { }
            }

            // 3) CPU
            var soCpu = new SessionOptions
            {
                GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL
            };
            Log("Creating CPU session...");
            var sCpu = new InferenceSession(modelPath, soCpu);
            Log("CPU session created.");
            return sCpu;
        }


        // ===== Utils =====
        private static Bitmap Letterbox(Bitmap src, int net, out float scale, out int padX, out int padY, out Size resized)
        {
            float r = Math.Min(net / (float)src.Width, net / (float)src.Height);
            int nw = (int)Math.Round(src.Width * r);
            int nh = (int)Math.Round(src.Height * r);

            padX = (net - nw) / 2;
            padY = (net - nh) / 2;
            scale = r;
            resized = new Size(nw, nh);

            var canvas = new Bitmap(net, net, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(canvas))
            {
                g.Clear(Color.Black);
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, padX, padY, nw, nh);
            }
            return canvas;
        }

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

        private static void DrawMaskOutline(Bitmap maskGray, Graphics g, Color color, int thickness, byte threshold, int step = 1)
        {
            var rect = new Rectangle(0, 0, maskGray.Width, maskGray.Height);
            BitmapData d = maskGray.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            try
            {
                using (var pen = new Pen(color, thickness))
                {
                    pen.Alignment = System.Drawing.Drawing2D.PenAlignment.Center;
                    unsafe
                    {
                        byte* basePtr = (byte*)d.Scan0;
                        int stride = d.Stride;
                        int w = rect.Width, h = rect.Height;

                        for (int y = 0; y < h; y += step)
                        {
                            byte* row = basePtr + y * stride;
                            byte* nextRow = (y + step < h) ? (basePtr + (y + step) * stride) : null;

                            for (int x = 0; x < w; x += step)
                            {
                                int idx = x * 4;
                                bool on = row[idx] >= threshold;

                                if (x + step < w)
                                {
                                    bool onR = row[(x + step) * 4] >= threshold;
                                    if (on != onR) g.DrawLine(pen, x + step, y, x + step, Math.Min(y + step, h));
                                }
                                if (nextRow != null)
                                {
                                    bool onD = nextRow[idx] >= threshold;
                                    if (on != onD) g.DrawLine(pen, x, y + step, Math.Min(x + step, w), y + step);
                                }
                            }
                        }
                    }
                }
            }
            finally { maskGray.UnlockBits(d); }
        }

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
            finally
            {
                dstRGBA.UnlockBits(dDst);
                maskGray.UnlockBits(dMsk);
            }
        }

        private static DenseTensor<float> ToCHWTensor(Bitmap bmp)
        {
            int w = bmp.Width, h = bmp.Height;
            var tensor = new DenseTensor<float>(new[] { 1, 3, h, w });

            var rect = new Rectangle(0, 0, w, h);
            var data = bmp.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
            try
            {
                unsafe
                {
                    byte* p = (byte*)data.Scan0;
                    int stride = data.Stride;
                    for (int y = 0; y < h; y++)
                    {
                        byte* row = p + y * stride;
                        for (int x = 0; x < w; x++)
                        {
                            int idx = x * 3;
                            byte b = row[idx + 0];
                            byte g = row[idx + 1];
                            byte r = row[idx + 2];

                            tensor[0, 0, y, x] = r / 255f;
                            tensor[0, 1, y, x] = g / 255f;
                            tensor[0, 2, y, x] = b / 255f;
                        }
                    }
                }
            }
            finally { bmp.UnlockBits(data); }
            return tensor;
        }

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

        private static Bitmap ResizeBitmap(Bitmap src, int w, int h)
        {
            var dst = new Bitmap(w, h, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = System.Drawing.Drawing2D.InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, 0, 0, w, h);
            }
            return dst;
        }

        private static Bitmap CropBitmap(Bitmap src, Rectangle roi)
        {
            var dst = new Bitmap(roi.Width, roi.Height, PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(dst))
            {
                g.DrawImage(src, new Rectangle(0, 0, roi.Width, roi.Height), roi, GraphicsUnit.Pixel);
            }
            return dst;
        }

        private static Color ClassColor(int cls)
        {
            var rng = new Random(cls * 123457);
            return Color.FromArgb(255, rng.Next(64, 255), rng.Next(64, 255), rng.Next(64, 255));
        }

        private static float Sigmoid(float x) => 1f / (1f + (float)Math.Exp(-x));
        private static float Clamp(float v, float min, float max) => v < min ? min : (v > max ? max : v);
        private static int Clamp(int v, int min, int max) => v < min ? min : (v > max ? max : v);
        private static byte ClampByte(int v) => (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));

        private static List<Det> Nms(List<Det> dets, float iouThr)
        {
            var keep = new List<Det>();
            var sorted = dets.OrderByDescending(d => d.Score).ToList();
            while (sorted.Count > 0)
            {
                var a = sorted[0];
                keep.Add(a);
                sorted.RemoveAt(0);
                for (int i = sorted.Count - 1; i >= 0; i--)
                {
                    if (IoU(a.Box, sorted[i].Box) > iouThr) sorted.RemoveAt(i);
                }
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
    }
}
