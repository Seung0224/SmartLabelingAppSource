// File: YoloSegOnnx.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.Linq;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp
{
    public static class YoloSegOnnx
    {
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


        // ===== 세션 캐싱 추가 (DML/CPU 공통) =====
        private static InferenceSession _cachedSession = null;
        private static string _cachedModelPath = null;
        private static readonly object _sessionLock = new object();

        private static InferenceSession GetOrCreateSession(string modelPath, out double initMs)
        {
            lock (_sessionLock)
            {
                if (_cachedSession != null && string.Equals(_cachedModelPath, modelPath, StringComparison.OrdinalIgnoreCase))
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

        public class SegResult
        {
            public Bitmap Overlayed;
            public List<Det> Dets;

            public double SessionMs;  // 세션 초기화(DML/CPU) 시간
            public double PreMs;      // 전처리 (letterbox+tensor)
            public double InferMs;    // 추론 (session.Run)
            public double PostMs;     // 후처리 (헤드 파싱, NMS 등)
            public double OverlayMs;  // 오버레이 합성
            public double TotalMs;    // 전체(함수 진입~종료)

            // 폼 타이틀에 바로 붙여 쓸 문자열
            public string TitleSuffix; // e.g. "session 320ms, pre 58ms, infer 167ms, post 7ms, overlay 267ms, total 799ms"
        }

        public class Det
        {
            public RectangleF Box;   // net 입력 좌표계(xyxy)
            public float Score;
            public int ClassId;
            public float[] Coeff;    // seg_dim
        }

        // YoloSegOnnx 클래스 안에 추가
        private static void DrawLabel(Graphics g, Rectangle anchor, string text, Color boxColor)
        {
            using (var font = new Font("Segoe UI", 12, FontStyle.Bold))
            {
                // 텍스트 크기 측정
                var sizeF = g.MeasureString(text, font);
                int pad = 4;
                int w = (int)Math.Ceiling(sizeF.Width) + pad * 2;
                int h = (int)Math.Ceiling(sizeF.Height) + pad * 2;

                // 라벨 위치(박스 좌상단 살짝 위로; 이미지 밖이면 아래로)
                int x = anchor.Left;
                int y = anchor.Top - h - 1;
                if (y < 0) y = anchor.Top + 1;

                var rect = new Rectangle(x, y, w, h);

                // 배경 (반투명 검정) + 테두리(클래스 색)
                using (var bg = new SolidBrush(Color.FromArgb(170, 0, 0, 0)))
                using (var pen = new Pen(boxColor, 2))
                using (var txt = new SolidBrush(Color.White))
                {
                    g.FillRectangle(bg, rect);
                    g.DrawRectangle(pen, rect);
                    g.DrawString(text, font, txt, rect.Left + pad, rect.Top + pad);
                }
            }
        }

        // ============ 메인 ============
        public static SegResult InferAndOverlay(
            string onnxPath,
            string imagePath,
            float conf = 0.25f,
            float iou = 0.45f,
            float maskThr = 0.5f,
            float alpha = 0.45f,
            int maxInstances = -1,
            float minBoxAreaRatio = 0f,
            float minMaskAreaRatio = 0f,
            bool discardTouchingBorder = false,
            bool drawBoxes = false,
            bool drawScores = true
        )
        {
            Log("InferAndOverlay() called");
            Log($"  Model:  {onnxPath}");
            Log($"  Image:  {imagePath}");
            Log($"  Params: conf={conf}, iou={iou}, maskThr={maskThr}, alpha={alpha}");

            // 타이밍 측정기
            var sw = Stopwatch.StartNew();
            double tSession = 0, tPrev = 0, tPre = 0, tInfer = 0, tPost = 0, tOverlay = 0;

            using (var orig = (Bitmap)Image.FromFile(imagePath))
            {
                // ===== 세션 생성 (DML → CPU 폴백) =====
                var session = GetOrCreateSession(onnxPath, out tSession);

                // 세션 초기화 시간
                tSession = sw.Elapsed.TotalMilliseconds;
                // 전처리 기준 시각을 "세션 생성 직후"로 리셋
                tPrev = sw.Elapsed.TotalMilliseconds;

                string inputName = session.InputMetadata.Keys.First();
                var inMeta = session.InputMetadata[inputName];
                Log($"Session OK. Input name: {inputName}");
                try { Log($"Input dims: [{JoinDims(inMeta.Dimensions)}]"); } catch { }

                // 입력 크기
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

                // Letterbox & 텐서
                using (var boxed = Letterbox(orig, netSize, out float scale, out int padX, out int padY, out Size resized))
                {
                    Log($"Letterbox: scale={scale:F6}, padX={padX}, padY={padY}, resized={resized.Width}x{resized.Height}");
                    var input = ToCHWTensor(boxed); // [1,3,net,net]
                    Log($"Input tensor ready: [1,3,{boxed.Height},{boxed.Width}]");

                    // --- 전처리 시간 기록
                    tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                    IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = null;
                    try
                    {
                        var inputsList = new List<NamedOnnxValue>(1)
                            {
                                NamedOnnxValue.CreateFromTensor<float>(inputName, input)
                            };

                        // 추론
                        outputs = session.Run(inputsList);
                        Log($"session.Run() returned {outputs.Count} outputs");

                        // --- 추론 시간 기록
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

                        // out0: det head (3D), out1: proto (4D)
                        var t3 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 3).AsTensor<float>();
                        var t4 = outputs.First(v => v.AsTensor<float>().Dimensions.Length == 4).AsTensor<float>();

                        var d3 = t3.Dimensions;    // e.g. [1,37,8400] or [1,8400,37]
                        var d4 = t4.Dimensions;    // [1, segDim, mh, mw]
                        int b = d3[0];
                        int a = d3[1];
                        int c = d3[2];

                        int segDim = d4[1];
                        int mh = d4[2];
                        int mw = d4[3];

                        // 채널-우선/후행 판별
                        bool channelsFirst = (a <= 512 && c >= 1000) || (a <= segDim + 4 + 256);
                        int channels = channelsFirst ? a : c;
                        int nPred = channelsFirst ? c : a;
                        int feat = channels;

                        // 클래스 수
                        int numClasses = feat - 4 - segDim;
                        if (numClasses < 0)
                        {
                            channelsFirst = !channelsFirst; // 반대로 시도
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

                        // 보통 proto는 입력의 1/4 해상도 → 안전하게 유지
                        netSize = Math.Max(netSize, Math.Max(mh, mw) * 4);

                        // 좌표 스케일(정규화 → 픽셀 변환 감지)
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

                        // --- 디텍션 수집
                        var dets = new List<Det>(256);
                        for (int i = 0; i < nPred; i++)
                        {
                            float x = (channelsFirst ? t3[0, 0, i] : t3[0, i, 0]) * coordScale;
                            float y = (channelsFirst ? t3[0, 1, i] : t3[0, i, 1]) * coordScale;
                            float w = (channelsFirst ? t3[0, 2, i] : t3[0, i, 2]) * coordScale;
                            float h = (channelsFirst ? t3[0, 3, i] : t3[0, i, 3]) * coordScale;

                            // class score
                            int bestC = 0; float bestS = 0f;
                            for (int cidx = 0; cidx < numClasses; cidx++)
                            {
                                float raw = channelsFirst
                                    ? t3[0, 4 + cidx, i]
                                    : t3[0, i, 4 + cidx];

                                // 시그모이드 전/후 모두 커버
                                float s = (raw < 0f || raw > 1f) ? Sigmoid(raw) : raw;
                                if (s > bestS) { bestS = s; bestC = cidx; }
                            }
                            if (bestS < conf) continue;

                            // 계수
                            var coeff = new float[segDim];
                            int baseIdx = 4 + numClasses;
                            for (int k = 0; k < segDim; k++)
                            {
                                coeff[k] = channelsFirst
                                    ? t3[0, baseIdx + k, i]
                                    : t3[0, i, baseIdx + k];
                            }

                            // xywh → xyxy (모델 입력 좌표계 픽셀)
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
                        if (dets.Count == 0)
                        {
                            // 후처리 시간까지 포함
                            tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;
                            var empty = new SegResult
                            {
                                Overlayed = (Bitmap)orig.Clone(),
                                Dets = dets,
                                SessionMs = tSession,
                                PreMs = tPre,
                                InferMs = tInfer,
                                PostMs = tPost,
                                OverlayMs = 0,
                                TotalMs = sw.Elapsed.TotalMilliseconds
                            };
                            empty.TitleSuffix =
                                $"session {empty.SessionMs:F0}ms, pre {empty.PreMs:F0}ms, infer {empty.InferMs:F0}ms, post {empty.PostMs:F0}ms, overlay {empty.OverlayMs:F0}ms, total {empty.TotalMs:F0}ms";
                            return empty;
                        }

                        // NMS
                        dets = Nms(dets, iou);
                        Log($"Detections after NMS (iou={iou}): {dets.Count}");

                        // --- 후처리 시간 기록 (오버레이 직전까지)
                        tPost = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

                        // --- 프로토타입 접근자
                        var proto = t4.ToArray(); // [1,segDim,mh,mw]
                        Func<int, int, int, float> ProtoAt = (k, y, x) => proto[(k * mh + y) * mw + x];

                        // --- 오버레이 합성
                        var over = new Bitmap(orig.Width, orig.Height, PixelFormat.Format32bppArgb);
                        using (var g = Graphics.FromImage(over))
                        {
                            g.DrawImage(orig, 0, 0, orig.Width, orig.Height);

                            int di = 0;
                            foreach (var d in dets)
                            {
                                // 1) m = sigmoid(sum_k coeff[k]*proto[k])
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

                                // 2) 업샘플 → netSize
                                using (var maskBmp = FloatMaskToBitmap(mask, mw, mh))
                                using (var up = ResizeBitmap(maskBmp, netSize, netSize))
                                {
                                    // 3) 레터박스 제거 → 원본 크기
                                    int cx = Clamp(padX, 0, up.Width);
                                    int cy = Clamp(padY, 0, up.Height);
                                    int cw = Math.Min(resized.Width, up.Width - cx);
                                    int ch = Math.Min(resized.Height, up.Height - cy);
                                    if (cw <= 0 || ch <= 0)
                                    {
                                        Log($" [det#{di}] invalid crop: cw={cw}, ch={ch}");
                                        di++;
                                        continue;
                                    }

                                    using (var cropped = CropBitmap(up, new Rectangle(cx, cy, cw, ch)))
                                    using (var toOrig = ResizeBitmap(cropped, orig.Width, orig.Height))
                                    {
                                        var boxOrig = NetBoxToOriginal(d.Box, scale, padX, padY, resized, orig.Size);
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
                                        if (drawScores)
                                        {
                                            DrawLabel(g, boxOrig, $"{d.Score:0.00}", color);
                                        }

                                        Log($" [det#{di}] class={d.ClassId}, score={d.Score:F3}, " +
                                            $"boxNet=({d.Box.Left:F1},{d.Box.Top:F1},{d.Box.Right:F1},{d.Box.Bottom:F1}), " +
                                            $"boxOrig=({boxOrig.X},{boxOrig.Y},{boxOrig.Width},{boxOrig.Height}), " +
                                            $"crop=({cx},{cy},{cw},{ch})");
                                    }
                                }
                                di++;
                            }
                        }

                        Log("Overlay composed successfully.");

                        // --- 오버레이 시간 기록 & 결과 작성
                        tOverlay = sw.Elapsed.TotalMilliseconds - tPrev;
                        double tTotal = sw.Elapsed.TotalMilliseconds;

                        var res = new SegResult
                        {
                            Overlayed = over,
                            Dets = dets,
                            SessionMs = tSession,
                            PreMs = tPre,
                            InferMs = tInfer,
                            PostMs = tPost,
                            OverlayMs = tOverlay,
                            TotalMs = tTotal
                        };
                        res.TitleSuffix =
                            $"session {res.SessionMs:F0}ms, pre {res.PreMs:F0}ms, infer {res.InferMs:F0}ms, post {res.PostMs:F0}ms, overlay {res.OverlayMs:F0}ms, total {res.TotalMs:F0}ms";
                        return res;
                    }
                    finally
                    {
                        if (outputs != null) outputs.Dispose();
                    }
                }

            }
        }

        // ===== 세션 생성 (DML → CPU 폴백) =====
        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            SessionOptions so = null;
            try
            {
                so = new SessionOptions();
                so.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

                try
                {
                    Log("Trying DML EP...");
                    so.AppendExecutionProvider_DML();
                    var s = new InferenceSession(modelPath, so);
                    Log("Session created with DML EP.");
                    return s;
                }
                catch (OnnxRuntimeException e)
                {
                    Log("DML EP failed, will fall back to CPU.");
                    Log(e.ToString());
                    if (so != null) { try { so.Dispose(); } catch { } }
                }
                catch (Exception e)
                {
                    Log("DML EP init failed (non-ORT), fall back to CPU.");
                    Log(e.ToString());
                    if (so != null) { try { so.Dispose(); } catch { } }
                }
            }
            catch (Exception e)
            {
                Log("SessionOptions init failed, fall back to CPU.");
                Log(e.ToString());
                if (so != null) { try { so.Dispose(); } catch { } }
            }

            var soCpu = new SessionOptions();
            soCpu.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
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

        // polyfills
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