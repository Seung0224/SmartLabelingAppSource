// File: YoloSegOnnx.cs
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Drawing.Imaging;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading;
using System.Windows.Forms;
using static SmartLabelingApp.ImageCanvas;

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

        private struct Seg { public PointF A, B; public Seg(PointF a, PointF b) { A = a; B = b; } }

        // 포인트를 키로 묶기 위한 양자화 (연결 시 사용)
        private static long QKey(PointF p, float scale = 100f)
        {
            int xi = (int)Math.Round(p.X * scale);
            int yi = (int)Math.Round(p.Y * scale);
            return ((long)xi << 32) ^ (uint)yi;
        }

        // 입력 텐서/NamedOnnxValue 재사용을 위한 캐시
        static string _inputName;
        static int _curNet = 0;
        static float[] _inBuf;
        static DenseTensor<float> _tensor;
        static NamedOnnxValue _nov;

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


        // (A) 버퍼 준비/교체 (netSize 바뀌면 새로 잡음)
        static void EnsureInputBuffers(InferenceSession s, int net)
        {
            if (_tensor != null && _curNet == net) return;

            _inputName = s.InputMetadata.Keys.First();
            _inBuf = new float[1 * 3 * net * net];
            _tensor = new DenseTensor<float>(_inBuf, new[] { 1, 3, net, net });

            _nov = NamedOnnxValue.CreateFromTensor(_inputName, _tensor);

            _curNet = net;
        }

        // (B) 비트맵을 레터박스 후 NCHW float[0..1]로 채우기
        static void FillTensorFromBitmap(Bitmap src, int net,
            out float scale, out int padX, out int padY, out Size resized)
        {
            int W = src.Width, H = src.Height;
            scale = Math.Min((float)net / W, (float)net / H);
            int rw = (int)Math.Round(W * scale);
            int rh = (int)Math.Round(H * scale);
            padX = (net - rw) / 2;
            padY = (net - rh) / 2;
            resized = new Size(rw, rh);

            // 레터박스된 이미지를 임시 비트맵에 그린 뒤 LockBits로 읽기
            var tmp = new Bitmap(net, net, PixelFormat.Format24bppRgb);
            using (var g = Graphics.FromImage(tmp))
            {
                g.Clear(Color.Black); // 필요하면 (114,114,114)
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
                            // BGR → NCHW(R,G,B)
                            byte b = row[x * 3 + 0];
                            byte g = row[x * 3 + 1];
                            byte r = row[x * 3 + 2];

                            int idx = y * net + x;
                            _inBuf[0 * plane + idx] = r * inv255;
                            _inBuf[1 * plane + idx] = g * inv255;
                            _inBuf[2 * plane + idx] = b * inv255;
                        }
                    }
                }
            }
            finally { tmp.UnlockBits(bd); }
        }



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

        private static List<PointF[]> ChainSegmentsToPolylines(List<Seg> segs, float tol = 0.01f)
        {
            var unused = new HashSet<int>(Enumerable.Range(0, segs.Count));
            var map = new Dictionary<long, List<int>>();

            void addEnd(PointF p, int idx)
            {
                var k = QKey(p, 100f);
                if (!map.TryGetValue(k, out var lst)) map[k] = lst = new List<int>();
                lst.Add(idx);
            }

            for (int i = 0; i < segs.Count; i++) { addEnd(segs[i].A, i); addEnd(segs[i].B, i); }

            var polylines = new List<PointF[]>();

            while (unused.Count > 0)
            {
                int seed = unused.First();
                unused.Remove(seed);

                var a = segs[seed].A; var b = segs[seed].B;
                var poly = new List<PointF> { a, b };

                // 한쪽 끝을 계속 확장
                bool extended = true;
                while (extended)
                {
                    extended = false;

                    // 끝점 b에서 이어지는 선분 찾기
                    var kb = QKey(b, 100f);
                    if (map.TryGetValue(kb, out var cand))
                    {
                        for (int i = cand.Count - 1; i >= 0; i--)
                        {
                            int si = cand[i];
                            if (!unused.Contains(si)) { cand.RemoveAt(i); continue; }
                            var s = segs[si];
                            // b==s.A 이면 s.B로, b==s.B 이면 s.A로
                            if (Math.Abs(b.X - s.A.X) < tol && Math.Abs(b.Y - s.A.Y) < tol)
                            { poly.Add(s.B); b = s.B; unused.Remove(si); extended = true; cand.RemoveAt(i); break; }
                            if (Math.Abs(b.X - s.B.X) < tol && Math.Abs(b.Y - s.B.Y) < tol)
                            { poly.Add(s.A); b = s.A; unused.Remove(si); extended = true; cand.RemoveAt(i); break; }
                        }
                    }

                    // 시작점 a쪽도 확장
                    if (!extended)
                    {
                        var ka = QKey(a, 100f);
                        if (map.TryGetValue(ka, out var cand2))
                        {
                            for (int i = cand2.Count - 1; i >= 0; i--)
                            {
                                int si = cand2[i];
                                if (!unused.Contains(si)) { cand2.RemoveAt(i); continue; }
                                var s = segs[si];
                                if (Math.Abs(a.X - s.A.X) < tol && Math.Abs(a.Y - s.A.Y) < tol)
                                { poly.Insert(0, s.B); a = s.B; unused.Remove(si); cand2.RemoveAt(i); extended = true; break; }
                                if (Math.Abs(a.X - s.B.X) < tol && Math.Abs(a.Y - s.B.Y) < tol)
                                { poly.Insert(0, s.A); a = s.A; unused.Remove(si); cand2.RemoveAt(i); extended = true; break; }
                            }
                        }
                    }
                }

                polylines.Add(poly.ToArray());
            }

            return polylines;
        }

        // 외곽선 수집: 경계는 "두 픽셀 사이" → 반 픽셀(0.5) 보정하여 정확 정렬
        private static void CollectMaskOutlineOverlaysExact(
    Bitmap maskGray,          // toOrig (원본 크기 마스크)
    byte thrByte,             // (byte)(maskThr*255+0.5)
    Color color,
    float widthPx,
    List<OverlayItem> overlaysOut)
        {
            if (overlaysOut == null) return;

            int w = maskGray.Width, h = maskGray.Height;
            var rect = new Rectangle(0, 0, w, h);
            var data = maskGray.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);

            try
            {
                unsafe
                {
                    bool[,] B = new bool[w, h];             // 이진 마스크 (채움과 동일 기준)
                    byte* basePtr = (byte*)data.Scan0;
                    int stride = data.Stride;

                    for (int y = 0; y < h; y++)
                    {
                        byte* row = basePtr + y * stride;
                        for (int x = 0; x < w; x++)
                            B[x, y] = row[x * 4] >= thrByte; // FloatMaskToBitmap가 그레이로 들어왔다는 전제
                    }

                    // 1) 정수 경계(픽셀 사이의 선이 아니라 "격자선 x, y")에서 on/off가 바뀌는 곳을 선분으로 수집
                    var segs = new List<(PointF A, PointF B)>();

                    // 수직 경계: (x-1,y) vs (x,y) 가 다르면 x에서 세로선
                    for (int y = 0; y < h; y++)
                        for (int x = 1; x < w; x++)
                            if (B[x - 1, y] != B[x, y])
                                segs.Add((new PointF(x, y), new PointF(x, y + 1)));

                    // 수평 경계: (x,y-1) vs (x,y) 가 다르면 y에서 가로선
                    for (int y = 1; y < h; y++)
                        for (int x = 0; x < w; x++)
                            if (B[x, y - 1] != B[x, y])
                                segs.Add((new PointF(x, y), new PointF(x + 1, y)));

                    // 2) 짧은 선분들을 연결해서 긴 폴리라인(폐곡선)으로 (정수 격자이므로 tol=0)
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

                        // 끝점 쪽으로 계속 확장
                        bool extended = true;
                        while (extended)
                        {
                            extended = false;

                            // 뒤쪽 끝
                            var end = path[path.Count - 1];
                            var key = ((int)end.X, (int)end.Y);
                            if (dict.TryGetValue(key, out var lst))
                            {
                                for (int k = lst.Count - 1; k >= 0; k--)
                                {
                                    int si = lst[k]; if (used[si]) continue;
                                    var s = segs[si];

                                    if ((int)s.A.X == (int)end.X && (int)s.A.Y == (int)end.Y)
                                    { path.Add(s.B); used[si] = true; extended = true; break; }
                                    if ((int)s.B.X == (int)end.X && (int)s.B.Y == (int)end.Y)
                                    { path.Add(s.A); used[si] = true; extended = true; break; }
                                }
                            }

                            // 앞쪽 끝
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

                                        if ((int)s.A.X == (int)beg.X && (int)s.A.Y == (int)beg.Y)
                                        { path.Insert(0, s.B); used[si] = true; extended = true; break; }
                                        if ((int)s.B.X == (int)beg.X && (int)s.B.Y == (int)beg.Y)
                                        { path.Insert(0, s.A); used[si] = true; extended = true; break; }
                                    }
                                }
                            }
                        }

                        polylines.Add(path.ToArray());
                    }

                    // 3) 통합 오버레이로 내보내기
                    foreach (var poly in polylines)
                    {
                        overlaysOut.Add(new OverlayItem
                        {
                            Kind = OverlayKind.Polyline,
                            PointsImg = poly,
                            Closed = true,
                            StrokeColor = color,
                            StrokeWidthPx = widthPx,      // 1~2px 권장
                        });
                    }
                }
            }
            finally { maskGray.UnlockBits(data); }
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

            EnsureInputBuffers(session, netSize);
            FillTensorFromBitmap(orig, netSize, out float scale, out int padX, out int padY, out Size resized);

            // 1) 전처리
            Log($"Letterbox: scale={scale:F6}, padX={padX}, padY={padY}, resized={resized.Width}x{resized.Height}");
            Log($"Input tensor ready: [1,3,{netSize},{netSize}]");

            tPre = sw.Elapsed.TotalMilliseconds - tPrev; tPrev = sw.Elapsed.TotalMilliseconds;

            // 2) 추론
            IDisposableReadOnlyCollection<DisposableNamedOnnxValue> outputs = null;
            try
            {
                outputs = session.Run(new[] { _nov });
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


        // === 완전 분리: 합성만 수행 (원본 + SegResult → 새 Bitmap 반환) ===
        public static Bitmap Overlay(Bitmap orig, SegResult r, float maskThr = 0.65f, float alpha = 0.45f, bool drawBoxes = false, bool drawScores = true, List<SmartLabelingApp.ImageCanvas.OverlayItem> overlaysOut = null)
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

                            // 외곽선: 채움과 동일 기준으로, 정확히 맞추기
                            if (overlaysOut != null)
                            {
                                CollectMaskOutlineOverlaysExact(toOrig, thrByte, color, 2f, overlaysOut);
                            }

                            if (drawScores && overlaysOut != null)
                            {
                                overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem
                                {
                                    Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Badge,
                                    Text = $"[{d.ClassId}], {d.Score:0.00}",
                                    BoxImg = boxOrig,
                                    StrokeColor = color
                                });
                            }

                            if (drawBoxes && overlaysOut != null)
                            {
                                overlaysOut.Add(new SmartLabelingApp.ImageCanvas.OverlayItem
                                {
                                    Kind = SmartLabelingApp.ImageCanvas.OverlayKind.Box,
                                    BoxImg = boxOrig,
                                    StrokeColor = color,
                                    StrokeWidthPx = 3f
                                });
                            }
                        }
                    }
                    di++;
                }
            }
            return over;
        }

        public struct OverlayResult
        {
            public Bitmap Image;
            public List<SmartLabelingApp.ImageCanvas.OverlayItem> Overlays;
        }

        public static OverlayResult Render(Bitmap orig, SegResult r,
            float maskThr = 0.65f, float alpha = 0.45f,
            bool drawBoxes = false, bool drawScores = true)
        {
            var list = new List<SmartLabelingApp.ImageCanvas.OverlayItem>();
            var bmp = Overlay(orig, r, maskThr, alpha, drawBoxes, drawScores, overlaysOut: list);

            return new OverlayResult { Image = bmp, Overlays = list };
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

        private static bool TryAppendCudaWithOptions(SessionOptions so)
        {
            try
            {
                // 케이스 A: 신형 API 타입이 존재하는 경우
                var optType = Type.GetType(
                    "Microsoft.ML.OnnxRuntime.Provider.OrtCUDAProviderOptions, Microsoft.ML.OnnxRuntime")
                    ?? Type.GetType("Microsoft.ML.OnnxRuntime.Providers.OrtCUDAProviderOptions, Microsoft.ML.OnnxRuntime");
                if (optType != null)
                {
                    var cuda = Activator.CreateInstance(optType);

                    // 속성들은 있는 경우에만 세팅
                    SetIfExists(optType, cuda, "DeviceId", 0);
                    var enumType = optType.Assembly.GetType(
                        "Microsoft.ML.OnnxRuntime.Provider.OrtCudnnConvAlgoSearch")
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

                // 케이스 B: 문자열 딕셔너리 방식(중간 버전)
                var mi2 = typeof(SessionOptions).GetMethod(
                    "AppendExecutionProvider", new[] { typeof(string), typeof(IDictionary<string, string>) });
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
            catch { /* 옵션 부착 실패 → 아래에서 기본 CUDA로 폴백 */ }
            return false;
        }

        private static void SetIfExists(Type t, object obj, string prop, object value)
        {
            var p = t.GetProperty(prop, BindingFlags.Public | BindingFlags.Instance);
            if (p != null && p.CanWrite) p.SetValue(obj, value, null);
        }

        private static void TryWarmup(InferenceSession s, int netSize)
        {
            try
            {
                var inputName = s.InputMetadata.Keys.FirstOrDefault();
                if (string.IsNullOrEmpty(inputName)) return;

                var dummy = new DenseTensor<float>(new[] { 1, 3, netSize, netSize }); // 0으로 충분

                // 입력 한 개만 using 으로 Dispose
                var input = NamedOnnxValue.CreateFromTensor(inputName, dummy);
                // 결과 컬렉션도 IDisposable이므로 using 으로 바로 버림
                using (var _ = s.Run(new[] { input }))
                {
                    // do nothing (warm-up)
                }
            }
            catch
            {
                // warm-up 실패해도 무시(환경 차이)
            }
        }

        static bool TryAppendTensorRT(SessionOptions so)
        {
            try
            {
                // 1) 전용 API가 있는 버전
                so.AppendExecutionProvider_Tensorrt(0); // 있으면 이 한 줄로 끝
                                                        // 옵션을 넣고 싶다면 별도 ProviderOptions 객체/사전이 필요
                return true;
            }
            catch
            {
                try
                {
                    // 2) 범용 AppendExecutionProvider(이름, 옵션 사전) 방식
                    var cacheDir = Path.Combine(AppContext.BaseDirectory, "trt_cache");
                    Directory.CreateDirectory(cacheDir);

                    var trtOpts = new Dictionary<string, string>
            {
                { "device_id", "0" },
                { "trt_fp16_enable", "1" },
                // { "trt_int8_enable", "1" }, // 교정 세트 있을 때만
                { "trt_engine_cache_enable", "1" },
                { "trt_engine_cache_path", cacheDir },
                { "trt_timing_cache_enable", "1" }
            };

                    so.AppendExecutionProvider("Tensorrt", trtOpts);
                    return true;
                }
                catch { return false; }
            }
        }



        private static InferenceSession CreateSessionWithFallback(string modelPath)
        {
            SessionOptions so = null;

            //// 1) TensorRT EP
            //try
            //{
            //    so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            //    Log("Trying TensorRT EP...");

            //    if (!TryAppendTensorRT(so))
            //        throw new NotSupportedException("TensorRT EP not available in this build.");

            //    var sess = new InferenceSession(modelPath, so);
            //    Log("Session created with TensorRT EP.");
            //    EnsureInputBuffers(sess, 640);
            //    TryWarmup(sess, 640);
            //    return sess;
            //}
            //catch (Exception ex)
            //{
            //    Log("TensorRT EP failed / unavailable, falling back to CUDA.");
            //    Log(ex.Message);
            //    try { so?.Dispose(); } catch { }
            //}

            // 2) CUDA EP
            try
            {
                so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                Log("Trying CUDA EP...");

                if (!TryAppendCudaWithOptions(so))      // 옵션 방식 시도
                    so.AppendExecutionProvider_CUDA(0); // 기본 방식

                var sess = new InferenceSession(modelPath, so);
                Log("Session created with CUDA EP.");
                EnsureInputBuffers(sess, 640);
                TryWarmup(sess, 640);
                return sess;
            }
            catch (Exception ex)
            {
                Log("CUDA EP failed, will try DML EP.");
                Log(ex.ToString());
                try { so?.Dispose(); } catch { }
            }

            // 3) DML EP
            try
            {
                so = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
                Log("Trying DML EP...");
                so.AppendExecutionProvider_DML(); // 또는 (0)
                var sess = new InferenceSession(modelPath, so);
                Log("Session created with DML EP.");
                EnsureInputBuffers(sess, 640);
                TryWarmup(sess, 640);
                return sess;
            }
            catch (Exception ex)
            {
                Log("DML EP failed, falling back to CPU.");
                Log(ex.ToString());
                try { so?.Dispose(); } catch { }
            }

            // 4) CPU
            var soCpu = new SessionOptions { GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL };
            Log("Creating CPU session...");
            var cpu = new InferenceSession(modelPath, soCpu);
            Log("CPU session created.");
            EnsureInputBuffers(cpu, 640);
            // (CPU에선 Warmup 별 의미 없음) 
            return cpu;
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
