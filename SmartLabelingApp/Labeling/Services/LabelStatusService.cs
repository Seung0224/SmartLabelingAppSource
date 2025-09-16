using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Text;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    /// <summary>
    /// TreeView 이미지 노드에 라벨 상태를 표시하는 유틸(확장판).
    /// - 상태: None, Labeled, Empty, Error
    /// - 뱃지 스타일 선택: Dot(기본), Ring, Square, Check
    /// - 색상 팔레트 커스터마이즈 가능 (static 필드)
    /// - (옵션) 텍스트 접미사, 굵기/색상 강조, 툴팁 포함
    /// </summary>
    public static class LabelStatusService
    {
        // ====== 공개 커스터마이즈 포인트 ======
        public enum LabelStatus { None = 0, Labeled = 1, Empty = 2, Error = 3 }
        public enum BadgeStyle { Dot, Ring, Square, Check }

        public static Color ColorLabeled = Color.FromArgb(52, 199, 89);   // 초록
        public static Color ColorEmpty = Color.FromArgb(255, 204, 0);   // 노랑
        public static Color ColorError = Color.FromArgb(255, 69, 58);   // 빨강
        public static Color ColorNone = Color.Transparent;

        // ====== Persisted labeled state (per original image path) ======
        // Storage: %LOCALAPPDATA%\SmartLabelingApp\label_state.db  (pipe-delimited)
        private sealed class LabelEntry
        {
            public bool Labeled;
            public int Count;
            public DateTime Last;
        }

        private static string _appDir = System.IO.Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "SmartLabelingApp");
        private static string _dbPath = System.IO.Path.Combine(_appDir, "label_state.db");
        private static readonly Dictionary<string, LabelEntry> _db =
            LoadDbInternal();

        private static System.Collections.Generic.Dictionary<string, LabelEntry> LoadDbInternal()
        {
            var map = new System.Collections.Generic.Dictionary<string, LabelEntry>(StringComparer.OrdinalIgnoreCase);
            try
            {
                if (!System.IO.File.Exists(_dbPath)) return map;
                foreach (var line in System.IO.File.ReadAllLines(_dbPath, System.Text.Encoding.UTF8))
                {
                    if (string.IsNullOrWhiteSpace(line) || line.StartsWith("#")) continue;
                    var parts = line.Split('|');
                    if (parts.Length < 4) continue;
                    var path = parts[0];
                    bool labeled = parts[1] == "1";
                    int count = 0; int.TryParse(parts[2], out count);
                    DateTime last = DateTime.MinValue; DateTime.TryParse(parts[3], out last);
                    map[path] = new LabelEntry { Labeled = labeled, Count = System.Math.Max(0, count), Last = last };
                }
            }
            catch { /* ignore */ }
            return map;
        }
        public static void SetStorageRoot(string projectDir)
        {
            if (string.IsNullOrWhiteSpace(projectDir)) return;
            _appDir = System.IO.Path.Combine(projectDir);                // ex) ...\AnnotationData
            _dbPath = System.IO.Path.Combine(_appDir, "label_state.db"); // ex) ...\AnnotationData\label_state.db
            try { System.IO.Directory.CreateDirectory(_appDir); } catch { }
        }

        private static void SaveDbInternal()
        {
            try
            {
                System.IO.Directory.CreateDirectory(_appDir);
                var sb = new System.Text.StringBuilder();
                sb.AppendLine("# SmartLabelingApp label state");
                foreach (var kv in _db)
                {
                    var path = kv.Key;
                    var e = kv.Value ?? new LabelEntry();
                    sb.Append(path).Append('|')
                      .Append(e.Labeled ? "1" : "0").Append('|')
                      .Append(e.Count).Append('|')
                      .Append(e.Last == default ? "" : e.Last.ToString("o"))
                      .AppendLine();
                }
                System.IO.File.WriteAllText(_dbPath, sb.ToString(), System.Text.Encoding.UTF8);
            }
            catch { /* ignore */ }
        }

        /// <summary>
        /// SAVE 직후 호출하여 현재 이미지의 라벨 상태(객체 개수) 영구 저장.
        /// 프로그램 재시작 후에도 표시 유지됨.
        /// </summary>
        public static void MarkLabeled(string imagePath, int count)
        {
            if (string.IsNullOrWhiteSpace(imagePath)) return;
            LabelEntry e;
            if (!_db.TryGetValue(imagePath, out e))
                e = _db[imagePath] = new LabelEntry();
            e.Labeled = count > 0;
            e.Count = System.Math.Max(0, count);
            e.Last = DateTime.Now;
            SaveDbInternal();
        }
        // 투명

        /// <summary>
        /// 트리뷰의 StateImageList에 연결할 아이콘 모음 생성
        /// </summary>
        /// <param name="style">Dot/Ring/Square/Check</param>
        /// <param name="size">아이콘 크기(px). 보통 16 또는 20</param>
        /// <param name="thickness">Ring 외곽선 두께 또는 Dot/Square 테두리 두께</param>
        public static ImageList BuildStateImageList(BadgeStyle style = BadgeStyle.Dot, int size = 16, int thickness = 1)
        {
            var list = new ImageList
            {
                ImageSize = new Size(size, size),
                ColorDepth = ColorDepth.Depth32Bit
            };

            // 0 None
            list.Images.Add(RenderNone(size));

            // 1 Labeled
            list.Images.Add(RenderBadge(style, ColorLabeled, size, thickness));

            // 2 Empty
            list.Images.Add(RenderBadge(style, ColorEmpty, size, thickness));

            // 3 Error
            list.Images.Add(RenderBadge(style, ColorError, size, thickness));

            return list;
        }

        public static void ApplyNodeState(TreeNode node, string imagePath, string lastExportRoot, bool showCountSuffix = true)
        {
            if (node == null) return;

            int count;
            string tooltip;
            var status = GetLabelStatus(imagePath, lastExportRoot, out count, out tooltip);

            node.StateImageIndex = (int)status;

            // 텍스트 "(n)" 접미사
            var baseText = StripCountSuffix(node.Text);
            node.Text = (showCountSuffix && status == LabelStatus.Labeled && count > 0)
                      ? $"{baseText} ({count})"
                      : baseText;

            // 색/굵기
            var baseFont = (node.TreeView != null) ? node.TreeView.Font : Control.DefaultFont;
            if (status == LabelStatus.Labeled)
            {
                node.ForeColor = Color.FromArgb(30, 90, 30);
                node.NodeFont = new Font(baseFont, FontStyle.Bold);
            }
            else if (status == LabelStatus.Empty)
            {
                node.ForeColor = Color.FromArgb(120, 100, 0);
                node.NodeFont = baseFont;
            }
            else if (status == LabelStatus.Error)
            {
                node.ForeColor = Color.FromArgb(150, 30, 30);
                node.NodeFont = baseFont;
            }
            else
            {
                node.ForeColor = SystemColors.WindowText;
                node.NodeFont = baseFont;
            }

            // 툴팁
            node.ToolTipText = tooltip ?? string.Empty;
        }

        public static LabelStatus GetLabelStatus(string imagePath, string lastExportRoot, out int count, out string tooltipText)
        {
            count = 0;
            tooltipText = string.Empty;
            if (string.IsNullOrEmpty(imagePath))
            {
                tooltipText = "No image";
                return LabelStatus.None;
            }

            // 0) Persisted DB first (set by MarkLabeled after SAVE)
            LabelEntry __e;
            if (_db.TryGetValue(imagePath, out __e) && __e != null && __e.Labeled)
            {
                count = __e.Count;
                tooltipText = (count > 0) ? $"Objects: {count} (saved)" : "Labeled (saved)";
                return LabelStatus.Labeled;
            }
            string labelPath;
            if (!TryResolveLabelPath(imagePath, lastExportRoot, out labelPath))
            {
                tooltipText = "No label file";
                return LabelStatus.None;
            }
            if (!File.Exists(labelPath))
            {
                tooltipText = "No label file";
                return LabelStatus.None;
            }

            try
            {
                var lines = File.ReadAllLines(labelPath, Encoding.UTF8);
                var objCount = 0;
                for (int i = 0; i < lines.Length; i++)
                {
                    var line = (lines[i] ?? string.Empty).Trim();
                    if (line.Length == 0 || line.StartsWith("#")) continue;
                    objCount++;
                }
                count = objCount;

                // 클래스별 요약 시도
                string classesPath = TryFindClassesForLabel(labelPath);
                if (!string.IsNullOrEmpty(classesPath) && File.Exists(classesPath))
                {
                    string summary = BuildClassSummary(lines, classesPath);
                    tooltipText = (objCount > 0)
                        ? $"Objects: {objCount}{(summary.Length > 0 ? "\r\n" : "")}{summary}"
                        : "Label file exists but empty";
                }
                else
                {
                    tooltipText = (objCount > 0) ? $"Objects: {objCount}" : "Label file exists but empty";
                }

                return (objCount > 0) ? LabelStatus.Labeled : LabelStatus.Empty;
            }
            catch
            {
                tooltipText = "Label file read error";
                return LabelStatus.Error;
            }
        }

        public static string StripCountSuffix(string text)
        {
            if (string.IsNullOrEmpty(text)) return text;
            int i = text.LastIndexOf('(');
            if (i > 0 && text.EndsWith(")"))
            {
                if (int.TryParse(text.Substring(i + 1, text.Length - i - 2), out _))
                    return text.Substring(0, i).TrimEnd();
            }
            return text;
        }

        public static bool TryResolveLabelPath(string imagePath, string lastExportRoot, out string labelPath)
        {
            labelPath = null;
            if (string.IsNullOrEmpty(imagePath)) return false;

            var baseName = Path.GetFileNameWithoutExtension(imagePath);

            // 1) 마지막 Export 루트 우선
            if (!string.IsNullOrEmpty(lastExportRoot))
            {
                var cp = Path.Combine(lastExportRoot, "classes.txt");
                var lp = Path.Combine(lastExportRoot, "labels", baseName + ".txt");
                if (File.Exists(cp) && File.Exists(lp)) { labelPath = lp; return true; }
            }

            try
            {
                var dir = new DirectoryInfo(Path.GetDirectoryName(imagePath) ?? ".");

                // 2) .../images/<file> 구조라면 images 상위가 루트
                if (dir.Name.Equals("images", StringComparison.OrdinalIgnoreCase) && dir.Parent != null)
                {
                    var root = dir.Parent.FullName;
                    var cp = Path.Combine(root, "classes.txt");
                    var lp = Path.Combine(root, "labels", baseName + ".txt");
                    if (File.Exists(cp) && File.Exists(lp)) { labelPath = lp; return true; }
                }

                // 3) 상위 3단계 내에서 classes.txt + labels/<file>.txt 찾기
                var walk = dir;
                for (int i = 0; i < 3 && walk != null; i++, walk = walk.Parent)
                {
                    var candidate = walk.FullName;
                    var cp = Path.Combine(candidate, "classes.txt");
                    var lp = Path.Combine(candidate, "labels", baseName + ".txt");
                    if (File.Exists(cp) && File.Exists(lp)) { labelPath = lp; return true; }

                    if (walk.Name.Equals("images", StringComparison.OrdinalIgnoreCase) && walk.Parent != null)
                    {
                        candidate = walk.Parent.FullName;
                        cp = Path.Combine(candidate, "classes.txt");
                        lp = Path.Combine(candidate, "labels", baseName + ".txt");
                        if (File.Exists(cp) && File.Exists(lp)) { labelPath = lp; return true; }
                    }
                }
            }
            catch { /* ignore */ }

            return false;
        }

        private static string TryFindClassesForLabel(string labelPath)
        {
            try
            {
                var dir = new DirectoryInfo(Path.GetDirectoryName(labelPath) ?? ".");
                if (dir.Name.Equals("labels", StringComparison.OrdinalIgnoreCase) && dir.Parent != null)
                {
                    var root = dir.Parent.FullName;
                    var cp = Path.Combine(root, "classes.txt");
                    if (File.Exists(cp)) return cp;
                }

                var walk = dir;
                for (int i = 0; i < 2 && walk != null; i++, walk = walk.Parent)
                {
                    var cp = Path.Combine(walk.FullName, "classes.txt");
                    if (File.Exists(cp)) return cp;
                }
            }
            catch { }
            return null;
        }

        private static string BuildClassSummary(string[] labelLines, string classesPath)
        {
            try
            {
                var classNames = new List<string>();
                foreach (var raw in File.ReadAllLines(classesPath, Encoding.UTF8))
                {
                    var s = (raw ?? string.Empty).Trim();
                    if (s.Length > 0) classNames.Add(s);
                }

                var counts = new Dictionary<int, int>();
                for (int i = 0; i < labelLines.Length; i++)
                {
                    var line = (labelLines[i] ?? string.Empty).Trim();
                    if (line.Length == 0 || line.StartsWith("#")) continue;

                    var tok = line.Split((char[])null, StringSplitOptions.RemoveEmptyEntries);
                    int cls;
                    if (tok.Length > 0 && int.TryParse(tok[0], out cls))
                    {
                        if (!counts.ContainsKey(cls)) counts[cls] = 0;
                        counts[cls]++;
                    }
                }

                if (counts.Count == 0) return string.Empty;

                var sb = new StringBuilder();
                bool first = true;
                foreach (var kv in counts)
                {
                    string name = (kv.Key >= 0 && kv.Key < classNames.Count) ? classNames[kv.Key] : ("cls_" + kv.Key.ToString());
                    if (!first) sb.Append(", ");
                    sb.AppendFormat("{0}: {1}", name, kv.Value);
                    first = false;
                }
                return sb.ToString();
            }
            catch
            {
                return string.Empty;
            }
        }

        // ====== Renderers ======
        private static Bitmap RenderNone(int size)
        {
            return new Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
        }

        private static Bitmap RenderBadge(BadgeStyle style, Color color, int size, int thickness)
        {
            switch (style)
            {
                case BadgeStyle.Ring: return RenderRing(color, size, Math.Max(2, thickness));
                case BadgeStyle.Square: return RenderSquare(color, size, Math.Max(1, thickness));
                case BadgeStyle.Check: return RenderCheck(color, size);
                default: return RenderDot(color, size, Math.Max(1, thickness));
            }
        }

        private static Bitmap RenderDot(Color color, int size, int border)
        {
            var bmp = new Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                float pad = border + 2;
                var r = new RectangleF(pad, pad, size - pad * 2, size - pad * 2);
                using (var br = new SolidBrush(color)) g.FillEllipse(br, r);
                using (var pen = new Pen(Color.White, border)) g.DrawEllipse(pen, r);
            }
            return bmp;
        }

        private static Bitmap RenderRing(Color color, int size, int thickness)
        {
            var bmp = new Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                float pad = thickness + 2;
                var r = new RectangleF(pad, pad, size - pad * 2, size - pad * 2);
                using (var penOuter = new Pen(Color.White, 1f)) g.DrawEllipse(penOuter, r);
                using (var pen = new Pen(color, thickness)) g.DrawEllipse(pen, r);
            }
            return bmp;
        }

        private static Bitmap RenderSquare(Color color, int size, int border)
        {
            var bmp = new Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                float pad = border + 2;
                var r = new RectangleF(pad, pad, size - pad * 2, size - pad * 2);
                using (var br = new SolidBrush(color)) g.FillRectangle(br, r);
                using (var pen = new Pen(Color.White, border)) g.DrawRectangle(pen, r.X, r.Y, r.Width, r.Height);
            }
            return bmp;
        }

        private static Bitmap RenderCheck(Color color, int size)
        {
            var bmp = new Bitmap(size, size, System.Drawing.Imaging.PixelFormat.Format32bppArgb);
            using (var g = Graphics.FromImage(bmp))
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                float pad = 3;
                var r = new RectangleF(pad, pad, size - pad * 2, size - pad * 2);
                using (var br = new SolidBrush(color)) g.FillEllipse(br, r);
                // 체크 표시
                using (var pen = new Pen(Color.White, Math.Max(2f, size / 8f)))
                {
                    pen.StartCap = System.Drawing.Drawing2D.LineCap.Round;
                    pen.EndCap = System.Drawing.Drawing2D.LineCap.Round;
                    var p1 = new PointF(r.Left + r.Width * 0.25f, r.Top + r.Height * 0.55f);
                    var p2 = new PointF(r.Left + r.Width * 0.45f, r.Top + r.Height * 0.75f);
                    var p3 = new PointF(r.Left + r.Width * 0.78f, r.Top + r.Height * 0.30f);
                    g.DrawLines(pen, new[] { p1, p2, p3 });
                }
            }
            return bmp;
        }
    }
}
