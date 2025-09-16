using SmartLabelingApp.AI; // IAISegmenter, AISegmenterPrompt, AISegmenterOptions
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    /// <summary>
    /// AI Rectangle tool (비동기 실행 + ROI 기반 세그멘터와 연동)
    /// - 드래그한 박스 내에서 "여러 개" 폴리곤을 프리뷰 후 한 번에 확정 가능하도록 업데이트
    /// </summary>
    public sealed class AITool : ITool
    {
        private volatile bool _aiBusy;
        private readonly IAISegmenter _segmenter;
        private readonly AISegmenterOptions _opts = new AISegmenterOptions();

        public AITool(IAISegmenter segmenter) { _segmenter = segmenter ?? throw new ArgumentNullException(nameof(segmenter)); }

        // --- 1) Drag rectangle state (image-space) ---
        private bool _dragging;
        private PointF _dragStartImg;
        private RectangleF _rectImg; // current rectangle in image coords

        // --- 2) Preview polygons (image-space) + UI buttons ---
        private List<List<PointF>> _previewPolys;    // null when no preview
        private RectangleF _previewUnionBoundsImg;   // union bounds of preview polygons
        private Rectangle _okBtnScr, _cancelBtnScr;  // last-drawn screen rects for buttons
        private const int BtnSize = 18;
        private const int BtnGap = 6;
        private int _previewVertexCount = 48; // 기본값 (각 폴리곤 리샘플 상한)

        public void SetPreviewVertexCount(int k)
        {
            if (k < 0) k = 0;          // 0 => 리샘플 생략(원본 유지)
            if (k > 256) k = 256;      // 안전 상한
            _previewVertexCount = k;
        }
        private static Bitmap CloneForWorker(Image img)
        {
            // Image → 독립된 Bitmap 복제 (인덱스 포맷은 32bpp로)
            var srcBmp = img as Bitmap;
            if (srcBmp == null) return new Bitmap(img); // 자동으로 32bppArgb 생성

            var fmt = srcBmp.PixelFormat;
            if ((fmt & PixelFormat.Indexed) != 0)
                fmt = PixelFormat.Format32bppArgb;

            var rect = new Rectangle(0, 0, srcBmp.Width, srcBmp.Height);
            return srcBmp.Clone(rect, fmt); // 완전한 딥 카피
        }

        public bool IsEditingActive => _dragging || (_previewPolys != null && _previewPolys.Count > 0);

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c == null) return;

            // Accept / Cancel click while preview is visible
            if (_previewPolys != null && e.Button == MouseButtons.Left)
            {
                if (_okBtnScr.Contains(e.Location))
                {
                    CommitPreview(c);
                    return;
                }
                if (_cancelBtnScr.Contains(e.Location))
                {
                    CancelPreview(c);
                    return;
                }
            }

            if (e.Button == MouseButtons.Left && c.Image != null)
            {
                // start drawing rectangle
                var imgPt = c.Transform.ScreenToImage(e.Location);
                _dragging = true;
                _dragStartImg = imgPt;
                _rectImg = new RectangleF(imgPt, SizeF.Empty);
                c.RaiseToolEditBegan();
                c.Invalidate();
                return;
            }
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (c == null) return;
            if (_dragging)
            {
                var pt = c.Transform.ScreenToImage(e.Location);
                _rectImg = RectFrom2Pts(_dragStartImg, pt);
                c.Invalidate();
            }
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (c == null) return;
            if (_dragging && e.Button == MouseButtons.Left)
            {
                _dragging = false;

                // tiny rect -> ignore
                if (_rectImg.Width < 4 || _rectImg.Height < 4)
                {
                    _rectImg = RectangleF.Empty;
                    c.Invalidate();
                    return;
                }

                RunSegmentation(c, _rectImg); // async
            }
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            if (c == null) return;

            if (_previewPolys != null && _previewPolys.Count > 0)
            {
                if (e.KeyCode == Keys.Enter)
                {
                    CommitPreview(c);
                    e.Handled = e.SuppressKeyPress = true; return;
                }
                if (e.KeyCode == Keys.Escape)
                {
                    CancelPreview(c);
                    e.Handled = e.SuppressKeyPress = true; return;
                }
            }
        }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            if (c == null || g == null) return;

            g.SmoothingMode = SmoothingMode.AntiAlias;

            // 1) dragging rectangle (dashed)
            if (_dragging && !_rectImg.IsEmpty)
            {
                var scr = c.Transform.ImageRectToScreen(_rectImg);
                using (var pen = new Pen(Color.Orange, 1f) { DashStyle = DashStyle.Dash })
                using (var br = new SolidBrush(Color.FromArgb(24, Color.Orange)))
                {
                    g.FillRectangle(br, scr);
                    g.DrawRectangle(pen, scr.X, scr.Y, scr.Width, scr.Height);
                }
            }

            // 2) preview polygons + floating buttons
            if (_previewPolys != null && _previewPolys.Count > 0)
            {
                RectangleF union = RectangleF.Empty;
                foreach (var poly in _previewPolys)
                {
                    var pts = poly.Select(p => ToScreen(c, p)).ToArray();
                    using (var gp = new GraphicsPath())
                    {
                        gp.AddPolygon(pts);
                        using (var br = new SolidBrush(Color.FromArgb(36, c.ActiveStrokeColor)))
                        using (var pen = new Pen(c.ActiveStrokeColor, 2f))
                        {
                            g.FillPath(br, gp);
                            g.DrawPath(pen, gp);
                        }
                    }

                    // vertices (handles)
                    using (var br = new SolidBrush(Color.LimeGreen))
                    using (var pen = new Pen(Color.Black, 1f))
                    {
                        for (int i = 0; i < pts.Length; i++)
                        {
                            var r = new RectangleF(pts[i].X - 3, pts[i].Y - 3, 6, 6);
                            g.FillRectangle(br, r); g.DrawRectangle(pen, r.X, r.Y, r.Width, r.Height);
                        }
                    }

                    var bb = RectFromPoints(poly);
                    union = union.IsEmpty ? bb : RectangleF.Union(union, bb);
                }

                _previewUnionBoundsImg = union;

                var bScr = c.Transform.ImageRectToScreen(_previewUnionBoundsImg);

                int okX = (int)Math.Round(bScr.Right + BtnGap);
                int okY = (int)Math.Round(bScr.Top - BtnSize - BtnGap);
                var ok = new Rectangle(okX, okY, BtnSize, BtnSize);
                var cancel = new Rectangle(ok.Right + BtnGap, ok.Top, BtnSize, BtnSize);

                DrawOkButton(g, ok);
                DrawCancelButton(g, cancel);
                _okBtnScr = ok; _cancelBtnScr = cancel;
            }
            else
            {
                _okBtnScr = _cancelBtnScr = Rectangle.Empty;
            }
        }

        // -------------------- 성능 개선 + 멀티폴리곤 --------------------
        private async void RunSegmentation(ImageCanvas c, RectangleF boxImg)
        {
            if (_aiBusy) return;    // 이미 실행 중이면 무시
            _aiBusy = true;

            var prompt = AISegmenterPrompt.FromBox(boxImg);
            List<List<PointF>> polys = null;

            Cursor old = c.Cursor;
            Bitmap bmpSafe = null;
            try
            {
                c.Cursor = Cursors.WaitCursor;

                bmpSafe = CloneForWorker(c.Image);

                polys = await Task.Run(() =>
                {
                    return _segmenter.Segment(bmpSafe, prompt, _opts);
                });
            }
            catch (Exception ex)
            {
                IWin32Window owner = c.FindForm() ?? (IWin32Window)c;
                MessageBox.Show(owner, "AI segmentation failed:\n" + ex.Message, "AI",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                _rectImg = RectangleF.Empty;
                c.Invalidate();
                return;
            }
            finally
            {
                // 작업용 복제 비트맵 정리
                if (bmpSafe != null) bmpSafe.Dispose();
                c.Cursor = old;
                _aiBusy = false;
            }

            if (polys == null || polys.Count == 0)
            {
                _rectImg = RectangleF.Empty;
                c.Invalidate();
                return;
            }

            // (변경) 모든 폴리곤을 대상으로 리샘플/정렬
            var outPolys = new List<List<PointF>>(polys.Count);
            foreach (var p in polys)
            {
                float peri = Perimeter(p);
                int desiredK = (int)Math.Max(8, Math.Min(128, Math.Round(peri / 6f)));
                var q = (_previewVertexCount > 0) ? SimplifyToK(p, Math.Min(_previewVertexCount, desiredK)) : p;
                outPolys.Add(q);
            }

            // 프리뷰에 모두 표시
            _previewPolys = outPolys;
            _previewUnionBoundsImg = UnionBounds(outPolys);

            _rectImg = RectangleF.Empty;
            c.Invalidate();
        }

        // -------------------------------------------------------

        private static RectangleF RectFrom2Pts(PointF a, PointF b)
        {
            float x1 = Math.Min(a.X, b.X), y1 = Math.Min(a.Y, b.Y);
            float x2 = Math.Max(a.X, b.X), y2 = Math.Max(a.Y, b.Y);
            return new RectangleF(x1, y1, x2 - x1, y2 - y1);
        }

        private static PointF ToScreen(ImageCanvas c, PointF imgPt)
        {
            var r = c.Transform.ImageRectToScreen(new RectangleF(imgPt.X, imgPt.Y, 0.01f, 0.01f));
            return new PointF(r.X, r.Y);
        }

        private static float Perimeter(List<PointF> poly)
        {
            if (poly == null || poly.Count < 2) return 0;
            float sum = 0;
            for (int i = 0; i < poly.Count; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % poly.Count];
                float dx = b.X - a.X, dy = b.Y - a.Y;
                sum += (float)Math.Sqrt(dx * dx + dy * dy);
            }
            return sum;
        }

        private void CommitPreview(ImageCanvas c)
        {
            if (_previewPolys == null || _previewPolys.Count == 0) { CancelPreview(c); return; }

            foreach (var polyPts in _previewPolys)
            {
                if (polyPts == null || polyPts.Count < 3) continue;

                var poly = new PolygonShape(new List<PointF>(polyPts))
                {
                    LabelName = c.ActiveLabelName,
                    StrokeColor = c.ActiveStrokeColor,
                    FillColor = c.ActiveFillColor
                };
                c.Shapes.Add(poly);
                c.Selection.Set(poly);
            }

            // reset preview
            _previewPolys = null;
            _previewUnionBoundsImg = RectangleF.Empty;
            c.Invalidate();
        }

        private void CancelPreview(ImageCanvas c)
        {
            _previewPolys = null;
            _previewUnionBoundsImg = RectangleF.Empty;
            c.Invalidate();
        }

        private static RectangleF RectFromPoints(IReadOnlyList<PointF> pts)
        {
            if (pts == null || pts.Count == 0) return RectangleF.Empty;
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < pts.Count; i++)
            {
                var p = pts[i];
                if (p.X < minX) minX = p.X;
                if (p.Y < minY) minY = p.Y;
                if (p.X > maxX) maxX = p.X;
                if (p.Y > maxY) maxY = p.Y;
            }
            return new RectangleF(minX, minY, Math.Max(1e-3f, maxX - minX), Math.Max(1e-3f, maxY - minY));
        }

        private static RectangleF UnionBounds(IEnumerable<List<PointF>> polys)
        {
            RectangleF u = RectangleF.Empty;
            foreach (var p in polys)
            {
                var r = RectFromPoints(p);
                u = u.IsEmpty ? r : RectangleF.Union(u, r);
            }
            return u;
        }

        /// <summary>
        /// Resample polygon perimeter to exactly K vertices (uniform arc-length sampling).
        /// Assumes closed polygon.
        /// </summary>
        private static List<PointF> SimplifyToK(List<PointF> poly, int k)
        {
            if (poly == null || poly.Count == 0 || k < 3) return poly ?? new List<PointF>();

            // build polyline of closed loop
            var pts = new List<PointF>(poly);
            if (pts[0] != pts[pts.Count - 1]) pts.Add(pts[0]);

            // cumulative lengths
            var seg = new List<float>();
            seg.Add(0f);
            float total = 0f;
            for (int i = 1; i < pts.Count; i++)
            {
                float dx = pts[i].X - pts[i - 1].X;
                float dy = pts[i].Y - pts[i - 1].Y;
                total += (float)Math.Sqrt(dx * dx + dy * dy);
                seg.Add(total);
            }
            if (total <= 1e-5f) return new List<PointF>(poly);

            var outPts = new List<PointF>(k);
            for (int i = 0; i < k; i++)
            {
                float t = i * (total / k);
                int idx = seg.BinarySearch(t);
                if (idx < 0) idx = ~idx;
                idx = Math.Min(Math.Max(1, idx), seg.Count - 1);
                float t0 = seg[idx - 1], t1 = seg[idx];
                float u = (t - t0) / Math.Max(1e-5f, t1 - t0);

                var A = pts[idx - 1];
                var B = pts[idx];
                outPts.Add(new PointF(A.X + (B.X - A.X) * u, A.Y + (B.Y - A.Y) * u));
            }
            return outPts;
        }

        private static void DrawOkButton(Graphics g, Rectangle r)
        {
            using (var br = new SolidBrush(Color.FromArgb(235, 255, 255, 255)))
            {
                g.FillRectangle(br, r);
                g.DrawRectangle(Pens.SeaGreen, r);
                using (var p = new Pen(Color.SeaGreen, 2f) { StartCap = LineCap.Round, EndCap = LineCap.Round })
                {
                    var p1 = new Point(r.Left + 4, r.Top + r.Height / 2);
                    var p2 = new Point(r.Left + 8, r.Bottom - 5);
                    var p3 = new Point(r.Right - 4, r.Top + 5);
                    g.DrawLines(p, new[] { p1, p2, p3 });
                }
            }
        }

        private static void DrawCancelButton(Graphics g, Rectangle r)
        {
            using (var br = new SolidBrush(Color.FromArgb(235, 255, 255, 255)))
            {
                g.FillRectangle(br, r);
                g.DrawRectangle(Pens.IndianRed, r);
                using (var p = new Pen(Color.IndianRed, 2f) { StartCap = LineCap.Round, EndCap = LineCap.Round })
                {
                    g.DrawLine(p, r.Left + 4, r.Top + 4, r.Right - 4, r.Bottom - 4);
                    g.DrawLine(p, r.Right - 4, r.Top + 4, r.Left + 4, r.Bottom - 4);
                }
            }
        }
    }
}
