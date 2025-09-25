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
    /// - (추가) ROI 모드(우클릭) + Ctrl+D 자동분류/자동커밋 + ROI 이동/리사이즈 + 정규화 ROI API
    /// </summary>
    public sealed class AITool : ITool
    {
        // === ADD: 프리셋/게이트 상태 ===
        private enum RetunePreset { Conservative = 0, Balanced = 1, Aggressive = 2 }
        private RetunePreset _preset = RetunePreset.Balanced;

        private Rectangle _retuneBtnScr;          // ↻ 버튼 화면좌표
        private readonly int BtnSize = 18;        // (기존 상수와 일치)
        private readonly int BtnGap = 6;

        private bool _gateAsForeground = true;
        private AISegmenterOptions.GateSpace _gateSpace = AISegmenterOptions.GateSpace.Lab;
        private int _gateClusterIndex = 0;
        private List<Color> _gatePalette = null;  // KMeans로 구한 ROI 색상 클러스터

        // 마지막 프롬프트(같은 영역 재세그 용)
        private AISegmenterPrompt _lastPrompt = null;


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
        private int _previewVertexCount = 48; // 기본값 (각 폴리곤 리샘플 상한)
        public bool HasActiveRoi => _roiMode && !_roiRectImg.IsEmpty;

        public void SetPreviewVertexCount(int k)
        {
            if (k < 0) k = 0;          // 0 => 리샘플 생략(원본 유지)
            if (k > 256) k = 256;      // 안전 상한
            _previewVertexCount = k;
        }
        // === ADD: 프리셋 값 적용 ===
        private void ApplyPresetValues(AISegmenterOptions opt, RetunePreset p)
        {
            if (opt == null) return;

            switch (p)
            {
                case RetunePreset.Conservative:
                    opt.MinAreaPx = 240;   // 큰 조각 위주
                    opt.CloseKernel = 7;   // 매끈하게/구멍메움
                    opt.ApproxEpsilon = 2.6;
                    opt.GrabCutIters = 3;
                    opt.GateTolerance = 20; // 색 범위 좁게
                    break;

                case RetunePreset.Balanced:
                    opt.MinAreaPx = 120;
                    opt.CloseKernel = 5;
                    opt.ApproxEpsilon = 2.0;
                    opt.GrabCutIters = 2;
                    opt.GateTolerance = 30;
                    break;

                case RetunePreset.Aggressive:
                    opt.MinAreaPx = 40;    // 작은 조각 허용
                    opt.CloseKernel = 3;   // 덜 메움
                    opt.ApproxEpsilon = 1.2;
                    opt.GrabCutIters = 1;
                    opt.GateTolerance = 38; // 색 범위 넓게
                    break;
            }
        }
        // === ADD: ROI 색 팔레트 추출 (간단 KMeans, 3개 권장) ===
        // ROI에서 대표색 팔레트 K=3 추출 (OpenCvSharp 호환 버전)
        private List<Color> ExtractRoiPaletteKMeans(Bitmap src, RectangleF roiImg, int k = 3)
        {
            if (src == null || k <= 0) return null;

            var rect = Rectangle.Round(roiImg);
            rect.Intersect(new Rectangle(0, 0, src.Width, src.Height));
            if (rect.Width < 2 || rect.Height < 2) return null;

            using (var roi = src.Clone(rect, System.Drawing.Imaging.PixelFormat.Format24bppRgb))
            using (var m = OpenCvSharp.Extensions.BitmapConverter.ToMat(roi))
            {
                // 다운샘플 (속도)
                int step = Math.Max(1, Math.Min(4, Math.Min(m.Cols, m.Rows) / 128));

                // 샘플 행렬 준비 (배열 생성자 쓰지 않고 안전하게 Set로 채움)
                // 수집 먼저
                var pts = new List<OpenCvSharp.Vec3f>();
                for (int y = 0; y < m.Rows; y += step)
                {
                    for (int x = 0; x < m.Cols; x += step)
                    {
                        var bgr = m.Get<OpenCvSharp.Vec3b>(y, x);
                        pts.Add(new OpenCvSharp.Vec3f(bgr.Item2, bgr.Item1, bgr.Item0)); // R,G,B
                    }
                }
                if (pts.Count < k) k = Math.Max(1, pts.Count);

                // pts → samples (CV_32FC1, rows = N, cols = 3)
                var samples = new OpenCvSharp.Mat(pts.Count, 3, OpenCvSharp.MatType.CV_32FC1);
                for (int i = 0; i < pts.Count; i++)
                {
                    var v = pts[i]; // R,G,B (float)
                    samples.Set(i, 0, v.Item0);
                    samples.Set(i, 1, v.Item1);
                    samples.Set(i, 2, v.Item2);
                }

                using (var labels = new OpenCvSharp.Mat())
                using (var centers = new OpenCvSharp.Mat())
                {
                    var term = new OpenCvSharp.TermCriteria(
                        OpenCvSharp.CriteriaTypes.Eps | OpenCvSharp.CriteriaTypes.MaxIter, 10, 1.0);

                    OpenCvSharp.Cv2.Kmeans(
                        samples, k, labels, term, 1,
                        OpenCvSharp.KMeansFlags.PpCenters, centers);

                    var palette = new List<Color>(k);
                    for (int iCenter = 0; iCenter < centers.Rows; iCenter++)
                    {
                        float r = centers.At<float>(iCenter, 0);
                        float g = centers.At<float>(iCenter, 1);
                        float b = centers.At<float>(iCenter, 2);
                        palette.Add(Color.FromArgb(
                            255,
                            (int)Math.Max(0, Math.Min(255, r)),
                            (int)Math.Max(0, Math.Min(255, g)),
                            (int)Math.Max(0, Math.Min(255, b))));
                    }
                    return palette;
                }
            }
        }


        // === ADD: 리튠 실행 ===
        private async Task RetuneAsync(ImageCanvas c, bool cyclePreset, bool cycleCluster, bool togglePolarity)
        {
            if (c == null || c.Image == null) return;
            if (_aiBusy) return;

            // 1) 상태 갱신
            if (cyclePreset)
            {
                _preset = (RetunePreset)(((int)_preset + 1) % 3);
            }
            if (togglePolarity) _gateAsForeground = !_gateAsForeground;
            if (cycleCluster && _gatePalette != null && _gatePalette.Count > 0)
            {
                _gateClusterIndex = (_gateClusterIndex + 1) % _gatePalette.Count;
            }

            // 2) 프롬프트 확보 (마지막 사용)
            var prompt = _lastPrompt;
            if (prompt == null)
            {
                // ROI 모드라면 ROI 박스를 사용
                if (_roiMode && !_roiRectImg.IsEmpty) prompt = AISegmenterPrompt.FromBox(_roiRectImg);
                else if (!_rectImg.IsEmpty) prompt = AISegmenterPrompt.FromBox(_rectImg);
                else return; // 재세그할 프롬프트가 없음
            }

            // 3) 팔레트 준비(최초 1회)
            if (_gatePalette == null || _gatePalette.Count == 0)
            {
                // 프롬프트가 Box인 경우에만 팔레트 추출; Points일 때는 UnionBounds 사용 가능
                RectangleF roiForPalette;
                if (prompt.Kind == PromptKind.Box && prompt.Box.HasValue) roiForPalette = prompt.Box.Value;
                else roiForPalette = _previewUnionBoundsImg.IsEmpty ? new RectangleF(0, 0, c.Image.Width, c.Image.Height) : _previewUnionBoundsImg;

                _gatePalette = ExtractRoiPaletteKMeans(c.Image as Bitmap, roiForPalette, 3) ?? new List<Color> { c.ActiveStrokeColor };
                _gateClusterIndex = Math.Min(_gateClusterIndex, _gatePalette.Count - 1);
            }

            // 4) 옵션 적용 (프리셋 + ColorGate)
            _opts.UseColorGate = true;
            _opts.GateColorSpace = _gateSpace;
            _opts.GateAsForeground = _gateAsForeground;
            _opts.GateColor = _gatePalette[Math.Max(0, Math.Min(_gateClusterIndex, _gatePalette.Count - 1))];
            ApplyPresetValues(_opts, _preset);

            // 5) 실행
            _aiBusy = true;
            Cursor old = c.Cursor;
            Bitmap bmpSafe = null;
            List<List<PointF>> polys = null;

            try
            {
                c.Cursor = Cursors.WaitCursor;
                bmpSafe = CloneForWorker(c.Image);
                polys = await Task.Run(() => _segmenter.Segment(bmpSafe, prompt, _opts));
            }
            catch (Exception ex)
            {
                var owner = c.FindForm() ?? (IWin32Window)c;
                MessageBox.Show(owner, "Retune failed:\n" + ex.Message, "AI",
                                MessageBoxButtons.OK, MessageBoxIcon.Error);
                return;
            }
            finally
            {
                if (bmpSafe != null) bmpSafe.Dispose();
                c.Cursor = old;
                _aiBusy = false;
            }

            if (polys == null || polys.Count == 0) return;

            // 6) 프리뷰 교체 (기존 로직과 동일)
            var outPolys = new List<List<PointF>>(polys.Count);
            foreach (var p in polys)
            {
                float peri = Perimeter(p);
                int desiredK = (int)Math.Max(8, Math.Min(128, Math.Round(peri / 6f)));
                var q = (_previewVertexCount > 0) ? SimplifyToK(p, Math.Min(_previewVertexCount, desiredK)) : p;
                outPolys.Add(q);
            }

            _previewPolys = outPolys;
            _previewUnionBoundsImg = UnionBounds(outPolys);
            c.Invalidate();
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

        public bool IsEditingActive => _dragging || (_previewPolys != null && _previewPolys.Count > 0) || _roiHandle != RoiHandle.None;

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
                if (_retuneBtnScr.Contains(e.Location))
                {
                    bool shift = (Control.ModifierKeys & Keys.Shift) == Keys.Shift;   // 색 클러스터 순환
                    bool ctrl = (Control.ModifierKeys & Keys.Control) == Keys.Control; // FG/BG 극성 토글
                    _ = RetuneAsync(c, cyclePreset: !shift && !ctrl, cycleCluster: shift, togglePolarity: ctrl);
                    return;
                }
            }

            // === ROI 편집 진입(좌클릭) ===
            if (_roiMode && c.Image != null && e.Button == MouseButtons.Left)
            {
                var h = HitTestRoi(c, e.Location);
                if (h != RoiHandle.None)
                {
                    _roiHandle = h;
                    _roiStartRectImg = _roiRectImg;
                    _roiDragStartScr = e.Location;
                    c.Capture = true;
                    return;
                }
            }

            // 기본 드래그 박스 시작(좌클릭)
            if (e.Button == MouseButtons.Left && c.Image != null)
            {
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

            // === ROI 이동/리사이즈 중 ===
            if (_roiMode && _roiHandle != RoiHandle.None)
            {
                var dx = e.Location.X - _roiDragStartScr.X;
                var dy = e.Location.Y - _roiDragStartScr.Y;

                // 스크린 델타 -> 이미지 델타
                var p0 = c.Transform.ScreenToImage(_roiDragStartScr);
                var p1 = c.Transform.ScreenToImage(new Point(_roiDragStartScr.X + dx, _roiDragStartScr.Y + dy));
                float ddx = p1.X - p0.X;
                float ddy = p1.Y - p0.Y;

                var r = _roiStartRectImg;

                switch (_roiHandle)
                {
                    case RoiHandle.Move: r.X += ddx; r.Y += ddy; break;
                    case RoiHandle.N: r.Y += ddy; r.Height -= ddy; break;
                    case RoiHandle.S: r.Height += ddy; break;
                    case RoiHandle.W: r.X += ddx; r.Width -= ddx; break;
                    case RoiHandle.E: r.Width += ddx; break;
                    case RoiHandle.NW: r.X += ddx; r.Width -= ddx; r.Y += ddy; r.Height -= ddy; break;
                    case RoiHandle.NE: r.Width += ddx; r.Y += ddy; r.Height -= ddy; break;
                    case RoiHandle.SW: r.X += ddx; r.Width -= ddx; r.Height += ddy; break;
                    case RoiHandle.SE: r.Width += ddx; r.Height += ddy; break;
                }

                // 최소 크기 + 경계 클램프
                if (r.Width < 4) r.Width = 4;
                if (r.Height < 4) r.Height = 4;
                r = ClampToImage(r, c.Transform.ImageSize);

                _roiRectImg = r;
                c.Invalidate();
                return;
            }

            // 기본 드래그 박스 갱신
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

            // === ROI 편집 종료 ===
            if (_roiMode && _roiHandle != RoiHandle.None && e.Button == MouseButtons.Left)
            {
                _roiHandle = RoiHandle.None;
                c.Capture = false;
                c.Invalidate();
                return;
            }

            // 기본 드래그 박스 확정
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

                RunSegmentation(c, _rectImg, false); // async
            }
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            if (c == null) return;

            // === Ctrl + D : ROI 모드에서 현재 ROI로 세그먼트 실행 + 자동 커밋 ===
            if (_roiMode && c.Image != null && e.Control && e.KeyCode == Keys.D)
            {
                if (_roiRectImg.Width >= 4 && _roiRectImg.Height >= 4)
                {
                    RunSegmentation(c, _roiRectImg, true); // 자동 커밋
                    e.Handled = e.SuppressKeyPress = true;
                    return;
                }
            }

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

            // === ROI overlay (남색) ===
            if (_roiMode && !_roiRectImg.IsEmpty)
            {
                var scr = c.Transform.ImageRectToScreen(_roiRectImg);

                // 1) 살짝 밝게 채워서 영역 식별
                using (var fill = new SolidBrush(Color.FromArgb(80, Color.White)))
                    g.FillRectangle(fill, scr);

                // 2) 무지개 테두리(두께 2~3px 권장)
                DrawRainbowRect(g, scr, 2f);

                // 3) 리사이즈 핸들(작은 사각형) - 테두리는 중립색으로
                var hs = 6f;
                void Handle(float x, float y)
                {
                    var r = new RectangleF(x - hs, y - hs, hs * 2f, hs * 2f);
                    g.FillRectangle(Brushes.White, r);
                    g.DrawRectangle(Pens.DimGray, r.X, r.Y, r.Width, r.Height);
                }

                // 모서리/중간점
                Handle(scr.Left, scr.Top);
                Handle(scr.Right, scr.Top);
                Handle(scr.Left, scr.Bottom);
                Handle(scr.Right, scr.Bottom);
                Handle((scr.Left + scr.Right) * 0.5f, scr.Top);
                Handle((scr.Left + scr.Right) * 0.5f, scr.Bottom);
                Handle(scr.Left, (scr.Top + scr.Bottom) * 0.5f);
                Handle(scr.Right, (scr.Top + scr.Bottom) * 0.5f);
            }

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
            // === ADD: Retune(↻) 버튼 및 상태 뱃지 ===
            if (_previewPolys != null && _previewPolys.Count > 0)
            {
                var bScr = c.Transform.ImageRectToScreen(_previewUnionBoundsImg);
                var retRect = new Rectangle(_cancelBtnScr.Right + BtnGap, _cancelBtnScr.Top, BtnSize, BtnSize);

                // ↻ 버튼
                DrawRetuneButton(g, retRect);
                _retuneBtnScr = retRect;

                // === 상태 배지 (Preset / Palette / FG/BG + 단축키 힌트) ===
                string presetText = _preset.ToString();            // Conservative / Balanced / Aggressive
                string pol = _gateAsForeground ? "FG" : "BG";
                string line1 = $"{presetText} · C{_gateClusterIndex} · {pol}";
                string line2 = "↻:Preset   Shift+↻:Color   Ctrl+↻:FG/BG";

                // 두 줄 레이아웃: 더 긴 줄 기준으로 폭 계산
                using (var f = new Font(SystemFonts.DefaultFont, FontStyle.Regular))
                {
                    var size1 = g.MeasureString(line1, f);
                    var size2 = g.MeasureString(line2, f);
                    float padX = 6f, padY = 3f, gapY = 1f;
                    float w = Math.Max(size1.Width, size2.Width) + padX * 2;
                    float h = size1.Height + gapY + size2.Height + padY * 2;

                    var badge = new RectangleF(_retuneBtnScr.Right + BtnGap, _retuneBtnScr.Top - 1, w, h);

                    using (var bg = new SolidBrush(Color.FromArgb(235, 255, 255, 255)))
                    using (var pen = new Pen(Color.DimGray))
                    using (var frg = new SolidBrush(Color.Black))
                    {
                        g.FillRectangle(bg, badge);
                        g.DrawRectangle(pen, badge.X, badge.Y, badge.Width, badge.Height);

                        float x = badge.X + padX;
                        float y = badge.Y + padY;
                        g.DrawString(line1, f, frg, new PointF(x, y));
                        y += size1.Height + gapY;
                        g.DrawString(line2, f, frg, new PointF(x, y));
                    }
                }

            }
            else
            {
                _retuneBtnScr = Rectangle.Empty;
            }

        }

        // === ADD: ↻ 버튼 그리기 ===
        private static void DrawRetuneButton(Graphics g, Rectangle r)
        {
            var br = new SolidBrush(Color.FromArgb(235, 255, 255, 255));
            g.FillRectangle(br, r);
            g.DrawRectangle(Pens.SteelBlue, r);
            var p = new Pen(Color.SteelBlue, 2f) { StartCap = LineCap.Round, EndCap = LineCap.Round };
            // 간단한 회전 아이콘
            int cx = r.Left + r.Width / 2;
            int cy = r.Top + r.Height / 2;
            g.DrawArc(p, r.Left + 3, r.Top + 3, r.Width - 6, r.Height - 6, 30, 270);
            g.DrawLine(p, cx + 4, r.Top + 6, r.Right - 4, r.Top + 8);
        }


        // 무지개 컬러 블렌드
        private static System.Drawing.Drawing2D.ColorBlend RainbowBlend()
        {
            return new System.Drawing.Drawing2D.ColorBlend
            {
                Positions = new[] { 0f, 0.2f, 0.4f, 0.6f, 0.8f, 1f },
                Colors = new[]
                {
            Color.FromArgb(255, 255,   0,   0), // Red
            Color.FromArgb(255, 255, 165,   0), // Orange
            Color.FromArgb(255, 255, 255,   0), // Yellow
            Color.FromArgb(255,   0, 128,   0), // Green
            Color.FromArgb(255,   0,   0, 255), // Blue
            Color.FromArgb(255, 128,   0, 128), // Purple
        }
            };
        }

        // 사각 ROI를 4변 각각에 맞는 방향으로 무지개 그라디언트로 그리기
        private static void DrawRainbowRect(Graphics g, RectangleF r, float thickness = 2f)
        {
            // Top (좌→우)
            using (var br = new System.Drawing.Drawing2D.LinearGradientBrush(
                   new RectangleF(r.Left, r.Top, r.Width, 1f), Color.Red, Color.Violet, 0f))
            {
                br.InterpolationColors = RainbowBlend();
                using (var p = new Pen(br, thickness)) g.DrawLine(p, r.Left, r.Top, r.Right - 1f, r.Top);
            }

            // Right (상→하)
            using (var br = new System.Drawing.Drawing2D.LinearGradientBrush(
                   new RectangleF(r.Right - 1f, r.Top, 1f, r.Height), Color.Red, Color.Violet, 90f))
            {
                br.InterpolationColors = RainbowBlend();
                using (var p = new Pen(br, thickness)) g.DrawLine(p, r.Right - 1f, r.Top, r.Right - 1f, r.Bottom - 1f);
            }

            // Bottom (우→좌)  ※ 방향 반전
            using (var br = new System.Drawing.Drawing2D.LinearGradientBrush(
                   new RectangleF(r.Left, r.Bottom - 1f, r.Width, 1f), Color.Violet, Color.Red, 0f))
            {
                br.InterpolationColors = RainbowBlend();
                using (var p = new Pen(br, thickness)) g.DrawLine(p, r.Right - 1f, r.Bottom - 1f, r.Left, r.Bottom - 1f);
            }

            // Left (하→상)  ※ 방향 반전
            using (var br = new System.Drawing.Drawing2D.LinearGradientBrush(
                   new RectangleF(r.Left, r.Top, 1f, r.Height), Color.Violet, Color.Red, 90f))
            {
                br.InterpolationColors = RainbowBlend();
                using (var p = new Pen(br, thickness)) g.DrawLine(p, r.Left, r.Bottom - 1f, r.Left, r.Top);
            }
        }


        // -------------------- 성능 개선 + 멀티폴리곤 --------------------
        private async void RunSegmentation(ImageCanvas c, RectangleF boxImg, bool autoCommit = false)
        {
            if (_aiBusy) return;    // 이미 실행 중이면 무시
            _aiBusy = true;
            _autoCommitNextPreview = autoCommit;

            var prompt = AISegmenterPrompt.FromBox(boxImg);
            _lastPrompt = prompt; // === ADD: 재세그용으로 보관
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

            if (_autoCommitNextPreview && _roiMode && !_roiRectImg.IsEmpty && c.Image != null)
            {
                var sz = c.Transform.ImageSize;            // SizeF
                _roiNormPersist = NormalizeRect(_roiRectImg, sz); // 내부 보관
            }

            // ★ Ctrl+D 등 autoCommit 요청 시 즉시 커밋
            if (_autoCommitNextPreview)
            {
                _autoCommitNextPreview = false;
                CommitPreview(c);

                // [ADD] 커밋 후에는 "현재 이미지에서만" ROI 오버레이를 숨김 (모드는 유지)
                _roiRectImg = RectangleF.Empty;
                _roiHandle = RoiHandle.None;
                c.Invalidate();
            }
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

        // [ADD] 현재 ROI 영역으로 GrabCut 실행 후 자동 커밋
        public async Task<bool> AutoLabelCurrentRoiAndCommitAsync(ImageCanvas c)
        {
            if (c == null || c.Image == null) return false;
            if (!_roiMode || _roiRectImg.IsEmpty) return false;

            var boxImg = _roiRectImg;   // ← 현재 ROI 박스 사용 (전체 프레임 아님)

            if (_aiBusy) return false;
            _aiBusy = true;

            List<List<PointF>> polys = null;
            Cursor old = c.Cursor;
            Bitmap bmpSafe = null;

            try
            {
                c.Cursor = Cursors.WaitCursor;
                bmpSafe = CloneForWorker(c.Image);

                polys = await Task.Run(() =>
                    _segmenter.Segment(bmpSafe, AISegmenterPrompt.FromBox(boxImg), _opts));
            }
            catch
            {
                return false;
            }
            finally
            {
                if (bmpSafe != null) bmpSafe.Dispose();
                c.Cursor = old;
                _aiBusy = false;
            }

            if (polys == null || polys.Count == 0) return false;

            // 프리뷰 구성 → 자동 커밋
            var outPolys = new List<List<PointF>>(polys.Count);
            foreach (var p in polys)
            {
                float peri = Perimeter(p);
                int desiredK = (int)Math.Max(8, Math.Min(128, Math.Round(peri / 6f)));
                var q = (_previewVertexCount > 0) ? SimplifyToK(p, Math.Min(_previewVertexCount, desiredK)) : p;
                outPolys.Add(q);
            }
            _previewPolys = outPolys;
            _previewUnionBoundsImg = UnionBounds(outPolys);
            _rectImg = RectangleF.Empty;

            // 커밋(현재 활성 라벨 사용)
            CommitPreview(c);

            // 저장 전에 “편집/선택 잔상” 없애기
            c.Selection?.Clear();
            c.Invalidate();

            return true;
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

        public async Task<bool> AutoLabelFullImageAndCommitAsync(ImageCanvas c)
        {
            if (c == null || c.Image == null) return false;

            var szf = c.Transform.ImageSize;
            // 이미지의 최소 변의 2%를 여백으로, 범위는 [2px, 32px]
            float pad = Math.Max(2f, Math.Min(32f, 0.02f * Math.Min(szf.Width, szf.Height)));
            float x = pad, y = pad;
            float w = Math.Max(8f, szf.Width - pad * 2f);
            float h = Math.Max(8f, szf.Height - pad * 2f);
            var full = new RectangleF(x, y, w, h);

            // RunSegmentation는 async void라 대기 불가 -> 여기서 동일 로직을 await로 실행
            if (_aiBusy) return false;
            _aiBusy = true;

            List<List<PointF>> polys = null;
            Cursor old = c.Cursor;
            Bitmap bmpSafe = null;

            try
            {
                c.Cursor = Cursors.WaitCursor;
                bmpSafe = CloneForWorker(c.Image);

                polys = await Task.Run(() => _segmenter.Segment(bmpSafe, AISegmenterPrompt.FromBox(full), _opts));
            }
            catch
            {
                return false;
            }
            finally
            {
                if (bmpSafe != null) bmpSafe.Dispose();
                c.Cursor = old;
                _aiBusy = false;
            }

            if (polys == null || polys.Count == 0) return false;

            // 프리뷰 구성 → 자동 커밋
            var outPolys = new List<List<PointF>>(polys.Count);
            foreach (var p in polys)
            {
                float peri = Perimeter(p);
                int desiredK = (int)Math.Max(8, Math.Min(128, Math.Round(peri / 6f)));
                var q = (_previewVertexCount > 0) ? SimplifyToK(p, Math.Min(_previewVertexCount, desiredK)) : p;
                outPolys.Add(q);
            }
            _previewPolys = outPolys;
            _previewUnionBoundsImg = UnionBounds(outPolys);
            _rectImg = RectangleF.Empty;

            // 커밋(현재 ActiveLabelName 사용)
            CommitPreview(c);

            // 커밋 후 선택/편집 잔상 없애기(저장 전에 깔끔)
            if (c.Selection != null) c.Selection.Clear();
            c.Invalidate();
            return true;
        }

        // ========================= [AI ROI mode] Public API =========================

        // 외부(MainForm 등)에서 ROI 모드를 켜며 초기 정규화 ROI를 전달 가능
        public void EnableRoiMode(ImageCanvas c, RectangleF? initialNorm = null)
        {
            _roiMode = true;
            if (c?.Image != null)
            {
                var sz = c.Transform.ImageSize;
                var seed = initialNorm ?? _roiNormPersist;   // [CHG] persist 우선 사용
                _roiRectImg = seed.HasValue ? DenormalizeRect(seed.Value, sz)
                                            : (_roiRectImg.IsEmpty ? DefaultCenterRoi(sz) : _roiRectImg);
                _roiRectImg = ClampToImage(_roiRectImg, sz);
                c.Invalidate();
            }
        }


        public void DisableRoiMode(ImageCanvas c)
        {
            _roiMode = false;
            _roiHandle = RoiHandle.None;
            c?.Invalidate();
        }

        public RectangleF? GetRoiNormalized(Size imgSize)
        {
            if (!_roiMode) return null;
            if (!_roiRectImg.IsEmpty) return NormalizeRect(_roiRectImg, imgSize);
            return _roiNormPersist;  // [CHG] 화면에 ROI가 사라졌어도 직전 비율 반환
        }

        // 새 이미지 로드시 직전 ROI를 동일 비율로 재생성
        public void EnsureRoiForCurrentImage(ImageCanvas c, RectangleF? lastNorm)
        {
            if (!_roiMode || c?.Image == null) return;
            var sz = c.Transform.ImageSize;
            var seed = lastNorm ?? _roiNormPersist;   // [CHG] persist 우선 사용
            if (seed.HasValue) _roiRectImg = DenormalizeRect(seed.Value, sz);
            if (_roiRectImg.IsEmpty) _roiRectImg = DefaultCenterRoi(sz);
            _roiRectImg = ClampToImage(_roiRectImg, sz);
            c.Invalidate();
        }

        // ========================= [AI ROI mode] Internals =========================

        private bool _roiMode = false;
        private RectangleF _roiRectImg = RectangleF.Empty;

        private enum RoiHandle { None, Move, N, S, E, W, NE, NW, SE, SW }
        private RoiHandle _roiHandle = RoiHandle.None;
        private Point _roiDragStartScr;
        private RectangleF _roiStartRectImg;

        // Ctrl+D auto-commit flag
        private bool _autoCommitNextPreview = false;
        private RectangleF? _roiNormPersist = null;

        private static RectangleF NormalizeRect(RectangleF r, SizeF img) =>
            img.Width <= 0 || img.Height <= 0 ? RectangleF.Empty :
            new RectangleF(r.X / img.Width, r.Y / img.Height, r.Width / img.Width, r.Height / img.Height);
        private static RectangleF NormalizeRect(RectangleF r, Size img) =>
            NormalizeRect(r, new SizeF(img.Width, img.Height));

        private static RectangleF DenormalizeRect(RectangleF rNorm, SizeF img) =>
            new RectangleF(rNorm.X * img.Width, rNorm.Y * img.Height, rNorm.Width * img.Width, rNorm.Height * img.Height);
        private static RectangleF DenormalizeRect(RectangleF rNorm, Size img) =>
            DenormalizeRect(rNorm, new SizeF(img.Width, img.Height));

        private static RectangleF DefaultCenterRoi(SizeF img)
        {
            float w = img.Width * 0.5f, h = img.Height * 0.5f;
            return new RectangleF((img.Width - w) * 0.5f, (img.Height - h) * 0.5f, w, h);
        }
        private static RectangleF DefaultCenterRoi(Size img) =>
            DefaultCenterRoi(new SizeF(img.Width, img.Height));

        private static RectangleF ClampToImage(RectangleF r, SizeF img)
        {
            float x = Math.Max(0, Math.Min(r.X, img.Width - 1));
            float y = Math.Max(0, Math.Min(r.Y, img.Height - 1));
            float w = Math.Max(1, Math.Min(r.Width, img.Width - x));
            float h = Math.Max(1, Math.Min(r.Height, img.Height - y));
            return new RectangleF(x, y, w, h);
        }
        private static RectangleF ClampToImage(RectangleF r, Size img) =>
            ClampToImage(r, new SizeF(img.Width, img.Height));

        private RoiHandle HitTestRoi(ImageCanvas c, Point screenPt)
        {
            if (_roiRectImg.IsEmpty) return RoiHandle.None;
            var scr = c.Transform.ImageRectToScreen(_roiRectImg);
            const int pad = 6;
            bool inside = scr.Contains(screenPt);
            bool L = Math.Abs(screenPt.X - scr.Left) <= pad;
            bool R = Math.Abs(screenPt.X - scr.Right) <= pad;
            bool T = Math.Abs(screenPt.Y - scr.Top) <= pad;
            bool B = Math.Abs(screenPt.Y - scr.Bottom) <= pad;

            if (L && T) return RoiHandle.NW;
            if (R && T) return RoiHandle.NE;
            if (L && B) return RoiHandle.SW;
            if (R && B) return RoiHandle.SE;
            if (T) return RoiHandle.N;
            if (B) return RoiHandle.S;
            if (L) return RoiHandle.W;
            if (R) return RoiHandle.E;
            return inside ? RoiHandle.Move : RoiHandle.None;
        }
    }
}
