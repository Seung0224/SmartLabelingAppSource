using Cyotek.Windows.Forms;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Reflection;
using System.Windows.Forms;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public class ImageCanvas : ImageBox
    {
        // === Feature Toggle (source-level) ===
        // Set this to true/false in code to enable/disable auto-union of same-label overlaps.
        public static bool AUTO_UNION_SAME_LABEL = true;


        // === Rotation (interactive handle) ===
        private bool _rotating = false;
        private PointF _rotCenterImg;
        private float _rotStartAngleRad = 0f;
        private float _rotCurrAngleRad = 0f;
        private List<IShape> _rotBase; // clones at rotation start (order aligned with Selection)
        private RectangleF _rotG0;     // union bounds at start

        private const int ROT_HANDLE_RADIUS_SCR = 8;   // px
        private const int ROT_HANDLE_OFFSET_SCR = 28;  // px above union top
        #region 1) Constants & Styling (상수/스타일)

        // 배지(라벨 태그) 레이아웃
        private const int LABEL_BADGE_GAP_PX = 2; // 도형과 배지 사이 간격
        private const int LABEL_BADGE_PADX = 4; // 텍스트 좌우 패딩
        private const int LABEL_BADGE_PADY = 3; // 텍스트 상하 패딩
        private const int LABEL_BADGE_BORDER_PX = 2; // 0이면 테두리 없음
        private const int LABEL_BADGE_ACCENT_W = 4; // 왼쪽 색 막대 너비(0이면 없음)
        private const int LABEL_BADGE_WIPE_PX = 1; // 배경 지우기용 여유

        // 핸들/히트 상수
        private float HandleSize => EditorUIConfig.HandleDrawSizePx;
        private float HandleHitSize => EditorUIConfig.CornerHitPx;
        private float MoveHitInflate => EditorUIConfig.EdgeBandPx;
        // 현재 활성 라벨 스타일(MainForm에서 주입)
        private Color _activeStroke = Color.DeepSkyBlue;
        private Color _activeFill = Color.FromArgb(72, Color.DeepSkyBlue);
        private string _activeLabel = null;

        public Color ActiveStrokeColor => _activeStroke;
        public Color ActiveFillColor => _activeFill;
        public string ActiveLabelName => _activeLabel;


        public struct InferenceBadge   // ← public, 그리고 ImageCanvas 내부
        {
            public RectangleF BoxImg;
            public string Text;
            public Color Accent;
        }

        private readonly List<InferenceBadge> _inferenceBadges = new List<InferenceBadge>();

        public void SetInferenceBadges(IEnumerable<InferenceBadge> badges)
        {
            _inferenceBadges.Clear();
            if (badges != null) _inferenceBadges.AddRange(badges);
            Invalidate();
        }

        public void ClearInferenceBadges()
        {
            _inferenceBadges.Clear();
            Invalidate();
        }

        public void SetActiveLabel(string labelName, Color strokeColor, Color fillColor)
        {
            _activeLabel = labelName;
            _activeStroke = strokeColor;
            _activeFill = fillColor;
            Invalidate();
        }

        // 새 도형에 1회 스타일 주입 추적
        private readonly HashSet<IShape> _styledShapes = new HashSet<IShape>();

        // 리플렉션 캐시(도형별 속성 이름 다를 수 있음)
        private sealed class StyleProps
        {
            public PropertyInfo Stroke;
            public PropertyInfo Fill;
            public PropertyInfo Color;     // 브러시류 등 단일 Color만 쓰는 경우
            public PropertyInfo LabelName;
        }
        private readonly Dictionary<Type, StyleProps> _stylePropCache = new Dictionary<Type, StyleProps>();

        private void ApplyStylesToNewShapes()
        {
            for (int i = 0; i < Shapes.Count; i++)
            {
                var s = Shapes[i];
                if (_styledShapes.Contains(s)) continue;
                ApplyActiveStyleTo(s);
                _styledShapes.Add(s);
            }
        }
        private void ApplyActiveStyleTo(IShape shape)
        {
            if (shape == null) return;

            var t = shape.GetType();
            if (!_stylePropCache.TryGetValue(t, out var props))
            {
                props = new StyleProps
                {
                    Stroke = t.GetProperty("StrokeColor"),
                    Fill = t.GetProperty("FillColor"),
                    Color = t.GetProperty("Color"),
                    LabelName = t.GetProperty("LabelName") ?? t.GetProperty("Name")
                };
                // 대체 이름
                if (props.Stroke == null) props.Stroke = t.GetProperty("PenColor");
                if (props.Fill == null) props.Fill = t.GetProperty("BrushColor");
                _stylePropCache[t] = props;
            }

            try
            {
                if (props.Stroke != null && props.Stroke.CanWrite) props.Stroke.SetValue(shape, _activeStroke);
                if (props.Fill != null && props.Fill.CanWrite) props.Fill.SetValue(shape, _activeFill);
                if (props.Color != null && props.Color.CanWrite) props.Color.SetValue(shape, _activeFill);
                if (props.LabelName != null && props.LabelName.CanWrite) props.LabelName.SetValue(shape, _activeLabel);
            }
            catch { /* 안전하게 무시 */ }
        }

        #endregion

        #region 2) Core Services & Public API (핵심 상태/서비스)

        public event Action ToolEditBegan;
        internal void RaiseToolEditBegan() => ToolEditBegan?.Invoke();

        public readonly List<IShape> Shapes = new List<IShape>();
        public readonly SelectionService Selection = new SelectionService();
        public readonly ClipboardService Clipboard = new ClipboardService();
        public readonly HistoryService History = new HistoryService();

        public IViewTransform Transform { get; private set; }
        public PointF LastMouseImg { get; set; } = new PointF(0, 0);

        public int BrushDiameterPx { get; private set; } = 18;
        public void SetBrushDiameter(int px)
        {
            if (px < 1) px = 1;
            BrushDiameterPx = px;
            Invalidate();
        }
        public float GetScreenScaleApprox() => Math.Max(0.5f, Zoom / 100f);

        // 현재 선택 존재 여부를 외부에서 편하게 확인
        public bool HasSelection => Selection != null && Selection.HasAny;

        /// <summary>
        /// 편집 상태(회전/러버밴드/멀티편집/팬)를 모두 종료하고
        /// 선택도 해제한 뒤 포인터 모드로 되돌립니다.
        /// Ctrl+S 저장 이후 호출하여 "기본 상태"로 복원할 때 사용하세요.
        /// </summary>
        public void ClearSelectionAndResetEditing()
        {
            // 회전 취소
            _rotating = false;
            _rotBase = null;
            _rotCurrAngleRad = 0f;

            // 멀티 편집 취소
            _msActive = false;
            _msHandle = HandleType.None;
            _msStartRects = null;

            // 러버밴드/캡처/커서 초기화
            _rbActive = false;
            _rbRectImg = RectangleF.Empty;
            Capture = false;
            Cursor = Cursors.Default;

            // 팬 상태 해제
            _isPanning = false;
            PanMode = false;

            // 선택 해제 & 포인터 모드
            if (Selection != null) Selection.Clear();
            Mode = ToolMode.Pointer;

            Invalidate();
        }
        public void ClearSelectionButKeepMode()
        {
            // 회전 취소
            _rotating = false;
            _rotBase = null;
            _rotCurrAngleRad = 0f;

            // 멀티 편집 취소
            _msActive = false;
            _msHandle = HandleType.None;
            _msStartRects = null;

            // 러버밴드/캡처/커서 초기화
            _rbActive = false;
            _rbRectImg = RectangleF.Empty;
            Capture = false;
            Cursor = Cursors.Default;

            // 팬 상태 해제
            _isPanning = false;
            PanMode = false;

            // 선택만 해제 (툴 모드는 절대 건드리지 않음)
            if (Selection != null) Selection.Clear();

            Invalidate();
        }


        public bool HasAnyShape => Shapes.Count > 0;

        public void ClearAllShapes()
        {
            if (Shapes.Count == 0 && !Selection.HasAny && !HasRubberBand) return;
            Shapes.Clear();
            Selection.Clear();
            ClearRubberBand();
            _styledShapes.Clear();
            Invalidate();
        }

        public void AddBox(RectangleF imgRect, string labelName, Color? stroke = null, Color? fill = null)
        {
            if (imgRect.Width <= 0 || imgRect.Height <= 0) return;

            var sh = new RectangleShape(imgRect);
            sh.LabelName = labelName;
            if (stroke.HasValue) sh.StrokeColor = stroke.Value;
            if (fill.HasValue) sh.FillColor = fill.Value;

            this.Shapes.Add(sh);
            _styledShapes.Add(sh); // prevent re-styling later
            this.Invalidate();
        }

        public void AddPolygon(List<PointF> imgPoints, string labelName, Color? stroke = null, Color? fill = null)
        {
            if (imgPoints == null || imgPoints.Count < 3) return;

            var sh = new PolygonShape(new List<PointF>(imgPoints));
            sh.LabelName = labelName;
            if (stroke.HasValue) sh.StrokeColor = stroke.Value;
            if (fill.HasValue) sh.FillColor = fill.Value;

            this.Shapes.Add(sh);
            _styledShapes.Add(sh); // prevent re-styling later
            this.Invalidate();
        }

        public void LoadImage(Image img)
        {
            if (Image != null) Image.Dispose();
            Image = img;

            Shapes.Clear();
            Selection.Clear();
            ClearRubberBand();
            _styledShapes.Clear();

            if (Image != null)
            {
                ZoomToFit();
                LastMouseImg = new PointF(Image.Width / 2f, Image.Height / 2f);
            }
            Invalidate();
        }

        #endregion

        #region 3) Tools & Mode (툴/모드 및 편집 상태)

        private readonly ITool _nullTool = new NullTool();
        private readonly Dictionary<ToolMode, ITool> _tools;
        private ToolMode _mode;

        public ToolMode Mode
        {
            get => _mode;
            set
            {
                _mode = value;
                ApplyPolygonPresetForMode(_mode);
                ModeChanged?.Invoke(_mode);
            }
        }
        public event Action<ToolMode> ModeChanged;

        private ITool ActiveTool
        {
            get
            {
                if (_tools != null && _tools.TryGetValue(Mode, out var t) && t != null) return t;
                return _tools != null && _tools.TryGetValue(ToolMode.Pointer, out var p) && p != null ? p : _nullTool;
            }
        }

        public new bool PanMode { get; set; }
        private bool _isPanning;
        private Point _panMouseDown;
        private Point _panScrollStart;

        // 멀티선택 편집 상태
        private bool _msActive;
        private HandleType _msHandle = HandleType.None;
        private RectangleF _msBoundsStartImg;
        private Dictionary<IShape, RectangleF> _msStartRects;
        private PointF _msDragStartImg;

        // 러버밴드
        private bool _rbActive;
        private PointF _rbStartImg;
        private RectangleF _rbRectImg = RectangleF.Empty;
        public bool HasRubberBand => !_rbRectImg.IsEmpty;
        public RectangleF RubberBandImageRect => _rbRectImg;
        public void ClearRubberBand() { _rbRectImg = RectangleF.Empty; Invalidate(); }

        // 복사-붙여넣기(멀티)
        private List<IShape> _multiCopyBuffer;
        private RectangleF _multiCopyBoundsImg;

        public ImageCanvas()
        {
            SetStyle(ControlStyles.AllPaintingInWmPaint | ControlStyles.UserPaint |
                     ControlStyles.OptimizedDoubleBuffer | ControlStyles.ResizeRedraw, true);
            UpdateStyles();

            BackColor = Color.FromArgb(248, 249, 252);
            AllowZoom = true;
            SizeMode = ImageBoxSizeMode.Normal;
            Zoom = 100;

            Transform = new ImageBoxTransform(this);

            var polyTool = new PolygonTool();
            _tools = new Dictionary<ToolMode, ITool>
            {
                { ToolMode.Pointer,  new PointerTool() },
                { ToolMode.Polygon,  polyTool },
                { ToolMode.Box,      polyTool },
                { ToolMode.Triangle, polyTool },
                { ToolMode.Ngon,     polyTool },
                { ToolMode.Circle,   new CircleTool() },
                { ToolMode.Brush,    new BrushTool() },
                { ToolMode.Eraser,   new EraserTool() },
                { ToolMode.Mask,     new MaskTool() },
                { ToolMode.AI,       new AITool(new AI.GrabCutSegmenter())},
            };

            _mode = ToolMode.Pointer;
            TabStop = true;
            SetStyle(ControlStyles.Selectable, true);
        }

        public ITool GetTool(ToolMode mode) => (_tools != null && _tools.TryGetValue(mode, out var t)) ? t : null;
        public void SetPolygonPreset(PolygonPreset preset, int regularSides)
        {
            if (GetTool(ToolMode.Polygon) is PolygonTool poly)
            {
                poly.Preset = preset;
                poly.RegularSides = regularSides;
            }
        }
        private void ApplyPolygonPresetForMode(ToolMode mode)
        {
            if (!(_tools.TryGetValue(ToolMode.Polygon, out var tool) && tool is PolygonTool poly)) return;

            switch (mode)
            {
                case ToolMode.Box: poly.Preset = PolygonPreset.RectBox; break;
                case ToolMode.Triangle: poly.Preset = PolygonPreset.Triangle; break;
                case ToolMode.Ngon:
                    poly.Preset = PolygonPreset.RegularN;
                    if (poly.RegularSides < 3) poly.RegularSides = 5;
                    break;
                default: poly.Preset = PolygonPreset.Free; break;
            }
        }

        private sealed class NullTool : ITool
        {
            public bool IsEditingActive => false;
            public void OnMouseDown(ImageCanvas c, MouseEventArgs e) { }
            public void OnMouseMove(ImageCanvas c, MouseEventArgs e) { }
            public void OnMouseUp(ImageCanvas c, MouseEventArgs e) { }
            public void OnKeyDown(ImageCanvas c, KeyEventArgs e) { }
            public void DrawOverlay(ImageCanvas c, Graphics g) { }
        }

        #endregion

        #region 4) Rendering (그리기)

        protected override void OnPaint(PaintEventArgs e)
        {
            base.OnPaint(e);

            var g = e.Graphics;
            g.SmoothingMode = SmoothingMode.AntiAlias;
            g.InterpolationMode = InterpolationMode.HighQualityBicubic;
            g.PixelOffsetMode = PixelOffsetMode.HighQuality;

            // 1) 저장된 도형
            for (int i = 0; i < Shapes.Count; i++)
                Shapes[i].Draw(g, Transform);

            // 2) 선택 오버레이
            if (Selection.Selected != null)
                Selection.Selected.DrawOverlay(g, Transform, Selection.SelectedVertexIndex);


            // === Single selection: show dashed bounds + rotation handle ===
            if (Selection.Selected != null && Selection.Multi.Count == 0)
            {
                var bImg = Selection.Selected.GetBoundsImg();
                if (!bImg.IsEmpty)
                {
                    var bScr = Transform.ImageRectToScreen(bImg);
                    using (var penSel = new Pen(Color.LimeGreen, 1f) { DashStyle = DashStyle.Dash })
                    using (var brSel = new SolidBrush(Color.FromArgb(30, Color.LimeGreen)))
                    {
                        g.FillRectangle(brSel, bScr);
                        g.DrawRectangle(penSel, bScr.X, bScr.Y, bScr.Width, bScr.Height);
                    }

                    // rotation handle & guide for single selection
                    var handle = GetRotateHandleCenterScreen();
                    if (handle != Point.Empty)
                    {
                        var centerScr = new Point((int)(bScr.X + bScr.Width * 0.5f), (int)(bScr.Y + bScr.Height * 0.5f));
                        using (var penGuide = new Pen(Color.Orange, 1f) { DashStyle = DashStyle.Dot })
                            g.DrawLine(penGuide, centerScr, handle);
                        using (var br = new SolidBrush(Color.Orange))
                            g.FillEllipse(br, handle.X - ROT_HANDLE_RADIUS_SCR, handle.Y - ROT_HANDLE_RADIUS_SCR, ROT_HANDLE_RADIUS_SCR * 2, ROT_HANDLE_RADIUS_SCR * 2);
                        using (var penH = new Pen(Color.Black, 1f))
                            g.DrawEllipse(penH, handle.X - ROT_HANDLE_RADIUS_SCR, handle.Y - ROT_HANDLE_RADIUS_SCR, ROT_HANDLE_RADIUS_SCR * 2, ROT_HANDLE_RADIUS_SCR * 2);

                        if (_rotating)
                        {
                            float deg = _rotCurrAngleRad * 180f / (float)Math.PI;
                            string txt = ((int)Math.Round(deg)).ToString() + "°";
                            using (var f = new Font("Segoe UI", 9f, FontStyle.Bold))
                            using (var bg = new SolidBrush(Color.FromArgb(160, 32, 32, 32)))
                            using (var fg = new SolidBrush(Color.White))
                            {
                                var sz = g.MeasureString(txt, f);
                                var rect = new RectangleF(handle.X + 10, handle.Y - sz.Height * 0.5f, sz.Width + 8, sz.Height);
                                g.FillRectangle(bg, rect);
                                g.DrawString(txt, f, fg, rect.X + 4, rect.Y);
                            }
                        }
                    }
                }
            }
            if (Selection.Multi.Count > 0)
            {
                for (int i = 0; i < Selection.Multi.Count; i++)
                    Selection.Multi[i].DrawOverlay(g, Transform, -1);

                var boundsImg = GetSelectionBoundsImg();
                var boundsScr = Transform.ImageRectToScreen(boundsImg);
                using (var pen = new Pen(Color.LimeGreen, 1f) { DashStyle = DashStyle.Dash })
                using (var br = new SolidBrush(Color.FromArgb(30, Color.LimeGreen)))
                {
                    g.FillRectangle(br, boundsScr);
                    g.DrawRectangle(pen, boundsScr.X, boundsScr.Y, boundsScr.Width, boundsScr.Height);
                    // Rotation handle & guide
                    if (Selection.HasAny)
                    {
                        var handle = GetRotateHandleCenterScreen();
                        if (handle != Point.Empty)
                        {
                            // guide line center -> handle
                            var unionImg = GetSelectionUnionImg();
                            var unionScr = Transform.ImageRectToScreen(unionImg);
                            var centerScr = new Point((int)(unionScr.X + unionScr.Width * 0.5f), (int)(unionScr.Y + unionScr.Height * 0.5f));
                            using (var penGuide = new Pen(Color.Orange, 1f) { DashStyle = DashStyle.Dot })
                                g.DrawLine(penGuide, centerScr, handle);
                            // handle circle
                            using (var br2 = new SolidBrush(Color.Orange))
                                g.FillEllipse(br2, handle.X - ROT_HANDLE_RADIUS_SCR, handle.Y - ROT_HANDLE_RADIUS_SCR, ROT_HANDLE_RADIUS_SCR * 2, ROT_HANDLE_RADIUS_SCR * 2);
                            using (var penH = new Pen(Color.Black, 1f))
                                g.DrawEllipse(penH, handle.X - ROT_HANDLE_RADIUS_SCR, handle.Y - ROT_HANDLE_RADIUS_SCR, ROT_HANDLE_RADIUS_SCR * 2, ROT_HANDLE_RADIUS_SCR * 2);

                            if (_rotating)
                            {
                                float deg = _rotCurrAngleRad * 180f / (float)Math.PI;
                                string txt = ((int)Math.Round(deg)).ToString() + "°";
                                using (var f = new Font("Segoe UI", 9f, FontStyle.Bold))
                                using (var bg = new SolidBrush(Color.FromArgb(160, 32, 32, 32)))
                                using (var fg = new SolidBrush(Color.White))
                                {
                                    var sz = g.MeasureString(txt, f);
                                    var rect = new RectangleF(handle.X + 10, handle.Y - sz.Height * 0.5f, sz.Width + 8, sz.Height);
                                    g.FillRectangle(bg, rect);
                                    g.DrawString(txt, f, fg, rect.X + 4, rect.Y);
                                }
                            }
                        }
                    }

                }
                DrawHandles(g, boundsScr);
            }

            // 3) 진행 중 툴 오버레이
            ActiveTool.DrawOverlay(this, g);

            // 4) 러버밴드
            if (_rbActive || !_rbRectImg.IsEmpty)
            {
                var sRect = Transform.ImageRectToScreen(_rbRectImg);
                float[] dash = { 4f, 4f };
                using (var penBlack = new Pen(Color.Black, 1f) { DashPattern = dash, DashOffset = 0f })
                using (var penWhite = new Pen(Color.White, 1f) { DashPattern = dash, DashOffset = 4f })
                {
                    g.DrawRectangle(penBlack, sRect.X, sRect.Y, sRect.Width, sRect.Height);
                    g.DrawRectangle(penWhite, sRect.X, sRect.Y, sRect.Width, sRect.Height);
                }
            }

            for (int i = 0; i < Shapes.Count; i++)
            {
                var s = Shapes[i];
                var bImg = s.GetBoundsImg();
                if (bImg.IsEmpty) continue;

                var bScr = Transform.ImageRectToScreen(bImg);
                if (TryGetShapeLabelAndStroke(s, out var lbl, out var stroke) && !string.IsNullOrWhiteSpace(lbl))
                    DrawLabelTag(g, bScr, lbl, stroke);
            }

            if (_inferenceBadges.Count > 0)
            {
                foreach (var b in _inferenceBadges)
                {
                    var bScr = Transform.ImageRectToScreen(b.BoxImg);
                    DrawLabelTag(g, bScr, b.Text, b.Accent);
                }
            }
        }

        private void DrawLabelTag(Graphics g, RectangleF shapeBoundsScreen, string label, Color accentColor)
        {
            if (string.IsNullOrWhiteSpace(label)) return;

            using (var font = new Font("Segoe UI", 9f, FontStyle.Bold))
            {
                var textSz = TextRenderer.MeasureText(label, font, new Size(int.MaxValue, int.MaxValue),
                                                      TextFormatFlags.NoPadding);

                int innerW = textSz.Width + LABEL_BADGE_PADX * 2
                           + (LABEL_BADGE_ACCENT_W > 0 ? (LABEL_BADGE_ACCENT_W + LABEL_BADGE_PADX) : 0);
                int innerH = textSz.Height + LABEL_BADGE_PADY * 2;

                var tagRect = new Rectangle(
                    (int)Math.Round(shapeBoundsScreen.Left),
                    (int)Math.Round(shapeBoundsScreen.Bottom) + LABEL_BADGE_GAP_PX,
                    innerW + LABEL_BADGE_BORDER_PX * 2,
                    innerH + LABEL_BADGE_BORDER_PX * 2
                );

                if (tagRect.Right > Width) tagRect.X = Math.Max(0, Width - tagRect.Width - 1);
                if (tagRect.Bottom > Height) tagRect.Y = Math.Max(0, Height - tagRect.Height - 1);

                if (LABEL_BADGE_WIPE_PX > 0)
                {
                    var wipeRect = Rectangle.Inflate(tagRect, LABEL_BADGE_WIPE_PX, LABEL_BADGE_WIPE_PX);
                    using (var wipe = new SolidBrush(Color.White)) g.FillRectangle(wipe, wipeRect);
                }

                using (var bg = new SolidBrush(Color.White)) g.FillRectangle(bg, tagRect);

                if (LABEL_BADGE_BORDER_PX > 0)
                {
                    using (var pen = new Pen(Color.FromArgb(180, 200, 210), LABEL_BADGE_BORDER_PX))
                    {
                        var br = (RectangleF)tagRect;

                        int bpx = LABEL_BADGE_BORDER_PX;
                        if ((bpx & 1) == 1)
                        {
                            br.X += .5f; br.Y += .5f; br.Width -= 1f; br.Height -= 1f;
                        }

                        g.DrawRectangle(pen, br.X, br.Y, br.Width, br.Height);
                    }
                }

                var innerRect = Rectangle.Inflate(tagRect, -LABEL_BADGE_BORDER_PX, -LABEL_BADGE_BORDER_PX);

                int textLeft = innerRect.Left + LABEL_BADGE_PADX;
                if (LABEL_BADGE_ACCENT_W > 0)
                {
                    var accRect = new Rectangle(innerRect.Left, innerRect.Top, LABEL_BADGE_ACCENT_W, innerRect.Height);
                    using (var acc = new SolidBrush(accentColor)) g.FillRectangle(acc, accRect);
                    textLeft = accRect.Right + LABEL_BADGE_PADX;
                }

                var textPt = new Point(textLeft, innerRect.Top + LABEL_BADGE_PADY - 1);
                TextRenderer.DrawText(g, label, font, textPt, Color.Black,
                    TextFormatFlags.NoPadding | TextFormatFlags.NoClipping);
            }
        }

        private void DrawHandles(Graphics g, RectangleF sRect)
        {
            var centers = GetHandleCenters(sRect);
            float hs = HandleSize;
            using (var b = new SolidBrush(Color.LimeGreen))
            using (var p = new Pen(Color.LimeGreen, 1f))
            {
                for (int i = 0; i < centers.Length; i++)
                {
                    var hr = new RectangleF(centers[i].X - hs / 2f, centers[i].Y - hs / 2f, hs, hs);
                    g.FillRectangle(b, hr);
                    g.DrawRectangle(p, hr.X, hr.Y, hr.Width, hr.Height);
                }
            }
        }
        private static PointF[] GetHandleCenters(RectangleF r)
        {
            float x1 = r.Left, y1 = r.Top, x2 = r.Right, y2 = r.Bottom;
            float cx = x1 + r.Width / 2f, cy = y1 + r.Height / 2f;
            return new[]
            {
                new PointF(x1, y1), new PointF(cx, y1), new PointF(x2, y1),
                new PointF(x1, cy),                       new PointF(x2, cy),
                new PointF(x1, y2), new PointF(cx, y2), new PointF(x2, y2)
            };
        }
        public Cursor CursorFromHandle(HandleType h)
        {
            switch (h)
            {
                case HandleType.N:
                case HandleType.S:
                    return Cursors.SizeNS;

                case HandleType.E:
                case HandleType.W:
                    return Cursors.SizeWE;

                case HandleType.NE:
                case HandleType.SW:
                    return Cursors.SizeNESW;

                case HandleType.NW:
                case HandleType.SE:
                    return Cursors.SizeNWSE;

                case HandleType.Move:
                    return Cursors.SizeAll;

                case HandleType.Vertex:
                    return Cursors.Hand;

                default:
                    return Cursors.Default;
            }
        }


        private bool TryGetShapeLabelAndStroke(IShape s, out string label, out Color stroke)
        {
            label = null; stroke = Color.Empty;

            if (s is BrushStrokeShape b) { label = b.LabelName; stroke = b.StrokeColor; return true; }
            if (s is RectangleShape r) { label = r.LabelName; stroke = r.StrokeColor; return true; }
            if (s is CircleShape c) { label = c.LabelName; stroke = c.StrokeColor; return true; }
            if (s is PolygonShape p) { label = p.LabelName; stroke = p.StrokeColor; return true; }
            if (s is TriangleShape t) { label = t.LabelName; stroke = t.StrokeColor; return true; }

            return false;
        }

        #endregion

        #region 5) Interaction & Editing (입력/편집/유틸)

        protected override void OnMouseDown(MouseEventArgs e)
        {
            // Rotation handle hit-test (start)
            if (Selection.HasAny && HitRotateHandle(e.Location))
            {
                var g0 = GetSelectionUnionImg();
                if (!g0.IsEmpty)
                {
                    _rotating = true;
                    _rotG0 = g0;
                    _rotCenterImg = new PointF(g0.X + g0.Width * 0.5f, g0.Y + g0.Height * 0.5f);
                    _rotStartAngleRad = GetAngleRadToMouse(e.Location, _rotCenterImg);
                    _rotCurrAngleRad = 0f;

                    // snapshot shapes in selection order
                    _rotBase = new List<IShape>();
                    if (Selection.Multi.Count > 0)
                    {
                        for (int i = 0; i < Selection.Multi.Count; i++)
                            _rotBase.Add(Selection.Multi[i].Clone());
                    }
                    else if (Selection.Selected != null)
                    {
                        _rotBase.Add(Selection.Selected.Clone());
                    }

                    Capture = true;
                    // handled
                    return;
                }
            }

            if (PanMode && e.Button == MouseButtons.Left && Image != null)
            {
                if (!AutoScroll) AutoScroll = true;
                _isPanning = true;
                _panMouseDown = e.Location;
                _panScrollStart = AutoScrollPosition;
                Cursor = Cursors.Hand;
                Capture = true;
                return;
            }

            bool ctrlDown = (ModifierKeys & Keys.Control) == Keys.Control;

            if (Mode == ToolMode.Pointer && e.Button == MouseButtons.Left && Selection.Multi.Count > 0 && !ctrlDown)
            {
                if (TryBeginMultiEdit(e)) { Invalidate(); return; }

                if (OverAnySelectedShape(e.Location))
                {
                    var groupImg = GetSelectionBoundsImg();
                    _msActive = true;
                    _msHandle = HandleType.Move;
                    _msBoundsStartImg = groupImg;
                    _msDragStartImg = Transform.ScreenToImage(e.Location);
                    Capture = true;
                    Invalidate();
                    return;
                }
            }

            if (Mode == ToolMode.Pointer && e.Button == MouseButtons.Left && ctrlDown)
            {
                var hit = HitTestShape(e.Location);
                if (hit != null)
                {
                    if (Selection.Multi.Count > 0)
                    {
                        if (Selection.Multi.Contains(hit))
                        {
                            Selection.Multi.Remove(hit);
                            if (Selection.Multi.Count == 1) Selection.Set(Selection.Multi[0]);
                        }
                        else Selection.Multi.Add(hit);
                    }
                    else if (Selection.Selected != null)
                    {
                        if (Selection.Selected != hit) Selection.SetMulti(new[] { Selection.Selected, hit });
                    }
                    else Selection.Set(hit);

                    if (Selection.Multi.Count > 0) TryBeginMultiEdit(e);
                    Invalidate();
                    return;
                }
            }

            if (Mode == ToolMode.Pointer && e.Button == MouseButtons.Left && Image != null)
            {
                var imgPt = Transform.ScreenToImage(e.Location);
                if (imgPt.X >= 0 && imgPt.Y >= 0 &&
                    imgPt.X < Transform.ImageSize.Width && imgPt.Y < Transform.ImageSize.Height)
                {
                    if (!IsOverAnyShape(e.Location))
                    {
                        if (!ctrlDown) Selection.Clear();

                        _rbActive = true;
                        _rbStartImg = imgPt;
                        _rbRectImg = new RectangleF(_rbStartImg, SizeF.Empty);
                        Capture = true;
                        Invalidate();
                        return;
                    }
                }
            }

            var tool = ActiveTool;
            tool.OnMouseDown(this, e);
            if (!tool.IsEditingActive) base.OnMouseDown(e);

            Invalidate();
        }

        protected override void OnMouseMove(MouseEventArgs e)
        {
            if (_rotating)
            {
                var ang = GetAngleRadToMouse(e.Location, _rotCenterImg);
                _rotCurrAngleRad = ang - _rotStartAngleRad;
                ApplyRotationDelta(_rotCurrAngleRad);
                Invalidate();
                return;
            }

            if (PanMode && _isPanning)
            {
                int startX = -_panScrollStart.X;
                int startY = -_panScrollStart.Y;

                int dx = e.X - _panMouseDown.X;
                int dy = e.Y - _panMouseDown.Y;

                AutoScrollPosition = new Point(startX - dx, startY - dy);

                if (Image != null) LastMouseImg = Transform.ScreenToImage(e.Location);
                return;
            }

            if (_rbActive)
            {
                var imgPt = Transform.ScreenToImage(e.Location);
                float x1 = Math.Min(_rbStartImg.X, imgPt.X);
                float y1 = Math.Min(_rbStartImg.Y, imgPt.Y);
                float x2 = Math.Max(_rbStartImg.X, imgPt.X);
                float y2 = Math.Max(_rbStartImg.Y, imgPt.Y);
                _rbRectImg = new RectangleF(x1, y1, x2 - x1, y2 - y1);
                Invalidate();
                return;
            }

            if (_msActive)
            {
                ContinueMultiEdit(e);
                Invalidate();
                return;
            }

            if (Mode == ToolMode.Pointer && Selection.Multi.Count > 0)
            {
                var groupImg = GetSelectionBoundsImg();
                var groupScr = Transform.ImageRectToScreen(groupImg);
                var h = HitTestHandle(groupScr, e.Location);
                Cursor = h != HandleType.None ? CursorFromHandle(h)
                       : groupScr.Contains(e.Location) ? Cursors.SizeAll
                       : Cursors.Default;
            }

            var tool = ActiveTool;
            tool.OnMouseMove(this, e);
            if (!tool.IsEditingActive) base.OnMouseMove(e);

            if (Image != null) LastMouseImg = Transform.ScreenToImage(e.Location);
        }

        protected override void OnMouseUp(MouseEventArgs e)
        {
            if (_rotating)
            {
                _rotating = false;
                _rotBase = null;
                Capture = false;
                /* history omitted (no PushChanged) */
                Invalidate();
                return;
            }

            if (PanMode && _isPanning && e.Button == MouseButtons.Left)
            {
                _isPanning = false;
                Cursor = Cursors.Hand;
                Capture = false;
                return;
            }

            if (_rbActive)
            {
                _rbActive = false;
                Capture = false;

                var selRect = _rbRectImg;

                if (selRect.Width < MinRectSizeImg || selRect.Height < MinRectSizeImg)
                {
                    _rbRectImg = RectangleF.Empty;
                    Invalidate();
                    return;
                }

                var picked = Shapes.Where(s => s.GetBoundsImg().IntersectsWith(selRect)).ToList();
                if ((ModifierKeys & Keys.Control) == Keys.Control)
                {
                    var union = new HashSet<IShape>(Selection.AllSelected());
                    foreach (var s in picked) union.Add(s);
                    if (union.Count > 0) Selection.SetMulti(union); else Selection.Clear();
                }
                else
                {
                    if (picked.Count > 0) Selection.SetMulti(picked); else Selection.Clear();
                }

                _rbRectImg = RectangleF.Empty;
                Invalidate();
                return;
            }

            if (_msActive) { EndMultiEdit(); Invalidate(); return; }

            var tool = ActiveTool;
            tool.OnMouseUp(this, e);
            if (!tool.IsEditingActive) base.OnMouseUp(e);

            // 방금 생성된 신규 도형들 스타일 주입
            ApplyStylesToNewShapes();

            Invalidate();
        }

        protected override void OnKeyDown(KeyEventArgs e)
        {
            if (e.KeyCode == Keys.Escape)
            {
                if (_rbActive || HasRubberBand)
                {
                    _rbActive = false;
                    _rbRectImg = RectangleF.Empty;
                    Capture = false;
                    Invalidate();
                    e.Handled = e.SuppressKeyPress = true; return;
                }
            }

            if (e.Control && e.KeyCode == Keys.Z)
            {
                if (History.UndoLastCreation(Shapes))
                {
                    if (Selection.Selected != null && !Shapes.Contains(Selection.Selected))
                        Selection.Clear();
                    Invalidate();
                }
                e.Handled = e.SuppressKeyPress = true; return;
            }

            if (e.Control && e.KeyCode == Keys.C)
            {
                if (Selection.Multi.Count > 0)
                {
                    _multiCopyBuffer = Selection.Multi.ToList();
                    _multiCopyBoundsImg = GetSelectionBoundsImg();
                    Clipboard.Copy(null);
                }
                else if (Selection.Selected != null)
                {
                    Clipboard.Copy(Selection.Selected);
                    _multiCopyBuffer = null;
                    _multiCopyBoundsImg = RectangleF.Empty; // ← 수정
                }
                e.Handled = e.SuppressKeyPress = true; return;
            }

            if (e.Control && e.KeyCode == Keys.V)
            {
                if (Image != null)
                {
                    if (_multiCopyBuffer != null && _multiCopyBuffer.Count > 0 && !_multiCopyBoundsImg.IsEmpty)
                    {
                        var dx = LastMouseImg.X - _multiCopyBoundsImg.X;
                        var dy = LastMouseImg.Y - _multiCopyBoundsImg.Y;

                        var pasted = new List<IShape>();
                        foreach (var src in _multiCopyBuffer)
                        {
                            var b = src.GetBoundsImg();
                            var target = new PointF(b.X + dx, b.Y + dy);

                            var clone = Clipboard.PasteAt(src, target, Transform.ImageSize);
                            if (clone != null)
                            {
                                Shapes.Add(clone);
                                History.PushCreated(clone);
                                pasted.Add(clone);
                            }
                        }

                        if (pasted.Count > 0)
                        {
                            Selection.SetMulti(pasted);
                            Mode = ToolMode.Pointer;
                            Invalidate();
                        }
                        e.Handled = e.SuppressKeyPress = true; return;
                    }

                    if (Clipboard.CopyShape != null)
                    {
                        var pasted = Clipboard.PasteAt(Clipboard.CopyShape, LastMouseImg, Transform.ImageSize);
                        if (pasted != null)
                        {
                            Shapes.Add(pasted);
                            History.PushCreated(pasted);
                            Selection.Set(pasted);
                            Mode = ToolMode.Pointer;
                            Invalidate();
                        }
                        e.Handled = e.SuppressKeyPress = true; return;
                    }
                }
                e.Handled = e.SuppressKeyPress = true; return;
            }

            if (e.Control && e.KeyCode == Keys.A)
            {
                if (Shapes.Count > 0)
                {
                    Selection.SetMulti(Shapes);
                    _rbActive = false;
                    _rbRectImg = RectangleF.Empty;
                    Capture = false;
                    Cursor = Cursors.Default;
                    Mode = ToolMode.Pointer;
                    Invalidate();
                }
                e.Handled = e.SuppressKeyPress = true; return;
            }

            if (e.KeyCode == Keys.Delete)
            {
                if (Selection.HasAny)
                {
                    var toRemove = Selection.AllSelected().ToList();
                    for (int i = 0; i < toRemove.Count; i++)
                        Shapes.Remove(toRemove[i]);
                    Selection.Clear();
                    Invalidate();
                }
                e.Handled = e.SuppressKeyPress = true; return;
            }


            if (e.Shift && (e.KeyCode == Keys.Up || e.KeyCode == Keys.Down))
            {
                if (Selection != null && Selection.HasAny)
                {
                    const float SCALE_STEP = 1.05f;
                    float factor = (e.KeyCode == Keys.Up) ? SCALE_STEP : 1f / SCALE_STEP;
                    ScaleSelectionUniform(factor);
                    Invalidate();
                    e.Handled = e.SuppressKeyPress = true; return;
                }
                // 선택이 없을 때는 폼에서 처리하도록 넘김
            }

            if (!e.Shift && (e.KeyCode == Keys.Left || e.KeyCode == Keys.Right || e.KeyCode == Keys.Up || e.KeyCode == Keys.Down))
            {
                if (Selection != null && Selection.HasAny)
                {
                    float step = e.Control ? 20f : 5f;
                    float dx = (e.KeyCode == Keys.Left) ? -step : (e.KeyCode == Keys.Right ? step : 0f);
                    float dy = (e.KeyCode == Keys.Up) ? -step : (e.KeyCode == Keys.Down ? step : 0f);

                    NudgeSelection(dx, dy);
                    Invalidate();
                    e.Handled = e.SuppressKeyPress = true; return;
                }
                // 선택이 없을 때는 폼(MainForm)에서 Ctrl+↑/↓ 등을 처리하도록 넘김
            }


            ActiveTool.OnKeyDown(this, e);
            base.OnKeyDown(e);
        }

        private IShape HitTestShape(Point pScreen)
        {
            for (int i = Shapes.Count - 1; i >= 0; i--)
            {
                var b = Shapes[i].GetBoundsImg();
                if (b.IsEmpty) continue;
                var bs = Transform.ImageRectToScreen(b);
                if (bs.Contains(pScreen)) return Shapes[i];
            }
            return null;
        }
        private bool OverAnySelectedShape(Point pScreen)
        {
            for (int i = 0; i < Selection.Multi.Count; i++)
            {
                var bImg = Selection.Multi[i].GetBoundsImg();
                if (bImg.IsEmpty) continue;
                var bScr = Transform.ImageRectToScreen(bImg);
                if (bScr.Contains(pScreen)) return true;
            }
            return false;
        }


        private RectangleF GetSelectionUnionImg()
        {
            RectangleF r = RectangleF.Empty;
            if (Selection.Selected != null) r = Selection.Selected.GetBoundsImg();
            for (int i = 0; i < Selection.Multi.Count; i++)
            {
                var b = Selection.Multi[i].GetBoundsImg();
                if (!b.IsEmpty) r = r.IsEmpty ? b : RectangleF.Union(r, b);
            }
            return r;
        }

        private Point GetRotateHandleCenterScreen()
        {
            var g = GetSelectionUnionImg();
            if (g.IsEmpty) return Point.Empty;
            var gs = Transform.ImageRectToScreen(g);
            int cx = (int)(gs.X + gs.Width * 0.5f);
            int cy = (int)(gs.Y - ROT_HANDLE_OFFSET_SCR);
            return new Point(cx, cy);
        }

        private bool HitRotateHandle(Point pScr)
        {
            var c = GetRotateHandleCenterScreen();
            if (c == Point.Empty) return false;
            int dx = pScr.X - c.X;
            int dy = pScr.Y - c.Y;
            return (dx * dx + dy * dy) <= (ROT_HANDLE_RADIUS_SCR * ROT_HANDLE_RADIUS_SCR);
        }

        private static float Atan2(float y, float x) => (float)Math.Atan2(y, x);

        private float GetAngleRadToMouse(Point mouseScr, PointF centerImg)
        {
            var imgPt = this.PointToImage(mouseScr);
            float dx = imgPt.X - centerImg.X;
            float dy = imgPt.Y - centerImg.Y;
            return Atan2(dy, dx);
        }

        private void ApplyRotationDelta(float deltaRad)
        {
            if (_rotBase == null || _rotBase.Count == 0) return;

            var list = Selection.Multi.Count > 0 ? Selection.Multi : (Selection.Selected != null ? new List<IShape> { Selection.Selected } : new List<IShape>());
            if (list.Count == 0) return;

            for (int i = 0; i < list.Count; i++)
            {
                var baseShape = _rotBase[i];
                var live = list[i];

                if (baseShape is RectangleShape rb && live is RectangleShape rbox)
                {
                    float deg = deltaRad * 180f / (float)Math.PI;
                    int k = (int)Math.Round(deg / 90f);
                    bool odd = Math.Abs(k) % 2 == 1;
                    var b0 = rb.RectImg;
                    float cx = b0.X + b0.Width * 0.5f;
                    float cy = b0.Y + b0.Height * 0.5f;
                    if (odd)
                        rbox.RectImg = Normalize(new RectangleF(cx - b0.Height * 0.5f, cy - b0.Width * 0.5f, b0.Height, b0.Width));
                    else
                        rbox.RectImg = b0;
                    continue;
                }

                if (baseShape is PolygonShape bp && live is PolygonShape poly && bp.PointsImg != null)
                {
                    for (int k = 0; k < bp.PointsImg.Count; k++)
                    {
                        var p0 = bp.PointsImg[k];
                        float dx = p0.X - _rotCenterImg.X;
                        float dy = p0.Y - _rotCenterImg.Y;
                        float x1 = _rotCenterImg.X + (dx * (float)Math.Cos(deltaRad) - dy * (float)Math.Sin(deltaRad));
                        float y1 = _rotCenterImg.Y + (dx * (float)Math.Sin(deltaRad) + dy * (float)Math.Cos(deltaRad));
                        if (k < poly.PointsImg.Count) poly.PointsImg[k] = new PointF(x1, y1);
                    }
                    continue;
                }

                if (baseShape is TriangleShape bt && live is TriangleShape tri && bt.PointsImg != null)
                {
                    for (int k = 0; k < bt.PointsImg.Count; k++)
                    {
                        var p0 = bt.PointsImg[k];
                        float dx = p0.X - _rotCenterImg.X;
                        float dy = p0.Y - _rotCenterImg.Y;
                        float x1 = _rotCenterImg.X + (dx * (float)Math.Cos(deltaRad) - dy * (float)Math.Sin(deltaRad));
                        float y1 = _rotCenterImg.Y + (dx * (float)Math.Sin(deltaRad) + dy * (float)Math.Cos(deltaRad));
                        if (k < tri.PointsImg.Count) tri.PointsImg[k] = new PointF(x1, y1);
                    }
                    continue;
                }

                if (baseShape is BrushStrokeShape bb && live is BrushStrokeShape brush)
                {
                    for (int k = 0; k < bb.PointsImg.Count; k++)
                    {
                        var p0 = bb.PointsImg[k];
                        float dx = p0.X - _rotCenterImg.X;
                        float dy = p0.Y - _rotCenterImg.Y;
                        float x1 = _rotCenterImg.X + (dx * (float)Math.Cos(deltaRad) - dy * (float)Math.Sin(deltaRad));
                        float y1 = _rotCenterImg.Y + (dx * (float)Math.Sin(deltaRad) + dy * (float)Math.Cos(deltaRad));
                        if (k < brush.PointsImg.Count) brush.PointsImg[k] = new PointF(x1, y1);
                    }
                    // Rebuild area via transforming existing path
                    using (var path = brush.GetAreaPathImgClone())
                    {
                        if (path != null)
                        {
                            using (var mtx = new System.Drawing.Drawing2D.Matrix())
                            {
                                mtx.Translate(-_rotCenterImg.X, -_rotCenterImg.Y, MatrixOrder.Append);
                                mtx.Rotate(_rotCurrAngleRad * 180f / (float)System.Math.PI, MatrixOrder.Append);
                                mtx.Translate(_rotCenterImg.X, _rotCenterImg.Y, MatrixOrder.Append);
                                path.Transform(mtx);
                                brush.ReplaceArea(path);
                            }
                        }
                    }
                    continue;
                }
                // shapes like circle unaffected by rotation
            }
        }
        private RectangleF GetSelectionBoundsImg()
        {
            RectangleF r = RectangleF.Empty;
            if (Selection.Selected != null) r = Selection.Selected.GetBoundsImg();
            for (int i = 0; i < Selection.Multi.Count; i++)
            {
                var b = Selection.Multi[i].GetBoundsImg();
                r = r.IsEmpty ? b : RectangleF.Union(r, b);
            }
            return r;
        }

        private void ScaleSelectionUniform(float factor)
        {
            var targets = new List<IShape>();
            if (Selection.Multi.Count > 0) targets.AddRange(Selection.Multi);
            else if (Selection.Selected != null) targets.Add(Selection.Selected);
            if (targets.Count == 0) return;

            RectangleF g0 = RectangleF.Empty;
            for (int i = 0; i < targets.Count; i++)
            {
                var b = targets[i].GetBoundsImg();
                if (b.IsEmpty) continue;
                g0 = g0.IsEmpty ? b : RectangleF.Union(g0, b);
            }
            if (g0.IsEmpty) return;

            float minW = MinRectSizeImg, minH = MinRectSizeImg;
            float newW = g0.Width * factor, newH = g0.Height * factor;
            if (newW < minW) { factor = minW / (g0.Width == 0 ? 1f : g0.Width); newW = minW; }
            if (newH < minH) { factor = Math.Max(factor, minH / (g0.Height == 0 ? 1f : g0.Height)); newH = g0.Height * factor; }

            float cx = g0.X + g0.Width * .5f, cy = g0.Y + g0.Height * .5f;
            var g1 = new RectangleF(cx - newW * .5f, cy - newH * .5f, newW, newH);

            var imgSz = Transform.ImageSize;
            if (!imgSz.IsEmpty)
            {
                float dx = 0f, dy = 0f;
                if (g1.Left < 0) dx = -g1.Left;
                else if (g1.Right > imgSz.Width) dx = imgSz.Width - g1.Right;
                if (g1.Top < 0) dy = -g1.Top;
                else if (g1.Bottom > imgSz.Height) dy = imgSz.Height - g1.Bottom;
                if (dx != 0f || dy != 0f) g1 = new RectangleF(g1.X + dx, g1.Y + dy, g1.Width, g1.Height);
            }

            for (int i = 0; i < targets.Count; i++)
            {
                var s = targets[i];

                if (s is RectangleShape rbox)
                {
                    var r0 = rbox.RectImg;
                    rbox.RectImg = Normalize(new RectangleF(
                        g1.X + (r0.X - g0.X) * factor,
                        g1.Y + (r0.Y - g0.Y) * factor,
                        r0.Width * factor, r0.Height * factor));
                    continue;
                }
                if (s is CircleShape circle)
                {
                    var r0 = circle.RectImg;
                    circle.RectImg = new RectangleF(
                        g1.X + (r0.X - g0.X) * factor,
                        g1.Y + (r0.Y - g0.Y) * factor,
                        r0.Width * factor, r0.Height * factor);
                    continue;
                }
                if (s is PolygonShape poly)
                {
                    for (int k = 0; k < poly.PointsImg.Count; k++)
                    {
                        var p0 = poly.PointsImg[k];
                        poly.PointsImg[k] = new PointF(
                            g1.X + (p0.X - g0.X) * factor,
                            g1.Y + (p0.Y - g0.Y) * factor);
                    }
                    continue;
                }
                if (s is TriangleShape tri)
                {
                    for (int k = 0; k < tri.PointsImg.Count; k++)
                    {
                        var p0 = tri.PointsImg[k];
                        tri.PointsImg[k] = new PointF(
                            g1.X + (p0.X - g0.X) * factor,
                            g1.Y + (p0.Y - g0.Y) * factor);
                    }
                    continue;
                }
                if (s is BrushStrokeShape brush)
                {
                    var path = brush.GetAreaPathImgClone();
                    if (path != null)
                    {
                        using (path)
                        using (var m = new Matrix())
                        {
                            m.Translate(-g0.X, -g0.Y, MatrixOrder.Append);
                            m.Scale(factor, factor, MatrixOrder.Append);
                            m.Translate(g1.X, g1.Y, MatrixOrder.Append);
                            path.Transform(m);
                            brush.ReplaceArea(path);
                        }
                    }
                    if (brush.PointsImg != null && brush.PointsImg.Count > 0)
                    {
                        for (int k = 0; k < brush.PointsImg.Count; k++)
                        {
                            var p0 = brush.PointsImg[k];
                            brush.PointsImg[k] = new PointF(
                                g1.X + (p0.X - g0.X) * factor,
                                g1.Y + (p0.Y - g0.Y) * factor);
                        }
                    }
                    continue;
                }

                var b0 = s.GetBoundsImg();
                if (!b0.IsEmpty)
                {
                    float tx = g1.X + (b0.X - g0.X) * factor - b0.X;
                    float ty = g1.Y + (b0.Y - g0.Y) * factor - b0.Y;
                    s.MoveBy(new SizeF(tx, ty));
                }
            }
        }

        private void NudgeSelection(float dx, float dy)
        {
            var targets = new List<IShape>();
            if (Selection.Multi.Count > 0) targets.AddRange(Selection.Multi);
            else if (Selection.Selected != null) targets.Add(Selection.Selected);
            if (targets.Count == 0) return;

            RectangleF g = RectangleF.Empty;
            for (int i = 0; i < targets.Count; i++)
            {
                var b = targets[i].GetBoundsImg();
                if (b.IsEmpty) continue;
                g = g.IsEmpty ? b : RectangleF.Union(g, b);
            }
            if (g.IsEmpty) return;

            var img = Transform.ImageSize;
            if (!img.IsEmpty)
            {
                float minDx = -g.Left;
                float maxDx = img.Width - g.Right;
                float minDy = -g.Top;
                float maxDy = img.Height - g.Bottom;

                dx = Clamp(dx, minDx, maxDx);
                dy = Clamp(dy, minDy, maxDy);
            }
            if (dx == 0f && dy == 0f) return;

            var delta = new SizeF(dx, dy);
            for (int i = 0; i < targets.Count; i++) targets[i].MoveBy(delta);
        }

        private HandleType HitTestHandle(RectangleF sRect, Point p)
        {
            float hs = HandleHitSize;
            var centers = GetHandleCenters(sRect);
            for (int i = 0; i < centers.Length; i++)
            {
                var hr = new RectangleF(centers[i].X - hs / 2f, centers[i].Y - hs / 2f, hs, hs);
                if (!hr.Contains(p)) continue;
                switch (i)
                {
                    case 0: return HandleType.NW;
                    case 1: return HandleType.N;
                    case 2: return HandleType.NE;
                    case 3: return HandleType.W;
                    case 4: return HandleType.E;
                    case 5: return HandleType.SW;
                    case 6: return HandleType.S;
                    case 7: return HandleType.SE;
                    default: return HandleType.None;
                }
            }
            return HandleType.None;
        }

        private bool TryBeginMultiEdit(MouseEventArgs e)
        {
            var groupImg = GetSelectionBoundsImg();
            var groupScr = Transform.ImageRectToScreen(groupImg);
            var groupScrInfl = RectangleF.Inflate(groupScr, MoveHitInflate, MoveHitInflate);

            var h = HitTestHandle(groupScr, e.Location);
            if (h != HandleType.None && e.Button == MouseButtons.Left)
            {
                _msActive = true;
                _msHandle = h;
                _msBoundsStartImg = groupImg;
                _msStartRects = new Dictionary<IShape, RectangleF>();
                for (int i = 0; i < Selection.Multi.Count; i++)
                    if (Selection.Multi[i] is RectangleShape rs) _msStartRects[rs] = rs.RectImg;
                Capture = true;
                return true;
            }

            if (groupScrInfl.Contains(e.Location) && e.Button == MouseButtons.Left)
            {
                _msActive = true;
                _msHandle = HandleType.Move;
                _msBoundsStartImg = groupImg;
                _msDragStartImg = Transform.ScreenToImage(e.Location);
                Capture = true;
                return true;
            }
            return false;
        }

        private void ContinueMultiEdit(MouseEventArgs e)
        {
            if (!_msActive) return;

            if (_msHandle == HandleType.Move)
            {
                var currImg = Transform.ScreenToImage(e.Location);
                var d = new SizeF(currImg.X - _msDragStartImg.X, currImg.Y - _msDragStartImg.Y);
                _msDragStartImg = currImg;

                for (int i = 0; i < Selection.Multi.Count; i++) Selection.Multi[i].MoveBy(d);
                return;
            }

            var m = Transform.ScreenToImage(e.Location);
            var newGroup = BuildResizedGroup(_msBoundsStartImg, m, _msHandle);

            float minW = MinRectSizeImg, minH = MinRectSizeImg;
            if (newGroup.Width < minW) newGroup.Width = minW;
            if (newGroup.Height < minH) newGroup.Height = minH;

            float sx = _msBoundsStartImg.Width == 0 ? 1f : newGroup.Width / _msBoundsStartImg.Width;
            float sy = _msBoundsStartImg.Height == 0 ? 1f : newGroup.Height / _msBoundsStartImg.Height;

            float ox = _msBoundsStartImg.X, oy = _msBoundsStartImg.Y;
            float nx = newGroup.X, ny = newGroup.Y;

            for (int i = 0; i < Selection.Multi.Count; i++)
            {
                var s = Selection.Multi[i];
                if (s is RectangleShape rs && _msStartRects != null && _msStartRects.ContainsKey(s))
                {
                    var r0 = _msStartRects[s];
                    rs.RectImg = Normalize(new RectangleF(
                        nx + (r0.X - ox) * sx,
                        ny + (r0.Y - oy) * sy,
                        r0.Width * sx, r0.Height * sy));
                }
                else
                {
                    var b0 = s.GetBoundsImg();
                    var tl0 = new PointF(b0.X, b0.Y);
                    var tl1 = new PointF(nx + (tl0.X - ox) * sx, ny + (tl0.Y - oy) * sy);
                    s.MoveBy(new SizeF(tl1.X - tl0.X, tl1.Y - tl0.Y));
                }
            }
        }

        private void EndMultiEdit()
        {
            _msActive = false;
            _msHandle = HandleType.None;
            _msStartRects = null;
            Capture = false;
        }

        private static RectangleF BuildResizedGroup(RectangleF start, PointF curr, HandleType h)
        {
            float x1 = start.Left, y1 = start.Top, x2 = start.Right, y2 = start.Bottom;
            switch (h)
            {
                case HandleType.NW: x1 = curr.X; y1 = curr.Y; break;
                case HandleType.N: y1 = curr.Y; break;
                case HandleType.NE: x2 = curr.X; y1 = curr.Y; break;
                case HandleType.W: x1 = curr.X; break;
                case HandleType.E: x2 = curr.X; break;
                case HandleType.SW: x1 = curr.X; y2 = curr.Y; break;
                case HandleType.S: y2 = curr.Y; break;
                case HandleType.SE: x2 = curr.X; y2 = curr.Y; break;
            }
            float nx = Math.Min(x1, x2), ny = Math.Min(y1, y2);
            return new RectangleF(nx, ny, Math.Abs(x2 - x1), Math.Abs(y2 - y1));
        }

        private bool IsOverAnyShape(Point pScreen)
        {
            for (int i = 0; i < Shapes.Count; i++)
            {
                var bImg = Shapes[i].GetBoundsImg();
                if (bImg.IsEmpty) continue;
                var bScr = Transform.ImageRectToScreen(bImg);
                if (bScr.Contains(pScreen)) return true;
            }
            return false;
        }

        // 마스크 반전: (전체 - 유니온) 을 하나의 BrushStrokeShape로 치환
        public void ApplyMaskInvert()
        {
            if (Image == null) return;

            using (var union = BuildUnionAreaOfAllShapes())
            {
                if (union == null) return;

                using (var full = new GraphicsPath(FillMode.Winding))
                {
                    full.AddRectangle(new RectangleF(0, 0, Image.Width, Image.Height));

                    var inverted = PathBoolean.Difference(full, union);
                    if (inverted == null) return;

                    var mask = new BrushStrokeShape { DiameterPx = BrushDiameterPx };
                    mask.ReplaceArea(inverted);

                    ApplyActiveStyleTo(mask);
                    _styledShapes.Add(mask);

                    Shapes.Clear();
                    Selection.Set(null);
                    Shapes.Add(mask);
                    Selection.Set(mask);

                    Invalidate();
                }
            }
        }

        private GraphicsPath BuildUnionAreaOfAllShapes()
        {
            GraphicsPath acc = null;
            foreach (var s in Shapes)
            {
                using (var area = GetShapeAreaPathClone(s))
                {
                    if (area == null) continue;
                    if (acc == null) acc = (GraphicsPath)area.Clone();
                    else
                    {
                        var newAcc = PathBoolean.Union(acc, area);
                        acc.Dispose();
                        acc = newAcc;
                    }
                }
            }
            return acc;
        }

        private GraphicsPath GetShapeAreaPathClone(IShape s)
        {
            if (s is BrushStrokeShape bs) return bs.GetAreaPathImgClone();
            if (s is RectangleShape rbox) { var gp = new GraphicsPath(FillMode.Winding); gp.AddRectangle(rbox.RectImg); return gp; }
            if (s is CircleShape circle) { var gp = new GraphicsPath(FillMode.Winding); gp.AddEllipse(circle.RectImg); return gp; }
            if (s is PolygonShape poly && poly.PointsImg?.Count >= 3)
            { var gp = new GraphicsPath(FillMode.Winding); gp.AddPolygon(poly.PointsImg.ToArray()); return gp; }
            if (s is TriangleShape tri && tri.PointsImg?.Count >= 3)
            { var gp = new GraphicsPath(FillMode.Winding); gp.AddPolygon(tri.PointsImg.ToArray()); return gp; }

            var b = s.GetBoundsImg();
            if (!b.IsEmpty) { var gp = new GraphicsPath(FillMode.Winding); gp.AddRectangle(b); return gp; }
            return null;
        }



        // 새 브러시 스트로크를 기존 브러시와 유니온

        public void MergeBrushWithExisting(BrushStrokeShape incoming)
        {
            // Apply style & toggle
            ApplyActiveStyleTo(incoming);
            _styledShapes.Add(incoming);

            if (!AUTO_UNION_SAME_LABEL)
            {
                Shapes.Add(incoming);
                History.PushCreated(incoming);
                Selection.Set(incoming);
                Invalidate();
                return;
            }

            using (var incArea = incoming.GetAreaPathImgClone())
            {
                if (incArea == null)
                {
                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                    return;
                }

                // Determine label
                string inLabel = incoming.LabelName;
                if (string.IsNullOrWhiteSpace(inLabel))
                {
                    if (!TryGetShapeLabelAndStroke(incoming, out inLabel, out _))
                        inLabel = null;
                }

                // Candidate targets: any shape with same label whose bounds intersect inflated incoming bounds
                var prox = incoming.DiameterPx + 2f;
                var proxBounds = RectangleF.Inflate(incArea.GetBounds(), prox, prox);

                var targets = new List<IShape>();
                if (!string.IsNullOrWhiteSpace(inLabel))
                {
                    foreach (var s in Shapes)
                    {
                        if (ReferenceEquals(s, incoming)) continue;
                        if (!TryGetShapeLabelAndStroke(s, out var lbl, out _)) continue;
                        if (!string.Equals(lbl, inLabel, StringComparison.Ordinal)) continue;
                        var b = s.GetBoundsImg();
                        if (!b.IsEmpty && b.IntersectsWith(proxBounds)) targets.Add(s);
                    }
                }

                if (targets.Count == 0)
                {
                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                    return;
                }

                GraphicsPath merged = (GraphicsPath)incArea.Clone();
                try
                {
                    foreach (var t in targets)
                    {
                        using (var tArea = GetShapeAreaPathClone(t))
                        {
                            if (tArea == null) continue;
                            var newMerged = PathBoolean.Union(merged, tArea);
                            merged.Dispose();
                            merged = newMerged;
                        }
                    }

                    try { merged.Flatten(null, 1.8f); } catch { /* ignore */ }

                    for (int k = Shapes.Count - 1; k >= 0; k--)
                        if (targets.Contains(Shapes[k])) Shapes.RemoveAt(k);

                    incoming.ReplaceArea(merged);
                    merged = null;

                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                }
                finally
                {
                    if (merged != null) merged.Dispose();
                }
            }
        }



        /// <summary>
        /// Merge the incoming shape with existing shapes that share the same label name
        /// when their areas overlap. Result is a single area shape; disjoint results
        /// remain multiple figures inside the same GraphicsPath. Holes are preserved.
        /// Simplification: GraphicsPath.Flatten with tolerance ~1.8px.
        /// </summary>
        public void MergeSameLabelOverlaps(IShape incoming)
        {
            // Feature toggle: if disabled, just add the incoming shape without merging
            if (!AUTO_UNION_SAME_LABEL)
            {
                ApplyActiveStyleTo(incoming);
                _styledShapes.Add(incoming);
                Shapes.Add(incoming);
                History.PushCreated(incoming);
                Selection.Set(incoming);
                Invalidate();
                return;
            }

            // Apply active style once to ensure label/colors are set
            ApplyActiveStyleTo(incoming);
            _styledShapes.Add(incoming);

            // Try to build area of incoming
            using (var incArea = GetShapeAreaPathClone(incoming))
            {
                if (incArea == null)
                {
                    // No area (shouldn't happen), just add as-is
                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                    return;
                }

                string inLabel = null; Color _;
                if (!TryGetShapeLabelAndStroke(incoming, out inLabel, out _))
                    inLabel = (incoming != null) ? incoming.LabelName : null;

                // Coarse bounds for target filtering
                var proxBounds = incArea.GetBounds();
                var targets = new List<IShape>();
                if (!string.IsNullOrWhiteSpace(inLabel))
                {
                    foreach (var s in Shapes)
                    {
                        if (ReferenceEquals(s, incoming)) continue;
                        if (!TryGetShapeLabelAndStroke(s, out var lbl, out _)) continue;
                        if (!string.Equals(lbl, inLabel, StringComparison.Ordinal)) continue;

                        var b = s.GetBoundsImg();
                        if (!b.IsEmpty && b.IntersectsWith(proxBounds))
                            targets.Add(s);
                    }
                }

                if (targets.Count == 0)
                {
                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                    return;
                }

                // Build merged union path
                GraphicsPath merged = (GraphicsPath)incArea.Clone();
                try
                {
                    foreach (var t in targets)
                    {
                        using (var tArea = GetShapeAreaPathClone(t))
                        {
                            if (tArea == null) continue;
                            var newMerged = PathBoolean.Union(merged, tArea);
                            merged.Dispose();
                            merged = newMerged;
                        }
                    }

                    // Simplify (flatten) to reduce vertices a bit (epsilon ≈ 1.8px)
                    try { merged.Flatten(null, 1.8f); } catch { /* ignore */ }

                    // Remove old targets
                    for (int i = Shapes.Count - 1; i >= 0; i--)
                        if (targets.Contains(Shapes[i]))
                            Shapes.RemoveAt(i);

                    // Replace incoming with a compound area shape that can preserve holes
                    BrushStrokeShape areaShape;
                    if (incoming is BrushStrokeShape bss)
                    {
                        areaShape = bss;
                    }
                    else
                    {
                        areaShape = new BrushStrokeShape
                        {
                            LabelName = inLabel,
                            StrokeColor = incoming.StrokeColor,
                            FillColor = incoming.FillColor,
                            DiameterPx = 1f // minimal; not used when drawing area
                        };
                    }
                    areaShape.ReplaceArea(merged);
                    merged = null; // ownership transferred

                    // Ensure incoming is not separately added
                    if (!(incoming is BrushStrokeShape))
                        incoming = areaShape;

                    Shapes.Add(incoming);
                    History.PushCreated(incoming);
                    Selection.Set(incoming);
                    Invalidate();
                }
                finally
                {
                    if (merged != null) merged.Dispose();
                }
            }
        }
        #endregion

        #region 6) Mask Rendering (8bpp grayscale PNG)

        /**
         * 현재 캔버스의 모든 라벨링 영역을 흰색(255), 배경을 검정(0)으로 만든
         * 이미지 크기 동일한 8bpp 그레이 마스크를 생성해 반환.
         * 반환 Bitmap은 PixelFormat.Format8bppIndexed, 회색 팔레트(0~255).
         */
        public Bitmap RenderBinaryMask8bpp()
        {
            if (Image == null) return null;

            int W = Image.Width;
            int H = Image.Height;

            // 1) 32bpp 작업용 캔버스에 도형 채우기(흰/검)
            using (var work = new Bitmap(W, H, PixelFormat.Format32bppArgb))
            {
                using (var g = Graphics.FromImage(work))
                {
                    g.Clear(Color.Black);
                    // 가장 또렷한 이진 마스크를 위해 안티앨리어싱 끔
                    g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.None;
                    g.CompositingMode = System.Drawing.Drawing2D.CompositingMode.SourceCopy;
                    g.PixelOffsetMode = System.Drawing.Drawing2D.PixelOffsetMode.Half;

                    // 모든 도형의 "면적"을 흰색으로 채움
                    for (int i = 0; i < Shapes.Count; i++)
                    {
                        using (var gp = GetShapeAreaPathClone(Shapes[i])) // ← 도형 면적 경로 (기존 함수)
                        {
                            if (gp == null) continue;
                            g.FillPath(Brushes.White, gp);
                        }
                    }
                }

                // 2) 32bpp → 8bpp 그레이(팔레트)로 변환(128 임계치로 0/255 이진화)
                return ConvertTo8bppBinary(work);
            }
        }

        /* 32bpp ARGB → 8bpp Indexed(그레이) 변환 헬퍼 */
        private static Bitmap ConvertTo8bppBinary(Bitmap src32)
        {
            int W = src32.Width;
            int H = src32.Height;

            var dst8 = new Bitmap(W, H, PixelFormat.Format8bppIndexed);

            // 회색 팔레트(0..255) 구성
            var pal = dst8.Palette;
            for (int i = 0; i < 256; i++)
                pal.Entries[i] = Color.FromArgb(i, i, i);
            dst8.Palette = pal;

            var r = new Rectangle(0, 0, W, H);
            var bdSrc = src32.LockBits(r, ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            var bdDst = dst8.LockBits(r, ImageLockMode.WriteOnly, PixelFormat.Format8bppIndexed);

            try
            {
                int srcStride = bdSrc.Stride;
                int dstStride = bdDst.Stride;
                var srcBytes = new byte[srcStride * H];
                var dstBytes = new byte[dstStride * H];

                Marshal.Copy(bdSrc.Scan0, srcBytes, 0, srcBytes.Length);

                for (int y = 0; y < H; y++)
                {
                    int sOff = y * srcStride;
                    int dOff = y * dstStride;

                    for (int x = 0; x < W; x++)
                    {
                        int si = sOff + x * 4; // B,G,R,A
                        int blue = srcBytes[si + 0];
                        int green = srcBytes[si + 1];
                        int red = srcBytes[si + 2];

                        // 표준 휘도 근사 후 임계치(128)로 0/255
                        int lum = (red * 299 + green * 587 + blue * 114 + 500) / 1000;
                        dstBytes[dOff + x] = (byte)(lum >= 128 ? 255 : 0);
                    }
                }

                Marshal.Copy(dstBytes, 0, bdDst.Scan0, dstBytes.Length);
            }
            finally
            {
                src32.UnlockBits(bdSrc);
                dst8.UnlockBits(bdDst);
            }

            return dst8;
        }

        #endregion
    }
}