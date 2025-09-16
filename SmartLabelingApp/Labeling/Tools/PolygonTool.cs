// PolygonTool.cs (통합본, C# 7.3)
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    public sealed class PolygonTool : ITool
    {
        // ===== 상태 =====
        public bool IsEditingActive
        {
            get
            {
                // Free 모드에서 점 추가 중이거나, RectBox/Triangle 드래그 중이면 편집 활성
                return (_current != null && _current.Count > 0) || _isDragging;
            }
        }

        // Free 모드용: 진행 중 점들 (이미지 좌표)
        private List<PointF> _current;               // 이미지 좌표
        private const float CloseRadiusScr = 8f;     // 첫 점 클릭으로 닫기 반경(픽셀)

        // 프리셋 (외부에서 MainForm/ImageCanvas 통해 설정)
        public PolygonPreset Preset = PolygonPreset.Free; // Free / RectBox / Triangle / RegularN
        public int RegularSides = 5;                       // RegularN일 때 변의 개수

        // RectBox / Triangle용 드래그 상태
        private bool _isDragging;
        private PointF _dragStartImg;
        private RectangleF _dragRectImg; // 이미지 좌표

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left) return;

            IViewTransform tr = c.Transform;
            PointF imgPt = tr.ScreenToImage(e.Location);
            SizeF imgSz = tr.ImageSize;

            if (imgPt.X < 0 || imgPt.Y < 0 || imgPt.X >= imgSz.Width || imgPt.Y >= imgSz.Height)
                return;

            if (!c.Focused) c.Focus(); // 포커스 확보

            // ===== RegularN: 클릭 즉시 정N각형 생성 (연속 생성) =====
            if (Preset == PolygonPreset.RegularN)
            {
                int sides = (RegularSides >= 3) ? RegularSides : 5;
                CreateRegularNgon(c, imgPt, sides);
                return;
            }

            // ===== RectBox / Triangle: 드래그 박스 지정 후 MouseUp에서 확정 =====
            if (Preset == PolygonPreset.RectBox || Preset == PolygonPreset.Triangle)
            {
                _isDragging = true;
                _dragStartImg = imgPt;
                _dragRectImg = new RectangleF(imgPt, SizeF.Empty);
                c.Capture = true;
                c.Cursor = Cursors.Cross;
                c.Invalidate();
                return;
            }

            // ===== Free(기존 폴리곤 작성) =====
            if (_current == null)
                _current = new List<PointF>();

            // 첫 점을 다시 클릭하면 닫기
            if (_current.Count >= 3)
            {
                PointF first = _current[0];
                PointF firstScr = tr.ImageToScreen(first);
                float dx = e.Location.X - firstScr.X;
                float dy = e.Location.Y - firstScr.Y;
                if (dx * dx + dy * dy <= CloseRadiusScr * CloseRadiusScr)
                {
                    PolygonShape polyClosed = new PolygonShape(_current);
                    c.MergeSameLabelOverlaps(polyClosed); // 생성 직후 복사 버퍼 채우기
                    _current = null;              // 연속 그리기 가능(다음 클릭부터 새 시작)
                    c.Invalidate();
                    return;
                }
            }

            _current.Add(imgPt);
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image != null)
            {
                PointF img = c.Transform.ScreenToImage(e.Location);
                if (img.X >= 0 && img.Y >= 0 && img.X < c.Transform.ImageSize.Width && img.Y < c.Transform.ImageSize.Height)
                    c.LastMouseImg = img; // 붙여넣기 기준
            }

            if (_isDragging)
            {
                // RectBox / Triangle 드래그 박스 갱신
                IViewTransform tr = c.Transform;
                PointF imgPt = tr.ScreenToImage(e.Location);
                _dragRectImg = GeometryUtil.Normalize(RectangleF.FromLTRB(
                    _dragStartImg.X, _dragStartImg.Y, imgPt.X, imgPt.Y));
                c.Invalidate();
            }
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging || e.Button != MouseButtons.Left) return;

            _isDragging = false;
            c.Capture = false;
            c.Cursor = Cursors.Default;

            // 너무 작은 박스는 취소
            if (_dragRectImg.Width >= GeometryUtil.MinRectSizeImg &&
                _dragRectImg.Height >= GeometryUtil.MinRectSizeImg)
            {
                if (Preset == PolygonPreset.RectBox)
                {
                    // 실제 Shape는 RectangleShape로 생성 → 엣지 리사이즈 UX 유지
                    RectangleShape rectShape = new RectangleShape(_dragRectImg);
                    c.MergeSameLabelOverlaps(rectShape);
                }
                else if (Preset == PolygonPreset.Triangle)
                {
                    // 드래그 박스 안에 정삼각형(위 꼭짓점이 위를 향함)
                    List<PointF> triPts = BuildEquilateralTriangle(_dragRectImg);
                    PolygonShape tri = new PolygonShape(triPts);
                    c.MergeSameLabelOverlaps(tri);
                }
            }

            _dragRectImg = RectangleF.Empty;
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            // ESC: 드래그/작성 취소
            if (e.KeyCode == Keys.Escape)
            {
                if (_isDragging)
                {
                    _isDragging = false;
                    c.Capture = false;
                    c.Cursor = Cursors.Default;
                    _dragRectImg = RectangleF.Empty;
                    c.Invalidate();
                    e.Handled = e.SuppressKeyPress = true;
                    return;
                }
                if (_current != null && _current.Count > 0)
                {
                    _current = null;
                    c.Invalidate();
                    e.Handled = e.SuppressKeyPress = true;
                    return;
                }
            }
        }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            // RectBox / Triangle 드래그 프리뷰
            if (_isDragging && (Preset == PolygonPreset.RectBox || Preset == PolygonPreset.Triangle))
            {
                using (var pen = new Pen(Color.Orange, 2f) { DashStyle = System.Drawing.Drawing2D.DashStyle.Dash })
                using (var fill = new SolidBrush(Color.FromArgb(24, Color.Orange)))
                {
                    if (Preset == PolygonPreset.RectBox)
                    {
                        // ✔ RectBox일 때만 사각형 프리뷰
                        var sr = c.Transform.ImageRectToScreen(_dragRectImg);
                        g.FillRectangle(fill, sr);
                        g.DrawRectangle(pen, sr.X, sr.Y, sr.Width, sr.Height);
                    }
                    else // Preset == Triangle
                    {
                        // ✔ Triangle일 땐 사각형 프리뷰를 그리지 않고, 삼각형만 프리뷰
                        var triImg = BuildEquilateralTriangle(_dragRectImg);
                        var triScr = triImg.Select(p => c.Transform.ImageToScreen(p)).ToArray();

                        // 필요 없으면 FillPolygon은 지우고 DrawPolygon만 남겨도 됨
                        g.FillPolygon(fill, triScr);
                        g.DrawPolygon(pen, triScr);
                    }
                }
                return;
            }

            // Free 모드 진행 중 프리뷰(기존 그대로)
            if (_current == null || _current.Count == 0) return;

            var pts = _current.Select(c.Transform.ImageToScreen).ToArray();
            using (var pen = new Pen(Color.Orange, 2f) { DashStyle = System.Drawing.Drawing2D.DashStyle.Dash })
            {
                if (pts.Length >= 2)
                    g.DrawLines(pen, pts);
            }
            // 첫 점/나머지 점 강조
            for (int i = 0; i < pts.Length; i++)
            {
                float r = (i == 0) ? 5f : 3f;
                using (var b = new SolidBrush(i == 0 ? Color.Red : Color.Orange))
                    g.FillEllipse(b, pts[i].X - r, pts[i].Y - r, r * 2f, r * 2f);
            }
        }


        // ===== Helper: 정삼각형(드래그 박스 내부) =====
        private static List<PointF> BuildEquilateralTriangle(RectangleF r)
        {
            float cx = r.X + r.Width / 2f;
            PointF top = new PointF(cx, r.Y);
            PointF left = new PointF(r.X, r.Bottom);
            PointF right = new PointF(r.Right, r.Bottom);
            return new List<PointF> { top, right, left };
        }

        // ===== Helper: 정N각형(마우스 클릭 중심) 생성 =====
        private void CreateRegularNgon(ImageCanvas c, PointF centerImg, int sides)
        {
            SizeF imgSz = c.Transform.ImageSize;

            // 반지름 결정(이미지 크기 대비 기본값; 경계 내로 클램프)
            float defaultR = (float)(System.Math.Min(imgSz.Width, imgSz.Height) * 0.12);
            float maxR =
                System.Math.Min(
                    System.Math.Min(centerImg.X, imgSz.Width - centerImg.X),
                    System.Math.Min(centerImg.Y, imgSz.Height - centerImg.Y)
                ) - 1f;
            float minR = SmartLabelingApp.GeometryUtil.MinRectSizeImg * 0.5f;
            float r = System.Math.Max(minR, System.Math.Min(defaultR, System.Math.Max(1f, maxR)));

            // 위쪽을 향하도록 -90° 오프셋
            List<PointF> pts = new List<PointF>(sides);
            double offset = -System.Math.PI / 2.0;
            for (int i = 0; i < sides; i++)
            {
                double ang = offset + (2.0 * System.Math.PI * i / sides);
                pts.Add(new PointF(
                    centerImg.X + (float)(r * System.Math.Cos(ang)),
                    centerImg.Y + (float)(r * System.Math.Sin(ang))
                ));
            }

            PolygonShape shape = new PolygonShape(pts);
            c.MergeSameLabelOverlaps(shape);
            c.Invalidate();
        }
    }
}
