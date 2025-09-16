using System;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace SmartLabelingApp
{
    // 이미지 좌표의 정사각형 바운딩(RectImg)을 외접원으로 사용하는 원형 도형
    // 편집 시: 둘레에 N개 버텍스를 노출하여 세그멘테이션 편집 UX 제공
    public sealed class CircleShape : IShape
    {
        public Color StrokeColor { get; set; } = Color.DeepSkyBlue;
        public Color FillColor { get; set; } = Color.FromArgb(72, Color.DeepSkyBlue);
        public string LabelName { get; set; }

        public RectangleF RectImg;

        // 원 버텍스 개수(기본 EditorUIConfig.CircleSegVertexCount 사용)
        // CircleTool에서 생성 시 필요하면 개별 갯수로 덮어써도 됨.
        public int VertexCount
        {
            get => (_vertexCount > 0) ? _vertexCount : Math.Max(8, EditorUIConfig.CircleSegVertexCount);
            set => _vertexCount = Math.Max(3, value);
        }
        private int _vertexCount = 0;

        public CircleShape(RectangleF rectImg)
        {
            RectImg = ToSquare(GeometryUtil.Normalize(rectImg));
        }

        public IShape Clone()
        {
            var c = new CircleShape(RectImg) { _vertexCount = _vertexCount };
            return c;
        }

        public RectangleF GetBoundsImg() => RectImg;

        public void MoveBy(SizeF deltaImg)
        {
            RectImg = new RectangleF(
                RectImg.X + deltaImg.Width,
                RectImg.Y + deltaImg.Height,
                RectImg.Width,
                RectImg.Height);
        }

        // 원 내부 히트 (바디 이동)
        public bool HitTestInterior(Point screenPt, IViewTransform t)
        {
            var s = t.ImageRectToScreen(RectImg);
            if (s.Width <= 0 || s.Height <= 0) return false;

            float cx = s.Left + s.Width / 2f;
            float cy = s.Top + s.Height / 2f;
            float rx = s.Width / 2f;
            float ry = s.Height / 2f;

            float dx = screenPt.X - cx;
            float dy = screenPt.Y - cy;

            // 타원 방정식
            float v = (dx * dx) / (rx * rx) + (dy * dy) / (ry * ry);
            return v <= 1.0f;
        }

        // 편집 핸들 히트: "둘레 버텍스"만 지원 (세그 편집 UX)
        public bool HitTestHandle(Point screenPt, IViewTransform t, out HandleType handle, out int vertexIndex)
        {
            handle = HandleType.None;
            vertexIndex = -1;

            int n = VertexCount;
            if (n < 3) return false;

            // 화면 좌표로 버텍스 후보 계산
            var sRect = t.ImageRectToScreen(RectImg);
            float cx = sRect.Left + sRect.Width * 0.5f;
            float cy = sRect.Top + sRect.Height * 0.5f;
            float r = Math.Min(sRect.Width, sRect.Height) * 0.5f;

            float hitR = EditorUIConfig.VertexHitRadiusPx;
            float hitR2 = hitR * hitR;

            // CCW로 0~2π 분배 (각도 기준은 0 rad = +X 방향)
            for (int i = 0; i < n; i++)
            {
                double th = 2.0 * Math.PI * i / n;
                float vx = cx + (float)(r * Math.Cos(th));
                float vy = cy + (float)(r * Math.Sin(th));

                float dx = vx - screenPt.X;
                float dy = vy - screenPt.Y;
                if (dx * dx + dy * dy <= hitR2)
                {
                    handle = HandleType.Vertex;
                    vertexIndex = i;
                    return true;
                }
            }

            return false;
        }

        // 핸들 드래그 리사이즈: Vertex 드래그 → 중심 고정, 반지름만 변경 (정원 유지)
        public void ResizeByHandle(HandleType h, PointF currImg, SizeF imageSize)
        {
            if (h != HandleType.Vertex) return;

            float cx = RectImg.Left + RectImg.Width * 0.5f;
            float cy = RectImg.Top + RectImg.Height * 0.5f;

            // 새 반지름 = 중심에서 드래그 위치까지의 거리
            float dx = currImg.X - cx;
            float dy = currImg.Y - cy;
            float newR = (float)Math.Sqrt(dx * dx + dy * dy);

            // 최소 크기 보정
            float minR = GeometryUtil.MinRectSizeImg * 0.5f;
            if (newR < minR) newR = minR;

            // 이미지 경계 보정 (필요시 조금 여유)
            float maxR = Math.Min(
                Math.Min(cx, imageSize.Width - cx),
                Math.Min(cy, imageSize.Height - cy)
            );
            if (maxR > 0) newR = Math.Min(newR, maxR);

            var side = newR * 2f;
            RectImg = new RectangleF(cx - newR, cy - newR, side, side);
        }

        public void Draw(Graphics g, IViewTransform t)
        {
            var s = t.ImageRectToScreen(RectImg);
            var fillCol = FillColor.A > 0 ? FillColor
                                          : Color.FromArgb(72, Color.FromArgb(StrokeColor.R, StrokeColor.G, StrokeColor.B));

            using (var pen = new Pen(StrokeColor, 2f))
            using (var fill = new SolidBrush(fillCol))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.FillEllipse(fill, s);
                g.DrawEllipse(pen, s);
            }

            ShapeAreaExtensions.DrawLabelBadge(g, t, RectImg, LabelName);
        }

        // 편집 오버레이: 점선 타원 + 둘레 버텍스 핸들
        public void DrawOverlay(Graphics g, IViewTransform t, int selectedVertexIndex)
        {
            var s = t.ImageRectToScreen(RectImg);

            using (var pen = new Pen(Color.Orange, 1f))
            {
                pen.DashStyle = DashStyle.Dash;
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.DrawEllipse(pen, s);
            }

            int n = VertexCount;
            if (n < 3) return;

            float cx = s.Left + s.Width * 0.5f;
            float cy = s.Top + s.Height * 0.5f;
            float r = Math.Min(s.Width, s.Height) * 0.5f;

            float hs = EditorUIConfig.HandleDrawSizePx;
            float half = hs * 0.5f;

            using (var b = new SolidBrush(Color.Orange))
            using (var sel = new SolidBrush(Color.Orange))
            using (var p = new Pen(Color.DarkOrange, 1f))
            {
                for (int i = 0; i < n; i++)
                {
                    double th = 2.0 * Math.PI * i / n;
                    float vx = cx + (float)(r * Math.Cos(th));
                    float vy = cy + (float)(r * Math.Sin(th));

                    var hr = new RectangleF(vx - half, vy - half, hs, hs);
                    g.FillRectangle(i == selectedVertexIndex ? sel : b, hr);
                    g.DrawRectangle(p, hr.X, hr.Y, hr.Width, hr.Height);
                }
            }
        }

        // ---- helpers ----
        private static RectangleF ToSquare(RectangleF r)
        {
            float side = r.Width > r.Height ? r.Width : r.Height;
            return new RectangleF(r.X, r.Y, side, side);
        }
    }
}
