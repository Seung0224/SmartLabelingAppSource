using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;

namespace SmartLabelingApp
{
    public sealed class TriangleShape : IShape
    {
        public Color StrokeColor { get; set; } = Color.DeepSkyBlue;
        public Color FillColor { get; set; } = Color.FromArgb(72, Color.DeepSkyBlue);
        public string LabelName { get; set; }
        public List<PointF> PointsImg { get; private set; } // 꼭짓점 3개 (이미지 좌표)

        public TriangleShape(RectangleF boundsImg)
        {
            // 드래그한 바운딩 within에서 위쪽 정삼각형(등변) 구성
            float x1 = boundsImg.Left, x2 = boundsImg.Right;
            float y1 = boundsImg.Top, y2 = boundsImg.Bottom;
            float cx = x1 + boundsImg.Width / 2f;

            PointsImg = new List<PointF>(3)
            {
                new PointF(cx, y1),       // top
                new PointF(x1, y2),       // left-bottom
                new PointF(x2, y2)        // right-bottom
            };
        }

        private TriangleShape(List<PointF> pts)
        {
            PointsImg = new List<PointF>(pts);
        }

        // --- IShape ---

        public RectangleF GetBoundsImg()
        {
            float minX = float.MaxValue, minY = float.MaxValue;
            float maxX = float.MinValue, maxY = float.MinValue;
            for (int i = 0; i < PointsImg.Count; i++)
            {
                PointF p = PointsImg[i];
                if (p.X < minX) minX = p.X; if (p.Y < minY) minY = p.Y;
                if (p.X > maxX) maxX = p.X; if (p.Y > maxY) maxY = p.Y;
            }
            return new RectangleF(minX, minY, maxX - minX, maxY - minY);
        }

        public void MoveBy(SizeF deltaImg)
        {
            for (int i = 0; i < PointsImg.Count; i++)
                PointsImg[i] = new PointF(PointsImg[i].X + deltaImg.Width, PointsImg[i].Y + deltaImg.Height);
        }

        public IShape Clone() => new TriangleShape(PointsImg);

        public void Draw(Graphics g, IViewTransform t)
        {
            if (PointsImg == null || PointsImg.Count < 3) return;
            var ptsScr = PointsImg.Select(p => t.ImageToScreen(p)).ToArray();
            var fillCol = FillColor.A > 0 ? FillColor
                                          : Color.FromArgb(72, Color.FromArgb(StrokeColor.R, StrokeColor.G, StrokeColor.B));

            using (var pen = new Pen(StrokeColor, 2f))
            using (var fill = new SolidBrush(fillCol))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.FillPolygon(fill, ptsScr);
                g.DrawPolygon(pen, ptsScr);
            }

            ShapeAreaExtensions.DrawLabelBadge(g, t, GetBoundsImg(), LabelName);
        }

        public void DrawOverlay(Graphics g, IViewTransform tr, int selectedVertexIndex)
        {
            // 화면 좌표로 변환
            var pts = new PointF[3];
            for (int i = 0; i < 3; i++)
                pts[i] = tr.ImageToScreen(PointsImg[i]);

            // 점선 외곽
            if (pts.Length >= 2)
            {
                using (var dash = new Pen(Color.Orange, 2f) { DashStyle = DashStyle.Dash })
                {
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.DrawPolygon(dash, pts);
                }
            }

            // 꼭짓점 핸들 (EditorUIConfig 기반)
            float handleSize = EditorUIConfig.HandleDrawSizePx;
            float half = handleSize / 2f;

            using (var brush = new SolidBrush(Color.Orange))
            using (var sel = new SolidBrush(Color.Orange))   // 선택/비선택 동일 색 (폴리곤과 통일)
            using (var pen = new Pen(Color.DarkOrange, 1f))
            {
                for (int i = 0; i < pts.Length; i++)
                {
                    var r = new RectangleF(pts[i].X - half, pts[i].Y - half, handleSize, handleSize);
                    g.FillRectangle(i == selectedVertexIndex ? sel : brush, r);
                    g.DrawRectangle(pen, r.X, r.Y, r.Width, r.Height);
                }
            }
        }

        public bool HitTestHandle(Point mouseScreen, IViewTransform tr, out HandleType handle, out int vertexIndex)
        {
            handle = HandleType.None;
            vertexIndex = -1;

            // 정점 히트 (EditorUIConfig 기반 반경)
            float r = EditorUIConfig.VertexHitRadiusPx; // px
            float r2 = r * r;

            for (int i = 0; i < 3; i++)
            {
                RectangleF sr = tr.ImageRectToScreen(new RectangleF(PointsImg[i].X - 0.5f, PointsImg[i].Y - 0.5f, 1f, 1f));
                PointF sp = new PointF(sr.X + sr.Width / 2f, sr.Y + sr.Height / 2f);
                float dx = sp.X - mouseScreen.X;
                float dy = sp.Y - mouseScreen.Y;
                if (dx * dx + dy * dy <= r2)
                {
                    handle = HandleType.Vertex;
                    vertexIndex = i;
                    return true;
                }
            }
            return false;
        }

        public bool HitTestInterior(Point mouseScreen, IViewTransform tr)
        {
            // 바리센트릭으로 삼각형 내부 판정
            PointF[] sp = new PointF[3];
            for (int i = 0; i < 3; i++)
            {
                RectangleF r = tr.ImageRectToScreen(new RectangleF(PointsImg[i].X - 0.5f, PointsImg[i].Y - 0.5f, 1f, 1f));
                sp[i] = new PointF(r.X + r.Width / 2f, r.Y + r.Height / 2f);
            }

            return IsPointInTriangle(mouseScreen, sp[0], sp[1], sp[2]);
        }

        public void ResizeByHandle(HandleType handle, PointF imgPoint, SizeF imageSize)
        {
            // Vertex만 지원(폴리곤과 동일)
            if (handle != HandleType.Vertex) return;

            // 가장 가까운 정점을 그 위치로 이동 (보조용)
            int idx = 0;
            float best = float.MaxValue;
            for (int i = 0; i < 3; i++)
            {
                float dx = PointsImg[i].X - imgPoint.X;
                float dy = PointsImg[i].Y - imgPoint.Y;
                float d2 = dx * dx + dy * dy;
                if (d2 < best) { best = d2; idx = i; }
            }
            float nx = Clamp(imgPoint.X, 0, imageSize.Width);
            float ny = Clamp(imgPoint.Y, 0, imageSize.Height);
            PointsImg[idx] = new PointF(nx, ny);
        }

        // --- helpers ---

        private static bool IsPointInTriangle(Point p, PointF a, PointF b, PointF c)
        {
            // 동일 방향성 검사
            bool b1 = Sign(p, a, b) < 0.0f;
            bool b2 = Sign(p, b, c) < 0.0f;
            bool b3 = Sign(p, c, a) < 0.0f;
            return (b1 == b2) && (b2 == b3);
        }

        private static float Sign(Point p1, PointF p2, PointF p3)
        {
            return (p1.X - p3.X) * (p2.Y - p3.Y) - (p2.X - p3.X) * (p1.Y - p3.Y);
        }

        private static float Clamp(float v, float min, float max)
        {
            if (v < min) return min;
            if (v > max) return max;
            return v;
        }
    }
}
