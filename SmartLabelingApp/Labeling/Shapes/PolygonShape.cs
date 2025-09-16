using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class PolygonShape : IShape
    {
        public Color StrokeColor { get; set; } = Color.DeepSkyBlue;
        public Color FillColor { get; set; } = Color.FromArgb(72, Color.DeepSkyBlue);
        public string LabelName { get; set; }

        public List<PointF> PointsImg = new List<PointF>(); // 이미지 좌표

        public PolygonShape(IEnumerable<PointF> pts) => PointsImg = pts.ToList();

        public RectangleF GetBoundsImg()
        {
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            foreach (var p in PointsImg)
            {
                if (p.X < minX) minX = p.X;
                if (p.Y < minY) minY = p.Y;
                if (p.X > maxX) maxX = p.X;
                if (p.Y > maxY) maxY = p.Y;
            }
            return new RectangleF(minX, minY, maxX - minX, maxY - minY);
        }

        public void MoveBy(SizeF d)
        {
            for (int i = 0; i < PointsImg.Count; i++)
                PointsImg[i] = new PointF(PointsImg[i].X + d.Width, PointsImg[i].Y + d.Height);
        }

        public IShape Clone() => new PolygonShape(PointsImg);

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
            var pts = PointsImg.Select(tr.ImageToScreen).ToArray();

            if (pts.Length >= 2)
            {
                using (var dash = new Pen(Color.Orange, 2f) { DashStyle = DashStyle.Dash })
                {
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.DrawPolygon(dash, pts);
                }
            }

            // 핸들 크기: EditorUIConfig로 통일
            float hs = EditorUIConfig.HandleDrawSizePx;
            float half = hs / 2f;
            using (var b = new SolidBrush(Color.Orange))
            using (var sel = new SolidBrush(Color.Orange))
            using (var p = new Pen(Color.DarkOrange, 1f))
            {
                for (int i = 0; i < pts.Length; i++)
                {
                    var r = new RectangleF(pts[i].X - half, pts[i].Y - half, hs, hs);
                    g.FillRectangle(i == selectedVertexIndex ? sel : b, r);
                    g.DrawRectangle(p, r.X, r.Y, r.Width, r.Height);
                }
            }
        }

        public bool HitTestHandle(Point screenPt, IViewTransform t, out HandleType handle, out int vertexIndex)
        {
            handle = HandleType.None;
            vertexIndex = -1;

            // 버텍스 히트 반경: EditorUIConfig 기반
            float r = EditorUIConfig.VertexHitRadiusPx; // px
            float r2 = r * r;

            for (int i = PointsImg.Count - 1; i >= 0; i--)
            {
                PointF p = PointsImg[i];

                // 이미지 점 → 화면 좌표(1x1 박스 중심)
                RectangleF sr = t.ImageRectToScreen(new RectangleF(p.X - 0.5f, p.Y - 0.5f, 1f, 1f));
                PointF sp = new PointF(sr.X + sr.Width / 2f, sr.Y + sr.Height / 2f);

                float dx = sp.X - screenPt.X;
                float dy = sp.Y - screenPt.Y;
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
            if (PointsImg.Count < 3) return false;
            var sPts = PointsImg.Select(tr.ImageToScreen).ToArray();
            return PointInPolygonScreen(mouseScreen, sPts);
        }

        public void ResizeByHandle(HandleType handle, PointF imgPoint, SizeF imageSize)
        {
            if (handle != HandleType.Vertex) return;
            // 어느 버텍스가 선택되었는지는 상위(PointerTool)에서 처리
            // 여기서는 no-op (실제 버텍스 세팅은 PointerTool에서 수행)
        }

        // ====================== 신규: Add Vertex / Contains (이미지 좌표) ======================

        /// <summary>
        /// 이미지 좌표 pImg에서 가장 가까운 "변(PointsImg[i]→PointsImg[i+1])"을 찾아
        /// 그 사이에 정점을 삽입한다. 성공 시 새로 삽입된 정점의 인덱스를 반환, 실패 시 -1.
        /// </summary>
        public int InsertVertexAtClosestEdge(PointF pImg)
        {
            if (PointsImg == null || PointsImg.Count < 2) return -1;

            int bestIdx = -1;      // 삽입할 기준 i (i 다음에 삽입)
            float bestDist2 = float.MaxValue;
            PointF bestProj = pImg;

            int n = PointsImg.Count;
            for (int i = 0; i < n; i++)
            {
                var a = PointsImg[i];
                var b = PointsImg[(i + 1) % n];

                // p를 선분 ab에 투영
                float vx = b.X - a.X, vy = b.Y - a.Y;
                float wx = pImg.X - a.X, wy = pImg.Y - a.Y;
                float vv = vx * vx + vy * vy;
                float t = vv > 1e-8f ? (vx * wx + vy * wy) / vv : 0f;
                if (t < 0f) t = 0f; else if (t > 1f) t = 1f;

                var proj = new PointF(a.X + t * vx, a.Y + t * vy);
                float dx = proj.X - pImg.X, dy = proj.Y - pImg.Y;
                float d2 = dx * dx + dy * dy;

                if (d2 < bestDist2)
                {
                    bestDist2 = d2;
                    bestIdx = i;
                    bestProj = proj;
                }
            }

            if (bestIdx >= 0)
            {
                PointsImg.Insert(bestIdx + 1, bestProj);
                return bestIdx + 1;
            }
            return -1;
        }

        /// <summary>
        /// 이미지 좌표에서 점이 폴리곤 내부인지 여부 (홀짝 규칙, 자체교차/홀 없음 가정).
        /// </summary>
        public bool Contains(PointF pImg)
        {
            var pts = PointsImg;
            int n = pts?.Count ?? 0;
            if (n < 3) return false;

            bool inside = false;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                var pi = pts[i];
                var pj = pts[j];
                bool intersect = ((pi.Y > pImg.Y) != (pj.Y > pImg.Y)) &&
                                 (pImg.X < (pj.X - pi.X) * (pImg.Y - pi.Y) / ((pj.Y - pi.Y) == 0 ? 1e-6f : (pj.Y - pi.Y)) + pi.X);
                if (intersect) inside = !inside;
            }
            return inside;
        }
    }
}
