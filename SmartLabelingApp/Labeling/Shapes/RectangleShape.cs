using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class RectangleShape : IShape
    {
        public Color StrokeColor { get; set; } = Color.DeepSkyBlue;
        public Color FillColor { get; set; } = Color.FromArgb(72, Color.DeepSkyBlue);
        public string LabelName { get; set; }

        public RectangleF RectImg; // 이미지 좌표

        public RectangleShape(RectangleF r) => RectImg = Normalize(r);
        public RectangleF GetBoundsImg() => RectImg;

        public void MoveBy(SizeF d)
        {
            RectImg = new RectangleF(RectImg.X + d.Width, RectImg.Y + d.Height, RectImg.Width, RectImg.Height);
        }

        public IShape Clone() => new RectangleShape(RectImg);

        public void Draw(Graphics g, IViewTransform t)
        {
            var s = t.ImageRectToScreen(RectImg);
            var fillCol = FillColor.A > 0 ? FillColor
                                          : Color.FromArgb(72, Color.FromArgb(StrokeColor.R, StrokeColor.G, StrokeColor.B));

            using (var pen = new Pen(StrokeColor, 2f))
            using (var fill = new SolidBrush(fillCol))
            {
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.FillRectangle(fill, s);
                g.DrawRectangle(pen, s.X, s.Y, s.Width, s.Height);
            }

            ShapeAreaExtensions.DrawLabelBadge(g, t, RectImg, LabelName);
        }

        public void DrawOverlay(Graphics g, IViewTransform tr, int _)
        {
            var sSel = tr.ImageRectToScreen(RectImg);
            using (var dash = new Pen(Color.Orange, 2f) { DashStyle = DashStyle.Dash })
                g.DrawRectangle(dash, sSel.X, sSel.Y, sSel.Width, sSel.Height);

            // 핸들 8개 (EditorUIConfig 기반)
            var centers = GetHandleCenters(sSel);
            float hs = EditorUIConfig.HandleDrawSizePx;
            float half = hs / 2f;
            using (var b = new SolidBrush(Color.Orange))
            using (var p = new Pen(Color.DarkOrange, 1f))
            {
                foreach (var c in centers)
                {
                    var r = new RectangleF(c.X - half, c.Y - half, hs, hs);
                    g.FillRectangle(b, r);
                    g.DrawRectangle(p, r.X, r.Y, r.Width, r.Height);
                }
            }
        }

        public bool HitTestHandle(Point screenPt, IViewTransform t, out HandleType handle, out int vertexIndex)
        {
            handle = HandleType.None;
            vertexIndex = -1;

            RectangleF sRect = t.ImageRectToScreen(RectImg);

            float cornerHit = EditorUIConfig.CornerHitPx;
            float edgeBand = EditorUIConfig.EdgeBandPx;

            float x1 = sRect.Left, y1 = sRect.Top;
            float x2 = sRect.Right, y2 = sRect.Bottom;
            float cx = x1 + sRect.Width / 2f;
            float cy = y1 + sRect.Height / 2f;

            // 1) 코너(우선)
            PointF[] corners = new[]
            {
                new PointF(x1, y1), new PointF(cx, y1), new PointF(x2, y1),
                new PointF(x1, cy),                      new PointF(x2, cy),
                new PointF(x1, y2), new PointF(cx, y2), new PointF(x2, y2)
            };
            for (int i = 0; i < corners.Length; i++)
            {
                RectangleF hr = new RectangleF(
                    corners[i].X - cornerHit / 2f,
                    corners[i].Y - cornerHit / 2f,
                    cornerHit, cornerHit);

                if (hr.Contains(screenPt))
                {
                    switch (i)
                    {
                        case 0: handle = HandleType.NW; break;
                        case 1: handle = HandleType.N; break;
                        case 2: handle = HandleType.NE; break;
                        case 3: handle = HandleType.W; break;
                        case 4: handle = HandleType.E; break;
                        case 5: handle = HandleType.SW; break;
                        case 6: handle = HandleType.S; break;
                        case 7: handle = HandleType.SE; break;
                    }
                    return true;
                }
            }

            // 2) 모서리 밴드(코너 제외)
            RectangleF topBand = new RectangleF(x1 + cornerHit * 0.5f, y1 - edgeBand / 2f, sRect.Width - cornerHit, edgeBand);
            RectangleF bottomBand = new RectangleF(x1 + cornerHit * 0.5f, y2 - edgeBand / 2f, sRect.Width - cornerHit, edgeBand);
            RectangleF leftBand = new RectangleF(x1 - edgeBand / 2f, y1 + cornerHit * 0.5f, edgeBand, sRect.Height - cornerHit);
            RectangleF rightBand = new RectangleF(x2 - edgeBand / 2f, y1 + cornerHit * 0.5f, edgeBand, sRect.Height - cornerHit);

            if (topBand.Contains(screenPt)) { handle = HandleType.N; return true; }
            if (bottomBand.Contains(screenPt)) { handle = HandleType.S; return true; }
            if (leftBand.Contains(screenPt)) { handle = HandleType.W; return true; }
            if (rightBand.Contains(screenPt)) { handle = HandleType.E; return true; }

            return false;
        }

        public bool HitTestInterior(Point mouseScreen, IViewTransform tr)
        {
            return tr.ImageRectToScreen(RectImg).Contains(mouseScreen);
        }

        public void ResizeByHandle(HandleType h, PointF imgPt, SizeF imgSize)
        {
            float left = RectImg.Left, top = RectImg.Top, right = RectImg.Right, bottom = RectImg.Bottom;

            switch (h)
            {
                case HandleType.N: top = imgPt.Y; break;
                case HandleType.S: bottom = imgPt.Y; break;
                case HandleType.W: left = imgPt.X; break;
                case HandleType.E: right = imgPt.X; break;
                case HandleType.NW: left = imgPt.X; top = imgPt.Y; break;
                case HandleType.NE: right = imgPt.X; top = imgPt.Y; break;
                case HandleType.SW: left = imgPt.X; bottom = imgPt.Y; break;
                case HandleType.SE: right = imgPt.X; bottom = imgPt.Y; break;
                default: return;
            }

            left = Clamp(left, 0, imgSize.Width);
            right = Clamp(right, 0, imgSize.Width);
            top = Clamp(top, 0, imgSize.Height);
            bottom = Clamp(bottom, 0, imgSize.Height);

            if (right - left < MinRectSizeImg)
            {
                if (h == HandleType.W || h == HandleType.NW || h == HandleType.SW) left = right + (-MinRectSizeImg);
                else right = left + MinRectSizeImg;
            }
            if (bottom - top < MinRectSizeImg)
            {
                if (h == HandleType.N || h == HandleType.NW || h == HandleType.NE) top = bottom + (-MinRectSizeImg);
                else bottom = top + MinRectSizeImg;
            }

            RectImg = Normalize(new RectangleF(left, top, right - left, bottom - top));
        }

        private static PointF[] GetHandleCenters(RectangleF sRect)
        {
            float x1 = sRect.Left, y1 = sRect.Top, x2 = sRect.Right, y2 = sRect.Bottom;
            float cx = x1 + sRect.Width / 2f, cy = y1 + sRect.Height / 2f;
            return new[]
            {
                new PointF(x1, y1), new PointF(cx, y1), new PointF(x2, y1),
                new PointF(x1, cy),                       new PointF(x2, cy),
                new PointF(x1, y2), new PointF(cx, y2), new PointF(x2, y2)
            };
        }
    }
}
