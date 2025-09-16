// ShapeAreaExtensions.cs
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace SmartLabelingApp
{
    public static class ShapeAreaExtensions
    {
        public static void DrawLabelBadge(Graphics g, IViewTransform t, RectangleF boundsImg, string label)
        {
            if (string.IsNullOrWhiteSpace(label)) return;

            var s = t.ImageRectToScreen(boundsImg);
            using (var font = new Font("Segoe UI", 9f, FontStyle.Bold))
            {
                var textSize = g.MeasureString(label, font);
                int pad = 4;
                var rect = new RectangleF(
                    s.Left,
                    s.Bottom + 2f,
                    textSize.Width + pad * 2,
                    textSize.Height + pad * 2
                );

                using (var bg = new SolidBrush(Color.White))
                using (var border = new Pen(Color.FromArgb(180, 180, 180)))
                using (var textBr = new SolidBrush(Color.Black))
                {
                    g.FillRectangle(bg, rect);
                    g.DrawRectangle(border, rect.X, rect.Y, rect.Width, rect.Height);
                    g.DrawString(label, font, textBr, rect.X + pad, rect.Y + pad);
                }
            }
        }

        // 모든 도형을 “채우기 경로(이미지 좌표)”로 변환
        public static GraphicsPath GetAreaPathImgClone(this IShape s)
        {
            if (s == null) return null;

            if (s is RectangleShape r)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddRectangle(r.RectImg);
                return gp;
            }
            if (s is CircleShape c)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddEllipse(c.RectImg);
                return gp;
            }
            if (s is PolygonShape poly && poly.PointsImg != null && poly.PointsImg.Count >= 3)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddPolygon(poly.PointsImg.ToArray());
                gp.CloseAllFigures();
                return gp;
            }
            if (s is TriangleShape tri && tri.PointsImg != null && tri.PointsImg.Count >= 3)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddPolygon(tri.PointsImg.ToArray());
                gp.CloseAllFigures();
                return gp;
            }
            if (s is BrushStrokeShape br)
            {
                return br.GetAreaPathImgClone(); // 이미 이미지좌표의 면적 패스
            }
            return null;
        }
    }
}
