using System;
using System.Drawing;

namespace SmartLabelingApp
{
    public static class GeometryUtil
    {
        public const float HandleSize = 8f;          // 화면 픽셀
        public const float MinRectSizeImg = 2f;      // 이미지 좌표

        public static float Clamp(float v, float min, float max)
            => v < min ? min : (v > max ? max : v);

        public static RectangleF Normalize(RectangleF r)
        {
            return new RectangleF(
                Math.Min(r.Left, r.Right),
                Math.Min(r.Top, r.Bottom),
                Math.Abs(r.Width),
                Math.Abs(r.Height));
        }

        public static bool PointInPolygonScreen(Point p, PointF[] sPts)
        {
            bool inside = false;
            int n = sPts.Length;
            for (int i = 0, j = n - 1; i < n; j = i++)
            {
                float xi = sPts[i].X, yi = sPts[i].Y;
                float xj = sPts[j].X, yj = sPts[j].Y;

                bool intersect = ((yi > p.Y) != (yj > p.Y)) &&
                                 (p.X < (xj - xi) * (p.Y - yi) / ((yj - yi) == 0 ? 1e-6f : (yj - yi)) + xi);
                if (intersect) inside = !inside;
            }
            return inside;
        }
    }
}