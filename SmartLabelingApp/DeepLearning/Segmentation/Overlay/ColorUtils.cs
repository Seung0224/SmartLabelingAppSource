using System;
using System.Drawing;

namespace SmartLabelingApp
{
    public static class ColorUtils
    {
        /// <summary>
        /// 클래스 ID에 대해 안정적인 팔레트 색을 생성합니다 (HSV 기반).
        /// </summary>
        public static Color ClassColor(int classId)
        {
            // 황금비 기반 분산
            const double phi = 0.6180339887498949;
            double h = (classId * phi) % 1.0; // [0,1)
            double s = 0.65;
            double v = 0.95;
            return HsvToRgb(h, s, v);
        }

        private static Color HsvToRgb(double h, double s, double v)
        {
            if (s <= 0.0001)
            {
                int gray = (int)(v * 255.0); // ← 'g' 대신 'gray'로 이름 충돌 방지
                return Color.FromArgb(gray, gray, gray);
            }

            h = (h - Math.Floor(h)) * 6.0;
            int sector = (int)Math.Floor(h);
            double f = h - sector;
            double p = v * (1.0 - s);
            double q = v * (1.0 - s * f);
            double t = v * (1.0 - s * (1.0 - f));

            double rD, gD, bD; // 더블 채널 임시값
            switch (sector)
            {
                case 0: rD = v; gD = t; bD = p; break;
                case 1: rD = q; gD = v; bD = p; break;
                case 2: rD = p; gD = v; bD = t; break;
                case 3: rD = p; gD = q; bD = v; break;
                case 4: rD = t; gD = p; bD = v; break;
                default: rD = v; gD = p; bD = q; break;
            }

            int r = MathUtils.Clamp((int)(rD * 255.0), 0, 255);
            int g = MathUtils.Clamp((int)(gD * 255.0), 0, 255);
            int b = MathUtils.Clamp((int)(bD * 255.0), 0, 255);
            return Color.FromArgb(r, g, b);
        }
    }
}
