using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public static class UiOverlayUtils
    {
        /// <summary>
        /// 이미지 사본 위에 상태 라벨과 테두리를 그려 반환합니다.
        /// 라벨 예: "NG → Score: 1.123 (1/20)"
        /// </summary>
        /// <param name="src">원본 Bitmap (수정하지 않음)</param>
        /// <param name="status">"OK" 또는 "NG" 등</param>
        /// <param name="score">점수(선택)</param>
        /// <param name="index">1-base 인덱스(선택, total과 함께 사용)</param>
        /// <param name="total">총 개수(선택, index와 함께 사용)</param>
        /// <param name="thickness">테두리 두께(픽셀)</param>
        /// <param name="marginX">오른쪽 여백</param>
        /// <param name="marginY">위쪽 여백</param>
        /// <param name="fontSize">폰트 크기</param>
        /// <param name="bgAlpha">배경박스 불투명도(0~255)</param>
        /// <param name="bgColor">배경박스 색상(기본: 흰색)</param>
        /// <returns>라벨/테두리가 그려진 새 Bitmap</returns>
        public static Bitmap DrawStatusFrame(
            Bitmap src,
            string status,
            float? score = null,
            int? index = null,
            int? total = null,
            int thickness = 10,
            int marginX = 20,
            int marginY = 16,
            int fontSize = 50,
            int bgAlpha = 20,
            Color? bgColor = null)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));

            int w = src.Width, h = src.Height;
            var dst = new Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            using (var g = Graphics.FromImage(dst))
            {
                // 원본 복사
                g.CompositingMode = CompositingMode.SourceOver;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

                g.DrawImage(src, 0, 0, w, h);

                // 색상: OK=초록, NG=빨강 (파이썬과 동일 계열)
                bool ok = string.Equals(status ?? "", "OK", StringComparison.OrdinalIgnoreCase);
                Color statusColor = ok ? Color.FromArgb(46, 204, 113) : Color.FromArgb(231, 76, 60);

                // 1) 외곽 테두리
                using (var pen = new Pen(statusColor, 1f))
                {
                    for (int t = 0; t < thickness; t++)
                    {
                        var rect = new Rectangle(t, t, w - 1 - (t * 2), h - 1 - (t * 2));
                        if (rect.Width <= 0 || rect.Height <= 0) break;
                        g.DrawRectangle(pen, rect);
                    }
                }

                // 2) 라벨 문자열 구성
                //  "NG → Score: 1.123 (1/20)"
                var label = (status ?? "").ToUpperInvariant();
                if (score.HasValue)
                {
                    // 파이썬 포맷과 비슷하게 소수 3자리
                    label += $" → Score: {score.Value.ToString("0.000", System.Globalization.CultureInfo.InvariantCulture)}";
                }
                if (index.HasValue && total.HasValue && total.Value > 0)
                {
                    label += $" ({index.Value}/{total.Value})";
                }

                // 3) 폰트/크기 측정
                using (var font = TryCreateFont("Arial", fontSize, FontStyle.Bold))
                using (var sf = new StringFormat(StringFormat.GenericTypographic))
                {
                    // MeasureString이 살짝 여유를 주므로, 더 정밀한 측정을 위해 GenericTypographic 사용
                    SizeF size = g.MeasureString(label, font, int.MaxValue, sf);
                    int tw = (int)Math.Ceiling(size.Width);
                    int th = (int)Math.Ceiling(size.Height);

                    int x = Math.Max(0, w - marginX - tw);
                    int y = Math.Max(0, marginY);

                    // 4) 반투명 배경 박스
                    int pad = 6;
                    if (bgAlpha > 0)
                    {
                        var bg = bgColor ?? Color.White;
                        using (var brush = new SolidBrush(Color.FromArgb(ClampByte(bgAlpha), bg)))
                        {
                            g.FillRectangle(brush, new Rectangle(x - pad, y - pad, tw + pad * 2, th + pad * 2));
                        }
                    }

                    // 5) 텍스트: 외곽선(네 방향) + 본문
                    DrawStringWithOutline(g, label, font, x, y, statusColor, Color.Black);
                }
            }

            return dst;
        }

        /// <summary>bool → OK/NG 라벨을 자동으로 만들고 위 함수 호출</summary>
        public static Bitmap DrawStatusFrameFromAnomaly(
            Bitmap src,
            bool isAnomaly,
            float? score = null,
            int? index = null,
            int? total = null,
            int thickness = 10,
            int marginX = 20,
            int marginY = 16,
            int fontSize = 40,
            int bgAlpha = 0,
            Color? bgColor = null)
        {
            var status = isAnomaly ? "NG" : "OK";
            return DrawStatusFrame(src, status, score, index, total, thickness, marginX, marginY, fontSize, bgAlpha, bgColor);
        }

        // ───────── helpers ─────────

        private static void DrawStringWithOutline(Graphics g, string text, Font font, int x, int y, Color fill, Color outline)
        {
            // 간단한 외곽선: 1px 네 방향
            using (var outlineBrush = new SolidBrush(outline))
            using (var fillBrush = new SolidBrush(fill))
            using (var sf = new StringFormat(StringFormat.GenericTypographic))
            {
                g.DrawString(text, font, outlineBrush, x + 1, y, sf);
                g.DrawString(text, font, outlineBrush, x - 1, y, sf);
                g.DrawString(text, font, outlineBrush, x, y + 1, sf);
                g.DrawString(text, font, outlineBrush, x, y - 1, sf);
                g.DrawString(text, font, fillBrush, x, y, sf);
            }
        }

        private static Font TryCreateFont(string family, int size, FontStyle style)
        {
            try { return new Font(family, size, style, GraphicsUnit.Pixel); }
            catch { return new Font(FontFamily.GenericSansSerif, size, style, GraphicsUnit.Pixel); }
        }

        private static int ClampByte(int v) => (v < 0) ? 0 : (v > 255 ? 255 : v);
    }
}
