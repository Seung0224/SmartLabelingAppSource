using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Text;
using System.Text;

namespace SmartLabelingApp
{
    public static class UiOverlayUtils
    {
        public static Bitmap DrawStatusFrame(
            Bitmap src,
            string status,
            float? score = null,
            int? index = null,
            int? total = null, int thickness = 10, int marginX = 20, int marginY = 16, int fontSize = 40, int bgAlpha = 0,
            Color? bgColor = null)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));

            int w = src.Width, h = src.Height;
            var dst = new Bitmap(w, h, System.Drawing.Imaging.PixelFormat.Format32bppArgb);

            using (var g = Graphics.FromImage(dst))
            {
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.AntiAlias;
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.PixelOffsetMode = PixelOffsetMode.HighQuality;
                g.TextRenderingHint = TextRenderingHint.ClearTypeGridFit;

                g.DrawImage(src, 0, 0, w, h);

                // 상태 색상
                bool ok = string.Equals(status, "OK", StringComparison.OrdinalIgnoreCase);
                Color statusColor = ok ? Color.FromArgb(46, 204, 113) : Color.FromArgb(231, 76, 60);

                // 1) 테두리
                using (var pen = new Pen(statusColor, thickness) { Alignment = PenAlignment.Inset })
                {
                    g.DrawRectangle(pen, 0, 0, w, h);
                }

                // 2) 라벨 문자열
                var sb = new StringBuilder(status?.ToUpperInvariant() ?? "");
                if (score.HasValue) sb.Append($" → Score: {score.Value:0.000}");
                if (index.HasValue && total.HasValue && total.Value > 0)
                    sb.Append($" ({index}/{total})");
                string label = sb.ToString();

                // 3) 위치/박스 계산
                using (var font = new Font("Arial", fontSize, FontStyle.Bold, GraphicsUnit.Pixel))
                using (var sf = new StringFormat(StringFormat.GenericTypographic))
                {
                    SizeF size = g.MeasureString(label, font, int.MaxValue, sf);
                    int x = Math.Max(0, w - marginX - (int)Math.Ceiling(size.Width));
                    int y = Math.Max(0, marginY);

                    // 배경
                    if (bgAlpha > 0)
                    {
                        var bg = bgColor ?? Color.White;
                        using (var brush = new SolidBrush(Color.FromArgb(bgAlpha, bg)))
                        {
                            g.FillRectangle(brush, new Rectangle(x - 6, y - 6,
                                (int)Math.Ceiling(size.Width) + 12,
                                (int)Math.Ceiling(size.Height) + 12));
                        }
                    }

                    // 텍스트 (GraphicsPath 외곽선 + 채움)
                    using (var path = new GraphicsPath())
                    {
                        path.AddString(label, font.FontFamily, (int)font.Style,
                            g.DpiY * font.Size / 72, new Point(x, y), sf);

                        using (var outlinePen = new Pen(Color.Black, 2) { LineJoin = LineJoin.Round })
                            g.DrawPath(outlinePen, path);
                        using (var brush = new SolidBrush(statusColor))
                            g.FillPath(brush, path);
                    }
                }
            }

            return dst;
        }

        public static Bitmap DrawStatusFrameFromAnomaly(Bitmap src, bool isAnomaly, float? score = null, int? index = null, int? total = null,
            int thickness = 10, int marginX = 100, int marginY = 16, int fontSize = 30, int bgAlpha = 0, Color? bgColor = null)
        {
            return DrawStatusFrame(src, isAnomaly ? "NG" : "OK", score, index, total,
                thickness, marginX, marginY, fontSize, bgAlpha, bgColor);
        }
    }
}