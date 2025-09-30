using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;

namespace SmartLabelingApp
{
    public static class BitmapUtils
    {
        public static Bitmap ResizeKeepAspect(Bitmap src, int shortSide)
        {
            int ow = src.Width, oh = src.Height;
            float scale = (float)shortSide / (ow < oh ? ow : oh);
            int nw = (int)System.Math.Round(ow * scale);
            int nh = (int)System.Math.Round(oh * scale);

            Bitmap dst = new Bitmap(nw, nh, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.CompositingQuality = CompositingQuality.HighQuality;
                g.SmoothingMode = SmoothingMode.HighQuality;
                g.DrawImage(src, new Rectangle(0, 0, nw, nh),
                    new Rectangle(0, 0, ow, oh), GraphicsUnit.Pixel);
            }
            return dst;
        }

        public static Bitmap CenterCrop(Bitmap src, int w, int h)
        {
            int x = System.Math.Max(0, (src.Width - w) / 2);
            int y = System.Math.Max(0, (src.Height - h) / 2);

            Bitmap dst = new Bitmap(w, h, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
            {
                g.InterpolationMode = InterpolationMode.HighQualityBicubic;
                g.DrawImage(src, new Rectangle(0, 0, w, h),
                    new Rectangle(x, y, w, h), GraphicsUnit.Pixel);
            }
            return dst;
        }
    }
}
