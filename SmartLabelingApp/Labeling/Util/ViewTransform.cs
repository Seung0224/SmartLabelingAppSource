using System.Drawing;
using Cyotek.Windows.Forms;

namespace SmartLabelingApp
{
    public interface IViewTransform
    {
        PointF ScreenToImage(Point screen);
        PointF ImageToScreen(PointF img);
        RectangleF ImageRectToScreen(RectangleF imgRect);
        SizeF ImageSize { get; }
    }

    public sealed class ImageBoxTransform : IViewTransform
    {
        private readonly ImageBox _box;
        public ImageBoxTransform(ImageBox box) => _box = box;

        public PointF ScreenToImage(Point screen)
        {
            var p = _box.PointToImage(screen);
            return new PointF(p.X, p.Y);
        }

        public PointF ImageToScreen(PointF img)
        {
            var rr = _box.GetOffsetRectangle(new Rectangle((int)img.X, (int)img.Y, 1, 1));
            return new PointF(rr.X, rr.Y);
        }

        public RectangleF ImageRectToScreen(RectangleF imgRect)
        {
            var ri = Rectangle.Round(imgRect);
            var rr = _box.GetOffsetRectangle(ri);
            return new RectangleF(rr.X, rr.Y, rr.Width, rr.Height);
        }

        public SizeF ImageSize
        {
            get
            {
                if (_box.Image == null) return SizeF.Empty;
                return new SizeF(_box.Image.Width, _box.Image.Height);
            }
        }
    }
}