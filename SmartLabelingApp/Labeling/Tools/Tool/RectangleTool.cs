using System.Drawing;
using System.Windows.Forms;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class RectangleTool : ITool
    {
        public bool IsEditingActive => _isDragging;

        private bool _isDragging;
        private PointF _dragStart;
        private RectangleF _currentImg;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left) return;

            var imgPt = c.Transform.ScreenToImage(e.Location);
            var imgSize = c.Transform.ImageSize;
            if (imgPt.X < 0 || imgPt.Y < 0 || imgPt.X >= imgSize.Width || imgPt.Y >= imgSize.Height) return;

            _isDragging = true;
            _dragStart = imgPt;
            _currentImg = new RectangleF(imgPt, SizeF.Empty);

            c.Capture = true;
            c.Cursor = Cursors.Cross;
            if (!c.Focused) c.Focus();
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging) return;

            var imgPt = c.Transform.ScreenToImage(e.Location);
            _currentImg = Normalize(new RectangleF(
                System.Math.Min(_dragStart.X, imgPt.X),
                System.Math.Min(_dragStart.Y, imgPt.Y),
                System.Math.Abs(_dragStart.X - imgPt.X),
                System.Math.Abs(_dragStart.Y - imgPt.Y)));

            c.Invalidate();
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging || e.Button != MouseButtons.Left) return;

            _isDragging = false;
            c.Capture = false;
            c.Cursor = Cursors.Default;

            if (_currentImg.Width >= MinRectSizeImg && _currentImg.Height >= MinRectSizeImg)
            {
                var rect = new RectangleShape(_currentImg);
                c.Shapes.Add(rect);
                c.History.PushCreated(rect);
                c.Clipboard.Copy(rect);
            }
            _currentImg = RectangleF.Empty;
            if (!c.Focused) c.Focus();
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e) { }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            if (!_isDragging) return;

            var sr = c.Transform.ImageRectToScreen(_currentImg);
            using (var pen = new Pen(Color.Orange, 2f) { DashStyle = System.Drawing.Drawing2D.DashStyle.Dash })
            using (var fill = new SolidBrush(Color.FromArgb(30, Color.Orange)))
            {
                g.FillRectangle(fill, sr);
                g.DrawRectangle(pen, sr.X, sr.Y, sr.Width, sr.Height);
            }
        }
    }
}
