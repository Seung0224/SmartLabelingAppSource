using System.Drawing;
using System.Windows.Forms;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class TriangleTool : ITool
    {
        public bool IsEditingActive => _isDragging;

        bool _isDragging;
        PointF _dragStartImg;
        RectangleF _currentImg;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left) return;

            var imgPt = c.Transform.ScreenToImage(e.Location);
            var imgSz = c.Transform.ImageSize;
            if (imgPt.X < 0 || imgPt.Y < 0 || imgPt.X >= imgSz.Width || imgPt.Y >= imgSz.Height) return;

            _isDragging = true;
            _dragStartImg = imgPt;
            _currentImg = new RectangleF(imgPt, SizeF.Empty);

            c.Capture = true;
            c.Cursor = Cursors.Cross;
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging) return;

            var imgPt = c.Transform.ScreenToImage(e.Location);
            _currentImg = Normalize(RectangleF.FromLTRB(
                _dragStartImg.X, _dragStartImg.Y, imgPt.X, imgPt.Y));

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
                var shape = new TriangleShape(_currentImg);
                c.Shapes.Add(shape);
                c.History.PushCreated(shape);
                c.Clipboard.Copy(shape); // 바로 Ctrl+V 가능
            }

            _currentImg = RectangleF.Empty;
            if (!c.Focused) c.Focus();
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            if (_isDragging && e.KeyCode == Keys.Escape)
            {
                _isDragging = false;
                c.Capture = false;
                c.Cursor = Cursors.Default;
                _currentImg = RectangleF.Empty;
                c.Invalidate();
                e.Handled = e.SuppressKeyPress = true;
            }
        }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            if (!_isDragging) return;

            var r = c.Transform.ImageRectToScreen(_currentImg);

            // 현재 프리뷰 삼각형(등변) 그리기
            PointF a = new PointF(r.Left + r.Width / 2f, r.Top);
            PointF b = new PointF(r.Left, r.Bottom);
            PointF d = new PointF(r.Right, r.Bottom);

            using (var fill = new SolidBrush(Color.FromArgb(30, Color.Orange)))
            using (var pen = new Pen(Color.Orange, 2f) { DashStyle = System.Drawing.Drawing2D.DashStyle.Dash })
            {
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.FillPolygon(fill, new[] { a, b, d });
                g.DrawPolygon(pen, new[] { a, b, d });
            }
        }
    }
}
