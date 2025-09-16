using System.Drawing;
using System.Windows.Forms;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class CircleTool : ITool
    {
        public bool IsEditingActive { get { return _isDragging; } }

        private bool _isDragging;
        private PointF _dragStartImg;
        private RectangleF _currentImg;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left)
                return;

            var imgPt = c.Transform.ScreenToImage(e.Location);
            var imgSz = c.Transform.ImageSize;
            if (imgPt.X < 0 || imgPt.Y < 0 || imgPt.X >= imgSz.Width || imgPt.Y >= imgSz.Height)
                return;

            _isDragging = true;
            _dragStartImg = imgPt;
            _currentImg = new RectangleF(imgPt, SizeF.Empty);

            c.Capture = true;
            c.Cursor = Cursors.Cross;
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging)
                return;

            var imgPt = c.Transform.ScreenToImage(e.Location);

            // 정사각형 강제(원형 미리보기 유지)
            float dx = imgPt.X - _dragStartImg.X;
            float dy = imgPt.Y - _dragStartImg.Y;
            float side = System.Math.Max(System.Math.Abs(dx), System.Math.Abs(dy));

            float x = _dragStartImg.X;
            float y = _dragStartImg.Y;
            if (dx < 0) x -= side;
            if (dy < 0) y -= side;

            _currentImg = Normalize(new RectangleF(x, y, side, side));

            c.Invalidate();
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (!_isDragging || e.Button != MouseButtons.Left)
                return;

            _isDragging = false;
            c.Capture = false;
            c.Cursor = Cursors.Default;

            if (_currentImg.Width >= MinRectSizeImg && _currentImg.Height >= MinRectSizeImg)
            {
                // 생성 & 히스토리 & 복사버퍼(즉시 Ctrl+V 가능)
                var shape = new CircleShape(_currentImg);
                c.MergeSameLabelOverlaps(shape);
            }

            _currentImg = RectangleF.Empty;

            if (!c.Focused) c.Focus();
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            // 드래그 중 Esc 취소
            if (_isDragging && e.KeyCode == Keys.Escape)
            {
                _isDragging = false;
                c.Capture = false;
                c.Cursor = Cursors.Default;
                _currentImg = RectangleF.Empty;
                c.Invalidate();

                e.Handled = true;
                e.SuppressKeyPress = true;
            }
        }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            if (!_isDragging)
                return;

            var sr = c.Transform.ImageRectToScreen(_currentImg);
            using (var pen = new Pen(Color.Orange, 2f))
            using (var fill = new SolidBrush(Color.FromArgb(30, Color.Orange)))
            {
                pen.DashStyle = System.Drawing.Drawing2D.DashStyle.Dash;
                g.SmoothingMode = System.Drawing.Drawing2D.SmoothingMode.AntiAlias;
                g.FillEllipse(fill, sr);
                g.DrawEllipse(pen, sr);
            }
        }
    }
}
