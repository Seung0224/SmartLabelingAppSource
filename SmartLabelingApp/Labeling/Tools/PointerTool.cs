using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Windows.Forms;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class PointerTool : ITool
    {
        public bool IsEditingActive =>
            _isMovingBox || _isResizingBox || _isMovingPoly || _isDraggingVertex
            || _isMovingCircle || _isResizingCircle
            || _isMovingBrush || _isResizingBrush;

        // Box
        bool _isMovingBox, _isResizingBox;
        RectangleF _dragBoxStart;

        // Polygon/Triangle
        bool _isMovingPoly, _isDraggingVertex;
        List<PointF> _dragPolyStart;

        // Circle
        bool _isMovingCircle, _isResizingCircle;
        RectangleF _dragCircleStart;

        // Brush
        bool _isMovingBrush, _isResizingBrush;
        RectangleF _dragBrushStartBounds;

        PointF _dragStartImg;
        private PointF _lastMoveImgForBrush;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left) return;

            var tr = c.Transform;
            var imgPt = tr.ScreenToImage(e.Location);

            for (int i = c.Shapes.Count - 1; i >= 0; i--)
            {
                var s = c.Shapes[i];

                // 1) 핸들 우선
                HandleType handle;
                int vtx;
                if (s.HitTestHandle(e.Location, tr, out handle, out vtx))
                {
                    c.Selection.Selected = s;
                    c.Selection.ActiveHandle = handle;
                    c.Selection.SelectedVertexIndex = vtx;
                    _dragStartImg = imgPt;

                    if (s is RectangleShape)
                    {
                        _dragBoxStart = ((RectangleShape)s).RectImg;
                        if (handle == HandleType.Move) { _isMovingBox = true; c.Cursor = Cursors.SizeAll; }
                        else { _isResizingBox = true; c.Cursor = c.CursorFromHandle(handle); }
                    }
                    else if (s is PolygonShape)
                    {
                        _dragPolyStart = new List<PointF>(((PolygonShape)s).PointsImg);
                        _isDraggingVertex = true; c.Cursor = Cursors.Hand;
                    }
                    else if (s is TriangleShape)
                    {
                        _dragPolyStart = new List<PointF>(((TriangleShape)s).PointsImg);
                        _isDraggingVertex = true; c.Cursor = Cursors.Hand;
                    }
                    else if (s is CircleShape)
                    {
                        _dragCircleStart = ((CircleShape)s).RectImg;
                        _isResizingCircle = true; c.Cursor = c.CursorFromHandle(handle);
                    }
                    else if (s is BrushStrokeShape)
                    {
                        var br = (BrushStrokeShape)s;
                        _dragBrushStartBounds = br.GetBoundsImg();
                        br.BeginResize(_dragBrushStartBounds); // 스냅샷
                        _isResizingBrush = true;
                        c.Cursor = c.CursorFromHandle(handle);
                    }
                    return;
                }

                // 2) 내부 (이동)
                if (s.HitTestInterior(e.Location, tr))
                {
                    c.Selection.Selected = s;
                    c.Selection.ActiveHandle = HandleType.Move;
                    c.Selection.SelectedVertexIndex = -1;

                    _dragStartImg = imgPt;

                    if (s is RectangleShape)
                    {
                        _dragBoxStart = ((RectangleShape)s).RectImg;
                        _isMovingBox = true; c.Cursor = Cursors.SizeAll;
                    }
                    else if (s is PolygonShape)
                    {
                        _dragPolyStart = new List<PointF>(((PolygonShape)s).PointsImg);
                        _isMovingPoly = true; c.Cursor = Cursors.SizeAll;
                    }
                    else if (s is TriangleShape)
                    {
                        _dragPolyStart = new List<PointF>(((TriangleShape)s).PointsImg);
                        _isMovingPoly = true; c.Cursor = Cursors.SizeAll;
                    }
                    else if (s is CircleShape)
                    {
                        _dragCircleStart = ((CircleShape)s).RectImg;
                        _isMovingCircle = true; c.Cursor = Cursors.SizeAll;
                    }
                    else if (s is BrushStrokeShape)
                    {
                        _dragBrushStartBounds = ((BrushStrokeShape)s).GetBoundsImg();
                        _isMovingBrush = true;
                        c.Cursor = Cursors.SizeAll;

                        _lastMoveImgForBrush = imgPt;
                    }
                    return;
                }
            }

            // 빈 공간
            c.Selection.Clear();
            c.Cursor = Cursors.Default;
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image != null)
            {
                var img = c.Transform.ScreenToImage(e.Location);
                if (img.X >= 0 && img.Y >= 0 && img.X < c.Transform.ImageSize.Width && img.Y < c.Transform.ImageSize.Height)
                    c.LastMouseImg = img;
            }

            var sel = c.Selection.Selected;
            if (!IsEditingActive)
            {
                if (sel != null)
                {
                    HandleType handle;
                    int _;
                    if (sel.HitTestHandle(e.Location, c.Transform, out handle, out _))
                        c.Cursor = c.CursorFromHandle(handle);
                    else if (sel.HitTestInterior(e.Location, c.Transform))
                        c.Cursor = Cursors.SizeAll;
                    else
                        c.Cursor = Cursors.Default;
                }
                return;
            }

            var tr = c.Transform;
            var imgPt = tr.ScreenToImage(e.Location);

            // Box 이동
            if (_isMovingBox && sel is RectangleShape)
            {
                var rbox = (RectangleShape)sel;
                var dx = imgPt.X - _dragStartImg.X;
                var dy = imgPt.Y - _dragStartImg.Y;
                float w = _dragBoxStart.Width, h = _dragBoxStart.Height;
                float nx = Clamp(_dragBoxStart.X + dx, 0, tr.ImageSize.Width - w);
                float ny = Clamp(_dragBoxStart.Y + dy, 0, tr.ImageSize.Height - h);
                rbox.RectImg = new RectangleF(nx, ny, w, h);
                c.Invalidate();
                return;
            }

            // Box 리사이즈
            if (_isResizingBox && sel is RectangleShape)
            {
                var rbox2 = (RectangleShape)sel;
                rbox2.RectImg = _dragBoxStart;
                rbox2.ResizeByHandle(c.Selection.ActiveHandle, imgPt, tr.ImageSize);
                c.Invalidate();
                return;
            }

            // 이동(폴리곤/삼각형)
            if (_isMovingPoly)
            {
                var dx = imgPt.X - _dragStartImg.X;
                var dy = imgPt.Y - _dragStartImg.Y;

                float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
                for (int i = 0; i < _dragPolyStart.Count; i++)
                {
                    var p = _dragPolyStart[i];
                    if (p.X < minX) minX = p.X; if (p.Y < minY) minY = p.Y;
                    if (p.X > maxX) maxX = p.X; if (p.Y > maxY) maxY = p.Y;
                }
                dx = Clamp(dx, -minX, tr.ImageSize.Width - maxX);
                dy = Clamp(dy, -minY, tr.ImageSize.Height - maxY);

                if (sel is PolygonShape)
                {
                    var poly = (PolygonShape)sel;
                    for (int i = 0; i < poly.PointsImg.Count; i++)
                        poly.PointsImg[i] = new PointF(_dragPolyStart[i].X + dx, _dragPolyStart[i].Y + dy);
                }
                else if (sel is TriangleShape)
                {
                    var tri = (TriangleShape)sel;
                    for (int i = 0; i < tri.PointsImg.Count; i++)
                        tri.PointsImg[i] = new PointF(_dragPolyStart[i].X + dx, _dragPolyStart[i].Y + dy);
                }
                c.Invalidate();
                return;
            }

            if (_isDraggingVertex)
            {
                int vi = c.Selection.SelectedVertexIndex;
                if (vi >= 0)
                {
                    float nx = Clamp(imgPt.X, 0, tr.ImageSize.Width);
                    float ny = Clamp(imgPt.Y, 0, tr.ImageSize.Height);

                    if (sel is PolygonShape)
                    {
                        var poly2 = (PolygonShape)sel;
                        if (vi < poly2.PointsImg.Count)
                            poly2.PointsImg[vi] = new PointF(nx, ny);
                    }
                    else if (sel is TriangleShape)
                    {
                        var tri2 = (TriangleShape)sel;
                        if (vi < tri2.PointsImg.Count)
                            tri2.PointsImg[vi] = new PointF(nx, ny);
                    }
                    c.Invalidate();
                }
                return;
            }

            // Circle 이동
            if (_isMovingCircle && sel is CircleShape)
            {
                var circleMove = (CircleShape)sel;
                var dx = imgPt.X - _dragStartImg.X;
                var dy = imgPt.Y - _dragStartImg.Y;
                float w = _dragCircleStart.Width, h = _dragCircleStart.Height;
                float nx = Clamp(_dragCircleStart.X + dx, 0, tr.ImageSize.Width - w);
                float ny = Clamp(_dragCircleStart.Y + dy, 0, tr.ImageSize.Height - h);
                circleMove.RectImg = new RectangleF(nx, ny, w, h);
                c.Invalidate();
                return;
            }

            // Circle 리사이즈
            if (_isResizingCircle && sel is CircleShape)
            {
                var circleResize = (CircleShape)sel;
                circleResize.RectImg = _dragCircleStart;
                circleResize.ResizeByHandle(c.Selection.ActiveHandle, imgPt, tr.ImageSize);
                c.Invalidate();
                return;
            }

            // Brush 이동
            if (_isMovingBrush && sel is BrushStrokeShape)
            {
                var brushMove = (BrushStrokeShape)sel;

                // 이번 프레임의 증분(delta) (이미지 좌표 기준)
                float ddx = imgPt.X - _lastMoveImgForBrush.X;
                float ddy = imgPt.Y - _lastMoveImgForBrush.Y;

                if (ddx != 0f || ddy != 0f)
                {
                    // 현재 바운즈를 기준으로 경계 체크 후 실제 적용할 증분 계산
                    var curr = brushMove.GetBoundsImg();

                    float nx = Clamp(curr.X + ddx, 0, tr.ImageSize.Width - curr.Width);
                    float ny = Clamp(curr.Y + ddy, 0, tr.ImageSize.Height - curr.Height);

                    // 클램프에 의해 줄어든 실제 이동량
                    float applyDx = nx - curr.X;
                    float applyDy = ny - curr.Y;

                    if (applyDx != 0f || applyDy != 0f)
                        brushMove.MoveBy(new SizeF(applyDx, applyDy));

                    // 다음 프레임을 위한 기준점 갱신(마우스 실제 위치로)
                    _lastMoveImgForBrush = imgPt;

                    c.Invalidate();
                }
                return;
            }

            // Brush 리사이즈
            if (_isResizingBrush && sel is BrushStrokeShape)
            {
                var brushResize = (BrushStrokeShape)sel;
                brushResize.ResizeByHandle(c.Selection.ActiveHandle, imgPt, tr.ImageSize);
                c.Invalidate();
                return;
            }
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left) return;

            if (_isResizingBrush && c.Selection.Selected is BrushStrokeShape)
            {
                var br = (BrushStrokeShape)c.Selection.Selected;
                br.EndResize();
            }

            _isMovingBox = _isResizingBox = _isMovingPoly = _isDraggingVertex = false;
            _isMovingCircle = _isResizingCircle = false;
            _isMovingBrush = _isResizingBrush = false;

            c.Cursor = Cursors.Default;
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e) { }

        public void DrawOverlay(ImageCanvas canvas, Graphics g)
        {
            var sel = canvas.Selection.Selected;
            if (sel != null)
                sel.DrawOverlay(g, canvas.Transform, canvas.Selection.SelectedVertexIndex);
        }

        private static float Clamp(float v, float min, float max)
        {
            if (v < min) return min;
            if (v > max) return max;
            return v;
        }
    }
}
