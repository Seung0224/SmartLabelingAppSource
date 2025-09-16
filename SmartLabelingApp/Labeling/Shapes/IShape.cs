using System.Drawing;

namespace SmartLabelingApp
{
    public interface IShape
    {
        Color StrokeColor { get; set; }
        Color FillColor { get; set; }
        string LabelName { get; set; }

        // 기초
        RectangleF GetBoundsImg();
        void MoveBy(SizeF deltaImg);
        IShape Clone();

        // 그리기
        void Draw(Graphics g, IViewTransform tr);
        void DrawOverlay(Graphics g, IViewTransform tr, int selectedVertexIndex);

        // 히트 테스트
        bool HitTestHandle(Point mouseScreen, IViewTransform tr, out HandleType handle, out int vertexIndex);
        bool HitTestInterior(Point mouseScreen, IViewTransform tr);

        // 리사이즈(핸들/버텍스별)
        void ResizeByHandle(HandleType handle, PointF imgPoint, SizeF imageSize);
    }
}
