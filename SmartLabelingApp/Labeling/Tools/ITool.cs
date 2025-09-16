using System.Drawing;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    public interface ITool
    {
        bool IsEditingActive { get; }
        void OnMouseDown(ImageCanvas canvas, MouseEventArgs e);
        void OnMouseMove(ImageCanvas canvas, MouseEventArgs e);
        void OnMouseUp(ImageCanvas canvas, MouseEventArgs e);
        void OnKeyDown(ImageCanvas canvas, KeyEventArgs e);
        void DrawOverlay(ImageCanvas canvas, Graphics g); // 진행중 도형 렌더
    }
}
