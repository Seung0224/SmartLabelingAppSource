using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    /// <summary>
    /// 현재 캔버스상의 모든 "면"을 합쳐서(Union) 이미지 전체에서 빼고(Difference)
    /// 남은 바깥 영역을 하나의 BrushStrokeShape로 생성한다.
    /// 이후 기존 도형들은 모두 삭제한다.
    /// </summary>
    public sealed class MaskTool : ITool
    {
        public bool IsEditingActive => false;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left || c.Image == null) return;

            // 1) 모든 도형의 면 GraphicsPath(이미지 좌표) 수집
            var parts = new List<GraphicsPath>();
            try
            {
                foreach (var s in c.Shapes)
                {
                    var a = s.GetAreaPathImgClone();      // ← 확장 메서드(각 도형→면 경로)
                    if (a != null && a.PointCount > 0) parts.Add(a);
                }

                // 2) 모든 면을 Union
                using (var eraseUnion = PathBoolean.UnionMany(parts))
                {
                    // 3) 이미지 전체 사각형 - eraseUnion = 바깥 영역
                    using (var full = new GraphicsPath())
                    {
                        var img = c.Transform.ImageSize;
                        full.AddRectangle(new RectangleF(0, 0, img.Width, img.Height));

                        using (var outside = (eraseUnion == null || eraseUnion.PointCount == 0)
                                ? (GraphicsPath)full.Clone()
                                : PathBoolean.Difference(full, eraseUnion))
                        {
                            if (outside == null || outside.PointCount == 0)
                            {
                                System.Media.SystemSounds.Asterisk.Play();
                                return;
                            }

                            // 4) 바깥 영역을 하나의 BrushStrokeShape로 생성
                            var maskShape = new BrushStrokeShape { DiameterPx = c.BrushDiameterPx };
                            maskShape.ReplaceArea(outside); // 소유권 이전

                            // 5) 기존 도형 전부 삭제 후 마스크만 남기기
                            c.Shapes.Clear();
                            c.Selection.Clear();
                            c.Shapes.Add(maskShape);
                            c.History.PushCreated(maskShape);
                            c.Selection.Set(maskShape);

                            // 마스크 적용 후 포인터로 전환(선택/이동 가능)
                            c.Mode = ToolMode.Pointer;
                            c.Invalidate();
                        }
                    }
                }
            }
            finally
            {
                // parts 리스트 내부 경로 Dispose
                for (int i = 0; i < parts.Count; i++) parts[i]?.Dispose();
            }
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e) { }
        public void OnMouseUp(ImageCanvas c, MouseEventArgs e) { }
        public void OnKeyDown(ImageCanvas c, KeyEventArgs e) { }
        public void DrawOverlay(ImageCanvas c, Graphics g) { }
    }
}
