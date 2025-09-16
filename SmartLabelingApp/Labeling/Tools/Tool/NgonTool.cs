using System;
using System.Collections.Generic;
using System.Drawing;
using System.Windows.Forms;
using static SmartLabelingApp.GeometryUtil;

namespace SmartLabelingApp
{
    public sealed class NgonTool : ITool
    {
        public bool IsEditingActive { get { return false; } }

        // 마지막으로 선택된 변의 수(3~50). 처음엔 미설정 → 첫 클릭에서 묻기
        private int _sides = 0;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (c.Image == null || e.Button != MouseButtons.Left)
                return;

            // 클릭 위치(이미지 좌표)
            var imgPt = c.Transform.ScreenToImage(e.Location);
            var imgSz = c.Transform.ImageSize;
            if (imgPt.X < 0 || imgPt.Y < 0 || imgPt.X >= imgSz.Width || imgPt.Y >= imgSz.Height)
                return;

            // Ctrl 누르고 클릭하면 다시 각 수 묻기
            bool ctrl = (Control.ModifierKeys & Keys.Control) == Keys.Control;
            if (_sides < 3 || ctrl)
            {
                int current = _sides > 0 ? _sides : 5; // 기본 5각형 제안
                var res = NgonSidesDialog.ShowDialogGetSides(c.FindForm(), current);
                if (!res.HasValue) return; // 취소
                _sides = Math.Max(3, Math.Min(50, res.Value));
            }

            CreateAt(c, imgPt, _sides); // ★ 클릭 위치에 생성(연속 그리기)
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e) { /* NOP */ }
        public void OnMouseUp(ImageCanvas c, MouseEventArgs e) { /* NOP */ }
        public void OnKeyDown(ImageCanvas c, KeyEventArgs e) { /* NOP */ }
        public void DrawOverlay(ImageCanvas c, Graphics g) { /* 즉시 생성이라 오버레이 없음 */ }

        private void CreateAt(ImageCanvas c, PointF centerImg, int sides)
        {
            var imgSz = c.Transform.ImageSize;

            // 기본 반지름: 이미지의 12% (최소/최대 보정)
            float defaultR = (float)(Math.Min(imgSz.Width, imgSz.Height) * 0.12);
            float maxR =
                Math.Min(
                    Math.Min(centerImg.X, imgSz.Width - centerImg.X),
                    Math.Min(centerImg.Y, imgSz.Height - centerImg.Y)
                ) - 1f; // 1px 여유
            float minR = MinRectSizeImg * 0.5f;

            float r = defaultR;
            if (maxR < minR) r = Math.Max(1f, maxR);   // 공간이 아주 좁을 때
            else r = Math.Max(minR, Math.Min(defaultR, maxR));

            // 정다각형 꼭짓점 생성 (-90°로 위쪽을 향하게)
            var pts = new List<PointF>(sides);
            double offset = -Math.PI / 2.0;
            for (int i = 0; i < sides; i++)
            {
                double ang = offset + (2.0 * Math.PI * i / sides);
                float x = centerImg.X + (float)(r * Math.Cos(ang));
                float y = centerImg.Y + (float)(r * Math.Sin(ang));
                pts.Add(new PointF(x, y));
            }

            var shape = new PolygonShape(pts);
            c.Shapes.Add(shape);
            c.History.PushCreated(shape);
            c.Clipboard.Copy(shape);   // 바로 Ctrl+V 가능
            c.Selection.Set(shape);    // 즉시 선택(이동/리사이즈/정점드래그 가능)

            if (!c.Focused) c.Focus();
            c.Invalidate();
        }
    }
}
