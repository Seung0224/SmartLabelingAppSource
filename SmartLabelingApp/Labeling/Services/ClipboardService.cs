using System;
using System.Drawing;

namespace SmartLabelingApp
{
    public sealed class ClipboardService
    {
        public IShape CopyShape;

        public void Copy(IShape shape)
        {
            // 원본이 null이면 그대로 null
            if (shape == null)
            {
                CopyShape = null;
                return;
            }

            // 1) 우선 Clone
            var clone = shape.Clone();

            // 2) 스타일(라벨/색) 보존을 강제
            try
            {
                if (clone != null)
                {
                    clone.LabelName = shape.LabelName;
                    clone.StrokeColor = shape.StrokeColor;
                    clone.FillColor = shape.FillColor;
                }
            }
            catch { /* ignore safe */ }

            CopyShape = clone;
        }


        public IShape PasteAt(IShape shape, PointF targetCenterImg, SizeF imageSize)
        {
            if (shape == null) return null;

            // 1) 우선 Clone
            var clone = shape.Clone();

            // 1.5) 스타일(라벨/색) 보존 강제
            try
            {
                if (clone != null)
                {
                    clone.LabelName = shape.LabelName;
                    clone.StrokeColor = shape.StrokeColor;
                    clone.FillColor = shape.FillColor;
                }
            }
            catch { /* ignore */ }

            // 2) 중앙 정렬 이동
            var bounds = clone.GetBoundsImg();
            var cx = bounds.Left + bounds.Width / 2f;
            var cy = bounds.Top + bounds.Height / 2f;
            var dx = targetCenterImg.X - cx;
            var dy = targetCenterImg.Y - cy;
            clone.MoveBy(new SizeF(dx, dy));

            // 3) 이미지 경계 클램프
            var b = clone.GetBoundsImg();
            float fixX = 0f, fixY = 0f;
            if (b.Left < 0) fixX = -b.Left;
            if (b.Top < 0) fixY = -b.Top;
            if (b.Right > imageSize.Width) fixX = Math.Min(fixX, imageSize.Width - b.Right);
            if (b.Bottom > imageSize.Height) fixY = Math.Min(fixY, imageSize.Height - b.Bottom);
            if (fixX != 0 || fixY != 0) clone.MoveBy(new SizeF(fixX, fixY));

            return clone;
        }

    }
}
