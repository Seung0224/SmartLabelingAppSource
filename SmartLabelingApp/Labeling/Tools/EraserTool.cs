using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Linq;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    /// <summary>
    /// 브러시처럼 긁어서 도형의 면적을 지우는 지우개 툴.
    /// - BrushStrokeShape: 면적에서 차집합 → 부분 지우기
    /// - Rectangle/Circle/Triangle/Polygon: 면적에서 차집합 → 남은 면이 있으면 BrushStrokeShape로 변환
    /// </summary>
    public sealed class EraserTool : ITool
    {
        public bool IsEditingActive => _erasing;

        private bool _erasing;
        private readonly List<PointF> _ptsImg = new List<PointF>();
        private const float MinSampleDistImg = 0.25f;

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left || c.Image == null) return;

            var ip = c.Transform.ScreenToImage(e.Location);
            var sz = c.Transform.ImageSize;
            if (ip.X < 0 || ip.Y < 0 || ip.X >= sz.Width || ip.Y >= sz.Height) return;

            _ptsImg.Clear();
            _ptsImg.Add(ip);
            _erasing = true;
            c.Capture = true;

            c.RaiseToolEditBegan(); // (브러시 크기 팝업 등) 닫기 신호
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            // 미리보기 링은 그리는 중이 아니어도 따라다님
            if (!_erasing && c.Image != null)
                c.Invalidate();

            if (!_erasing || c.Image == null || e.Button != MouseButtons.Left) return;

            var ip = c.Transform.ScreenToImage(e.Location);
            var last = _ptsImg[_ptsImg.Count - 1];
            float dx = ip.X - last.X, dy = ip.Y - last.Y;

            if (dx * dx + dy * dy >= MinSampleDistImg * MinSampleDistImg)
            {
                _ptsImg.Add(ip);
                c.Invalidate();
            }
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (!_erasing || e.Button != MouseButtons.Left) return;

            // 1) 지우개 영역(이미지 좌표) 생성
            using (var eraserArea = BuildEraserAreaImageSpace(_ptsImg, c.BrushDiameterPx))
            {
                if (eraserArea != null && !IsPathEmpty(eraserArea))
                {
                    // 2) 모든 도형에 차집합 적용
                    //    (성능: 바운딩 박스 충돌 시에만 Difference)
                    var toRemove = new List<IShape>();
                    var toAdd = new List<IShape>(); // 변환(도형→브러시영역) 결과

                    for (int i = 0; i < c.Shapes.Count; i++)
                    {
                        var s = c.Shapes[i];

                        using (var shapeArea = TryBuildShapeAreaImageSpace(s))
                        {
                            if (shapeArea == null || IsPathEmpty(shapeArea))
                                continue;

                            // 빠른 바운딩 체크
                            if (!shapeArea.GetBounds().IntersectsWith(eraserArea.GetBounds()))
                                continue;

                            // 실제 차집합
                            using (var diff = PathBoolean.Difference(shapeArea, eraserArea))
                            {
                                if (diff == null || IsPathEmpty(diff))
                                {
                                    // 완전히 지워짐 → 제거
                                    toRemove.Add(s);
                                }
                                else
                                {
                                    if (s is BrushStrokeShape bs)
                                    {
                                        // 기존 브러시 면에서 일부만 지워짐 → 면적 교체
                                        // ReplaceArea가 소유권을 가져가므로 복제 전달
                                        bs.ReplaceArea((GraphicsPath)diff.Clone());
                                    }
                                    else
                                    {
                                        // 기하 도형은 남은 면적을 BrushStrokeShape로 변환
                                        var newBrush = new BrushStrokeShape
                                        {
                                            DiameterPx = c.BrushDiameterPx
                                        };
                                        newBrush.ReplaceArea((GraphicsPath)diff.Clone());

                                        toRemove.Add(s);
                                        toAdd.Add(newBrush);
                                    }
                                }
                            }
                        }
                    }

                    // 3) 일괄 반영
                    if (toRemove.Count > 0 || toAdd.Count > 0)
                    {
                        // 제거
                        for (int i = 0; i < toRemove.Count; i++)
                            c.Shapes.Remove(toRemove[i]);

                        // 추가
                        for (int i = 0; i < toAdd.Count; i++)
                        {
                            c.Shapes.Add(toAdd[i]);
                            // 생성 히스토리(있다면) 남기기
                            c.History.PushCreated(toAdd[i]);
                        }

                        // 선택은 안전하게 초기화(삭제/변환으로 인한 dangling 방지)
                        c.Selection.Clear();
                    }
                }
            }

            _erasing = false;
            _ptsImg.Clear();
            c.Capture = false;
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            if (_erasing && e.KeyCode == Keys.Escape)
            {
                _erasing = false;
                _ptsImg.Clear();
                c.Capture = false;
                c.Invalidate();
                e.Handled = e.SuppressKeyPress = true;
            }
        }
        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            if (!_erasing || _ptsImg == null || _ptsImg.Count == 0 || c.Image == null)
                return;

            // 이미지 지름(px) -> 화면 지름(px)
            float wScr = c.Transform.ImageRectToScreen(
                            new RectangleF(0, 0, c.BrushDiameterPx, c.BrushDiameterPx)).Width;
            if (wScr < 1f) wScr = 1f;
            if (wScr > 256f) wScr = 256f;

            int maxPts = 600;
            int step = Math.Max(1, _ptsImg.Count / maxPts);

            g.SmoothingMode = SmoothingMode.AntiAlias;

            using (var pen = new Pen(Color.FromArgb(80, Color.OrangeRed), wScr))
            {
                pen.LineJoin = LineJoin.Round;
                pen.StartCap = LineCap.Round;
                pen.EndCap = LineCap.Round;

                if (_ptsImg.Count == 1)
                {
                    var sp = c.Transform.ImageToScreen(_ptsImg[0]);
                    using (var br = new SolidBrush(Color.FromArgb(80, Color.OrangeRed)))
                        g.FillEllipse(br, sp.X - wScr / 2f, sp.Y - wScr / 2f, wScr, wScr);
                }
                else
                {
                    var ptsScr = new List<PointF>((_ptsImg.Count + step - 1) / step);
                    for (int i = 0; i < _ptsImg.Count; i += step)
                        ptsScr.Add(c.Transform.ImageToScreen(_ptsImg[i]));

                    g.DrawLines(pen, ptsScr.ToArray());
                }
            }
        }

        // ===== helpers =====

        private static bool IsPathEmpty(GraphicsPath p)
        {
            if (p == null) return true;
            var b = p.GetBounds();
            return b.Width <= 0f || b.Height <= 0f;
        }

        private static GraphicsPath BuildEraserAreaImageSpace(IReadOnlyList<PointF> ptsImg, float diameterPx)
        {
            if (ptsImg == null || ptsImg.Count == 0) return null;

            var path = new GraphicsPath(FillMode.Winding);

            if (ptsImg.Count == 1)
            {
                var p = ptsImg[0];
                path.AddEllipse(p.X - diameterPx / 2f, p.Y - diameterPx / 2f, diameterPx, diameterPx);
                return path;
            }

            path.AddLines(ptsImg.ToArray());
            using (var pen = new Pen(Color.Black, Math.Max(1f, diameterPx)))
            {
                pen.LineJoin = LineJoin.Round;
                pen.StartCap = LineCap.Round;
                pen.EndCap = LineCap.Round;
                path.Widen(pen);
            }
            return path;
        }

        private static GraphicsPath TryBuildShapeAreaImageSpace(IShape s)
        {
            if (s == null) return null;

            // 1) 브러시 면
            if (s is BrushStrokeShape bs)
            {
                var area = bs.GetAreaPathImgClone();
                return area; // 이미 Clone 반환
            }

            // 2) 사각형
            if (s is RectangleShape r)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddRectangle(r.RectImg);
                return gp;
            }

            // 3) 원 (타원)
            if (s is CircleShape c)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddEllipse(c.RectImg);
                return gp;
            }

            // 4) 삼각형
            if (s is TriangleShape tri && tri.PointsImg != null && tri.PointsImg.Count >= 3)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddPolygon(tri.PointsImg.ToArray());
                return gp;
            }

            // 5) 폴리곤 / N-gon
            if (s is PolygonShape poly && poly.PointsImg != null && poly.PointsImg.Count >= 3)
            {
                var gp = new GraphicsPath(FillMode.Winding);
                gp.AddPolygon(poly.PointsImg.ToArray());
                return gp;
            }

            // 그 외 타입은 바운즈 기반 간단 근사라도 만들 수 있지만,
            // 안전하게 null 반환(해당 타입은 지우개 제외)
            return null;
        }
    }
}
