using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    public sealed class BrushTool : ITool
    {
        public bool IsEditingActive => _drawing;

        private bool _drawing;
        private readonly List<PointF> _ptsImg = new List<PointF>();
        private PointF _lastImg; // 마지막 기록 지점(이미지 좌표)

        public void OnMouseDown(ImageCanvas c, MouseEventArgs e)
        {
            if (e.Button != MouseButtons.Left || c.Image == null) return;

            var ip = c.Transform.ScreenToImage(e.Location);
            var sz = c.Transform.ImageSize;
            if (ip.X < 0 || ip.Y < 0 || ip.X >= sz.Width || ip.Y >= sz.Height) return;

            _ptsImg.Clear();
            ip = ClampToImage(ip, sz);
            _ptsImg.Add(ip);
            _lastImg = ip;

            _drawing = true;
            c.Capture = true;

            c.RaiseToolEditBegan(); // 팝업 닫기 신호
            c.Invalidate();
        }

        public void OnMouseMove(ImageCanvas c, MouseEventArgs e)
        {
            // 그리는 중이 아닐 때는 미리보기 링만 갱신
            if (!_drawing || c.Image == null)
            {
                if (c.Image != null) c.Invalidate();
                return;
            }

            // ※ MouseMove는 보통 e.Button == None 이므로, 버튼 체크를 하지 말고 _drawing 플래그만 사용
            var ip = c.Transform.ScreenToImage(e.Location);
            var sz = c.Transform.ImageSize;
            ip = ClampToImage(ip, sz);

            // 일정 간격으로 포인트 추가 (보간 포함)
            float minDist = Math.Max(0.5f, EditorUIConfig.BrushVertexSpacingPx); // 안전 하한
            AddWithSpacing(_ptsImg, ref _lastImg, ip, minDist);

            // 프리뷰는 매 프레임 갱신
            c.Invalidate();
        }

        public void OnMouseUp(ImageCanvas c, MouseEventArgs e)
        {
            if (!_drawing || e.Button != MouseButtons.Left) return;

            if (_ptsImg.Count > 0)
            {
                // 단순화 (RDP)
                float eps = Math.Max(0f, EditorUIConfig.BrushSimplifyEpsPx);
                var simplified = (eps > 0f) ? RdpSimplify(_ptsImg, eps) : new List<PointF>(_ptsImg);

                // 너무 적으면 점 하나라도 남기기
                if (simplified.Count == 0 && _ptsImg.Count > 0)
                    simplified.Add(_ptsImg[0]);

                var shape = new BrushStrokeShape
                {
                    DiameterPx = (c.BrushDiameterPx > 0f) ? c.BrushDiameterPx : EditorUIConfig.BrushDefaultDiameterPx
                };
                shape.PointsImg.AddRange(simplified);

                // 기존 브러시와 합치기(앱 로직 유지)
                c.MergeBrushWithExisting(shape);

                // 선택 해제(앱 UX에 맞춰 유지)
                c.Selection.Clear();
            }

            _drawing = false;
            _ptsImg.Clear();
            c.Capture = false;
            c.Invalidate();
        }

        public void OnKeyDown(ImageCanvas c, KeyEventArgs e)
        {
            if (_drawing && e.KeyCode == Keys.Escape)
            {
                _drawing = false;
                _ptsImg.Clear();
                c.Capture = false;
                c.Invalidate();
                e.Handled = e.SuppressKeyPress = true;
            }
        }

        public void DrawOverlay(ImageCanvas c, Graphics g)
        {
            // === 그리는 중이 아닐 때: 브러시 미리보기 링 ===
            if (!_drawing && c.Image != null)
            {
                var imgSz = c.Transform.ImageSize;
                var p = c.LastMouseImg;
                if (p.X >= 0 && p.Y >= 0 && p.X < imgSz.Width && p.Y < imgSz.Height)
                {
                    float d = (c.BrushDiameterPx > 0f) ? c.BrushDiameterPx : EditorUIConfig.BrushDefaultDiameterPx;
                    var rImg = new RectangleF(p.X - d / 2f, p.Y - d / 2f, d, d);
                    var rScr = c.Transform.ImageRectToScreen(rImg);

                    using (var fill = new SolidBrush(Color.FromArgb(64, Color.DeepSkyBlue)))
                    using (var pen = new Pen(Color.DeepSkyBlue, 1.5f))
                    {
                        g.SmoothingMode = SmoothingMode.AntiAlias;
                        g.FillEllipse(fill, rScr);
                        g.DrawEllipse(pen, rScr);
                    }
                }
                return;
            }

            // === 그리는 중 프리뷰: 굵기 정확(이미지 px → 화면 px), 라운드 조인/캡 ===
            if (!_drawing || _ptsImg.Count == 0 || c.Image == null) return;

            // 이미지 지름(px) -> 화면 지름(px)
            float brushD = (c.BrushDiameterPx > 0f) ? c.BrushDiameterPx : EditorUIConfig.BrushDefaultDiameterPx;
            float wScr = c.Transform.ImageRectToScreen(new RectangleF(0, 0, brushD, brushD)).Width;
            if (wScr < 1f) wScr = 1f;
            if (wScr > 256f) wScr = 256f; // GDI+ 안전상한

            // 프리뷰용 다운샘플 (많이 그려도 가볍게)
            int maxPts = 600;
            int step = Math.Max(1, _ptsImg.Count / maxPts);

            g.SmoothingMode = SmoothingMode.AntiAlias;

            using (var pen = new Pen(Color.FromArgb(80, Color.DeepSkyBlue), wScr))
            {
                pen.LineJoin = LineJoin.Round;
                pen.StartCap = LineCap.Round;
                pen.EndCap = LineCap.Round;

                if (_ptsImg.Count == 1)
                {
                    var sp = c.Transform.ImageToScreen(_ptsImg[0]);
                    using (var br = new SolidBrush(Color.FromArgb(80, Color.DeepSkyBlue)))
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

        // ───────────────────────────────────── helpers ─────────────────────────────────────

        private static PointF ClampToImage(PointF p, SizeF imgSize)
        {
            float x = p.X, y = p.Y;
            if (x < 0) x = 0; else if (x >= imgSize.Width) x = imgSize.Width - 1e-3f;
            if (y < 0) y = 0; else if (y >= imgSize.Height) y = imgSize.Height - 1e-3f;
            return new PointF(x, y);
        }

        /// <summary>
        /// 마지막 기록 지점에서 target까지의 선분을 일정 간격(minDist)으로 보간하며 포인트를 추가.
        /// </summary>
        private static void AddWithSpacing(List<PointF> pts, ref PointF last, PointF target, float minDist)
        {
            float dx = target.X - last.X;
            float dy = target.Y - last.Y;
            float dist = (float)Math.Sqrt(dx * dx + dy * dy);
            if (dist < minDist) return;

            int steps = Math.Max(1, (int)Math.Floor(dist / minDist));
            float inv = 1f / steps;

            for (int i = 1; i <= steps; i++)
            {
                float t = i * inv;
                var p = new PointF(last.X + dx * t, last.Y + dy * t);
                pts.Add(p);
            }
            last = target;
        }

        // 라머–더글라스–피커 단순화 (2D polyline)
        private static List<PointF> RdpSimplify(List<PointF> pts, float eps)
        {
            if (pts == null || pts.Count < 3 || eps <= 0f) return pts ?? new List<PointF>();
            var outPts = new List<PointF>();
            RdpRec(pts, 0, pts.Count - 1, eps, outPts);

            // 시작/끝 보정
            if (outPts.Count == 0 || outPts[0] != pts[0]) outPts.Insert(0, pts[0]);
            if (outPts[outPts.Count - 1] != pts[pts.Count - 1]) outPts.Add(pts[pts.Count - 1]);

            return outPts;
        }

        private static void RdpRec(List<PointF> pts, int s, int e, float eps, List<PointF> outPts)
        {
            if (e <= s + 1) return;

            float maxDist = -1f; int idx = -1;
            var a = pts[s]; var b = pts[e];
            for (int i = s + 1; i < e; i++)
            {
                float d = PerpDistance(pts[i], a, b);
                if (d > maxDist) { maxDist = d; idx = i; }
            }
            if (maxDist > eps)
            {
                RdpRec(pts, s, idx, eps, outPts);
                if (outPts.Count == 0 || outPts[outPts.Count - 1] != pts[idx]) outPts.Add(pts[idx]);
                RdpRec(pts, idx, e, eps, outPts);
            }
        }

        private static float PerpDistance(PointF p, PointF a, PointF b)
        {
            float dx = b.X - a.X, dy = b.Y - a.Y;
            if (dx == 0f && dy == 0f)
            {
                float vx = p.X - a.X, vy = p.Y - a.Y;
                return (float)Math.Sqrt(vx * vx + vy * vy);
            }
            float t = ((p.X - a.X) * dx + (p.Y - a.Y) * dy) / (dx * dx + dy * dy);
            float px = a.X + t * dx, py = a.Y + t * dy;
            float ex = p.X - px, ey = p.Y - py;
            return (float)Math.Sqrt(ex * ex + ey * ey);
        }
    }
}
