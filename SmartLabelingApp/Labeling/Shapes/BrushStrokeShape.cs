using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace SmartLabelingApp
{
    public class BrushStrokeShape : IShape
    {
        public Color StrokeColor { get; set; } = Color.DeepSkyBlue;
        public Color FillColor { get; set; } = Color.FromArgb(72, Color.DeepSkyBlue);
        public string LabelName { get; set; }

        public List<PointF> PointsImg { get; } = new List<PointF>();
        public float DiameterPx { get; set; } = 18f;

        private GraphicsPath _areaImg; // 이미지 좌표 경로(외곽+홀)

        // 리사이즈 스냅샷
        private GraphicsPath _resizeStartPath;
        private List<PointF> _resizeStartPoints;
        private RectangleF _resizeStartBounds;

        public GraphicsPath GetAreaPathImgClone()
        {
            EnsureAreaPath();
            return _areaImg == null ? null : (GraphicsPath)_areaImg.Clone();
        }

        public void ReplaceArea(GraphicsPath newArea)
        {
            if (_areaImg != null) _areaImg.Dispose();
            _areaImg = (newArea != null) ? (GraphicsPath)newArea.Clone() : null;
        }

        private void EnsureAreaPath()
        {
            if (_areaImg != null) return;
            if (PointsImg == null || PointsImg.Count == 0) return;

            var p = new GraphicsPath(FillMode.Winding);

            if (PointsImg.Count == 1)
            {
                var c = PointsImg[0];
                p.AddEllipse(c.X - DiameterPx * 0.5f, c.Y - DiameterPx * 0.5f, DiameterPx, DiameterPx);
            }
            else
            {
                p.AddLines(PointsImg.ToArray());
                using (var pen = new Pen(Color.DeepSkyBlue, DiameterPx))
                {
                    pen.LineJoin = LineJoin.Round;
                    pen.StartCap = LineCap.Round;
                    pen.EndCap = LineCap.Round;
                    p.Widen(pen);
                }
            }

            _areaImg = p;
        }

        private static bool IsUsable(GraphicsPath gp)
        {
            try { return gp != null && gp.PointCount > 1; }
            catch { return false; }
        }

        private GraphicsPath BuildScreenPath(IViewTransform tr)
        {
            EnsureAreaPath();
            if (!IsUsable(_areaImg)) return null;

            try
            {
                var pts = _areaImg.PathPoints;
                var types = _areaImg.PathTypes;

                var gp = new GraphicsPath(FillMode.Winding);

                int start = 0;
                for (int i = 0; i < types.Length; i++)
                {
                    bool isStart = ((types[i] & (byte)PathPointType.PathTypeMask) == (byte)PathPointType.Start);
                    bool isLast = (i == types.Length - 1);
                    bool nextIsStart = !isLast && ((types[i + 1] & (byte)PathPointType.PathTypeMask) == (byte)PathPointType.Start);

                    if (isStart) start = i;

                    if (isLast || nextIsStart)
                    {
                        int len = i - start + 1;
                        if (len < 2) continue;

                        var buf = new PointF[len];
                        for (int k = 0; k < len; k++)
                            buf[k] = tr.ImageToScreen(pts[start + k]);

                        if (buf.Length >= 3)
                        {
                            if (buf[0] != buf[buf.Length - 1])
                            {
                                var closed = new List<PointF>(buf);
                                closed.Add(buf[0]);
                                buf = closed.ToArray();
                            }
                            gp.AddPolygon(buf);
                        }
                        else
                        {
                            gp.AddLines(buf);
                        }
                    }
                }

                return IsUsable(gp) ? gp : null;
            }
            catch { return null; }
        }

        public RectangleF GetBoundsImg()
        {
            EnsureAreaPath();
            return IsUsable(_areaImg) ? _areaImg.GetBounds() : RectangleF.Empty;
        }

        public void MoveBy(SizeF deltaImg)
        {
            if (IsUsable(_areaImg))
            {
                try
                {
                    using (var m = new Matrix())
                    {
                        m.Translate(deltaImg.Width, deltaImg.Height);
                        _areaImg.Transform(m);
                    }
                }
                catch { /* ignore */ }
            }
            if (PointsImg != null)
            {
                for (int i = 0; i < PointsImg.Count; i++)
                    PointsImg[i] = new PointF(PointsImg[i].X + deltaImg.Width, PointsImg[i].Y + deltaImg.Height);
            }
        }

        public IShape Clone()
        {
            var c = new BrushStrokeShape { DiameterPx = DiameterPx };
            if (PointsImg != null) c.PointsImg.AddRange(PointsImg);
            if (IsUsable(_areaImg)) c._areaImg = (GraphicsPath)_areaImg.Clone();
            return c;
        }

        public void Draw(Graphics g, IViewTransform tr)
        {
            using (var sPath = BuildScreenPath(tr))
            {
                if (!IsUsable(sPath)) return;

                g.SmoothingMode = SmoothingMode.AntiAlias;

                var fillCol = (FillColor.A > 0)
                    ? FillColor
                    : Color.FromArgb(72, Color.FromArgb(StrokeColor.R, StrokeColor.G, StrokeColor.B));

                using (var fill = new SolidBrush(fillCol))
                using (var pen = new Pen(StrokeColor, 2f))
                {
                    g.FillPath(fill, sPath);
                    g.DrawPath(pen, sPath);
                }
            }

            ShapeAreaExtensions.DrawLabelBadge(g, tr, GetBoundsImg(), LabelName);
        }

        public void DrawOverlay(Graphics g, IViewTransform t, int selectedVertexIndex)
        {
            var outline = GetOuterOutlineScreenPoints(t, 96); // 균등 샘플
            if (outline != null && outline.Count >= 2)
            {
                using (var dash = new Pen(Color.Orange, 2f) { DashStyle = DashStyle.Dash })
                {
                    g.SmoothingMode = SmoothingMode.AntiAlias;
                    g.DrawPolygon(dash, outline.ToArray());
                }

                float hs = EditorUIConfig.HandleDrawSizePx;
                float half = hs / 2f;
                using (var b = new SolidBrush(Color.Orange))
                using (var sel = new SolidBrush(Color.Orange))
                using (var p = new Pen(Color.DarkOrange, 1f))
                {
                    for (int i = 0; i < outline.Count; i++)
                    {
                        var c = outline[i];
                        var r = new RectangleF(c.X - half, c.Y - half, hs, hs);
                        g.FillRectangle(i == selectedVertexIndex ? sel : b, r);
                        g.DrawRectangle(p, r.X, r.Y, r.Width, r.Height);
                    }
                }
            }
        }




        private static PointF[] GetHandleCenters(RectangleF sRect)
        {
            float x1 = sRect.Left, y1 = sRect.Top, x2 = sRect.Right, y2 = sRect.Bottom;
            float cx = x1 + sRect.Width / 2f, cy = y1 + sRect.Height / 2f;
            return new[]
            {
                new PointF(x1, y1), new PointF(cx, y1), new PointF(x2, y1),
                new PointF(x1, cy),                      new PointF(x2, cy),
                new PointF(x1, y2), new PointF(cx, y2), new PointF(x2, y2)
            };
        }

        public bool HitTestHandle(Point mouseScreen, IViewTransform t, out HandleType handle, out int vertexIndex)
        {
            handle = HandleType.None; vertexIndex = -1;

            // 외곽선 균등 샘플 버텍스 우선 검사
            var outline = GetOuterOutlineScreenPoints(t, 96);
            if (outline != null && outline.Count > 0)
            {
                float r = EditorUIConfig.VertexHitRadiusPx;
                float r2 = r * r;

                for (int i = outline.Count - 1; i >= 0; i--)
                {
                    float dx = outline[i].X - mouseScreen.X;
                    float dy = outline[i].Y - mouseScreen.Y;
                    if (dx * dx + dy * dy <= r2)
                    {
                        handle = HandleType.Vertex;
                        vertexIndex = i; // (실제 이동은 ResizeByHandle에서 최근접 PointsImg 이동)
                        return true;
                    }
                }
            }

            // (B) 이하 기존 스케일 핸들 로직 그대로
            var bounds = GetBoundsImg();
            if (bounds.IsEmpty)
                return false;

            // 스크린 바운즈
            var sRect = t.ImageRectToScreen(bounds);

            // 1) 작은 도형 보정(픽셀 단위 최소 크기 보장)
            const float minPickW = 30f;
            const float minPickH = 30f;
            if (sRect.Width < minPickW)
            {
                float grow = (minPickW - sRect.Width) * 0.5f;
                sRect = RectangleF.Inflate(sRect, grow, 0f);
            }
            if (sRect.Height < minPickH)
            {
                float grow = (minPickH - sRect.Height) * 0.5f;
                sRect = RectangleF.Inflate(sRect, 0f, grow);
            }

            // 2) 살짝 바깥으로도 허용
            var hitRect = RectangleF.Inflate(sRect, 6f, 6f);

            // 3) 큰 코너 / 넓은 엣지 밴드 (EditorUIConfig로 통일)
            float cornerHit = EditorUIConfig.CornerHitPx; // 코너 박스
            float edgeBand = EditorUIConfig.EdgeBandPx;  // 엣지 밴드

            float x1 = hitRect.Left, y1 = hitRect.Top;
            float x2 = hitRect.Right, y2 = hitRect.Bottom;
            float cx = x1 + hitRect.Width / 2f, cy = y1 + hitRect.Height / 2f;

            // 코너(8개)
            PointF[] corners = new[]
            {
        new PointF(x1, y1), new PointF(cx, y1), new PointF(x2, y1),
        new PointF(x1, cy),                      new PointF(x2, cy),
        new PointF(x1, y2), new PointF(cx, y2), new PointF(x2, y2)
    };

            for (int i = 0; i < corners.Length; i++)
            {
                RectangleF hr = new RectangleF(
                    corners[i].X - cornerHit / 2f,
                    corners[i].Y - cornerHit / 2f,
                    cornerHit, cornerHit);

                if (hr.Contains(mouseScreen))
                {
                    switch (i)
                    {
                        case 0: handle = HandleType.NW; break;
                        case 1: handle = HandleType.N; break;
                        case 2: handle = HandleType.NE; break;
                        case 3: handle = HandleType.W; break;
                        case 4: handle = HandleType.E; break;
                        case 5: handle = HandleType.SW; break;
                        case 6: handle = HandleType.S; break;
                        case 7: handle = HandleType.SE; break;
                        default: handle = HandleType.None; break;
                    }
                    return true;
                }
            }

            // 엣지 밴드(코너 제외)
            RectangleF topBand = new RectangleF(x1 + cornerHit * 0.5f, y1 - edgeBand / 2f, hitRect.Width - cornerHit, edgeBand);
            RectangleF bottomBand = new RectangleF(x1 + cornerHit * 0.5f, y2 - edgeBand / 2f, hitRect.Width - cornerHit, edgeBand);
            RectangleF leftBand = new RectangleF(x1 - edgeBand / 2f, y1 + cornerHit * 0.5f, edgeBand, hitRect.Height - cornerHit);
            RectangleF rightBand = new RectangleF(x2 - edgeBand / 2f, y1 + cornerHit * 0.5f, edgeBand, hitRect.Height - cornerHit);

            if (topBand.Contains(mouseScreen)) { handle = HandleType.N; return true; }
            if (bottomBand.Contains(mouseScreen)) { handle = HandleType.S; return true; }
            if (leftBand.Contains(mouseScreen)) { handle = HandleType.W; return true; }
            if (rightBand.Contains(mouseScreen)) { handle = HandleType.E; return true; }

            return false;
        }



        public bool HitTestInterior(Point mouseScreen, IViewTransform tr)
        {
            using (var sPath = BuildScreenPath(tr))
            {
                return IsUsable(sPath) && sPath.IsVisible(mouseScreen);
            }
        }

        public void BeginResize(RectangleF startBounds)
        {
            _resizeStartBounds = startBounds;
            _resizeStartPath = GetAreaPathImgClone();
            _resizeStartPoints = (PointsImg != null && PointsImg.Count > 0)
                ? new List<PointF>(PointsImg)
                : new List<PointF>();
        }

        public void EndResize()
        {
            if (_resizeStartPath != null) _resizeStartPath.Dispose();
            _resizeStartPath = null;
            _resizeStartPoints = null;
            _resizeStartBounds = RectangleF.Empty;
        }

        public void ResizeByHandle(HandleType handle, PointF imgPoint, SizeF imageSize)
        {
            // ─────────────────────────────────────────────────────
            // 1) 버텍스 이동(PointsImg) — 스냅샷 불필요, 바로 점만 이동
            // ─────────────────────────────────────────────────────
            if (handle == HandleType.Vertex)
            {
                if (PointsImg == null || PointsImg.Count == 0) return;

                // 가장 가까운 버텍스를 찾아 이동 (TriangleShape와 동일 전략)
                int idx = 0;
                float best = float.MaxValue;
                for (int i = 0; i < PointsImg.Count; i++)
                {
                    float dx = PointsImg[i].X - imgPoint.X;
                    float dy = PointsImg[i].Y - imgPoint.Y;
                    float d2 = dx * dx + dy * dy;
                    if (d2 < best) { best = d2; idx = i; }
                }

                float nx = Clamp(imgPoint.X, 0f, imageSize.Width);
                float ny = Clamp(imgPoint.Y, 0f, imageSize.Height);
                PointsImg[idx] = new PointF(nx, ny);

                // 영역 경로를 다시 만들 수 있도록 무효화
                _areaImg = null;

                return;
            }

            // ─────────────────────────────────────────────────────
            // 2) 스케일(코너/엣지 핸들) — 기존 스냅샷 기반 로직 그대로
            // ─────────────────────────────────────────────────────
            if (!IsUsable(_resizeStartPath) || _resizeStartBounds.IsEmpty) return;

            RectangleF start = _resizeStartBounds;

            float x1 = start.Left, y1 = start.Top, x2 = start.Right, y2 = start.Bottom;
            switch (handle)
            {
                case HandleType.NW: x1 = imgPoint.X; y1 = imgPoint.Y; break;
                case HandleType.N: y1 = imgPoint.Y; break;
                case HandleType.NE: x2 = imgPoint.X; y1 = imgPoint.Y; break;
                case HandleType.W: x1 = imgPoint.X; break;
                case HandleType.E: x2 = imgPoint.X; break;
                case HandleType.SW: x1 = imgPoint.X; y2 = imgPoint.Y; break;
                case HandleType.S: y2 = imgPoint.Y; break;
                case HandleType.SE: x2 = imgPoint.X; y2 = imgPoint.Y; break;
            }
            float nx2 = x1 < x2 ? x1 : x2;
            float ny2 = y1 < y2 ? y1 : y2;
            float nw = System.Math.Abs(x2 - x1);
            float nh = System.Math.Abs(y2 - y1);
            if (nw < 1f) nw = 1f;
            if (nh < 1f) nh = 1f;

            nx2 = Clamp(nx2, 0f, imageSize.Width - nw);
            ny2 = Clamp(ny2, 0f, imageSize.Height - nh);

            RectangleF newRect = new RectangleF(nx2, ny2, nw, nh);

            float sx = start.Width <= 0 ? 1f : newRect.Width / start.Width;
            float sy = start.Height <= 0 ? 1f : newRect.Height / start.Height;

            using (var path = (GraphicsPath)_resizeStartPath.Clone())
            using (var m = new Matrix())
            {
                m.Translate(-start.X, -start.Y, MatrixOrder.Append);
                m.Scale(sx, sy, MatrixOrder.Append);
                m.Translate(newRect.X, newRect.Y, MatrixOrder.Append);

                path.Transform(m);
                ReplaceArea(path);
            }

            if (_resizeStartPoints != null && _resizeStartPoints.Count > 0)
            {
                PointsImg.Clear();
                for (int i = 0; i < _resizeStartPoints.Count; i++)
                {
                    var p0 = _resizeStartPoints[i];
                    float x = newRect.X + (p0.X - start.X) * sx;
                    float y = newRect.Y + (p0.Y - start.Y) * sy;
                    PointsImg.Add(new PointF(x, y));
                }
            }
        }
        // BrushStrokeShape 클래스 내부에 추가
        // BrushStrokeShape.cs 내부

        private static double PolyAreaAbs(IList<PointF> poly)
        {
            if (poly == null || poly.Count < 3) return 0;
            double a = 0;
            for (int i = 0, j = poly.Count - 1; i < poly.Count; j = i++)
                a += (double)(poly[j].X * poly[i].Y - poly[i].X * poly[j].Y);
            return Math.Abs(a) * 0.5;
        }

        /// <summary>
        /// 브러시 영역의 화면 좌표 외곽선을 구해 targetVerts 개로 "둘레 길이 균등" 리샘플링.
        /// </summary>
        private List<PointF> GetOuterOutlineScreenPoints(IViewTransform tr, int targetVerts = 96)
        {
            using (var sPath = BuildScreenPath(tr))
            {
                if (sPath == null || sPath.PointCount < 3) return null;

                // 안전하게 선분화
                sPath.Flatten();

                var pts = sPath.PathPoints;
                var types = sPath.PathTypes;
                if (pts == null || types == null || pts.Length < 3) return null;

                // 1) 가장 큰 폐곡선(외곽) 고르기
                var bestClosed = new List<PointF>();
                double bestArea = 0.0;

                int start = 0;
                byte mask = (byte)PathPointType.PathTypeMask;
                for (int i = 0; i < types.Length; i++)
                {
                    bool isStart = ((types[i] & mask) == (byte)PathPointType.Start);
                    bool isLast = (i == types.Length - 1);
                    bool nextIsStart = !isLast && ((types[i + 1] & mask) == (byte)PathPointType.Start);

                    if (isStart) start = i;

                    if (isLast || nextIsStart)
                    {
                        int len = i - start + 1;
                        if (len >= 3)
                        {
                            var seg = new List<PointF>(len + 1);
                            for (int k = 0; k < len; k++) seg.Add(pts[start + k]);

                            // 닫혀있지 않으면 닫기
                            if (seg[0] != seg[seg.Count - 1]) seg.Add(seg[0]);

                            double area = PolyAreaAbs(seg);
                            if (area > bestArea)
                            {
                                bestArea = area;
                                bestClosed = seg;
                            }
                        }
                    }
                }
                if (bestClosed.Count < 4) return null; // (시작=끝 포함 최소 4)

                // 2) 둘레 길이 균등 리샘플링
                //    bestClosed는 닫힌 형태(맨 앞점 == 맨 끝점) 기준
                int nClosed = bestClosed.Count;
                // 길이 누적 배열
                double total = 0;
                var cum = new double[nClosed];
                cum[0] = 0;
                for (int i = 1; i < nClosed; i++)
                {
                    var a = bestClosed[i - 1];
                    var b = bestClosed[i];
                    double dx = b.X - a.X, dy = b.Y - a.Y;
                    double seg = Math.Sqrt(dx * dx + dy * dy);
                    total += seg;
                    cum[i] = total;
                }
                if (total <= 1e-6) return null;

                targetVerts = Math.Max(8, targetVerts);
                var outPts = new List<PointF>(targetVerts);

                // 균등 간격: s = (i * total) / targetVerts
                int segIdx = 1; // 누적 배열 탐색 포인터
                for (int i = 0; i < targetVerts; i++)
                {
                    double s = (i * total) / targetVerts;

                    while (segIdx < nClosed && cum[segIdx] < s) segIdx++;
                    if (segIdx >= nClosed) segIdx = nClosed - 1;

                    var a = bestClosed[segIdx - 1];
                    var b = bestClosed[segIdx];

                    double segStart = cum[segIdx - 1];
                    double segLen = Math.Max(1e-9, cum[segIdx] - segStart);
                    double t = (s - segStart) / segLen;
                    float x = (float)(a.X + (b.X - a.X) * t);
                    float y = (float)(a.Y + (b.Y - a.Y) * t);

                    outPts.Add(new PointF(x, y));
                }

                return outPts;
            }
        }



        private static float Clamp(float v, float min, float max)
        {
            if (v < min) return min;
            if (v > max) return max;
            return v;
        }
    }
}
