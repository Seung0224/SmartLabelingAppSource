using Clipper2Lib;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Drawing.Drawing2D;

namespace SmartLabelingApp
{
    public static class PathBoolean
    {
        private const double SCALE = 1024.0; // 이미지좌표 → 정수좌표 스케일

        // GraphicsPath(이미지 좌표) → Clipper Paths64
        public static Paths64 ToClipperPaths(GraphicsPath gp)
        {
            var result = new Paths64();
            if (gp == null || gp.PointCount < 3) return result;

            using (var flat = (GraphicsPath)gp.Clone())
            {
                flat.FillMode = FillMode.Winding;
                flat.Flatten(null, 0.25f);

                var pts = flat.PathPoints;

                using (var it = new GraphicsPathIterator(flat))
                {
                    int start, end;
                    bool isClosed;

                    // ★ 3개 인수 오버로드 사용 (삼각형 포함 모든 서브패스 분리)
                    while (it.NextSubpath(out start, out end, out isClosed) > 0)
                    {
                        int len = end - start + 1;          // start..end 포함
                        if (len < 3) continue;

                        var poly = new Path64(len + (isClosed ? 0 : 1));
                        for (int i = start; i <= end; i++)
                        {
                            var p = pts[i];
                            poly.Add(new Point64(
                                (long)Math.Round(p.X * SCALE),
                                (long)Math.Round(p.Y * SCALE)));
                        }

                        // 닫힘정보가 없으면 강제 닫기
                        if (!isClosed && poly.Count > 0)
                            poly.Add(poly[0]);

                        if (poly.Count >= 3)
                            result.Add(poly);
                    }
                }
            }
            return result;
        }

        // Clipper Paths64 → GraphicsPath(이미지 좌표)
        public static GraphicsPath ToGraphicsPath(Paths64 paths)
        {
            var gp = new GraphicsPath(FillMode.Winding);
            foreach (var poly in paths)
            {
                if (poly.Count < 3) continue;
                var pts = new PointF[poly.Count];
                for (int i = 0; i < poly.Count; i++)
                    pts[i] = new PointF((float)(poly[i].X / SCALE), (float)(poly[i].Y / SCALE));
                gp.AddPolygon(pts);
            }
            return gp;
        }

        // A ∪ B
        public static GraphicsPath Union(GraphicsPath a, GraphicsPath b)
        {
            var pa = ToClipperPaths(a);
            var pb = ToClipperPaths(b);
            var res = Clipper.Union(pa, pb, FillRule.NonZero);
            return ToGraphicsPath(res);
        }

        // A − B
        public static GraphicsPath Difference(GraphicsPath subject, GraphicsPath clip)
        {
            var s = ToClipperPaths(subject);
            var c = ToClipperPaths(clip);
            var res = Clipper.Difference(s, c, FillRule.NonZero);
            return ToGraphicsPath(res);
        }

        // 여러 개 경로 전체 유니온
        // PathBoolean.cs
        public static GraphicsPath UnionMany(IEnumerable<GraphicsPath> parts)
        {
            var all = new Paths64();

            foreach (var gp in parts)
            {
                if (gp == null) continue;
                var p = ToClipperPaths(gp);
                if (p.Count > 0) all.AddRange(p);
            }

            if (all.Count == 0) return null;

            // ▼ 여기 한 줄이면 충분합니다 (버전과 무관)
            var united = Clipper.Union(all, FillRule.NonZero);

            return ToGraphicsPath(united);
        }


        // subject − (clip1 ∪ clip2 ∪ …)
        public static GraphicsPath DifferenceMany(GraphicsPath subject, IEnumerable<GraphicsPath> clips)
        {
            var s = ToClipperPaths(subject);
            var c = new Paths64();
            foreach (var gp in clips)
            {
                if (gp == null) continue;
                var p = ToClipperPaths(gp);
                if (p.Count > 0) c.AddRange(p);
            }
            var res = Clipper.Difference(s, c, FillRule.NonZero);
            return ToGraphicsPath(res);
        }
    }
}
