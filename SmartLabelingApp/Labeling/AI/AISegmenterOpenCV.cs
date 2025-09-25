// NuGet: OpenCvSharp4, OpenCvSharp4.runtime.windows
using OpenCvSharp;
using OpenCvSharp.Extensions;
using SmartLabelingApp.AI;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;

namespace SmartLabelingApp.AI
{
    // OpenCV GrabCut 함수를 사용
    // 딥러닝이 아니라 통계적 최적화, 전경과 배경을 각각 GMM (혼합 가우시안)으로 표현하고 파라메타 값을 반복적으로 추정함

    // 1. GMM 파라미터 업데이트
    // → 현재 전경/배경 라벨에 따라 각 픽셀을 해당 클래스의 K 성분중 하나에 할당
    // → 혼합비, 평균, 공분산
    // 2. 라벨(전경/배경 재할당)
    // → 새롭게 추정된 GMM을 가지고 외곽부의 픽셀과 라벨이 매끄러워지도록 매끄러움을 더함

    // 위 과정 반복 그러면 4가지 결과가 나오는데(확실한 전경/ 확실한 배경 / 전경 같은것/ 배경같은것)
    // 확실한 전경과 전경같은것 둘다 채택함으로써 최적화해냄


    /// <summary>
    /// OpenCV GrabCut 구현(박스/포인트 프롬프트 지원) — 성능개선판
    /// 변경점:
    /// 1) ROI 기반 처리: 전체 이미지 대신 프롬프트 주변 영역만 잘라서 GrabCut 수행 → 큰 이미지에서 속도 향상
    /// 2) 메모리 절감: ROI 크기만큼의 Mask/Mat만 생성
    /// 3) 나머지 파이프라인(후처리/컨투어/다각형 변환)은 동일
    /// </summary>
    /// 
    public sealed class GrabCutSegmenter : IAISegmenter
    {
        public List<List<PointF>> Segment(Bitmap image, AISegmenterPrompt prompt, AISegmenterOptions options = null)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));
            if (options == null) options = new AISegmenterOptions();

            using (var src0 = BitmapConverter.ToMat(image))
            using (var src = EnsureBgr8UC3(src0))
            {
                int w = src.Cols, h = src.Rows;

                // ===== 0) ROI 계산 =====
                Rect roi;
                if (prompt.Kind == PromptKind.Box)
                {
                    var rect = ClampRectToImage(prompt.Box ?? RectangleF.Empty, w, h);
                    if (rect.Width < 1 || rect.Height < 1)
                        return new List<List<PointF>>();

                    roi = ExpandRectToInt(rect, w, h, padPx: Math.Max(6, (int)Math.Round(Math.Min(w, h) * 0.01))); // 1% 패딩(최소 6px)
                }
                else if (prompt.Kind == PromptKind.Points)
                {
                    if (prompt.Points == null || prompt.Points.Count == 0)
                        return new List<List<PointF>>();

                    // 포인트들의 바운딩박스 + 패딩
                    var bb = BoundsOfPoints(prompt.Points.Select(p => p.Point));
                    roi = ExpandRectToInt(bb, w, h, padPx: Math.Max(6, (int)Math.Round(Math.Min(w, h) * 0.015))); // 1.5% 패딩
                }
                else
                {
                    return new List<List<PointF>>();
                }

                // ROI가 너무 작으면 최소 크기 보장
                if (roi.Width < 8 || roi.Height < 8)
                    roi = InflateClamped(roi, w, h, 4);

                // ===== 1) ROI 추출 =====
                using (var srcRoi = new Mat(src, roi))
                using (var maskRoi = new Mat(srcRoi.Rows, srcRoi.Cols, MatType.CV_8UC1, Scalar.All((int)GrabCutClasses.PR_BGD)))
                using (var bgd = new Mat())
                using (var fgd = new Mat())
                {
                    // ---- 초기화 (딱 한 번만) ----
                    bool usedMaskInit = false;

                    // (1) Color Gate가 켜져 있으면 먼저 씨앗을 마스크에 채움 → InitWithMask
                    if (options.UseColorGate)
                    {
                        ApplyColorGateSeeds(srcRoi, maskRoi, options);
                        usedMaskInit = true;
                    }

                    // (2) 포인트 프롬프트면 점 씨앗 추가 → InitWithMask
                    if (prompt.Kind == PromptKind.Points && prompt.Points != null && prompt.Points.Count > 0)
                    {
                        foreach (var pt in prompt.Points)
                        {
                            var pAbs = ClampPoint(pt.Point, w, h);
                            var p = new OpenCvSharp.Point((int)Math.Round(pAbs.X) - roi.X, (int)Math.Round(pAbs.Y) - roi.Y);
                            p.X = Math.Max(0, Math.Min(maskRoi.Cols - 1, p.X));
                            p.Y = Math.Max(0, Math.Min(maskRoi.Rows - 1, p.Y));
                            byte val = pt.IsForeground ? (byte)GrabCutClasses.FGD : (byte)GrabCutClasses.BGD;
                            int radius = Math.Max(1, (int)Math.Round(Math.Min(roi.Width, roi.Height) * 0.01)); // ROI 기준 1%
                            Cv2.Circle(maskRoi, p, radius, new Scalar(val), thickness: -1);
                        }
                        usedMaskInit = true;
                    }

                    // (3) GrabCut 실행
                    if (!usedMaskInit && prompt.Kind == PromptKind.Box)
                    {
                        // Rect 기반 초기화 (ColorGate OFF & Points 아님)
                        var rect = ClampRectToImage(prompt.Box ?? RectangleF.Empty, w, h);
                        var rLocal = new Rect(
                            (int)Math.Round(rect.X) - roi.X,
                            (int)Math.Round(rect.Y) - roi.Y,
                            (int)Math.Round(rect.Width),
                            (int)Math.Round(rect.Height));
                        rLocal = ClampRectToRoi(rLocal, srcRoi.Cols, srcRoi.Rows);

                        Cv2.GrabCut(srcRoi, maskRoi, rLocal, bgd, fgd,
                                    Math.Max(1, options.GrabCutIters),
                                    GrabCutModes.InitWithRect);
                    }
                    else
                    {
                        // ColorGate 또는 Points가 개입된 경우 → 마스크 기반 초기화
                        Cv2.GrabCut(srcRoi, maskRoi, new Rect(), bgd, fgd,
                                    Math.Max(1, options.GrabCutIters),
                                    GrabCutModes.InitWithMask);
                    }

                    // ---- 2) FGD 바이너리 ----
                    using (var isFG = new Mat())
                    using (var isFG2 = new Mat())
                    using (var bin = new Mat())
                    using (var binU8 = new Mat())
                    {
                        Cv2.InRange(maskRoi, new Scalar((int)GrabCutClasses.FGD), new Scalar((int)GrabCutClasses.FGD), isFG);
                        Cv2.InRange(maskRoi, new Scalar((int)GrabCutClasses.PR_FGD), new Scalar((int)GrabCutClasses.PR_FGD), isFG2);
                        Cv2.BitwiseOr(isFG, isFG2, bin);
                        bin.ConvertTo(binU8, MatType.CV_8UC1, 255.0);

                        // ---- 3) 후처리(옵션) ----
                        if (options.Smooth)
                        {
                            int k = options.CloseKernel;
                            if (k % 2 == 0) k++;
                            k = Math.Max(1, k);
                            using (var kernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new OpenCvSharp.Size(k, k)))
                            {
                                Cv2.MorphologyEx(binU8, binU8, MorphTypes.Close, kernel);
                            }
                        }

                        // ---- 4) 컨투어 → 폴리곤(이미지 좌표로 오프셋 추가) ----
                        OpenCvSharp.Point[][] contours;
                        OpenCvSharp.HierarchyIndex[] hierarchy;
                        Cv2.FindContours(binU8, out contours, out hierarchy, RetrievalModes.External, ContourApproximationModes.ApproxSimple);

                        var polys = new List<List<PointF>>();
                        foreach (var cnt in contours)
                        {
                            double area = Cv2.ContourArea(cnt);
                            if (area < options.MinAreaPx) continue;

                            var approx = Cv2.ApproxPolyDP(cnt, options.ApproxEpsilon, true);
                            if (approx.Length >= 3)
                            {
                                var poly = new List<PointF>(approx.Length);
                                for (int i = 0; i < approx.Length; i++)
                                {
                                    poly.Add(new PointF(approx[i].X + roi.X, approx[i].Y + roi.Y));
                                }
                                polys.Add(poly);
                            }
                        }

                        // 큰 것 우선 정렬
                        polys = polys.OrderByDescending(p => Math.Abs(PolygonArea(p))).ToList();
                        return polys;
                    }
                }
            }
        }


        // ---------- 유틸 ----------
        private static Mat EnsureBgr8UC3(Mat m)
        {
            if (m.Type() == MatType.CV_8UC3)
                return m.Clone();

            var outMat = new Mat();

            if (m.Type() == MatType.CV_8UC4 || m.Channels() == 4)
            {
                Cv2.CvtColor(m, outMat, ColorConversionCodes.BGRA2BGR);
                return outMat;
            }
            if (m.Type() == MatType.CV_8UC1 || m.Channels() == 1)
            {
                Cv2.CvtColor(m, outMat, ColorConversionCodes.GRAY2BGR);
                return outMat;
            }

            var tmp8 = new Mat();
            m.ConvertTo(tmp8, MatType.CV_8U);
            if (tmp8.Channels() == 1)
                Cv2.CvtColor(tmp8, outMat, ColorConversionCodes.GRAY2BGR);
            else if (tmp8.Channels() == 4)
                Cv2.CvtColor(tmp8, outMat, ColorConversionCodes.BGRA2BGR);
            else if (tmp8.Channels() == 3)
                outMat = tmp8.Clone();
            else
            {
                var plane = new Mat();
                Cv2.ExtractChannel(tmp8, plane, 0);
                Cv2.CvtColor(plane, outMat, ColorConversionCodes.GRAY2BGR);
                plane.Dispose();
            }
            tmp8.Dispose();
            return outMat;
        }

        // === ADD: 색 기반 씨앗 생성 ===
        private static void ApplyColorGateSeeds(Mat srcRoiBgr, Mat maskRoi, AISegmenterOptions opt)
        {
            if (srcRoiBgr == null || maskRoi == null || opt == null) return;
            if (!opt.UseColorGate) return;

            // 1) 색공간 변환
            var conv = new Mat();
            if (opt.GateColorSpace == AISegmenterOptions.GateSpace.Lab)
                Cv2.CvtColor(srcRoiBgr, conv, ColorConversionCodes.BGR2Lab);
            else
                Cv2.CvtColor(srcRoiBgr, conv, ColorConversionCodes.BGR2HSV);

            // 2) 타겟색을 같은 공간으로 변환
            var one = new Mat(1, 1, MatType.CV_8UC3, new Scalar(opt.GateColor.B, opt.GateColor.G, opt.GateColor.R));
            var one2 = new Mat();
            if (opt.GateColorSpace == AISegmenterOptions.GateSpace.Lab)
                Cv2.CvtColor(one, one2, ColorConversionCodes.BGR2Lab);
            else
                Cv2.CvtColor(one, one2, ColorConversionCodes.BGR2HSV);
            var tv = one2.Get<Vec3b>(0, 0);
            var tgt = new Scalar(tv.Item0, tv.Item1, tv.Item2);

            // 3) 거리맵 (채널 L1 합)
            var ch = conv.Split();
            var a0 = new Mat(); var a1 = new Mat(); var a2 = new Mat();
            Cv2.Absdiff(ch[0], new Scalar(tgt.Val0), a0);
            Cv2.Absdiff(ch[1], new Scalar(tgt.Val1), a1);
            Cv2.Absdiff(ch[2], new Scalar(tgt.Val2), a2);
            var sum = new Mat();
            Cv2.Add(a0, a1, sum);
            Cv2.Add(sum, a2, sum);

            // 4) 임계값으로 근접/원거리 분리
            var near = new Mat(); var far = new Mat();
            Cv2.Threshold(sum, near, opt.GateTolerance, 255, ThresholdTypes.BinaryInv); // 가까움=255
            Cv2.Threshold(sum, far, opt.GateTolerance, 255, ThresholdTypes.Binary);   // 멂=255

            // 5) 마스크 라벨 적용
            byte fgVal = opt.GateAsForeground ? (byte)GrabCutClasses.FGD : (byte)GrabCutClasses.PR_FGD;
            byte bgVal = opt.GateAsForeground ? (byte)GrabCutClasses.BGD : (byte)GrabCutClasses.PR_BGD;

            maskRoi.SetTo(fgVal, near);
            maskRoi.SetTo(bgVal, far);
        }


        private static RectangleF ClampRectToImage(RectangleF r, int w, int h)
        {
            if (r.IsEmpty) return RectangleF.Empty;
            float x1 = Math.Max(0, Math.Min(w, r.Left));
            float y1 = Math.Max(0, Math.Min(h, r.Top));
            float x2 = Math.Max(0, Math.Min(w, r.Right));
            float y2 = Math.Max(0, Math.Min(h, r.Bottom));
            return new RectangleF(Math.Min(x1, x2), Math.Min(y1, y2), Math.Abs(x2 - x1), Math.Abs(y2 - y1));
        }

        private static System.Drawing.RectangleF BoundsOfPoints(IEnumerable<PointF> pts)
        {
            float minX = float.MaxValue, minY = float.MaxValue, maxX = float.MinValue, maxY = float.MinValue;
            bool any = false;
            foreach (var p in pts)
            {
                any = true;
                if (p.X < minX) minX = p.X;
                if (p.Y < minY) minY = p.Y;
                if (p.X > maxX) maxX = p.X;
                if (p.Y > maxY) maxY = p.Y;
            }
            return any ? new System.Drawing.RectangleF(minX, minY, Math.Max(1e-3f, maxX - minX), Math.Max(1e-3f, maxY - minY)) : System.Drawing.RectangleF.Empty;
        }

        private static Rect ExpandRectToInt(System.Drawing.RectangleF rf, int w, int h, int padPx)
        {
            int x = (int)Math.Floor(rf.X) - padPx;
            int y = (int)Math.Floor(rf.Y) - padPx;
            int r = (int)Math.Ceiling(rf.Right) + padPx;
            int b = (int)Math.Ceiling(rf.Bottom) + padPx;

            x = Math.Max(0, x); y = Math.Max(0, y);
            r = Math.Min(w, r); b = Math.Min(h, b);

            int ww = Math.Max(1, r - x);
            int hh = Math.Max(1, b - y);
            return new Rect(x, y, ww, hh);
        }

        private static Rect InflateClamped(Rect roi, int w, int h, int px)
        {
            int x = Math.Max(0, roi.X - px);
            int y = Math.Max(0, roi.Y - px);
            int r = Math.Min(w, roi.Right + px);
            int b = Math.Min(h, roi.Bottom + px);
            return new Rect(x, y, Math.Max(1, r - x), Math.Max(1, b - y));
        }

        private static Rect ClampRectToRoi(Rect r, int w, int h)
        {
            int x = Math.Max(0, Math.Min(w - 1, r.X));
            int y = Math.Max(0, Math.Min(h - 1, r.Y));
            int x2 = Math.Max(0, Math.Min(w, r.X + r.Width));
            int y2 = Math.Max(0, Math.Min(h, r.Y + r.Height));
            int ww = Math.Max(1, x2 - x);
            int hh = Math.Max(1, y2 - y);
            return new Rect(x, y, ww, hh);
        }

        private static PointF ClampPoint(PointF p, int w, int h)
        {
            return new PointF(
                Math.Max(0, Math.Min(w - 1, p.X)),
                Math.Max(0, Math.Min(h - 1, p.Y)));
        }

        /// <summary>신발끈 공식(면적)</summary>
        private static double PolygonArea(IReadOnlyList<PointF> poly)
        {
            if (poly == null || poly.Count < 3) return 0;
            double sum = 0;
            for (int i = 0; i < poly.Count; i++)
            {
                var a = poly[i];
                var b = poly[(i + 1) % poly.Count];
                sum += (double)a.X * b.Y - (double)a.Y * b.X;
            }
            return 0.5 * sum;
        }
    }

    // GrabCut 라벨 값 (OpenCV 정의와 동일)
    internal enum GrabCutClasses
    {
        BGD = 0,        // 확실한 배경
        FGD = 1,        // 확실한 전경
        PR_BGD = 2,     // 배경 같음
        PR_FGD = 3      // 전경 같음
    }
}
