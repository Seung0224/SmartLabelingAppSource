using System;
using System.Collections.Generic;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmartLabelingApp
{    /// <summary>
     /// 후처리 공통 모듈:
     /// - IoU 계산
     /// - Greedy NMS (점수 내림차순)
     /// - 레터박스 복원: net 좌표계를 원본 좌표계로 변환
     /// </summary>
    public static class Postprocess
    {
        /// <summary>
        /// 두 사각형(같은 좌표계)의 IoU를 계산합니다.
        /// </summary>
        public static float IoU(in RectangleF a, in RectangleF b)
        {
            float xx0 = Math.Max(a.Left, b.Left);
            float yy0 = Math.Max(a.Top, b.Top);
            float xx1 = Math.Min(a.Right, b.Right);
            float yy1 = Math.Min(a.Bottom, b.Bottom);

            float w = Math.Max(0f, xx1 - xx0);
            float h = Math.Max(0f, yy1 - yy0);
            float inter = w * h;
            float areaA = Math.Max(0f, a.Width * a.Height);
            float areaB = Math.Max(0f, b.Width * b.Height);
            float denom = areaA + areaB - inter + 1e-6f;
            return inter / denom;
        }

        /// <summary>
        /// Greedy NMS (점수 내림차순).
        /// 제네릭 버전으로, 임의의 타입 T에 대해 Box/Score 추출 람다를 받습니다.
        /// </summary>
        /// <param name="items">대상 목록</param>
        /// <param name="getBox">사각형 추출기</param>
        /// <param name="getScore">점수 추출기</param>
        /// <param name="iouThr">IoU 임계치 (기본 0.45)</param>
        /// <typeparam name="T">탐지(Det)와 유사한 임의 타입</typeparam>
        /// <returns>억제 후 남긴 항목 목록(새 리스트)</returns>
        public static List<T> Nms<T>(
            IList<T> items,
            Func<T, RectangleF> getBox,
            Func<T, float> getScore,
            float iouThr = 0.45f)
        {
            if (items == null || items.Count <= 1)
                return items?.ToList() ?? new List<T>();

            // 점수 내림차순 정렬된 인덱스
            var order = Enumerable.Range(0, items.Count)
                .OrderByDescending(i => getScore(items[i]))
                .ToArray();

            var keep = new List<T>(items.Count);
            var suppressed = new bool[items.Count];

            for (int oi = 0; oi < order.Length; oi++)
            {
                int i = order[oi];
                if (suppressed[i]) continue;

                var a = getBox(items[i]);
                keep.Add(items[i]);

                float aArea = a.Width * a.Height;

                for (int oj = oi + 1; oj < order.Length; oj++)
                {
                    int j = order[oj];
                    if (suppressed[j]) continue;

                    var b = getBox(items[j]);
                    float xx0 = Math.Max(a.Left, b.Left);
                    float yy0 = Math.Max(a.Top, b.Top);
                    float xx1 = Math.Min(a.Right, b.Right);
                    float yy1 = Math.Min(a.Bottom, b.Bottom);
                    float w = Math.Max(0f, xx1 - xx0);
                    float h = Math.Max(0f, yy1 - yy0);
                    float inter = w * h;
                    float ovr = inter / (aArea + b.Width * b.Height - inter + 1e-6f);

                    if (ovr > iouThr) suppressed[j] = true;
                }
            }
            return keep;
        }

        /// <summary>
        /// net(정사각) 좌표계 사각형을 원본 이미지 좌표로 변환합니다.
        /// scale/padX/padY는 전처리(레터박스)에서 계산된 값이어야 합니다.
        /// </summary>
        /// <param name="netBox">넷 입력 좌표계 박스(l,t,w,h 또는 ltrb 형태의 RectangleF)</param>
        /// <param name="scale">전처리에서 사용한 축소 배율</param>
        /// <param name="padX">좌측 패딩(px)</param>
        /// <param name="padY">상단 패딩(px)</param>
        /// <param name="resized">스케일 적용 후(패딩 전) 크기</param>
        /// <param name="origSize">원본 이미지 크기</param>
        /// <returns>원본 좌표계 정수 사각형(Rectangle). 유효하지 않으면 Rectangle.Empty</returns>
        public static Rectangle NetBoxToOriginal(
            in RectangleF netBox,
            float scale,
            int padX,
            int padY,
            in Size resized,
            in Size origSize)
        {
            float invScale = 1f / Math.Max(1e-6f, scale);
            float l = (netBox.Left - padX) * invScale;
            float t = (netBox.Top - padY) * invScale;
            float r = (netBox.Right - padX) * invScale;
            float b = (netBox.Bottom - padY) * invScale;

            // 원본 경계로 클램프
            l = Math.Max(0, Math.Min(origSize.Width - 1, l));
            r = Math.Max(0, Math.Min(origSize.Width - 1, r));
            t = Math.Max(0, Math.Min(origSize.Height - 1, t));
            b = Math.Max(0, Math.Min(origSize.Height - 1, b));

            // 뒤집힌 경우 보정
            if (r < l) { var tmp = l; l = r; r = tmp; }
            if (b < t) { var tmp = t; t = b; b = tmp; }

            int x = (int)Math.Floor(l);
            int y = (int)Math.Floor(t);
            int w = (int)Math.Ceiling(r - l);
            int h = (int)Math.Ceiling(b - t);

            if (w <= 0 || h <= 0) return Rectangle.Empty;

            // 최종 보정(경계 초과 시 잘라냄)
            if (x + w > origSize.Width) w = origSize.Width - x;
            if (y + h > origSize.Height) h = origSize.Height - y;

            return new Rectangle(x, y, w, h);
        }
    }
}
