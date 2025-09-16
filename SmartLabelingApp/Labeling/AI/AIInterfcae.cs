using System;
using System.Collections.Generic;
using System.Drawing;

namespace SmartLabelingApp.AI
{
    /// <summary>프롬프트 종류</summary>
    public enum PromptKind { Box, Points }

    /// <summary>포인트 프롬프트(전경/배경 클릭)</summary>
    public struct PromptPoint
    {
        public PointF Point;       // 이미지 좌표(px)
        public bool IsForeground;  // true=전경, false=배경
        public PromptPoint(PointF p, bool fg) { Point = p; IsForeground = fg; }
    }

    /// <summary>AI 세그멘테이션 입력 프롬프트</summary>
    public sealed class AISegmenterPrompt
    {
        public PromptKind Kind { get; set; }
        public RectangleF? Box { get; set; } // Kind==Box일 때 사용
        public List<PromptPoint> Points { get; set; } = new List<PromptPoint>(); // Kind==Points일 때 사용

        public static AISegmenterPrompt FromBox(RectangleF box)
            => new AISegmenterPrompt { Kind = PromptKind.Box, Box = box };

        public static AISegmenterPrompt FromPoints(IEnumerable<PromptPoint> pts)
            => new AISegmenterPrompt { Kind = PromptKind.Points, Points = new List<PromptPoint>(pts) };
    }

    /// <summary>
    /// 공통 옵션(알고리즘마다 일부만 사용 가능).
    /// ③(SAM)에서는 임베딩 캐시/입력 스케일 등 다른 항목을 추가로 무시/확장해도 무방.
    /// </summary>
    public sealed class AISegmenterOptions
    {
        /// <summary>후처리: 너무 작은 폴리곤 제거(픽셀 단위)</summary>
        public double MinAreaPx { get; set; } = 64;

        /// <summary>후처리: 폴리곤 단순화(epsilon, px)</summary>
        public double ApproxEpsilon { get; set; } = 2.0;

        /// <summary>후처리: 마스크 클로징 수행 여부</summary>
        public bool Smooth { get; set; } = true;

        /// <summary>후처리: 클로징 커널(홀수)</summary>
        public int CloseKernel { get; set; } = 3;

        // -------- 알고리즘 특화(필요 시 무시 가능) --------
        /// <summary>GrabCut 전용: 반복 횟수(1~5 권장)</summary>
        public int GrabCutIters { get; set; } = 2;
    }

    /// <summary>
    /// 세그멘테이션 공통 인터페이스
    /// - 반환: 다각형 리스트(각 점은 "이미지 좌표")
    /// </summary>
    public interface IAISegmenter
    {
        List<List<PointF>> Segment(Bitmap image, AISegmenterPrompt prompt, AISegmenterOptions options = null);
    }
}
