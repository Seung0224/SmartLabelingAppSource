using System.Collections.Generic;
using System.Drawing;

namespace SmartLabelingApp
{
    public sealed class SegResult
    {
        // 전처리 메타
        public int NetSize { get; set; } = 640;
        public float Scale { get; set; }
        public int PadX { get; set; }
        public int PadY { get; set; }
        public Size Resized { get; set; }
        public Size OrigSize { get; set; }

        // 세그 헤드
        public int SegDim { get; set; }
        public int MaskH { get; set; }
        public int MaskW { get; set; }
        public float[] ProtoFlat { get; set; }   // KHW([K,H,W]) 표준

        // 디텍션
        public List<Det> Dets { get; set; }

        // 타이밍 (선택)
        public double PreMs { get; set; }
        public double InferMs { get; set; }
        public double PostMs { get; set; }
        public double TotalMs { get; set; }
    }
}
