using System.Drawing;

namespace SmartLabelingApp
{
    public sealed class Det
    {
        public RectangleF Box { get; set; }   // net 좌표계 (l,t,w,h로 사용)
        public float Score { get; set; }      // 0..1
        public int ClassId { get; set; }      // 클래스 인덱스
        public float[] Coeff { get; set; }    // 길이 = SegDim
    }
}
