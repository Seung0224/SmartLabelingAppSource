namespace SmartLabelingApp
{
    public class PreprocessConfig
    {
        public int resize = 256;    // 짧은 변 기준 리사이즈
        public int crop = 224;    // 센터 크롭 크기
        public float[] mean = { 0.485f, 0.456f, 0.406f };
        public float[] std = { 0.229f, 0.224f, 0.225f };
    }
}