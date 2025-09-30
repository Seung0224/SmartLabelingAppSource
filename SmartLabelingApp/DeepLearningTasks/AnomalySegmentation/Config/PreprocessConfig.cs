namespace SmartLabelingApp
{
    public class PreprocessConfig
    {
        public int resize = 0;
        public int crop = 0;
        public float[] mean = { 0.485f, 0.456f, 0.406f };
        public float[] std = { 0.229f, 0.224f, 0.225f };
    }
}