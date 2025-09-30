using System.Drawing;
using System.Drawing.Drawing2D;
using System.Drawing.Imaging;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp
{
    public static class ImagePreprocessor
    {
        public static DenseTensor<float> PreprocessToCHW(Bitmap src, PreprocessConfig pp)
        {
            using (var resized = BitmapUtils.ResizeKeepAspect(src, pp.resize))
            using (var cropped = BitmapUtils.CenterCrop(resized, pp.crop, pp.crop))
            {
                int H = pp.crop, W = pp.crop;
                var tensor = new DenseTensor<float>(new[] { 1, 3, H, W });

                // ---- 1) 채널 LUT 미리 계산 ----
                float inv255 = 1.0f / 255.0f;
                float rMul = inv255 / pp.std[0], gMul = inv255 / pp.std[1], bMul = inv255 / pp.std[2];
                float rAdd = -pp.mean[0] / pp.std[0], gAdd = -pp.mean[1] / pp.std[1], bAdd = -pp.mean[2] / pp.std[2];

                float[] lutR = new float[256], lutG = new float[256], lutB = new float[256];
                for (int i = 0; i < 256; i++)
                {
                    lutR[i] = i * rMul + rAdd;
                    lutG[i] = i * gMul + gAdd;
                    lutB[i] = i * bMul + bAdd;
                }

                // ---- 2) 텐서 버퍼에 직접 쓰기 (CHW) ----
                var buf = tensor.Buffer.Span;
                int plane = H * W;
                int rOff = 0, gOff = plane, bOff = 2 * plane;

                var rect = new Rectangle(0, 0, W, H);
                var bmpData = cropped.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    unsafe
                    {
                        byte* scan0 = (byte*)bmpData.Scan0;
                        int stride = bmpData.Stride;

                        int idxR = rOff, idxG = gOff, idxB = bOff;

                        for (int y = 0; y < H; y++)
                        {
                            byte* row = scan0 + y * stride;

                            int x3 = 0;
                            for (int x = 0; x < W; x++, x3 += 3)
                            {
                                byte b = row[x3 + 0];
                                byte g = row[x3 + 1];
                                byte r = row[x3 + 2];

                                buf[idxR++] = lutR[r];
                                buf[idxG++] = lutG[g];
                                buf[idxB++] = lutB[b];
                            }
                        }
                    }
                }
                finally
                {
                    cropped.UnlockBits(bmpData);
                }

                return tensor;
            }
        }

        public static DenseTensor<float> PreprocessToCHW(string imagePath, PreprocessConfig pp)
        {
            using (var src = new Bitmap(imagePath))
            using (var resized = BitmapUtils.ResizeKeepAspect(src, pp.resize))
            using (var cropped = BitmapUtils.CenterCrop(resized, pp.crop, pp.crop))
            {
                int H = pp.crop, W = pp.crop;
                var tensor = new DenseTensor<float>(new[] { 1, 3, H, W });

                // ---- 1) 채널 LUT(0..255) 미리 계산: (v/255 - mean)/std ----
                float inv255 = 1.0f / 255.0f;
                float rMul = inv255 / pp.std[0], gMul = inv255 / pp.std[1], bMul = inv255 / pp.std[2];
                float rAdd = -pp.mean[0] / pp.std[0], gAdd = -pp.mean[1] / pp.std[1], bAdd = -pp.mean[2] / pp.std[2];

                // LUT는 float[256] 하나로 충분하지만, 채널별 계수가 달라서 3개 만듦
                float[] lutR = new float[256], lutG = new float[256], lutB = new float[256];
                for (int i = 0; i < 256; i++)
                {
                    lutR[i] = i * rMul + rAdd;
                    lutG[i] = i * gMul + gAdd;
                    lutB[i] = i * bMul + bAdd;
                }

                // ---- 2) 텐서 버퍼에 직접 쓰기 (CHW) ----
                // CHW의 연속 버퍼: [R( H*W ), G( H*W ), B( H*W )]
                var buf = tensor.Buffer.Span; // .NET 4.8에서도 사용 가능
                int plane = H * W;
                int rOff = 0, gOff = plane, bOff = 2 * plane;

                var rect = new Rectangle(0, 0, W, H);
                var bmpData = cropped.LockBits(rect, ImageLockMode.ReadOnly, PixelFormat.Format24bppRgb);
                try
                {
                    unsafe
                    {
                        byte* scan0 = (byte*)bmpData.Scan0;
                        int stride = bmpData.Stride;

                        int idxR = rOff, idxG = gOff, idxB = bOff;

                        for (int y = 0; y < H; y++)
                        {
                            byte* row = scan0 + y * stride;

                            // x를 0..W-1 순회하며 BGR 읽고 LUT로 바로 쓰기
                            int x3 = 0;
                            for (int x = 0; x < W; x++, x3 += 3)
                            {
                                byte b = row[x3 + 0];
                                byte g = row[x3 + 1];
                                byte r = row[x3 + 2];

                                buf[idxR++] = lutR[r];
                                buf[idxG++] = lutG[g];
                                buf[idxB++] = lutB[b];
                            }
                        }
                    }
                }
                finally
                {
                    cropped.UnlockBits(bmpData);
                }

                return tensor;
            }
        }
    }
}
