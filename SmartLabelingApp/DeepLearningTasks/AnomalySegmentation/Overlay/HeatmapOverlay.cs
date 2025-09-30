using System;
using System.Drawing;
using System.Drawing.Imaging;

namespace SmartLabelingApp
{
    public static class HeatmapOverlay
    {
        public static Bitmap MakeOverlay(Bitmap src, float[] patchMin, int gridH, int gridW,
            float clipQ = 0.98f, float gamma = 1.8f,
            float alphaMin = 0.02f, float alphaMax = 0.5f)
        {
            int srcW = src.Width, srcH = src.Height;

            // 1) 14x14 → 2D 배열
            float[,] map = new float[gridH, gridW];
            int p = 0;
            for (int i = 0; i < gridH; i++)
                for (int j = 0; j < gridW; j++)
                    map[i, j] = patchMin[p++];

            // 2) 업샘플 (nearest bilinear 간단구현)
            float[,] big = BilinearResize(map, srcW, srcH);

            // 3) 정규화 + 클리핑
            float[] flat = new float[srcW * srcH];
            Buffer.BlockCopy(big, 0, flat, 0, flat.Length * sizeof(float));
            Array.Sort(flat);
            float thr = flat[(int)(flat.Length * clipQ)];
            for (int i = 0; i < srcH; i++)
                for (int j = 0; j < srcW; j++)
                {
                    float v = big[i, j];
                    v = Math.Min(v, thr) / thr;       // [0,1]
                    v = (float)Math.Pow(v, gamma);   // 감마 강조
                    big[i, j] = v;
                }

            // 4) Overlay
            Bitmap dst = new Bitmap(srcW, srcH, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
                g.DrawImage(src, 0, 0, srcW, srcH);

            BitmapData bd = dst.LockBits(new Rectangle(0, 0, srcW, srcH),
                                         ImageLockMode.ReadWrite,
                                         PixelFormat.Format24bppRgb);
            unsafe
            {
                byte* ptr = (byte*)bd.Scan0;
                for (int y = 0; y < srcH; y++)
                {
                    for (int x = 0; x < srcW; x++)
                    {
                        float v = big[y, x];
                        var (r, g, b) = Jet(v);
                        float alpha = alphaMin + v * (alphaMax - alphaMin);

                        byte* pPix = ptr + y * bd.Stride + x * 3;
                        pPix[2] = (byte)(pPix[2] * (1 - alpha) + r * alpha); // R
                        pPix[1] = (byte)(pPix[1] * (1 - alpha) + g * alpha); // G
                        pPix[0] = (byte)(pPix[0] * (1 - alpha) + b * alpha); // B
                    }
                }
            }
            dst.UnlockBits(bd);
            return dst;
        }

        private static float[,] BilinearResize(float[,] src, int W, int H)
        {
            int h = src.GetLength(0), w = src.GetLength(1);
            float[,] dst = new float[H, W];
            for (int y = 0; y < H; y++)
            {
                float fy = (float)(y) / (H - 1) * (h - 1);
                int y0 = (int)Math.Floor(fy), y1 = Math.Min(y0 + 1, h - 1);
                float ly = fy - y0;
                for (int x = 0; x < W; x++)
                {
                    float fx = (float)(x) / (W - 1) * (w - 1);
                    int x0 = (int)Math.Floor(fx), x1 = Math.Min(x0 + 1, w - 1);
                    float lx = fx - x0;

                    float v00 = src[y0, x0], v01 = src[y0, x1];
                    float v10 = src[y1, x0], v11 = src[y1, x1];
                    float v0 = v00 * (1 - lx) + v01 * lx;
                    float v1 = v10 * (1 - lx) + v11 * lx;
                    dst[y, x] = v0 * (1 - ly) + v1 * ly;
                }
            }
            return dst;
        }

        private static (byte r, byte g, byte b) Jet(float v)
        {
            v = Math.Max(0, Math.Min(1, v));
            double four = 4.0 * v;
            byte r = (byte)(255 * Math.Min(Math.Max(Math.Min(four - 1.5, -four + 4.5), 0), 1));
            byte g = (byte)(255 * Math.Min(Math.Max(Math.Min(four - 0.5, -four + 3.5), 0), 1));
            byte b = (byte)(255 * Math.Min(Math.Max(Math.Min(four + 0.5, -four + 2.5), 0), 1));
            return (r, g, b);
        }
    }
}
