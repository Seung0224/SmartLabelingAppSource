using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public static class HeatmapOverlay
    {
        // === 공용 엔트리: 시그니처 동일 ===
        public static Bitmap MakeOverlay(
            Bitmap src, float[] patchMin, int gridH, int gridW,
            float clipQ = 0.98f, float gamma = 1.8f,
            float alphaMin = 0.02f, float alphaMax = 0.5f)
        {
            int W = src.Width, H = src.Height;

            // 1) (gridH x gridW) -> (H x W) 분리형 Bilinear 업샘플 (1D 버퍼)
            float[] up = SeparableBilinearResize(patchMin, gridH, gridW, H, W);

            // 2) clipQ 분위수(thr) 계산 (QuickSelect; 정렬 없음)
            float thr = QuantileSelect(up, clipQ);
            if (thr <= 1e-12f) thr = 1e-12f;

            // 3) LUT 준비 (감마/컬러맵) — 호출마다 생성해도 가볍지만,
            //    빈번히 쓰면 static 캐시로 바꿔도 OK
            const int LUTN = 1024;
            var gammaLut = BuildGammaLut(LUTN, gamma);
            var jetLut = BuildJetLut(LUTN); // (r,g,b) byte LUT

            // 4) 오버레이 (행 병렬)
            Bitmap dst = new Bitmap(W, H, PixelFormat.Format24bppRgb);
            using (Graphics g = Graphics.FromImage(dst))
                g.DrawImage(src, 0, 0, W, H);

            BitmapData bd = dst.LockBits(new Rectangle(0, 0, W, H),
                                         ImageLockMode.ReadWrite,
                                         PixelFormat.Format24bppRgb);

            float alphaScale = (alphaMax - alphaMin);
            try
            {
                unsafe
                {
                    byte* basePtr = (byte*)bd.Scan0;
                    int stride = bd.Stride;

                    Parallel.For(0, H, y =>
                    {
                        byte* row = basePtr + y * stride;
                        int idx = y * W;
                        for (int x = 0; x < W; x++, idx++)
                        {
                            // 정규화 + 클리핑
                            float v = up[idx];
                            if (v > thr) v = thr;
                            v = v / thr; // 0..1

                            // 감마 LUT
                            int li = (int)(v * (LUTN - 1) + 0.5f);
                            if ((uint)li >= LUTN) li = (li < 0) ? 0 : (LUTN - 1);
                            float vg = gammaLut[li];

                            // alpha
                            float a = alphaMin + vg * alphaScale;

                            // JET LUT
                            var (r, gch, b) = jetLut[li];

                            // blend
                            byte* pPix = row + x * 3;
                            pPix[2] = (byte)(pPix[2] * (1 - a) + r * a); // R
                            pPix[1] = (byte)(pPix[1] * (1 - a) + gch * a); // G
                            pPix[0] = (byte)(pPix[0] * (1 - a) + b * a); // B
                        }
                    });
                }
            }
            finally
            {
                dst.UnlockBits(bd);
            }
            return dst;
        }

        // === 분리형 Bilinear 업샘플 (grid -> 이미지) ===
        // in: srcGrid (row-major, length = gridH*gridW)
        private static float[] SeparableBilinearResize(float[] srcGrid, int gh, int gw, int H, int W)
        {
            // 가로 리사이즈: gh 행 각각 gw -> W
            float[] tmp = new float[gh * W];

            // 미리 가로 매핑 인덱스/가중치 계산
            var x0 = new int[W]; var x1 = new int[W]; var lx = new float[W];
            for (int x = 0; x < W; x++)
            {
                float fx = (W == 1) ? 0f : (float)x / (W - 1) * (gw - 1);
                int ix0 = (int)Math.Floor(fx);
                int ix1 = Math.Min(ix0 + 1, gw - 1);
                x0[x] = ix0; x1[x] = ix1; lx[x] = fx - ix0;
            }

            for (int y = 0; y < gh; y++)
            {
                int rowIn = y * gw;
                int rowOut = y * W;
                for (int x = 0; x < W; x++)
                {
                    float v0 = srcGrid[rowIn + x0[x]];
                    float v1 = srcGrid[rowIn + x1[x]];
                    tmp[rowOut + x] = v0 * (1 - lx[x]) + v1 * lx[x];
                }
            }

            // 세로 리사이즈: gh -> H (각 열 독립)
            float[] dst = new float[H * W];
            var y0 = new int[H]; var y1 = new int[H]; var ly = new float[H];
            for (int y = 0; y < H; y++)
            {
                float fy = (H == 1) ? 0f : (float)y / (H - 1) * (gh - 1);
                int iy0 = (int)Math.Floor(fy);
                int iy1 = Math.Min(iy0 + 1, gh - 1);
                y0[y] = iy0; y1[y] = iy1; ly[y] = fy - iy0;
            }

            for (int y = 0; y < H; y++)
            {
                int rowOut = y * W;
                int row0 = y0[y] * W;
                int row1 = y1[y] * W;
                float wy = ly[y];
                float wy0 = 1 - wy, wy1 = wy;
                for (int x = 0; x < W; x++)
                {
                    float v0 = tmp[row0 + x];
                    float v1 = tmp[row1 + x];
                    dst[rowOut + x] = v0 * wy0 + v1 * wy1;
                }
            }
            return dst;
        }

        // === clipQ 분위수 (nth_element/QuickSelect) ===
        private static float QuantileSelect(float[] data, float q)
        {
            int n = data.Length;
            int k = (int)(q * (n - 1) + 0.5f);
            // 복사본 위에서 선택 (원본 보존)
            var arr = new float[n];
            Array.Copy(data, arr, n);
            return QuickSelect(arr, 0, n - 1, k);
        }

        private static float QuickSelect(float[] a, int left, int right, int k)
        {
            while (true)
            {
                if (left == right) return a[left];
                int pivotIndex = Partition(a, left, right, (left + right) >> 1);
                if (k == pivotIndex) return a[k];
                if (k < pivotIndex) right = pivotIndex - 1;
                else left = pivotIndex + 1;
            }
        }

        private static int Partition(float[] a, int left, int right, int pivotIndex)
        {
            float pivot = a[pivotIndex];
            (a[pivotIndex], a[right]) = (a[right], a[pivotIndex]);
            int store = left;
            for (int i = left; i < right; i++)
            {
                if (a[i] < pivot)
                {
                    (a[store], a[i]) = (a[i], a[store]);
                    store++;
                }
            }
            (a[right], a[store]) = (a[store], a[right]);
            return store;
        }

        // === 감마 LUT (0..1 -> 0..1) ===
        private static float[] BuildGammaLut(int N, float gamma)
        {
            var lut = new float[N];
            float inv = 1f / (N - 1);
            for (int i = 0; i < N; i++)
            {
                float v = i * inv;                 // 0..1
                lut[i] = (float)Math.Pow(v, gamma);
            }
            return lut;
        }

        // === JET LUT (0..1 -> (r,g,b) byte) : 원래와 동일한 수식 기반 ===
        private static (byte r, byte g, byte b)[] BuildJetLut(int N)
        {
            var lut = new (byte r, byte g, byte b)[N];
            double fourStep = 4.0;
            double inv = 1.0 / (N - 1);
            for (int i = 0; i < N; i++)
            {
                double v = i * inv;               // 0..1
                double four = fourStep * v;
                byte r = (byte)(255 * Clamp01(Math.Min(four - 1.5, -four + 4.5)));
                byte g = (byte)(255 * Clamp01(Math.Min(four - 0.5, -four + 3.5)));
                byte b = (byte)(255 * Clamp01(Math.Min(four + 0.5, -four + 2.5)));
                lut[i] = (r, g, b);
            }
            return lut;

            double Clamp01(double x) => x < 0 ? 0 : (x > 1 ? 1 : x);
        }
    }
}
