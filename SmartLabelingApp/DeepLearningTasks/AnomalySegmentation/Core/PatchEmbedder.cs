using System;
using System.Linq;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp
{
    /// <summary>
    /// L3-only PatchCore 임베딩 생성 유틸.
    /// ONNX 출력(layer3: [1, 1024, 14, 14]) → (196 × 1024) row-major + 행 L2 정규화.
    /// </summary>
    public static class PatchEmbedder
    {
        /// <summary>
        /// ONNX의 layer3 텐서(형상 [1, C=1024, H=14, W=14])에서
        /// 패치 임베딩 (P=H*W=196, D=1024) 를 생성한다. 각 행은 L2 정규화된다.
        /// </summary>
        public static PatchEmbeddings BuildFromL3(Tensor<float> l3t)
        {
            var l3 = l3t as DenseTensor<float> ?? ToDense(l3t);
            CheckShape4D(l3, "l3");

            int n = l3.Dimensions[0];               // 1
            int c = l3.Dimensions[1];               // 1024
            int h = l3.Dimensions[2];               // 14
            int w = l3.Dimensions[3];               // 14
            if (n != 1) throw new ArgumentException($"Expected batch=1, but got {n}.");

            // CHW contiguous 배열로 평탄화
            float[] chw = ToCHWArray(l3);

            // (H*W, C) = (196, 1024) row-major
            float[] rows = CHW_to_PatchesRowMajor(chw, c, h, w);

            // 행별 L2 정규화
            L2NormalizeRowsInPlace(rows, h * w, c);

            return new PatchEmbeddings
            {
                GridH = h,
                GridW = w,
                Dim = c,           // 1024
                Patches = h * w,   // 196
                RowsRowMajor = rows
            };
        }

        // -------------------- helpers --------------------

        private static DenseTensor<float> ToDense(Tensor<float> t)
        {
            // Tensor<float> 를 DenseTensor<float>로 복제
            var dims = t.Dimensions.ToArray();
            var dense = new DenseTensor<float>(dims);
            var spanDst = dense.Buffer.Span;
            int idx = 0;
            foreach (var v in t) spanDst[idx++] = v;
            return dense;
        }

        private static void CheckShape4D(DenseTensor<float> t, string name)
        {
            if (t.Rank != 4)
                throw new ArgumentException($"{name} must be rank-4 tensor, but got rank={t.Rank}.");

            if (t.Dimensions[1] <= 0 || t.Dimensions[2] <= 0 || t.Dimensions[3] <= 0)
                throw new ArgumentException($"{name} has invalid dims: [{DimsToString(t.Dimensions)}]");
        }

        private static string DimsToString(ReadOnlySpan<int> dims)
        {
            var arr = new int[dims.Length];
            dims.CopyTo(arr);
            // .NET Framework에선 string.Join<T>(...) 오버로드가 부족하므로 문자열로 변환
            return string.Join(",", arr.Select(x => x.ToString()).ToArray());
        }

        /// <summary>
        /// DenseTensor(N, C, H, W)를 CHW 순으로 1D 배열로 평탄화 (N=1 전제).
        /// 메모리 레이아웃이 이미 연속(contiguous)이지만, 안전하게 새 배열 생성.
        /// </summary>
        private static float[] ToCHWArray(DenseTensor<float> t)
        {
            int n = t.Dimensions[0];
            int c = t.Dimensions[1];
            int h = t.Dimensions[2];
            int w = t.Dimensions[3];
            if (n != 1) throw new ArgumentException($"Expected batch=1, but got {n}.");

            int total = c * h * w;
            var arr = new float[total];

            // DenseTensor는 기본적으로 row-major이지만, 인덱서로 안전하게 복사
            // 인덱스: [0, c, h, w]
            int k = 0;
            for (int ci = 0; ci < c; ci++)
                for (int hi = 0; hi < h; hi++)
                    for (int wi = 0; wi < w; wi++)
                        arr[k++] = t[0, ci, hi, wi];

            return arr;
        }

        /// <summary>
        /// CHW 배열(C,H,W)을 (H*W, C) row-major로 변환한다.
        /// 즉, 각 (h,w) 위치의 채널 벡터 길이 C를 한 행으로 나열.
        /// </summary>
        private static float[] CHW_to_PatchesRowMajor(float[] chw, int c, int h, int w)
        {
            int patches = h * w;     // 196
            var rows = new float[patches * c];

            // CHW에서 픽셀 순서: [c, h, w]
            // 행 인덱스 p = h*W + w
            // CHW index base = c*(h*W + w)
            int p = 0;
            for (int hi = 0; hi < h; hi++)
            {
                for (int wi = 0; wi < w; wi++)
                {
                    int baseCHW = (hi * w + wi) * c; // but current chw layout is (c,h,w) flattened as looped above
                    // 위 ToCHWArray에서 순서가 ci→hi→wi 이므로,
                    // 현재 chw의 인덱스는 [ci*(h*w) + hi*w + wi].
                    // 따라서 여기선 다시 계산:
                    int rowOff = p * c;
                    for (int ci = 0; ci < c; ci++)
                    {
                        int idxCHW = ci * (h * w) + hi * w + wi;
                        rows[rowOff + ci] = chw[idxCHW];
                    }
                    p++;
                }
            }
            return rows;
        }

        /// <summary>
        /// rows: (Patches × Dim) row-major 배열의 각 행을 L2 정규화(in-place).
        /// </summary>
        public static void L2NormalizeRowsInPlace(float[] rows, int patches, int dim, float eps = 1e-12f)
        {
            int off = 0;
            for (int p = 0; p < patches; p++)
            {
                double ss = 0.0;
                int end = off + dim;
                for (int i = off; i < end; i++) ss += (double)rows[i] * rows[i];
                float inv = (float)(1.0 / Math.Sqrt(Math.Max(ss, eps)));
                for (int i = off; i < end; i++) rows[i] *= inv;
                off = end;
            }
        }
    }

    /// <summary>
    /// 패치 임베딩 컨테이너: (Patches × Dim) row-major.
    /// </summary>
    public sealed class PatchEmbeddings
    {
        public int GridH { get; set; }     // 14
        public int GridW { get; set; }     // 14
        public int Dim { get; set; }     // 1024
        public int Patches { get; set; }   // 196
        public float[] RowsRowMajor { get; set; } // length = Patches * Dim
    }
}
