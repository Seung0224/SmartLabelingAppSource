using System;
using System.Linq;

namespace SmartLabelingApp
{
    public sealed class AnomalyResult
    {
        public float ImageScore;   // 패치 min-dist의 max
        public float[] PatchMin;     // 길이 = patches
        public bool IsAnomaly;
    }

    public static class AnomalyScorer
    {
        /// <summary>
        /// rowsRowMajor: (patches × d) row-major, 각 행은 L2 정규화(코사인 가정).
        /// gallery: (ntotal × d) row-major, 각 행은 L2 정규화(코사인 가정).
        /// metric: "ip"(=cosine: 1 - dot) 또는 기존 Distance에서 지원하는 값.
        /// </summary>
        public static AnomalyResult ScoreImage(
            float[] rowsRowMajor, int patches, int d,
            float[] gallery, int ntotal,
            string metric, float threshold)
        {
            if (rowsRowMajor == null || rowsRowMajor.Length == 0)
                throw new ArgumentException("rowsRowMajor is empty");
            if (gallery == null || gallery.Length == 0)
                throw new ArgumentException("gallery is empty");
            if (d <= 0) throw new ArgumentException("d must be > 0");
            if (ntotal <= 0) throw new ArgumentException("ntotal must be > 0");

            // rowsRowMajor 길이로부터 patches 재확인
            if (rowsRowMajor.Length % d != 0)
                throw new InvalidOperationException(
                    $"rowsRowMajor length {rowsRowMajor.Length} not divisible by dim {d}");
            int inferredPatches = rowsRowMajor.Length / d;
            if (patches != inferredPatches)
            {
                Console.WriteLine($"[WARN] patches({patches}) → inferred({inferredPatches})로 수정");
                patches = inferredPatches;
            }

            // gallery 길이로부터 dim 재확인
            if (gallery.Length % ntotal != 0)
                throw new InvalidOperationException(
                    $"gallery length {gallery.Length} not divisible by ntotal {ntotal}");
            int inferredDim = gallery.Length / ntotal;
            if (inferredDim != d)
                throw new InvalidOperationException(
                    $"[DIM MISMATCH] patch dim(d)={d}, gallery dim={inferredDim} — " +
                    $"export 산출물과 현재 ONNX/임베딩이 일치하는지 확인하세요.");

            var patchMin = new float[patches];

            // === 거리 계산 엔진 선택 ===
            metric = (metric ?? "ip").Trim().ToLowerInvariant();

            if (metric == "ip")
            {
                // 1) MKL GEMM (가장 빠름) — Distance.cs에 RowwiseMinDistancesIP_MKL가 추가되어 있어야 함
                if (HasMethod(nameof(Distance.RowwiseMinDistancesIP_MKL)))
                {
                    try
                    {
                        Distance.RowwiseMinDistancesIP_MKL(
                            rowsRowMajor, patches, d,
                            gallery, ntotal,
                            patchMin);
                    }
                    catch (TypeLoadException) // MathNet.MKL이 실제로 없을 때 대비
                    {
                        FallbackSIMDOrNaive(rowsRowMajor, patches, d, gallery, ntotal, patchMin);
                    }
                }
                else
                {
                    FallbackSIMDOrNaive(rowsRowMajor, patches, d, gallery, ntotal, patchMin);
                }
            }
            else
            {
                // (필요 시 다른 metric 경로 그대로 유지)
                Distance.RowwiseMinDistances(
                    rowsRowMajor, patches, d,
                    gallery, ntotal,
                    metric, patchMin);
            }

            float imgScore = patchMin.Max();
            bool isAnom = imgScore > threshold;

            return new AnomalyResult
            {
                ImageScore = imgScore,
                PatchMin = patchMin,
                IsAnomaly = isAnom
            };
        }

        private static void FallbackSIMDOrNaive(
            float[] Q, int P, int D,
            float[] G, int N,
            float[] outMin)
        {
            // 2) SIMD 최적화 버전이 있으면 사용
            if (HasMethod(nameof(Distance.RowwiseMinDistancesIP_Optimized)))
            {
                Distance.RowwiseMinDistancesIP_Optimized(Q, P, D, G, N, outMin);
                return;
            }
            // 3) 마지막 폴백: 기존 브루트포스 (metric="ip")
            Distance.RowwiseMinDistances(Q, P, D, G, N, "ip", outMin);
        }

        private static bool HasMethod(string name)
        {
            var m = typeof(Distance).GetMethod(name,
                System.Reflection.BindingFlags.Public |
                System.Reflection.BindingFlags.Static);
            return m != null;
        }
    }
}
