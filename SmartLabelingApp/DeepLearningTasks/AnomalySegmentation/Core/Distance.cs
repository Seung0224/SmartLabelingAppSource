using MathNet.Numerics.LinearAlgebra;
using System;
using System.Numerics;
using System.Threading.Tasks;

namespace SmartLabelingApp
{ 
    public static class Distance
    {
        // rowsA: (na x d), rowsB: (nb x d) — 모두 row-major
        // metric = "ip" (cosine/IP: A,B는 L2 정규화 전제 → 1 - dot), or "l2"
        public static void RowwiseMinDistances(
            float[] rowsA, int na, int d,
            float[] rowsB, int nb,
            string metric,
            float[] outMin) // length = na
        {
            if (metric == null) metric = "ip";
            bool useCos = metric.Equals("ip", StringComparison.OrdinalIgnoreCase);

            for (int i = 0; i < na; i++)
            {
                int offA = i * d;
                float best = float.PositiveInfinity;

                for (int j = 0; j < nb; j++)
                {
                    int offB = j * d;

                    float acc = 0f;
                    if (useCos)
                    {
                        // cosine/IP: 거리 = 1 - dot(A,B)  (A,B는 이미 L2-normalized)
                        for (int k = 0; k < d; k++) acc += rowsA[offA + k] * rowsB[offB + k];
                        float dist = 1f - acc;
                        if (dist < best) best = dist;
                    }
                    else
                    {
                        // L2: ||A - B||2
                        for (int k = 0; k < d; k++)
                        {
                            float diff = rowsA[offA + k] - rowsB[offB + k];
                            acc += diff * diff;
                        }
                        float dist = (float)Math.Sqrt(acc);
                        if (dist < best) best = dist;
                    }
                }
                outMin[i] = best;
            }
        }

        /// <summary>
        /// 코사인 거리(=1 - dot), 쿼리/갤러리 모두 L2-정규화 전제.
        /// Q: P×D (row-major), G: N×D (row-major)
        /// outMin[p] = 1 - max_n dot(Q[p], G[n])
        /// </summary>
        public static void RowwiseMinDistancesIP_Optimized(
            float[] Q, int P, int D,
            float[] G, int N,
            float[] outMin)
        {
            if (outMin.Length < P) throw new ArgumentException(nameof(outMin));

            int W = System.Numerics.Vector<float>.Count; // 보통 AVX2면 8
            Parallel.For(0, P, p =>
            {
                int qOff = p * D;
                float maxDot = float.NegativeInfinity;

                // 갤러리 한 행씩 훑기
                for (int n = 0; n < N; n++)
                {
                    int gOff = n * D;
                    float dot = 0f;

                    int k = 0;
                    // SIMD 구간
                    for (; k <= D - W; k += W)
                    {
                        var vq = new System.Numerics.Vector<float>(Q, qOff + k);
                        var vg = new System.Numerics.Vector<float>(G, gOff + k);
                        dot += Vector.Dot(vq, vg);
                    }
                    // 꼬리 처리
                    for (; k < D; k++)
                        dot += Q[qOff + k] * G[gOff + k];

                    if (dot > maxDot) maxDot = dot;
                }
                outMin[p] = 1.0f - maxDot; // cosine distance
            });
        }

        /// <summary>
        /// 코사인 거리(정규화 전제): S = Q·Gᵀ (MKL GEMM) 후, outMin[p] = 1 - max_j S[p,j]
        /// Q: (P×D) row-major, G: (N×D) row-major
        /// </summary>
        public static void RowwiseMinDistancesIP_MKL(float[] Q, int P, int D, float[] G, int N, float[] outMin)
        {
            if (outMin.Length < P) throw new ArgumentException(nameof(outMin));

            // Math.NET 내부는 column-major이지만, 
            // G(row-major, N×D)를 그대로 넘기면 "GT(column-major, D×N)"로 해석됩니다.
            // 즉: DenseOfColumnMajor(D, N, G) == (Gᵀ)의 열메모리 레이아웃과 동일!
            var M = Matrix<float>.Build;
            var Qm = M.DenseOfRowMajor(P, D, Q);      // (P×D)
            var GTm = M.DenseOfColumnMajor(D, N, G);   // (D×N) == Gᵀ

            // S = Q · Gᵀ  → (P×N)
            var S = Qm * GTm;

            // 각 행의 max → 1 - max
            for (int p = 0; p < P; p++)
            {
                var row = S.Row(p);
                float maxDot = float.NegativeInfinity;
                for (int j = 0; j < row.Count; j++)
                    if (row[j] > maxDot) maxDot = row[j];
                outMin[p] = 1.0f - maxDot;
            }
        }
    }
}
