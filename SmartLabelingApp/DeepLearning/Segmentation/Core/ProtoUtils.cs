using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace SmartLabelingApp
{
    public enum ProtoLayout { Unknown = 0, KHW = 1, HWK = 2 }

    public static class ProtoUtils
    {
        /// <summary>
        /// HWK([H, W, K]) → KHW([K, H, W]) 전치. dst.Length == K*H*W
        /// </summary>
        public static void TransposeHWKtoKHW(float[] srcHWK, int segDim, int mw, int mh, float[] dstKHW)
        {
            if (srcHWK == null || dstKHW == null)
                throw new ArgumentNullException("src/dst must not be null.");
            if (srcHWK.Length < segDim * mw * mh || dstKHW.Length < segDim * mw * mh)
                throw new ArgumentException("buffer length mismatch");

            // src: idxHWK = ((y * mw) + x) * segDim + k
            // dst: idxKHW = ((k * mh) + y) * mw + x
            int idx = 0;
            for (int y = 0; y < mh; y++)
            {
                for (int x = 0; x < mw; x++, idx++)
                {
                    int baseHWK = idx * segDim;
                    for (int k = 0; k < segDim; k++)
                    {
                        int dst = ((k * mh) + y) * mw + x;
                        dstKHW[dst] = srcHWK[baseHWK + k];
                    }
                }
            }
        }

        /// <summary>
        /// 두 레이아웃 중 어느 쪽이 "구조성"(분산) 높게 나오는지 비교해 레이아웃을 추정합니다.
        /// (감지는 1회만 호출하고 결과를 캐시하세요)
        /// </summary>
        public static ProtoLayout DetectByVariance(
            float[] coeffSample, float[] protoFlat, int segDim, int mw, int mh)
        {
            int len = mw * mh;
            var a = new float[len];
            var b = new float[len];

            MaskSynth.ComputeMask_KHW_NoAlloc(coeffSample, protoFlat, segDim, mw, mh, a);
            MaskSynth.ComputeMask_HWK_NoAlloc(coeffSample, protoFlat, segDim, mw, mh, b);

            double meanA = 0, meanB = 0;
            for (int i = 0; i < len; i++) { meanA += a[i]; meanB += b[i]; }
            meanA /= len; meanB /= len;

            double varA = 0, varB = 0;
            for (int i = 0; i < len; i++)
            {
                double da = a[i] - meanA; varA += da * da;
                double db = b[i] - meanB; varB += db * db;
            }
            varA /= len; varB /= len;

            return varA >= varB ? ProtoLayout.KHW : ProtoLayout.HWK;
        }
    }
}
