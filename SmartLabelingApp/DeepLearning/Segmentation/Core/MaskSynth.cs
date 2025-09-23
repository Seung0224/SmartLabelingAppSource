using System;
using System.Numerics;
using System.Threading.Tasks;

namespace SmartLabelingApp
{    /// <summary>
     /// Proto·Coeff → Mask 합성 공통 모듈.
     /// 표준 레이아웃은 KHW([K, H, W])를 사용합니다.
     /// </summary>
    public static class MaskSynth
    {
        /// <summary>
        /// KHW([K, H, W]) 레이아웃에서 마스크를 계산하여 maskOut(H*W)에 기록합니다.
        /// mask[y,x] = sigmoid( Σ_k coeff[k] * proto[k, y, x] )
        /// </summary>
        public static void ComputeMask_KHW_NoAlloc(
            float[] coeff, float[] protoFlatKHW, int segDim, int mw, int mh, float[] maskOut)
        {
            if (coeff == null || protoFlatKHW == null || maskOut == null)
                throw new ArgumentNullException("coeff/proto/maskOut must not be null.");
            if (coeff.Length < segDim) throw new ArgumentException("coeff length < segDim");
            if (protoFlatKHW.Length < segDim * mw * mh) throw new ArgumentException("proto length mismatch");
            if (maskOut.Length < mw * mh) throw new ArgumentException("maskOut length mismatch");

            int vec = Vector<float>.Count;

            Parallel.For(0, mh, y =>
            {
                int rowOff = y * mw;
                int x = 0;

                for (; x <= mw - vec; x += vec)
                {
                    var sumV = new Vector<float>(0f);

                    for (int k = 0; k < segDim; k++)
                    {
                        int off = ((k * mh + y) * mw) + x;           // [K, H, W] → flat index
                        var pV = new Vector<float>(protoFlatKHW, off);
                        sumV += pV * new Vector<float>(coeff[k]);
                    }

                    for (int t = 0; t < vec; t++)
                    {
                        float s = sumV[t];
                        maskOut[rowOff + x + t] = MathUtils.Sigmoid(s);
                    }
                }

                for (; x < mw; x++)
                {
                    float sum = 0f;
                    int baseIdx = y * mw + x;
                    for (int k = 0; k < segDim; k++)
                        sum += coeff[k] * protoFlatKHW[(k * mh + y) * mw + x];
                    maskOut[rowOff + x] = MathUtils.Sigmoid(sum);
                }
            });
        }

        /// <summary>
        /// HWK([H, W, K]) 레이아웃에서 마스크 계산(감지/전치 전 1회 비교용). 일반 사용은 지양.
        /// </summary>
        public static void ComputeMask_HWK_NoAlloc(
            float[] coeff, float[] protoFlatHWK, int segDim, int mw, int mh, float[] maskOut)
        {
            if (coeff == null || protoFlatHWK == null || maskOut == null)
                throw new ArgumentNullException("coeff/proto/maskOut must not be null.");
            if (coeff.Length < segDim) throw new ArgumentException("coeff length < segDim");
            if (protoFlatHWK.Length < segDim * mw * mh) throw new ArgumentException("proto length mismatch");
            if (maskOut.Length < mw * mh) throw new ArgumentException("maskOut length mismatch");

            Parallel.For(0, mh, y =>
            {
                int row = y * mw;
                for (int x = 0; x < mw; x++)
                {
                    float sum = 0f;
                    int baseIdx = (y * mw + x) * segDim; // [H, W, K]
                    for (int k = 0; k < segDim; k++)
                        sum += coeff[k] * protoFlatHWK[baseIdx + k];
                    maskOut[row + x] = MathUtils.Sigmoid(sum);
                }
            });
        }
    }
}
