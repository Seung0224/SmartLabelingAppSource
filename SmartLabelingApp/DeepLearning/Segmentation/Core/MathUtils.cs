using System;
using System.Runtime.CompilerServices;

namespace SmartLabelingApp
{
    /// <summary>
    /// 엔진/ONNX 공통으로 사용하는 순수 수학/보조 함수 모음.
    /// 상태가 없고, 스레드-세이프합니다.
    /// </summary>
    public static class MathUtils
    {
        /// <summary>
        /// 시그모이드: 1 / (1 + exp(-x))
        /// 매우 큰 양/음수에서도 안정적으로 동작하도록 분기합니다.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Sigmoid(float x)
        {
            // 수치 안정성 분기: 큰 음수에서 exp(-x) overflow 방지
            if (x >= 0f)
            {
                float z = (float)Math.Exp(-x);
                return 1f / (1f + z);
            }
            else
            {
                float z = (float)Math.Exp(x);
                return z / (1f + z);
            }
        }

        /// <summary>정수 클램프</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static int Clamp(int v, int lo, int hi)
        {
#if DEBUG
            if (lo > hi) throw new ArgumentException("Clamp bounds invalid: lo > hi");
#endif
            return v < lo ? lo : (v > hi ? hi : v);
        }

        /// <summary>부동소수점 클램프</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static float Clamp(float v, float lo, float hi)
        {
#if DEBUG
            if (lo > hi) throw new ArgumentException("Clamp bounds invalid: lo > hi");
#endif
            return v < lo ? lo : (v > hi ? hi : v);
        }

        /// <summary>0~255 범위로 고정하고 byte로 변환</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ClampByte(int v)
        {
            return (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }
    }
}
