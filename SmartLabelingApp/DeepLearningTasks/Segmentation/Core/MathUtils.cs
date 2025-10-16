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
        /// <summary>실수(double) 클램프</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Clamp(double v, double lo, double hi)
        {
#if DEBUG
            if (lo > hi) throw new ArgumentException("Clamp bounds invalid: lo > hi");
#endif
            return v < lo ? lo : (v > hi ? hi : v);
        }

        /// <summary>0~1 범위로 고정 (double)</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static double Clamp(double v) => (v < 0) ? 0 : (v > 1 ? 1 : v);

        /// <summary>0~255 범위로 고정하고 byte로 변환</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static byte ClampByte(int v)
        {
            return (byte)(v < 0 ? 0 : (v > 255 ? 255 : v));
        }

        /// <summary>
        /// Math.Clamp() 대체용 — C# 8 이하 환경에서도 사용 가능.
        /// 제네릭으로 int, float, double 등 모든 비교형 지원.
        /// </summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        public static T MathClamp<T>(T value, T min, T max) where T : IComparable<T>
        {
#if DEBUG
            if (min.CompareTo(max) > 0)
                throw new ArgumentException("MathClamp bounds invalid: min > max");
#endif
            if (value.CompareTo(min) < 0) return min;
            if (value.CompareTo(max) > 0) return max;
            return value;
        }
    }
}
