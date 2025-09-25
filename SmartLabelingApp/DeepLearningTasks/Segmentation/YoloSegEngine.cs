using System;
using System.Runtime.InteropServices;

namespace SmartLabelingApp
{
    /// <summary>
    /// TensorRT 엔진 래퍼: 생성/해제 + 네이티브 핸들/입력 크기 노출만 담당
    /// (추론은 SegmentationInfer가 전담)
    /// </summary>
    public sealed class YoloSegEngine : IDisposable
    {
        private const string DllName = "TensorRTRunner";

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl, CharSet = CharSet.Ansi)]
        private static extern IntPtr trt_create_engine(string enginePath, int deviceId);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern void trt_destroy_engine(IntPtr handle);

        [DllImport(DllName, CallingConvention = CallingConvention.Cdecl)]
        private static extern int trt_get_input_size(IntPtr handle);

        private IntPtr _handle = IntPtr.Zero;

        /// <summary>네이티브 TensorRT 핸들(IntPtr)</summary>
        public IntPtr Handle => _handle;

        /// <summary>정사각 고정 입력이면 사이즈, 동적이면 -1</summary>
        public int FixedInputNet { get; }

        public int DeviceId { get; }

        public YoloSegEngine(string enginePath, int deviceId = 0)
        {
            DeviceId = deviceId;
            _handle = trt_create_engine(enginePath, deviceId);
            if (_handle == IntPtr.Zero)
                throw new InvalidOperationException("TensorRT 엔진 생성 실패");

            try
            {
                FixedInputNet = trt_get_input_size(_handle);
            }
            catch
            {
                FixedInputNet = -1;
            }
        }

        public void Dispose()
        {
            if (_handle != IntPtr.Zero)
            {
                try { trt_destroy_engine(_handle); } catch { }
                _handle = IntPtr.Zero;
            }
        }
    }
}
