using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace SmartLabelingApp
{
    public static class PatchCoreOnnx
    {
        public static InferenceSession CreateSession(string onnxPath)
        {
            var so = new SessionOptions();
            // 필요 시: so.AppendExecutionProvider_CUDA();
            return new InferenceSession(onnxPath, so);
        }

        public static IDisposableReadOnlyCollection<DisposableNamedOnnxValue>
            Run(InferenceSession session, DenseTensor<float> input)
        {
            string inputName = session.InputMetadata.Keys.First();
            var inputValue = NamedOnnxValue.CreateFromTensor(inputName, input);
            return session.Run(new[] { inputValue });
        }
    }
}
