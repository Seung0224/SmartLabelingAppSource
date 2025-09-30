using System;
using System.IO;
using System.Linq;
using System.Text;
using System.Globalization;
using Newtonsoft.Json;

namespace SmartLabelingApp
{
    public sealed class Artifacts
    {
        public string ExportDir { get; private set; }
        public string OnnxPath { get; private set; }   // wrn50_l3.onnx

        // Gallery (N x Dim), row-major, already L2-normalized
        public float[] Gallery { get; private set; }
        public int GalleryRows { get; private set; } // N
        public int Dim { get; private set; } // 1024 (L3 only)

        // Threshold (cosine distance)
        public float Threshold { get; private set; }
        public string Metric { get; private set; } = "cosine"; // "cosine" expected

        // Preprocess / meta
        public int InputSize { get; private set; }
        public int GridH { get; private set; }
        public int GridW { get; private set; }
        public float[] Mean { get; private set; }
        public float[] Std { get; private set; }

        public static Artifacts Load(string exportDir)
        {
            if (string.IsNullOrWhiteSpace(exportDir))
                throw new ArgumentException("exportDir is null/empty.");

            exportDir = Path.GetFullPath(exportDir);
            if (!Directory.Exists(exportDir))
                throw new DirectoryNotFoundException($"Export dir not found: {exportDir}");

            var A = new Artifacts { ExportDir = exportDir };

            // ---- 1) ONNX ----
            A.OnnxPath = Path.Combine(exportDir, "wrn50_l3.onnx");
            if (!File.Exists(A.OnnxPath))
                throw new FileNotFoundException($"wrn50_l3.onnx not found in: {exportDir}", A.OnnxPath);

            // ---- 2) threshold.json ----
            var thrPath = Path.Combine(exportDir, "threshold.json");
            if (!File.Exists(thrPath))
                throw new FileNotFoundException($"threshold.json not found in: {exportDir}", thrPath);

            var thr = ReadJson<ThresholdJson>(thrPath);
            if (thr == null) throw new InvalidDataException("threshold.json parse failed.");
            if (!string.IsNullOrWhiteSpace(thr.metric))
                A.Metric = thr.metric.Trim().ToLowerInvariant();
            A.Threshold = (float)thr.value;

            // ---- 3) meta.json (optional but recommended) ----
            var metaPath = Path.Combine(exportDir, "meta.json");
            if (File.Exists(metaPath))
            {
                var meta = ReadJson<MetaJson>(metaPath);
                if (meta != null)
                {
                    if (meta.input_size > 0) A.InputSize = meta.input_size;
                    if (meta.grid != null && meta.grid.Length == 2)
                    {
                        A.GridH = meta.grid[0];
                        A.GridW = meta.grid[1];
                    }
                    if (meta.mean != null && meta.mean.Length == 3)
                        A.Mean = meta.mean.Select(x => (float)x).ToArray();
                    if (meta.std != null && meta.std.Length == 3)
                        A.Std = meta.std.Select(x => (float)x).ToArray();
                    if (!string.IsNullOrWhiteSpace(meta.metric))
                        A.Metric = meta.metric.Trim().ToLowerInvariant();
                }
            }

            // ---- 4) gallery_f32.bin ----
            var galPath = Path.Combine(exportDir, "gallery_f32.bin");
            if (!File.Exists(galPath))
                throw new FileNotFoundException($"gallery_f32.bin not found in: {exportDir}", galPath);

            // For L3-only, Dim is 1024. We can validate with file length.
            A.Dim = 1024; // constant for L3-only pipeline
            var bytes = File.ReadAllBytes(galPath);
            if ((bytes.Length % 4) != 0)
                throw new InvalidDataException($"gallery_f32.bin size not multiple of 4 bytes: {bytes.Length}");

            int totalFloats = bytes.Length / 4;
            if ((totalFloats % A.Dim) != 0)
                throw new InvalidDataException($"gallery_f32.bin float count {totalFloats} not divisible by dim={A.Dim}.");

            A.GalleryRows = totalFloats / A.Dim;
            A.Gallery = new float[totalFloats];
            Buffer.BlockCopy(bytes, 0, A.Gallery, 0, bytes.Length);

            // Quick sanity: gallery rows > 0 and metric must be cosine
            if (A.GalleryRows <= 0)
                throw new InvalidDataException("gallery_f32.bin seems empty.");
            if (A.Metric != "cosine")
                Console.WriteLine($"[WARN] Metric is '{A.Metric}', expected 'cosine'. The app will still run using cosine.");

            return A;
        }

        private static T ReadJson<T>(string path) where T : class
        {
            var json = File.ReadAllText(path, Encoding.UTF8);
            var obj = JsonConvert.DeserializeObject<T>(json);
            return obj;
        }

        // ----- JSON DTOs -----
        private sealed class ThresholdJson
        {
            public string metric { get; set; } = "cosine";
            public double q { get; set; } = 0.998;
            public double value { get; set; } = 0.05;
            public int ntotal { get; set; } = 0;
        }

        private sealed class MetaJson
        {
            public string backbone { get; set; }
            public string[] layers { get; set; }
            public int input_size { get; set; } = 224;
            public double[] mean { get; set; }
            public double[] std { get; set; }
            public int[] grid { get; set; }  // [H,W] = [14,14]
            public string metric { get; set; } = "cosine";
            public string data_name { get; set; }
            public string model_dir { get; set; }
        }
    }
}
