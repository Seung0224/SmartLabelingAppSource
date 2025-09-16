using System;
using System.Drawing;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public sealed class BrushSizeWindow : Form
    {
        private readonly Guna2BorderlessForm _borderless;

        private readonly Label _title;
        private readonly Label _lblMin;
        private readonly Label _lblMax;
        private readonly Label _lblValue;
        private readonly Guna2TrackBar _track;

        // 폼 기본 SizeChanged와 혼동 피하려고 이름 변경
        public event Action<int> BrushSizeChanged;

        public int MinimumPx { get; set; } = 2;    // 지름(px)
        public int MaximumPx { get; set; } = 256;  // 지름(px)

        private int _valuePx = 18;                 // 지름(px)
        public int ValuePx
        {
            get { return _valuePx; }
            set
            {
                int v = Math.Max(MinimumPx, Math.Min(MaximumPx, value));
                if (_valuePx == v) return;
                _valuePx = v;
                SyncUI();
                if (BrushSizeChanged != null) BrushSizeChanged(_valuePx);
            }
        }

        public BrushSizeWindow()
        {
            Text = "Brush Size";
            StartPosition = FormStartPosition.Manual;
            FormBorderStyle = FormBorderStyle.None;
            TopMost = true;
            ShowInTaskbar = false;
            BackColor = Color.White;
            Size = new Size(200, 120);

            _borderless = new Guna2BorderlessForm();
            _borderless.ContainerControl = this;
            _borderless.BorderRadius = 12;
            _borderless.TransparentWhileDrag = true;
            _borderless.ResizeForm = false;

            var root = new Guna2Panel();
            root.Dock = DockStyle.Fill;
            root.Padding = new Padding(32, 10, 32, 12);
            root.BorderThickness = 0; // <- 테두리 선 제거
            root.BorderColor = Color.Transparent;
            root.BorderRadius = 12;
            root.FillColor = Color.White;
            Controls.Add(root);

            _title = new Label();
            _title.Text = "Brush size";
            _title.Dock = DockStyle.Top;
            _title.Height = 24;
            _title.Font = new Font("Segoe UI", 10, FontStyle.Bold);
            _title.TextAlign = ContentAlignment.MiddleLeft;
            root.Controls.Add(_title);

            _lblValue = new Label();
            _lblValue.Dock = DockStyle.Top;
            _lblValue.Height = 28;
            _lblValue.Font = new Font("Segoe UI", 12, FontStyle.Bold);
            _lblValue.TextAlign = ContentAlignment.MiddleCenter;
            _lblValue.Text = "18 px";
            root.Controls.Add(_lblValue);

            _track = new Guna2TrackBar();
            _track.Dock = DockStyle.Top;
            _track.Height = 36;
            _track.Minimum = MinimumPx;
            _track.Maximum = MaximumPx;
            _track.Value = _valuePx;
            _track.HoverState.ThumbColor = Color.DeepSkyBlue;
            _track.ThumbColor = Color.DeepSkyBlue;
            _track.FillColor = Color.FromArgb(230, 230, 230);
            _track.ValueChanged += delegate
            {
                _valuePx = _track.Value;
                SyncUI();
                if (BrushSizeChanged != null) BrushSizeChanged(_valuePx);
            };
            root.Controls.Add(_track);

            var tickRow = new Panel { Dock = DockStyle.Top, Height = 16, Padding = new Padding(2, 0, 2, 0) };
            _lblMin = new Label { Dock = DockStyle.Left, Width = 60, TextAlign = ContentAlignment.MiddleLeft, Font = new Font("Segoe UI", 8f), Text = MinimumPx + " px" };
            _lblMax = new Label { Dock = DockStyle.Right, Width = 60, TextAlign = ContentAlignment.MiddleRight, Font = new Font("Segoe UI", 8f), Text = MaximumPx + " px" };
            tickRow.Controls.Add(_lblMax);
            tickRow.Controls.Add(_lblMin);
            root.Controls.Add(tickRow);

            // ESC 닫기
            KeyPreview = true;
            KeyDown += (s, e) => { if (e.KeyCode == Keys.Escape) Close(); };
        }

        private void SyncUI()
        {
            _track.Minimum = MinimumPx;
            _track.Maximum = MaximumPx;
            if (_track.Value != _valuePx)
                _track.Value = Math.Max(_track.Minimum, Math.Min(_track.Maximum, _valuePx));
            _lblValue.Text = _track.Value + " px";
            _lblMin.Text = MinimumPx + " px";
            _lblMax.Text = MaximumPx + " px";
        }
    }
}
