using System;
using System.Drawing;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public class ColorMapWindow : Form
    {
        private readonly Guna2HtmlLabel _lblMin, _lblMax;
        private readonly Guna2NumericUpDown _nudMin, _nudMax;
        private readonly Guna2TrackBar _trkMin, _trkMax;
        private readonly Guna2Button _btnOK, _btnCancel;

        public Action<ushort, ushort> OnLiveUpdate;  // MainForm으로 (min, max) 전달
        public ushort SelectedMin => (ushort)_trkMin.Value;
        public ushort SelectedMax => (ushort)_trkMax.Value;
        public bool Confirmed { get; private set; }

        public ColorMapWindow()
        {
            Text = "Color Map";
            FormBorderStyle = FormBorderStyle.FixedToolWindow;
            StartPosition = FormStartPosition.CenterParent;
            ClientSize = new Size(400, 180);
            ShowInTaskbar = false;
            TopMost = false;

            _lblMin = new Guna2HtmlLabel() { Text = "Min", Location = new Point(20, 20), BackColor = Color.Transparent };
            _lblMax = new Guna2HtmlLabel() { Text = "Max", Location = new Point(20, 75), BackColor = Color.Transparent };

            _nudMin = new Guna2NumericUpDown()
            {
                Location = new Point(65, 15),
                Size = new Size(80, 28),
                Minimum = 0,
                Maximum = 65535,
                Value = 0,
                BorderRadius = 6
            };
            _nudMax = new Guna2NumericUpDown()
            {
                Location = new Point(65, 70),
                Size = new Size(80, 28),
                Minimum = 0,
                Maximum = 65535,
                Value = 65535,
                BorderRadius = 6
            };

            _trkMin = new Guna2TrackBar()
            {
                Location = new Point(160, 20),
                Size = new Size(220, 24),
                Minimum = 0,
                Maximum = 65535,
                Value = 0
            };
            _trkMax = new Guna2TrackBar()
            {
                Location = new Point(160, 75),
                Size = new Size(220, 24),
                Minimum = 0,
                Maximum = 65535,
                Value = 65535
            };

            _btnOK = new Guna2Button()
            {
                Text = "OK",
                Size = new Size(120, 36),
                BorderRadius = 6,
                Location = new Point(150, 125),
                FillColor = Color.White,
                BorderColor = Color.Silver,
                BorderThickness = 1,
                HoverState = { FillColor = Color.Gainsboro },
                ForeColor = Color.Black
            };
            _btnCancel = new Guna2Button()
            {
                Text = "Cancel",
                Size = new Size(120, 36),
                BorderRadius = 6,
                Location = new Point(280, 125),
                FillColor = Color.White,
                BorderColor = Color.Silver,
                BorderThickness = 1,
                HoverState = { FillColor = Color.Gainsboro },
                ForeColor = Color.Black
            };

            Controls.AddRange(new Control[] {
                _lblMin, _lblMax, _nudMin, _nudMax, _trkMin, _trkMax, _btnOK, _btnCancel
            });

            // 이벤트 바인딩
            _trkMin.ValueChanged += (s, e) =>
            {
                if (_trkMin.Value > _trkMax.Value)
                    _trkMax.Value = _trkMin.Value;
                _nudMin.Value = _trkMin.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            };
            _trkMax.ValueChanged += (s, e) =>
            {
                if (_trkMax.Value < _trkMin.Value)
                    _trkMin.Value = _trkMax.Value;
                _nudMax.Value = _trkMax.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            };
            _nudMin.ValueChanged += (s, e) =>
            {
                _trkMin.Value = (int)_nudMin.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            };
            _nudMax.ValueChanged += (s, e) =>
            {
                _trkMax.Value = (int)_nudMax.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            };

            _btnOK.Click += (s, e) => { Confirmed = true; Close(); };
            _btnCancel.Click += (s, e) => { Confirmed = false; Close(); };
        }
    }
}
