using Guna.UI2.WinForms;
using System;
using System.Drawing;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    public class ExportSplitDialog : Form
    {
        private readonly Guna2NumericUpDown _numTrain;
        private readonly Guna2NumericUpDown _numVal;
        private readonly Guna2NumericUpDown _numTest;
        private readonly Guna2Button _btnOk;
        private readonly Guna2Button _btnCancel;
        private readonly Label _lblSum;

        public int TrainPercent => (int)_numTrain.Value;
        public int ValPercent   => (int)_numVal.Value;
        public int TestPercent  => (int)_numTest.Value;

        public ExportSplitDialog(int defTrain = 90, int defVal = 5, int defTest = 5)
        {
            Text = "Export Split";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = false;
            MinimizeBox = false;
            ClientSize = new Size(320, 190);
            BackColor = Color.White;

            var lblTrain = new Label { Text = "Train %", AutoSize = true, Location = new Point(24, 24) };
            var lblVal   = new Label { Text = "Validation %", AutoSize = true, Location = new Point(24, 62) };
            var lblTest  = new Label { Text = "Test %", AutoSize = true, Location = new Point(24, 100) };

            _numTrain = new Guna2NumericUpDown
            {
                Minimum = 0, Maximum = 100, Value = defTrain,
                Location = new Point(140, 20), Size = new Size(140, 30)
            };
            _numVal = new Guna2NumericUpDown
            {
                Minimum = 0, Maximum = 100, Value = defVal,
                Location = new Point(140, 58), Size = new Size(140, 30)
            };
            _numTest = new Guna2NumericUpDown
            {
                Minimum = 0, Maximum = 100, Value = defTest,
                Location = new Point(140, 96), Size = new Size(140, 30)
            };

            _lblSum = new Label
            {
                AutoSize = true,
                Location = new Point(24, 134),
                ForeColor = Color.DimGray
            };

            _btnOk = new Guna2Button
            {
                Text = "OK",
                BorderRadius = 10,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.White,
                ForeColor = Color.Black,
                Size = new Size(100, 32),
                Location = new Point(96, 150),
                DialogResult = DialogResult.OK
            };
            _btnCancel = new Guna2Button
            {
                Text = "Cancel",
                BorderRadius = 10,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.White,
                ForeColor = Color.Black,
                Size = new Size(100, 32),
                Location = new Point(206, 150),
                DialogResult = DialogResult.Cancel
            };

            Controls.AddRange(new Control[]
            {
                lblTrain, lblVal, lblTest, _numTrain, _numVal, _numTest, _lblSum, _btnOk, _btnCancel
            });

            _numTrain.ValueChanged += OnValueChanged;
            _numVal.ValueChanged += OnValueChanged;
            _numTest.ValueChanged += OnValueChanged;
            this.Shown += (s, e) => UpdateSum();

            AcceptButton = _btnOk;
            CancelButton = _btnCancel;

            _btnOk.Click += (s, e) => OnOk();
            _btnCancel.Click += (s, e) => { this.DialogResult = DialogResult.Cancel; this.Close(); };
        }

        private void OnValueChanged(object sender, EventArgs e) => UpdateSum();
        private void OnOk()
        {
            int sum = (int)(_numTrain.Value + _numVal.Value + _numTest.Value);
            if (sum != 100)
            {
                MessageBox.Show(this, "세 비율의 합이 100%가 아닙니다.", "EXPORT",
                                MessageBoxButtons.OK, MessageBoxIcon.Warning);
                return;
            }
            this.DialogResult = DialogResult.OK;
            this.Close();
        }

        private void UpdateSum()
        {
            int sum = (int)(_numTrain.Value + _numVal.Value + _numTest.Value);
            _lblSum.Text = $"합계: {sum}%";
            bool ok = (sum == 100);
            _btnOk.Enabled = ok;
            _lblSum.ForeColor = ok ? Color.SeaGreen : Color.IndianRed;
        }
    }
}