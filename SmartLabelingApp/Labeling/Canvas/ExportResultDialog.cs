using System;
using System.Drawing;
using System.IO;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public class ExportResultDialog : Form
    {
        public ExportResultDialog(int total, int nTrain, int nVal, int nTest,
                                  string resultRoot, string zipPath = null)
        {
            Text = "EXPORT";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = MinimizeBox = false;
            BackColor = Color.White;

            ClientSize = new Size(540, 160);

            var panel = new Guna2Panel
            {
                Dock = DockStyle.Fill,
                BorderThickness = 0,      // 테두리 없음
                BorderRadius = 12,
                FillColor = Color.White,
                Padding = new Padding(12)
            };
            Controls.Add(panel);

            var title = new Label
            {
                AutoSize = true,
                Text = "✅ Success",
                ForeColor = Color.LimeGreen,
                Font = new Font("Segoe UI", 11.5f, FontStyle.Bold),
                Location = new Point(12, 10)
            };
            panel.Controls.Add(title);

            // 한 줄 요약 (굵고 크게)
            var line1 = new Label
            {
                AutoSize = true,
                Font = new Font("Segoe UI", 10.5f, FontStyle.Bold),
                Location = new Point(12, 42),
                Text = $"총 샘플 {total}EA · Train {nTrain}EA · Val {nVal}EA · Test {nTest}EA",
                MaximumSize = new Size(panel.ClientSize.Width - panel.Padding.Horizontal, 0)
            };
            panel.Controls.Add(line1);

            var info = $"결과 폴더: {resultRoot}";
            if (!string.IsNullOrEmpty(zipPath)) info += $"\nZIP: {zipPath}";

            var line2 = new Label
            {
                AutoSize = true,
                Font = new Font("Segoe UI", 9f),
                Location = new Point(12, 70),
                Text = info,
                MaximumSize = new Size(panel.ClientSize.Width - panel.Padding.Horizontal, 0)
            };
            // 컨테이너 폭이 변하면 줄바꿈 폭도 갱신
            panel.Resize += (s, e) =>
            {
                int w = panel.ClientSize.Width - panel.Padding.Horizontal;
                line1.MaximumSize = new Size(w, 0);
                line2.MaximumSize = new Size(w, 0);
            };
            panel.Controls.Add(line2);

            var btnOk = new Guna2Button
            {
                Text = "OK",
                BorderRadius = 10,
                BorderThickness = 2,
                BorderColor = Color.LightGray,
                FillColor = Color.White,
                ForeColor = Color.Black,
                Size = new Size(92, 30),
                Anchor = AnchorStyles.Bottom | AnchorStyles.Right,
                Location = new Point(ClientSize.Width - 12 - 92, ClientSize.Height - 12 - 30)
            };
            btnOk.Click += (s, e) => { DialogResult = DialogResult.OK; Close(); };
            panel.Controls.Add(btnOk);
        }
    }
}
