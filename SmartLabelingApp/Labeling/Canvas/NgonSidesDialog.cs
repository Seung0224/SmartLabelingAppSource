using System;
using System.Drawing;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public sealed class NgonSidesDialog : Form
    {
        private readonly Guna2NumericUpDown _nud;
        private readonly Guna2Button _ok;
        private readonly Guna2Button _cancel;

        public int Sides { get { return (int)_nud.Value; } }

        public NgonSidesDialog(int initialSides)
        {
            // 창 기본 설정
            this.Text = "N-gon 설정";
            this.FormBorderStyle = FormBorderStyle.FixedDialog;
            this.StartPosition = FormStartPosition.CenterParent;
            this.MinimizeBox = false;
            this.MaximizeBox = false;
            this.ClientSize = new Size(320, 50); // 타이틀 제외 영역 크기

            // 제목 외 UI 글꼴 크게
            var uiFont = new Font("Segoe UI", 12f, FontStyle.Regular);

            // 숫자 입력
            _nud = new Guna2NumericUpDown();
            _nud.Minimum = 3;
            _nud.Maximum = 36;
            _nud.Value = Math.Max(3, Math.Min(50, initialSides));
            _nud.Font = uiFont;
            _nud.BorderRadius = 6;
            _nud.Size = new Size(100, 36);

            // 확인 버튼
            _ok = new Guna2Button();
            _ok.Text = "확인";
            _ok.Font = uiFont;
            _ok.BorderRadius = 8;
            _ok.Size = new Size(90, 36);
            _ok.Click += delegate { this.DialogResult = DialogResult.OK; };

            // 취소 버튼
            _cancel = new Guna2Button();
            _cancel.Text = "취소";
            _cancel.Font = uiFont;
            _cancel.BorderRadius = 8;
            _cancel.Size = new Size(90, 36);
            _cancel.Click += delegate { this.DialogResult = DialogResult.Cancel; };

            // 한 줄 배치
            const int margin = 12;
            const int gap = 10;
            int top = (this.ClientSize.Height - _nud.Height) / 2;
            int x = margin;

            _nud.Location = new Point(x, top); x += _nud.Width + gap;
            _ok.Location = new Point(x, top); x += _ok.Width + gap;
            _cancel.Location = new Point(x, top);

            // 컨트롤 추가
            this.Controls.AddRange(new Control[] { _nud, _ok, _cancel });

            // Enter/ESC 동작
            this.AcceptButton = _ok;
            this.CancelButton = _cancel;

            // 입력란에서 Enter/ESC가 바로 동작하도록
            _nud.KeyDown += (s, e) =>
            {
                if (e.KeyCode == Keys.Enter) { this.DialogResult = DialogResult.OK; }
                else if (e.KeyCode == Keys.Escape) { this.DialogResult = DialogResult.Cancel; }
            };
        }

        public static int? ShowDialogGetSides(IWin32Window owner, int defaultSides)
        {
            using (var dlg = new NgonSidesDialog(defaultSides))
            {
                return dlg.ShowDialog(owner) == DialogResult.OK
                    ? (int?)dlg.Sides
                    : null;
            }
        }
    }
}
