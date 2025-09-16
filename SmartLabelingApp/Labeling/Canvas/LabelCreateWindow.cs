using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public class LabelCreateWindow : Form
    {
        // UI: Guna2
        private Guna2HtmlLabel _lblName;
        private Guna2TextBox _txtName;

        private Guna2HtmlLabel _lblColorTitle;
        private Guna2Panel _pnlGradient;   // 확장 스펙트럼 그라디언트 바 (클릭/드래그 픽)
        private Guna2TrackBar _trkColor;   // 색상 위치 트랙바 (클릭/드래그 즉시 반영)

        private Guna2HtmlLabel _lblR;
        private Guna2HtmlLabel _lblG;
        private Guna2HtmlLabel _lblB;
        private Guna2NumericUpDown _nudR, _nudG, _nudB;

        private Guna2Panel _pnlPreview;

        private Guna2Button _btnOK, _btnCancel;

        // 상태
        private bool _updating;

        // ---- 확장 스펙트럼 스톱(화이트→레인보우→그레이→블랙)
        // HSV UI 없이, 단일 1D 그라디언트로 선택 범위를 넓힌다.
        private readonly Color[] _stops = new Color[]
        {
            Color.White,
            Color.FromArgb(255, 255, 0, 0),      // Red
            Color.FromArgb(255, 255, 128, 0),    // Orange
            Color.FromArgb(255, 255, 255, 0),    // Yellow
            Color.FromArgb(255, 128, 255, 0),    // Yellow-Green
            Color.FromArgb(255, 0, 255, 0),      // Green
            Color.FromArgb(255, 0, 255, 128),    // Spring Green
            Color.FromArgb(255, 0, 255, 255),    // Cyan
            Color.FromArgb(255, 0, 128, 255),    // DeepSky-ish
            Color.FromArgb(255, 0, 0, 255),      // Blue
            Color.FromArgb(255, 75, 0, 130),     // Indigo
            Color.FromArgb(255, 128, 0, 255),    // Purple
            Color.FromArgb(255, 255, 0, 255),    // Magenta
            Color.FromArgb(255, 255, 128, 192),  // Pink
            Color.FromArgb(255, 160, 160, 160),  // Gray
            Color.Black
        };

        public string LabelName { get { return _txtName.Text.Trim(); } }
        public Color SelectedColor
        {
            get { return Color.FromArgb((int)_nudR.Value, (int)_nudG.Value, (int)_nudB.Value); }
        }

        public LabelCreateWindow()
        {
            // 창 기본
            FormBorderStyle = FormBorderStyle.FixedToolWindow;
            Text = "New Label";
            ShowInTaskbar = false;
            TopMost = false;

            // 높이/폭 확장: OK/CANCEL이 가려지지 않도록
            ClientSize = new Size(360, 295);
            StartPosition = FormStartPosition.Manual;

            BuildUI();
            WireEvents();

            // 트랙바 해상도: 스톱 사이당 200 스텝 (넓고 세밀)
            _trkColor.Minimum = 0;
            _trkColor.Maximum = Math.Max(10, (_stops.Length - 1) * 200);
            _trkColor.SmallChange = 1;
            _trkColor.LargeChange = 10;

            // 초기 상태
            _trkColor.Value = 0;
            ApplyTrackToRgb();
        }

        public void ResetForNewLabel()
        {
            _txtName.Text = "";
            _trkColor.Value = 0;
            ApplyTrackToRgb();
        }

        private void BuildUI()
        {
            // 레이아웃 상수
            int margin = 16;
            int fullW = ClientSize.Width - margin * 2;

            // ====== 상단: 라벨 이름 ======
            _lblName = new Guna2HtmlLabel();
            _lblName.Text = "Label Name";
            _lblName.BackColor = Color.Transparent;
            _lblName.Location = new Point(margin, 12);
            _lblName.AutoSize = true;

            _txtName = new Guna2TextBox();
            _txtName.Location = new Point(margin, 32);
            _txtName.Size = new Size(fullW, 28);
            _txtName.BorderRadius = 6;

            // ====== 색상 (그라디언트 바 + 트랙바) ======
            _lblColorTitle = new Guna2HtmlLabel();
            _lblColorTitle.Text = "Color";
            _lblColorTitle.BackColor = Color.Transparent;
            _lblColorTitle.Location = new Point(margin, 66);
            _lblColorTitle.AutoSize = true;

            _pnlGradient = new Guna2Panel();
            _pnlGradient.Location = new Point(margin, 86);
            _pnlGradient.Size = new Size(fullW, 16); // 살짝 키워서 클릭 영역 여유
            _pnlGradient.BorderColor = Color.Silver;
            _pnlGradient.BorderThickness = 1;
            _pnlGradient.CustomBorderThickness = new Padding(1);
            _pnlGradient.Paint += GradientPaint;

            _trkColor = new Guna2TrackBar();
            _trkColor.Location = new Point(margin, 106);
            _trkColor.Size = new Size(fullW, 24);

            // ====== RGB + 미리보기 ======
            int rgbTop = 136;
            int lblW = 14;
            int nudW = 90;
            int colGap = 6;
            int rowGap = 8;

            // R
            _lblR = new Guna2HtmlLabel();
            _lblR.Text = "R";
            _lblR.BackColor = Color.Transparent;
            _lblR.Location = new Point(margin, rgbTop);
            _lblR.AutoSize = true;

            _nudR = new Guna2NumericUpDown();
            _nudR.Location = new Point(margin + lblW + colGap, rgbTop - 2);
            _nudR.Size = new Size(nudW, 28);
            _nudR.Minimum = 0;
            _nudR.Maximum = 255;
            _nudR.BorderRadius = 6;

            // G
            _lblG = new Guna2HtmlLabel();
            _lblG.Text = "G";
            _lblG.BackColor = Color.Transparent;
            _lblG.Location = new Point(margin, rgbTop + 28 + rowGap);
            _lblG.AutoSize = true;

            _nudG = new Guna2NumericUpDown();
            _nudG.Location = new Point(margin + lblW + colGap, rgbTop + 28 + rowGap - 2);
            _nudG.Size = new Size(nudW, 28);
            _nudG.Minimum = 0;
            _nudG.Maximum = 255;
            _nudG.BorderRadius = 6;

            // B
            _lblB = new Guna2HtmlLabel();
            _lblB.Text = "B";
            _lblB.BackColor = Color.Transparent;
            _lblB.Location = new Point(margin, rgbTop + (28 + rowGap) * 2);
            _lblB.AutoSize = true;

            _nudB = new Guna2NumericUpDown();
            _nudB.Location = new Point(margin + lblW + colGap, rgbTop + (28 + rowGap) * 2 - 2);
            _nudB.Size = new Size(nudW, 28);
            _nudB.Minimum = 0;
            _nudB.Maximum = 255;
            _nudB.BorderRadius = 6;

            // Preview (오른쪽 넓게)
            int previewLeft = margin + lblW + colGap + nudW + 16;
            int previewW = ClientSize.Width - previewLeft - margin;
            _pnlPreview = new Guna2Panel();
            _pnlPreview.Location = new Point(previewLeft, rgbTop);
            _pnlPreview.Size = new Size(previewW, 28 * 3 + rowGap * 2);
            _pnlPreview.BorderThickness = 1;
            _pnlPreview.BorderColor = Color.Silver;
            _pnlPreview.CustomBorderThickness = new Padding(1);
            _pnlPreview.FillColor = Color.Red;

            // ====== OK / Cancel ======
            int btnW = 160;
            int btnH = 34;
            int btnGap = 12;
            int btnY = ClientSize.Height - margin - btnH;

            _btnCancel = new Guna2Button();
            _btnCancel.Text = "Cancel";
            _btnCancel.Size = new Size(btnW, btnH);
            _btnCancel.BorderRadius = 6;
            _btnCancel.Location = new Point(ClientSize.Width - margin - btnW, btnY);
            _btnCancel.FillColor = Color.White;
            _btnCancel.BorderThickness = 1;
            _btnCancel.BorderColor = Color.Silver;
            _btnCancel.HoverState.FillColor = Color.Gainsboro;
            _btnCancel.ForeColor = Color.Black;

            _btnOK = new Guna2Button();
            _btnOK.Text = "OK";
            _btnOK.Size = new Size(btnW, btnH);
            _btnOK.BorderRadius = 6;
            _btnOK.Location = new Point(_btnCancel.Left - btnGap - btnW, btnY);
            _btnOK.FillColor = Color.White;
            _btnOK.BorderThickness = 1;
            _btnOK.BorderColor = Color.Silver;
            _btnOK.HoverState.FillColor = Color.Gainsboro;
            _btnOK.ForeColor = Color.Black;

            _btnOK.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;
            _btnCancel.Anchor = AnchorStyles.Bottom | AnchorStyles.Right;

            this.AcceptButton = _btnOK;
            this.CancelButton = _btnCancel;

            // Controls 추가
            Controls.Add(_lblName);
            Controls.Add(_txtName);
            Controls.Add(_lblColorTitle);
            Controls.Add(_pnlGradient);
            Controls.Add(_trkColor);

            Controls.Add(_lblR);
            Controls.Add(_lblG);
            Controls.Add(_lblB);
            Controls.Add(_nudR);
            Controls.Add(_nudG);
            Controls.Add(_nudB);
            Controls.Add(_pnlPreview);

            Controls.Add(_btnOK);
            Controls.Add(_btnCancel);
        }

        private void WireEvents()
        {
            // 트랙바 값이 바뀌면 RGB/미리보기 반영 + 그라디언트 커서 갱신
            _trkColor.ValueChanged += (s, e) =>
            {
                if (_updating) return;
                ApplyTrackToRgb();
                _pnlGradient.Invalidate();
            };

            // 트랙바를 "어디든 클릭"하거나 "드래그"해도 즉시 해당 위치로 점프
            _trkColor.MouseDown += TrkColorImmediateJump;
            _trkColor.MouseMove += TrkColorImmediateJump;

            // 그라디언트 패널을 클릭/드래그 시에도 동일하게 동작
            _pnlGradient.MouseDown += GradientPick;
            _pnlGradient.MouseMove += GradientPick;

            _nudR.ValueChanged += (s, e) => { if (!_updating) SyncPreviewFromRgb(); };
            _nudG.ValueChanged += (s, e) => { if (!_updating) SyncPreviewFromRgb(); };
            _nudB.ValueChanged += (s, e) => { if (!_updating) SyncPreviewFromRgb(); };

            _btnOK.Click += (s, e) =>
            {
                this.DialogResult = DialogResult.OK;
                this.Close();
            };
            _btnCancel.Click += (s, e) =>
            {
                this.DialogResult = DialogResult.Cancel;
                this.Close();
            };
        }

        // ====== 즉시 점프: 트랙바 클릭/드래그 ======
        private void TrkColorImmediateJump(object sender, MouseEventArgs e)
        {
            bool leftDown = (e.Button == MouseButtons.Left) || ((Control.MouseButtons & MouseButtons.Left) == MouseButtons.Left);
            if (!leftDown) return;

            int w = Math.Max(1, _trkColor.Width - 1);
            float pos = Math.Max(0f, Math.Min(1f, e.X / (float)w));
            int newVal = (int)Math.Round(pos * _trkColor.Maximum);

            if (newVal < _trkColor.Minimum) newVal = _trkColor.Minimum;
            if (newVal > _trkColor.Maximum) newVal = _trkColor.Maximum;

            if (_trkColor.Value != newVal)
            {
                _trkColor.Value = newVal; // ValueChanged → ApplyTrackToRgb 호출됨
            }
        }

        // ====== 즉시 점프: 그라디언트 바 클릭/드래그 ======
        private void GradientPick(object sender, MouseEventArgs e)
        {
            bool leftDown = (e.Button == MouseButtons.Left) || ((Control.MouseButtons & MouseButtons.Left) == MouseButtons.Left);
            if (!leftDown) return;

            int w = Math.Max(1, _pnlGradient.Width - 1);
            int x = Math.Max(0, Math.Min(w, e.X));
            float pos = x / (float)w;

            int newVal = (int)Math.Round(pos * _trkColor.Maximum);
            newVal = Math.Max(_trkColor.Minimum, Math.Min(_trkColor.Maximum, newVal));

            if (_trkColor.Value != newVal)
                _trkColor.Value = newVal;
            else
                ApplyTrackToRgb(); // 같은 값이라도 즉시 반영되도록
        }

        // 트랙바 위치를 스펙트럼 스톱 사이의 선형보간으로 RGB에 반영
        private void ApplyTrackToRgb()
        {
            Color c = SampleGradient(_trkColor.Value, _trkColor.Maximum);
            _updating = true;
            _nudR.Value = c.R;
            _nudG.Value = c.G;
            _nudB.Value = c.B;
            _updating = false;
            _pnlPreview.FillColor = c;
        }

        private void SyncPreviewFromRgb()
        {
            Color c = Color.FromArgb((int)_nudR.Value, (int)_nudG.Value, (int)_nudB.Value);
            _pnlPreview.FillColor = c;
        }

        // 0..max 범위의 값 -> 0..1 위치 -> 스톱 사이 선형보간
        private Color SampleGradient(int value, int max)
        {
            if (max <= 0 || _stops == null || _stops.Length == 0) return Color.Black;

            float pos = (float)value / (float)max; // 0..1
            float scaled = pos * (_stops.Length - 1);
            int i = (int)Math.Floor(scaled);

            if (i < 0) i = 0;
            if (i >= _stops.Length - 1) return _stops[_stops.Length - 1];

            float t = scaled - i;
            return Lerp(_stops[i], _stops[i + 1], t);
        }

        private Color Lerp(Color a, Color b, float t)
        {
            if (t < 0f) t = 0f; else if (t > 1f) t = 1f;
            int r = a.R + (int)Math.Round((b.R - a.R) * t);
            int g = a.G + (int)Math.Round((b.G - a.G) * t);
            int b2 = a.B + (int)Math.Round((b.B - a.B) * t);
            return Color.FromArgb(r, g, b2);
        }

        private void GradientPaint(object sender, PaintEventArgs e)
        {
            Rectangle r = _pnlGradient.ClientRectangle;
            if (r.Width <= 0 || r.Height <= 0) return;

            using (var br = new LinearGradientBrush(r, Color.Black, Color.Black, 0f))
            {
                var cb = new ColorBlend();
                cb.Colors = _stops;

                // 스톱 개수에 맞춰 등간격 포지션
                int n = _stops.Length;
                float[] pos = new float[n];
                for (int i = 0; i < n; i++)
                    pos[i] = (float)i / (float)(n - 1);
                cb.Positions = pos;

                br.InterpolationColors = cb;
                e.Graphics.FillRectangle(br, r);

                // 현재 선택 위치 커서(얇은 라인)
                float x = (_trkColor.Maximum > 0) ? (r.Width - 1) * (_trkColor.Value / (float)_trkColor.Maximum) : 0f;
                using (var pen = new Pen(Color.Silver))
                {
                    e.Graphics.DrawRectangle(pen, 0, 0, r.Width - 1, r.Height - 1);
                }
                using (var pSel = new Pen(Color.White, 1f))
                using (var pSel2 = new Pen(Color.Black, 1f))
                {
                    int xi = (int)Math.Round(x);
                    e.Graphics.DrawLine(pSel, xi, 1, xi, r.Height - 2);
                    e.Graphics.DrawLine(pSel2, xi + 1, 1, xi + 1, r.Height - 2);
                }
            }
        }
    }
}
