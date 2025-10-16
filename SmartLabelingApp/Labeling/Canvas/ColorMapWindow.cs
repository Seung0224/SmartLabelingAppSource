using System;
using System.Drawing;
using System.Drawing.Drawing2D;
using System.Windows.Forms;
using Guna.UI2.WinForms;

namespace SmartLabelingApp
{
    public class ColorMapWindow : Form
    {
        // ===== Range(8/16bit) =====
        private readonly Guna2HtmlLabel _lblMax, _lblMin;
        private readonly Guna2NumericUpDown _nudMax, _nudMin;
        private readonly Guna2TrackBar _trkMax, _trkMin;

        // ===== Middle: Gradient only (click/drag to pick color) =====
        private readonly Guna2Panel _pnlGradient;

        // 상태
        private int _bitDepth = 16; // 8 or 16
        private int _colorPos;      // 0.._colorMax (그라디언트 내 선택 위치)
        private int _colorMax;      // 스톱 사이 200 스텝 × (N-1)

        // JET-like 스톱 (Blue→Cyan→Green→Yellow→Orange→Red)
        private Color[] _stops = new Color[]
        {
            Color.FromArgb(  0,   0, 130),
            Color.FromArgb(  0,  80, 255),
            Color.FromArgb(  0, 220, 255),
            Color.FromArgb( 10, 255,  60),
            Color.FromArgb(230, 255,  20),
            Color.FromArgb(255, 160,   0),
            Color.FromArgb(255,  40,   0)
        };

        // 콜백
        public Action<ushort, ushort> OnLiveUpdate;  // (min,max) 변경 알림
        public Action<Color> OnColorChanged;         // 선택색 변경 알림

        public ushort SelectedMin => (ushort)_trkMin.Value;
        public ushort SelectedMax => (ushort)_trkMax.Value;

        public ColorMapWindow()
        {
            Text = "Color Map";
            FormBorderStyle = FormBorderStyle.FixedToolWindow;
            StartPosition = FormStartPosition.CenterParent;
            ShowInTaskbar = false;
            TopMost = false;

            // 레이아웃: Top(Max) → Middle(Gradient) → Bottom(Min)
            ClientSize = new Size(520, 130);

            // ===== Max(맨위) =====
            _lblMax = new Guna2HtmlLabel() { Text = "Max", Location = new Point(18, 16), AutoSize = true, BackColor = Color.Transparent };
            _nudMax = new Guna2NumericUpDown()
            {
                Location = new Point(52, 8),
                Size = new Size(90, 28),
                Minimum = 0,
                Maximum = 65535,
                Value = 65535,
                BorderRadius = 6
            };
            _trkMax = new Guna2TrackBar()
            {
                Location = new Point(150, 16),
                Size = new Size(350, 20),
                Minimum = 0,
                Maximum = 65535,
                Value = 65535
            };

            // ===== Gradient (중앙) =====
            _pnlGradient = new Guna2Panel()
            {
                Location = new Point(16, 54),
                Size = new Size(484, 18),
                BorderColor = Color.Silver,
                BorderThickness = 1,
                CustomBorderThickness = new Padding(1)
            };
            _pnlGradient.Paint += GradientPaint;
            _pnlGradient.MouseDown += GradientPick;
            _pnlGradient.MouseMove += GradientPick;

            // ===== Min(맨아래) =====
            _lblMin = new Guna2HtmlLabel() { Text = "Min", Location = new Point(18, 92), AutoSize = true, BackColor = Color.Transparent };
            _nudMin = new Guna2NumericUpDown()
            {
                Location = new Point(52, 84),
                Size = new Size(90, 28),
                Minimum = 0,
                Maximum = 65535,
                Value = 0,
                BorderRadius = 6
            };
            _trkMin = new Guna2TrackBar()
            {
                Location = new Point(150, 92),
                Size = new Size(350, 20),
                Minimum = 0,
                Maximum = 65535,
                Value = 0
            };

            Controls.AddRange(new Control[]
            {
                _lblMax, _nudMax, _trkMax,
                _pnlGradient,
                _lblMin, _nudMin, _trkMin
            });

            // 이벤트(범위)
            _trkMax.ValueChanged += (s, e) =>
            {
                if (_trkMax.Value < _trkMin.Value) _trkMin.Value = _trkMax.Value;
                _nudMax.Value = _trkMax.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
                NotifyColorChanged();
                _pnlGradient.Invalidate();
            };
            _nudMax.ValueChanged += (s, e) =>
            {
                _trkMax.Value = (int)_nudMax.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
                NotifyColorChanged();
                _pnlGradient.Invalidate();
            };

            _trkMin.ValueChanged += (s, e) =>
            {
                if (_trkMin.Value > _trkMax.Value) _trkMax.Value = _trkMin.Value;
                _nudMin.Value = _trkMin.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
                NotifyColorChanged();
                _pnlGradient.Invalidate();
            };
            _nudMin.ValueChanged += (s, e) =>
            {
                _trkMin.Value = (int)_nudMin.Value;
                OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
                NotifyColorChanged();
                _pnlGradient.Invalidate();
            };

            // 초기 설정
            _colorMax = Math.Max(10, (_stops.Length - 1) * 200);
            _colorPos = 0; // 왼쪽 시작
            NotifyColorChanged();
        }

        // ===== 공개 API =====

        /// 8 또는 16 지정 → Min/Max 범위 자동 설정(0~255 / 0~65535)
        public void ConfigureForBitDepth(int bitDepth)
        {
            _bitDepth = (bitDepth == 8) ? 8 : 16;
            int max = (_bitDepth == 8) ? 255 : 65535;

            _nudMin.Maximum = max; _nudMax.Maximum = max;
            _trkMin.Maximum = max; _trkMax.Maximum = max;

            if (_nudMin.Value > max) _nudMin.Value = max;
            if (_nudMax.Value > max) _nudMax.Value = max;
            if (_trkMin.Value > max) _trkMin.Value = max;
            if (_trkMax.Value > max) _trkMax.Value = max;

            if (_trkMin.Value > _trkMax.Value) _trkMin.Value = _trkMax.Value;

            OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            NotifyColorChanged();
            _pnlGradient.Invalidate();
        }

        /// 초기 Min/Max 세팅(현재 비트 심도 범위로 클램프)
        public void SetInitialRange(ushort min, ushort max)
        {
            int vmax = (_bitDepth == 8) ? 255 : 65535;
            int vmin = 0;

            int a = Math.Max(vmin, Math.Min(vmax, min));
            int b = Math.Max(vmin, Math.Min(vmax, max));
            if (a > b) (a, b) = (b, a);

            _nudMin.Value = a; _nudMax.Value = b;
            _trkMin.Value = a; _trkMax.Value = b;

            OnLiveUpdate?.Invoke((ushort)_trkMin.Value, (ushort)_trkMax.Value);
            NotifyColorChanged();
            _pnlGradient.Invalidate();
        }

        /// 컬러스톱 프리셋 변경
        public void SetColorStopsPreset(string preset)
        {
            preset = (preset ?? "").Trim().ToUpperInvariant();
            if (preset == "RAINBOW_EXT")
            {
                _stops = new Color[]
                {
                    Color.White,
                    Color.FromArgb(255,255,0,0), Color.FromArgb(255,255,128,0), Color.FromArgb(255,255,255,0),
                    Color.FromArgb(255,128,255,0), Color.FromArgb(255,0,255,0),   Color.FromArgb(255,0,255,128),
                    Color.FromArgb(255,0,255,255), Color.FromArgb(255,0,128,255), Color.FromArgb(255,0,0,255),
                    Color.FromArgb(255,75,0,130),  Color.FromArgb(255,128,0,255), Color.FromArgb(255,255,0,255),
                    Color.FromArgb(255,255,128,192), Color.FromArgb(255,160,160,160), Color.Black
                };
            }
            else // default JET-like
            {
                _stops = new Color[]
                {
                    Color.FromArgb(  0,   0, 130),
                    Color.FromArgb(  0,  80, 255),
                    Color.FromArgb(  0, 220, 255),
                    Color.FromArgb( 10, 255,  60),
                    Color.FromArgb(230, 255,  20),
                    Color.FromArgb(255, 160,   0),
                    Color.FromArgb(255,  40,   0)
                };
            }

            _colorMax = Math.Max(10, (_stops.Length - 1) * 200);
            _colorPos = Math.Min(_colorPos, _colorMax);
            _pnlGradient.Invalidate();
            NotifyColorChanged();
        }

        // ===== 내부 로직 =====

        private void GradientPick(object sender, MouseEventArgs e)
        {
            bool leftDown = (e.Button == MouseButtons.Left) || ((Control.MouseButtons & MouseButtons.Left) == MouseButtons.Left);
            if (!leftDown) return;

            Rectangle r = _pnlGradient.ClientRectangle;
            int w = Math.Max(1, r.Width - 1);
            int x = Math.Max(0, Math.Min(w, e.X));
            float pos = x / (float)w;

            int newPos = (int)Math.Round(pos * _colorMax);
            newPos = Math.Max(0, Math.Min(_colorMax, newPos));
            if (_colorPos != newPos)
            {
                _colorPos = newPos;
                NotifyColorChanged();
                _pnlGradient.Invalidate();
            }
            else
            {
                // 같은 값이어도 즉시 반영
                NotifyColorChanged();
            }
        }

        private void NotifyColorChanged()
        {
            if (OnColorChanged == null) return;

            float pos01 = (_colorMax > 0) ? _colorPos / (float)_colorMax : 0f;

            // 현재 Min/Max 범위(정규화)
            float minPos = GetNorm((int)_trkMin.Value);
            float maxPos = GetNorm((int)_trkMax.Value);
            if (minPos > maxPos) { var t = minPos; minPos = maxPos; maxPos = t; }

            Color c = (pos01 < minPos || pos01 > maxPos)
                    ? Color.Black           // 범위 밖 → 검정
                    : SampleGradientBy01(pos01);

            OnColorChanged?.Invoke(c);
        }

        private float GetNorm(int v)
        {
            int max = (_bitDepth == 8) ? 255 : 65535;
            if (max <= 0) return 0f;
            if (v < 0) v = 0; if (v > max) v = max;
            return v / (float)max;
        }

        // === Gradient 샘플링/보간 ===
        private Color SampleGradientBy01(float pos01)
        {
            if (_stops == null || _stops.Length == 0) return Color.Black;
            pos01 = Math.Max(0f, Math.Min(1f, pos01));
            float scaled = pos01 * (_stops.Length - 1);
            int i = (int)Math.Floor(scaled);
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

        // Cognex 스타일: 범위 밖 검정, 범위 안만 컬러 + 선택 위치 듀얼 라인 표시
        private void GradientPaint(object sender, PaintEventArgs e)
        {
            Rectangle r = _pnlGradient.ClientRectangle;
            if (r.Width <= 0 || r.Height <= 0) return;

            // 배경 검정
            using (var bg = new SolidBrush(Color.Black))
                e.Graphics.FillRectangle(bg, r);

            // Min/Max → X
            float minPos = GetNorm((int)_trkMin.Value);
            float maxPos = GetNorm((int)_trkMax.Value);
            if (minPos > maxPos) { var t = minPos; minPos = maxPos; maxPos = t; }

            int x0 = r.X + (int)Math.Round((r.Width - 1) * minPos);
            int x1 = r.X + (int)Math.Round((r.Width - 1) * maxPos);
            if (x1 < x0) (x0, x1) = (x1, x0);

            // 유효 대역만 컬러
            Rectangle band = Rectangle.FromLTRB(x0, r.Y, Math.Max(x0 + 1, x1), r.Bottom);
            if (band.Width > 0)
            {
                using (var br = new LinearGradientBrush(band, Color.Black, Color.Black, 0f))
                {
                    br.InterpolationColors = new ColorBlend
                    {
                        Colors = _stops,
                        Positions = BuildEvenPositions(_stops.Length)
                    };
                    e.Graphics.FillRectangle(br, band);
                }
            }

            // 외곽선
            using (var pen = new Pen(Color.Silver))
                e.Graphics.DrawRectangle(pen, r.X, r.Y, r.Width - 1, r.Height - 1);

            // 선택 위치 듀얼 라인
            float sel01 = (_colorMax > 0) ? _colorPos / (float)_colorMax : 0f;
            int xs = r.X + (int)Math.Round((r.Width - 1) * sel01);
            using (var p1 = new Pen(Color.White, 1f))
            using (var p2 = new Pen(Color.Black, 1f))
            {
                e.Graphics.DrawLine(p1, xs, r.Y + 1, xs, r.Bottom - 2);
                e.Graphics.DrawLine(p2, xs + 1, r.Y + 1, xs + 1, r.Bottom - 2);
            }
        }

        private float[] BuildEvenPositions(int n)
        {
            if (n <= 1) return new float[] { 0f };
            float[] pos = new float[n];
            for (int i = 0; i < n; i++) pos[i] = i / (float)(n - 1);
            return pos;
        }
    }
}
