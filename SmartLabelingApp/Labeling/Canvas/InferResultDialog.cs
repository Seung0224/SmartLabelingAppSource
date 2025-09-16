using System;
using System.Drawing;
using System.Windows.Forms;
using Cyotek.Windows.Forms;
using Guna.UI2.WinForms;
using Guna.UI2.WinForms.Enums;

namespace SmartLabelingApp
{
    public class InferResultDialog : Form
    {
        // --- 스타일 상수
        private const int TOPBAR_H = 36;   // 더 얇게
        private const int PAD_H = 12;
        private const int PAD_V = 2;
        private const int GAP = 2;

        private readonly Guna2BorderlessForm _borderless;
        private readonly Guna2Elipse _elipse;
        private readonly Guna2ShadowForm _shadow;

        // Top bar
        private readonly Guna2GradientPanel _topBar;
        private readonly Label _lblTitle;
        private readonly Guna2ControlBox _btnMin;
        private readonly Guna2ControlBox _btnMax;
        private readonly Guna2ControlBox _btnClose;
        private readonly Guna2DragControl _dragTop;
        private readonly Guna2DragControl _dragTitle;

        // 중앙 호스트(화이트 카드)
        private readonly Guna2Panel _host;

        // 뷰어 패널 & 이미지박스
        private readonly Guna2Panel _viewerPanel;
        private readonly ImageBox _imageBox;

        public InferResultDialog(Image imageToShow, string title = null)
        {
            // ---- 기본 폼
            Text = string.IsNullOrWhiteSpace(title) ? "SmartLabelingApp" : title;
            StartPosition = FormStartPosition.CenterParent;
            ShowInTaskbar = false;
            DoubleBuffered = true;

            MinimumSize = new Size(800, 560);
            BackColor = Color.White;
            FormBorderStyle = FormBorderStyle.None;

            // 창 효과
            _elipse = new Guna2Elipse { BorderRadius = 2, TargetControl = this };
            _borderless = new Guna2BorderlessForm
            {
                ContainerControl = this,
                BorderRadius = 2,
                TransparentWhileDrag = true,
                ResizeForm = true
            };
            _shadow = new Guna2ShadowForm { ShadowColor = Color.Black };

            // 중앙 호스트(화이트 카드)
            _host = new Guna2Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(8, 0, 8, 8),
                BorderColor = Color.FromArgb(220, 224, 230),
                BorderThickness = 1,
                BorderRadius = 12,
                FillColor = Color.White
            };
            _host.ShadowDecoration.Parent = _host;
            Controls.Add(_host);

            // 상단바 (초록 그라데이션)
            _topBar = new Guna2GradientPanel
            {
                Dock = DockStyle.Top,
                Height = TOPBAR_H,
                FillColor = Color.FromArgb(66, 181, 137), // 초록
                FillColor2 = Color.FromArgb(134, 236, 179),
                Padding = new Padding(PAD_H, PAD_V, PAD_H, PAD_V)
            };
            _topBar.DoubleClick += (_, __) => ToggleMaximizeRestore();
            Controls.Add(_topBar);

            _dragTop = new Guna2DragControl
            {
                TargetControl = _topBar,
                DockIndicatorTransparencyValue = 0.6f,
                UseTransparentDrag = true
            };

            // 우측 컨트롤박스 영역
            int toolEdge = TOPBAR_H - PAD_V * 2;
            var rightPanel = new Panel
            {
                Dock = DockStyle.Right,
                Width = toolEdge * 3 + GAP * 4,
                BackColor = Color.Transparent
            };
            _topBar.Controls.Add(rightPanel);

            int y = PAD_V;
            int cbEdge = toolEdge;

            _btnMin = new Guna2ControlBox
            {
                ControlBoxType = ControlBoxType.MinimizeBox,
                FillColor = Color.Transparent,
                IconColor = Color.White,
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge * 3 + GAP * 3), y)
            };
            _btnMax = new Guna2ControlBox
            {
                ControlBoxType = ControlBoxType.MaximizeBox,
                FillColor = Color.Transparent,
                IconColor = Color.White,
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge * 2 + GAP * 2), y)
            };
            _btnClose = new Guna2ControlBox
            {
                FillColor = Color.Transparent,
                IconColor = Color.White,
                HoverState = { FillColor = Color.FromArgb(255, 80, 80), IconColor = Color.White },
                BorderRadius = 2,
                UseTransparentBackground = true,
                Size = new Size(cbEdge, cbEdge),
                Location = new Point(rightPanel.Width - (cbEdge + GAP), y)
            };
            rightPanel.Controls.AddRange(new Control[] { _btnMin, _btnMax, _btnClose });

            // 타이틀(좌측)
            _lblTitle = new Label
            {
                Text = Text,
                Dock = DockStyle.Fill,
                TextAlign = ContentAlignment.MiddleLeft,
                AutoEllipsis = true,
                BackColor = Color.Transparent,
                ForeColor = Color.White,
                Font = new Font("Segoe UI", 10, FontStyle.Regular)
            };
            _lblTitle.DoubleClick += (s, e) => ToggleMaximizeRestore();
            _topBar.Controls.Add(_lblTitle);

            _dragTitle = new Guna2DragControl
            {
                TargetControl = _lblTitle,
                DockIndicatorTransparencyValue = 0.6f,
                UseTransparentDrag = true
            };

            // 뷰어 패널(화이트)
            _viewerPanel = new Guna2Panel
            {
                Dock = DockStyle.Fill,
                Padding = new Padding(8),
                FillColor = Color.White,
                BorderRadius = 8
            };
            _host.Controls.Add(_viewerPanel);

            // 이미지 박스
            _imageBox = new ImageBox
            {
                Dock = DockStyle.Fill,
                Image = imageToShow,
                BackColor = Color.White,
                AutoCenter = true,
                GridDisplayMode = ImageBoxGridDisplayMode.Client, // 체크무늬 유지
                GridColor = Color.Gainsboro
            };
            _viewerPanel.Controls.Add(_imageBox);

            // 사이즈/줌 적용
            Shown += (_, __) =>
            {
                AdjustSizeToImage();
                ApplyCoverZoom();  // 체크무늬 보이지 않도록 약간 더 확대
                BringToFront();
            };
            _viewerPanel.Resize += (_, __) => ApplyCoverZoom(); // 창 크기 변경시에도 유지
        }

        private void ToggleMaximizeRestore()
        {
            WindowState = (WindowState == FormWindowState.Maximized)
                ? FormWindowState.Normal
                : FormWindowState.Maximized;
        }

        private void AdjustSizeToImage()
        {
            if (_imageBox.Image == null) return;

            var img = _imageBox.Image.Size;
            var wa = Screen.FromControl(this).WorkingArea;

            // 패딩/상단바 여백 포함 대략치
            int marginW = Padding.Left + Padding.Right + 8 + 8 + 16;
            int marginH = Padding.Top + Padding.Bottom + TOPBAR_H + 8 + 16;

            int targetW = img.Width + marginW;
            int targetH = img.Height + marginH;

            targetW = Math.Min(targetW, wa.Width - 24);
            targetH = Math.Min(targetH, wa.Height - 24);

            targetW = Math.Max(targetW, MinimumSize.Width);
            targetH = Math.Max(targetH, MinimumSize.Height);

            Size = new Size(targetW, targetH);
            CenterToParentSafe();
        }

        // 화면을 "덮는" 방식으로 살짝(2%) 더 확대해서 배경 체크무늬가 안 보이게
        private void ApplyCoverZoom()
        {
            if (_imageBox.Image == null) return;

            Size area = _viewerPanel.ClientSize;
            if (area.Width < 1 || area.Height < 1) return;

            Size img = _imageBox.Image.Size;
            float scaleX = (float)area.Width / img.Width;
            float scaleY = (float)area.Height / img.Height;

            float cover = Math.Max(scaleX, scaleY) * 1.02f; // 2% 여유 확대
            int zoomPercent = Math.Max(10, (int)Math.Round(cover * 100f));

            // ImageBox는 Zoom(%) 사용
            if (_imageBox.Zoom != zoomPercent)
            {
                _imageBox.Zoom = zoomPercent;
                // 가운데 정렬 유지
                _imageBox.CenterAt(new Point(img.Width / 2, img.Height / 2));
            }
        }

        private void CenterToParentSafe()
        {
            try
            {
                if (Owner != null) CenterToParent();
                else
                {
                    var wa = Screen.FromControl(this).WorkingArea;
                    Location = new Point(
                        wa.Left + (wa.Width - Width) / 2,
                        wa.Top + (wa.Height - Height) / 2);
                }
            }
            catch { /* ignore */ }
        }

        protected override void OnTextChanged(EventArgs e)
        {
            base.OnTextChanged(e);
            if (_lblTitle != null && !_lblTitle.IsDisposed)
                _lblTitle.Text = Text ?? string.Empty;
        }

        protected override void OnFormClosed(FormClosedEventArgs e)
        {
            if (_imageBox?.Image != null)
            {
                _imageBox.Image.Dispose();
                _imageBox.Image = null;
            }
            base.OnFormClosed(e);
        }
    }
}
