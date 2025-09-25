using Guna.UI2.WinForms;
using System;
using System.Drawing;
using System.Windows.Forms;

public class ProgressOverlay : IDisposable
{
    private readonly Form _overlay;
    private readonly Guna2ProgressBar _bar;
    private readonly Label _title;
    private readonly Label _status;
    private readonly Form _owner;
    private readonly Cursor _oldCursor;
    private readonly string _baseTitle;
    private readonly bool _showPercentInTitle;

    // ★ 추가: 중앙정렬/추적용 핸들러 보관(Dispose에서 해제)
    private EventHandler _ownerMoveHandler;
    private EventHandler _ownerSizeHandler;
    private FormClosedEventHandler _ownerClosedHandler;
    private EventHandler _ownerActivatedHandler;

    public ProgressOverlay(Form owner, string title, bool showPercentInTitle = true)
    {
        _showPercentInTitle = showPercentInTitle;

        _owner = owner;
        _oldCursor = owner.Cursor;
        owner.Cursor = Cursors.WaitCursor;

        _overlay = new Form();
        _overlay.FormBorderStyle = FormBorderStyle.None;
        _overlay.StartPosition = FormStartPosition.Manual;
        _overlay.ShowInTaskbar = false;
        _overlay.TopMost = true;                 // ★ 항상 최상위로
        _overlay.BackColor = Color.Black;
        _overlay.Opacity = 0.85;
        _overlay.Owner = owner;

        Rectangle r = owner.RectangleToScreen(owner.ClientRectangle);
        _overlay.Bounds = r;

        var card = new Guna2Panel();
        card.Size = new Size(420, 140);
        card.BorderRadius = 12;
        card.BorderThickness = 1;
        card.BorderColor = Color.Silver;
        card.FillColor = Color.White;
        card.ShadowDecoration.Enabled = true;
        card.ShadowDecoration.Depth = 12;
        card.Padding = new Padding(16, 14, 16, 14);

        _title = new Label();
        _title.Dock = DockStyle.Top;
        _title.Height = 36;
        _title.TextAlign = ContentAlignment.MiddleCenter;
        _title.Font = new Font("Segoe UI", 11f, FontStyle.Bold);

        _baseTitle = string.IsNullOrEmpty(title) ? "Working..." : title;
        _title.Text = _baseTitle;

        _bar = new Guna2ProgressBar();
        _bar.Dock = DockStyle.Top;
        _bar.Height = 14;
        _bar.Minimum = 0;
        _bar.Maximum = 100;
        _bar.Value = 0;
        _bar.BorderRadius = 6;
        _bar.FillColor = Color.Gainsboro;

        _status = new Label();
        _status.Dock = DockStyle.Top;
        _status.Height = 28;
        _status.TextAlign = ContentAlignment.MiddleCenter;
        _status.Font = new Font("Segoe UI", 9f, FontStyle.Regular);
        _status.Text = string.Empty;

        card.Controls.Add(_status);
        card.Controls.Add(_bar);
        card.Controls.Add(_title);
        _overlay.Controls.Add(card);

        // ★ 추가: 중앙 정렬 함수
        void CenterToOwnerNow()
        {
            if (_overlay.IsDisposed) return;

            Rectangle targetBounds;
            if (_owner != null && !_owner.IsDisposed && _owner.Visible)
            {
                targetBounds = _owner.RectangleToScreen(_owner.ClientRectangle);
            }
            else
            {
                // 오너가 없거나 보이지 않을 때는 커서가 있는 모니터 기준
                targetBounds = Screen.FromPoint(Cursor.Position).WorkingArea;
            }

            // 오버레이를 오너에 맞춰 갱신
            _overlay.Bounds = targetBounds;

            // 카드 중앙 배치
            int x = (_overlay.ClientSize.Width - card.Width) / 2;
            int y = (_overlay.ClientSize.Height - card.Height) / 2;
            if (x < 0) x = 0; if (y < 0) y = 0;
            card.Left = x;
            card.Top = y;
        }

        _overlay.Shown += (s, e) => CenterToOwnerNow();  // ★ 표시 직후 중앙
        _overlay.Load += (s, e) => CenterToOwnerNow();   // ★ 일부 DPI/레이아웃 케이스 보강

        // ★ 추가: 오너 이동/리사이즈/활성화/종료 시 다시 중앙
        _ownerMoveHandler = (s, e) => CenterToOwnerNow();
        _ownerSizeHandler = (s, e) => CenterToOwnerNow();
        _ownerClosedHandler = (s, e) => { try { _overlay.Close(); } catch { } };
        _ownerActivatedHandler = (s, e) => { try { _overlay.BringToFront(); } catch { } };

        if (_owner != null && !_owner.IsDisposed)
        {
            _owner.Move += _ownerMoveHandler;
            _owner.SizeChanged += _ownerSizeHandler;
            _owner.FormClosed += _ownerClosedHandler;
            _owner.Activated += _ownerActivatedHandler;
        }

        _overlay.Show(owner);
        _overlay.BringToFront();
        CenterToOwnerNow(); // ★ 최종 한 번 더
    }

    public void Report(int percent, string status)
    {
        if (_overlay == null || _overlay.IsDisposed) return;

        if (_overlay.InvokeRequired)
        {
            _overlay.BeginInvoke((Action)(() => Report(percent, status)));
            return;
        }

        if (percent < 0) percent = 0;
        if (percent > 100) percent = 100;

        if (_showPercentInTitle)
            _title.Text = _baseTitle + " " + percent.ToString() + "%";

        _bar.Value = percent;
        _status.Text = string.IsNullOrEmpty(status) ? "" : status;
    }

    public void Dispose()
    {
        try
        {
            if (_overlay != null)
            {
                if (_overlay.InvokeRequired)
                    _overlay.Invoke((Action)(() => _overlay.Close()));
                else
                    _overlay.Close();
            }
        }
        finally
        {
            // ★ 추가: 이벤트 해제
            if (_owner != null && !_owner.IsDisposed)
            {
                if (_ownerMoveHandler != null) _owner.Move -= _ownerMoveHandler;
                if (_ownerSizeHandler != null) _owner.SizeChanged -= _ownerSizeHandler;
                if (_ownerClosedHandler != null) _owner.FormClosed -= _ownerClosedHandler;
                if (_ownerActivatedHandler != null) _owner.Activated -= _ownerActivatedHandler;
                _owner.Cursor = _oldCursor;
            }
        }
    }
}
