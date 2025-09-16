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
        _overlay.TopMost = owner.TopMost;
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
        _bar.FillColor = Color.Gainsboro; // 은은한 바탕

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

        _overlay.Shown += (s, e) =>
        {
            card.Left = (_overlay.ClientSize.Width - card.Width) / 2;
            card.Top = (_overlay.ClientSize.Height - card.Height) / 2;
        };

        _overlay.Show(owner);
        _overlay.BringToFront();
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
            if (_owner != null) _owner.Cursor = _oldCursor;
        }
    }
}