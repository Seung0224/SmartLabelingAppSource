using Guna.UI2.WinForms;
using System;
using System.Drawing;
using System.IO;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace SmartLabelingApp
{
    /// <summary>
    /// 프리트레인 가중치(yolo11x-seg.pt) 미존재 시 사용자에게 알리는 전용 다이얼로그.
    /// - ExportResultDialog 유사 카드 UI
    /// - 경로는 한 줄(TextBox, 가로 스크롤)
    /// - 하단 가로 버튼: Search / Download / Close
    /// - Download: 온라인에서 yolo11x-seg.pt 다운로드 → %LOCALAPPDATA%\\SmartLabelingApp\\weights 저장 → 닫기
    /// - Search: 로컬 .pt 선택 후 표준 위치로 복사(ProgressOverlay 표시) → 닫기
    /// </summary>
    public sealed class PretrainedWeightsDialog : Form
    {
        private readonly Guna2Panel _panel;
        private readonly Label _title;
        private readonly Label _line1;
        private readonly Label _line2;
        private readonly Panel _buttonHost;
        private readonly Guna2Button _btnSearch;
        private readonly Guna2Button _btnDownload;
        private readonly Guna2Button _btnClose;

        private const int MIN_DOWNLOAD_SHOW_MS = 3000; // 다운로드 진행 표시 최소 3초 보장
        private const int MIN_COPY_SHOW_MS = 2200; // 로컬 복사 진행 표시 최소 2.2초 보장
        private const int ANIM_TICK_MS = 18;   // 진행바 1%씩 올라가는 틱 속도(밀리초)


        public string WeightsPath { get; }

        // 단일 공식 URL (실패 시 재시도/미러 사용 안 함)
        private const string DOWNLOAD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11x-seg.pt";


        public PretrainedWeightsDialog(string weightsPath)
        {
            WeightsPath = string.IsNullOrWhiteSpace(weightsPath) ? GetDefaultPretrainedPath() : weightsPath;

            Text = "PRETRAINED";
            StartPosition = FormStartPosition.CenterParent;
            FormBorderStyle = FormBorderStyle.FixedDialog;
            MaximizeBox = MinimizeBox = false;
            BackColor = Color.White;
            Font = new Font("Segoe UI", 9f);

            ClientSize = new Size(640, 230);

            _panel = new Guna2Panel
            {
                Dock = DockStyle.Fill,
                BorderThickness = 0,
                BorderRadius = 12,
                FillColor = Color.White,
                Padding = new Padding(12)
            };
            Controls.Add(_panel);

            _title = new Label
            {
                AutoSize = true,
                Text = "⚠ Pretrained required",
                ForeColor = Color.DarkOrange,
                Font = new Font("Segoe UI", 11.5f, FontStyle.Bold),
                Location = new Point(12, 10)
            };
            _panel.Controls.Add(_title);

            _line1 = new Label
            {
                AutoSize = true,
                Font = new Font("Segoe UI", 10.5f, FontStyle.Bold),
                Location = new Point(12, 42),
                Text = "세그멘테이션 학습을 위해 yolo11x-seg.pt 파일이 필요합니다.",
                MaximumSize = new Size(_panel.ClientSize.Width - _panel.Padding.Horizontal, 0)
            };
            _panel.Controls.Add(_line1);


            _line2 = new Label
            {
                AutoSize = true,
                Font = new Font("Segoe UI", 9f),
                Location = new Point(12, 70), Text = "다음 경로에 파일이 없습니다: \n →" + (string.IsNullOrWhiteSpace(WeightsPath) ? "(경로 정보 없음)" : WeightsPath),
                MaximumSize = new Size(_panel.ClientSize.Width - _panel.Padding.Horizontal, 0) };
            _panel.Controls.Add(_line2);

            _panel.Resize += (s, e) =>
            {
                int w = _panel.ClientSize.Width - _panel.Padding.Horizontal;
                _line1.MaximumSize = new Size(w, 0);
                _line2.MaximumSize = new Size(w, 0);
            };

            // 하단 버튼(가로)
            _buttonHost = new Panel
            {
                Dock = DockStyle.Bottom,
                Height = 70,
                Padding = new Padding(10, 10, 10, 10)
            };
            _buttonHost.Resize += (s, e) => LayoutButtons();
            _panel.Controls.Add(_buttonHost);

            Guna2Button MakePill(string text) => new Guna2Button
            {
                AutoRoundedCorners = true,
                BorderRadius = 22,
                Height = 45,
                Text = text,
                Font = new Font("Segoe UI", 10f, FontStyle.Bold),
                FillColor = Color.White,
                ForeColor = Color.FromArgb(34, 38, 45),
                BorderColor = Color.LightGray,
                BorderThickness = 2,
                HoverState = { FillColor = Color.FromArgb(245, 245, 245) },
                PressedColor = Color.FromArgb(240, 240, 240),
                Cursor = Cursors.Hand
            };

            _btnSearch = MakePill("Search");
            _btnDownload = MakePill("Download");
            _btnClose = MakePill("Close");

            _buttonHost.Controls.Add(_btnSearch);
            _buttonHost.Controls.Add(_btnDownload);
            _buttonHost.Controls.Add(_btnClose);

            _btnClose.Click += (s, e) => this.Close();
            _btnDownload.Click += async (s, e) => await DownloadWithOverlayAndCloseAsync();
            _btnSearch.Click += (s, e) => SearchCopyWithOverlayAndClose();

            LayoutButtons();
        }

        private void LayoutButtons()
        {
            if (_buttonHost == null || _buttonHost.Controls.Count == 0) return;

            int gap = 8;
            int leftPadding = _buttonHost.Padding.Left;
            int rightPadding = _buttonHost.Padding.Right;
            int top = _buttonHost.Padding.Top;

            int innerWidth = _buttonHost.ClientSize.Width - leftPadding - rightPadding;
            int btnCount = 3;
            int btnWidth = Math.Max(100, (innerWidth - gap * (btnCount - 1)) / btnCount);
            int x = leftPadding;

            foreach (Control c in new[] { _btnSearch, _btnDownload, _btnClose })
            {
                c.SetBounds(x, top, btnWidth, 45);
                x += btnWidth + gap;
            }
        }

        private static string GetDefaultPretrainedPath()
        {
            var root = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SmartLabelingApp", "weights");
            return Path.Combine(root, "yolo11x-seg.pt");
        }

        // === DOWNLOAD (with ProgressOverlay) ===
        private async Task DownloadWithOverlayAndCloseAsync()
        {
            try
            {
                string dest = WeightsPath;
                Directory.CreateDirectory(Path.GetDirectoryName(dest) ?? ".");

                using (var overlay = new ProgressOverlay(this, "Downloading pretrained...", true))
                {
                    var t0 = DateTime.UtcNow;

                    int shownPct = 0;        // 화면에 실제로 표시되는 퍼센트
                    int realTargetPct = 0;   // 네트워크 콜백이 알려주는 실제 진행률(0~100)
                    long totalBytes = -1;    // Content-Length (없으면 -1)
                    var cts = new CancellationTokenSource();

                    // ⏱ 시간 캡 + 부드러운 1% 단위 애니메이션
                    var anim = Task.Run(async () =>
                    {
                        while (!cts.IsCancellationRequested)
                        {
                            // “표시 상한선”: 최소 표시 시간(MIN_DOWNLOAD_SHOW_MS) 동안 99%를 넘지 않도록 제한
                            int cap = (int)Math.Min(99, (DateTime.UtcNow - t0).TotalMilliseconds * 100.0 / MIN_DOWNLOAD_SHOW_MS);

                            // 용량 미상(totalBytes<=0)일 때는 시간만으로 90%까지 끌어가고, 완료 시 100% 처리
                            int desired = (totalBytes > 0)
                                ? Math.Min(realTargetPct, cap)
                                : Math.Min(cap, 90);

                            if (shownPct < desired) shownPct++;

                            string tip;
                            if (totalBytes > 0)
                                tip = $"{shownPct}%  (~{(shownPct * totalBytes / 100.0) / 1048576.0:0.0} / {(totalBytes / 1048576.0):0.0} MB)";
                            else
                                tip = shownPct < 95 ? $"{shownPct}%" : "Finalizing...";

                            overlay.Report(shownPct, tip);
                            await Task.Delay(ANIM_TICK_MS);
                        }
                    });

                    overlay.Report(0, "Connecting...");

                    // 실제 다운로드(실제 진행률은 realTargetPct에 반영, 표시 속도는 위 anim에서 시간으로 제한)
                    await DownloadFileAsync(
                        DOWNLOAD_URL,
                        dest,
                        (pct, transferred, total) =>
                        {
                            totalBytes = total;
                            realTargetPct = pct; // anim 루프에서 cap으로 표시 속도를 제한
                        });

                    // ✅ 완료 처리: 100%까지 올리고, 최소 노출 시간 보장
                    int elapsedMs = (int)(DateTime.UtcNow - t0).TotalMilliseconds;
                    int remain = Math.Max(0, MIN_DOWNLOAD_SHOW_MS - elapsedMs);

                    realTargetPct = 100; // 이제 100%까지 표시 허용
                    if (remain > 0) await Task.Delay(remain);
                    await Task.Delay(200); // 100% 상태 잠깐 유지

                    cts.Cancel();
                    try { await anim; } catch { /* ignore */ }
                    overlay.Report(100, "Completed");
                }

                this.DialogResult = DialogResult.OK;
                this.Close();
            }
            catch
            {
                var dlg = new Guna.UI2.WinForms.Guna2MessageDialog
                {
                    Parent = this,
                    Caption = "다운로드 실패",
                    Text = "다운로드가 되지 않습니다. 관리자에게 연락하세요.",
                    Buttons = Guna.UI2.WinForms.MessageDialogButtons.OK,
                    Icon = Guna.UI2.WinForms.MessageDialogIcon.Error,
                    Style = Guna.UI2.WinForms.MessageDialogStyle.Light
                };
                dlg.Show();
            }
        }



        private static async Task DownloadFileAsync(string url, string destPath,
            Action<int, long, long> onProgress)
        {
            var temp = destPath + ".part";
            if (File.Exists(temp)) File.Delete(temp);

            using (var http = new HttpClient())
            using (var resp = await http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead))
            {
                resp.EnsureSuccessStatusCode();
                var total = resp.Content.Headers.ContentLength ?? -1L;

                using (var src = await resp.Content.ReadAsStreamAsync())
                using (var dst = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true))
                {
                    var buf = new byte[81920];
                    long read = 0;
                    int n, last = -1;
                    while ((n = await src.ReadAsync(buf, 0, buf.Length)) > 0)
                    {
                        await dst.WriteAsync(buf, 0, n);
                        read += n;
                        int pct = total > 0 ? (int)(read * 100 / total) : 0;
                        if (pct != last) { last = pct; onProgress?.Invoke(pct, read, total); }
                    }
                }
            }

            if (File.Exists(destPath)) File.Delete(destPath);
            File.Move(temp, destPath);
        }

        // === SEARCH (copy with ProgressOverlay) ===
        private async void SearchCopyWithOverlayAndClose()
        {
            using (var ofd = new OpenFileDialog())
            {
                ofd.Title = "pretrained .pt 선택";
                ofd.Filter = "PyTorch Weights (*.pt)|*.pt|All files (*.*)|*.*";
                ofd.CheckFileExists = true;
                if (ofd.ShowDialog(this) != DialogResult.OK) return;

                try
                {
                    var src = ofd.FileName;
                    var dest = WeightsPath; // 표준 캐시 위치
                    Directory.CreateDirectory(Path.GetDirectoryName(dest) ?? ".");

                    var fi = new FileInfo(src);
                    long total = fi.Length;

                    using (var overlay = new ProgressOverlay(this, "Copying pretrained...", true))
                    {
                        var t0 = DateTime.UtcNow;
                        overlay.Report(0, "Preparing...");

                        int shownPct = 0;
                        int realTargetPct = 0; // 실제 복사 진행률
                        var cts = new CancellationTokenSource();

                        var anim = Task.Run(async () =>
                        {
                            while (!cts.IsCancellationRequested)
                            {
                                // 복사도 표시 속도를 시간으로 제한(최소 노출 시간 보장)
                                int cap = (int)Math.Min(99, (DateTime.UtcNow - t0).TotalMilliseconds * 100.0 / MIN_COPY_SHOW_MS);
                                int desired = Math.Min(realTargetPct, cap);

                                if (shownPct < desired) shownPct++;

                                string tip = $"{shownPct}%  ({(shownPct * total / 100.0) / 1048576.0:0.0} / {(total / 1048576.0):0.0} MB)";
                                overlay.Report(shownPct, tip);
                                await Task.Delay(ANIM_TICK_MS);
                            }
                        });

                        // 실제 복사(빠르게 끝나도 위 anim의 시간 캡 때문에 눈에 보이게 진행됨)
                        CopyFileWithProgress(src, dest, (read) =>
                        {
                            if (total > 0)
                            {
                                int pct = (int)(read * 100 / total);
                                if (pct > realTargetPct) realTargetPct = pct;
                            }
                        });

                        // ✅ 완료 처리: 100%까지 올리고, 최소 노출 시간 보장
                        int elapsedMs = (int)(DateTime.UtcNow - t0).TotalMilliseconds;
                        int remain = Math.Max(0, MIN_COPY_SHOW_MS - elapsedMs);

                        realTargetPct = 100;
                        if (remain > 0) await Task.Delay(remain);
                        await Task.Delay(150);

                        cts.Cancel();
                        try { await anim; } catch { /* ignore */ }
                        overlay.Report(100, "Completed");
                    }

                    if (_line2 != null) _line2.Text = dest;
                    this.DialogResult = DialogResult.OK;
                    this.Close(); // 닫고 메인 UI로 복귀
                }
                catch (Exception ex)
                {
                    var dlg = new Guna2MessageDialog
                    {
                        Caption = "복사 실패",
                        Text = ex.Message,
                        Buttons = MessageDialogButtons.OK,
                        Icon = MessageDialogIcon.Error,
                        Style = MessageDialogStyle.Light
                    };
                    dlg.Show();
                }
            }
        }


        private static void CopyFileWithProgress(string src, string dest, Action<long> onProgress)
        {
            string temp = dest + ".part";
            if (File.Exists(temp)) File.Delete(temp);

            using (var input = new FileStream(src, FileMode.Open, FileAccess.Read, FileShare.Read))
            using (var output = new FileStream(temp, FileMode.Create, FileAccess.Write, FileShare.None, 81920, useAsync: true))
            {
                byte[] buffer = new byte[81920];
                int bytesRead;
                long totalRead = 0;
                while ((bytesRead = input.Read(buffer, 0, buffer.Length)) > 0)
                {
                    output.Write(buffer, 0, bytesRead);
                    totalRead += bytesRead;
                    onProgress?.Invoke(totalRead);
                }
            }

            if (File.Exists(dest)) File.Delete(dest);
            File.Move(temp, dest);
        }

        public static DialogResult ShowForMissingDefault(IWin32Window owner, string weightsPath)
        {
            using (var dlg = new PretrainedWeightsDialog(weightsPath))
                return dlg.ShowDialog(owner);
        }
    }
}
