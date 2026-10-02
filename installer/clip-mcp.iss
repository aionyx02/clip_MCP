; The clip-mcp installer. Built by .github/workflows/release.yml, which passes:
;   AppVersion    the version, from pyproject.toml
;   Wheel         the clip-mcp wheel's file name, in ..\dist
;   FfmpegUrl     the pinned FFmpeg build, downloaded at install time
;   FfmpegSha256  its checksum, so a changed or broken download is refused
;   FfmpegFolder  the folder inside that zip
;
; Everything goes into one folder of the user's own (no administrator rights): uv, Python,
; clip-mcp's packages and FFmpeg. Uninstalling takes clip-mcp out of every AI client first,
; asks before deleting the user's projects, and then removes that folder whole.
; No AI client is connected here: the user chooses them in the editor.

#ifndef AppVersion
  #error AppVersion is passed by the build
#endif

#define AppName "clip-mcp 剪輯"

[Setup]
AppId={{E83AD81B-2150-4ABA-A8BA-27C3B683041B}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher=aionyx
AppPublisherURL=https://github.com/aionyx02/clip_MCP
AppSupportURL=https://github.com/aionyx02/clip_MCP/issues
DefaultDirName={localappdata}\Programs\clip-mcp
DisableDirPage=yes
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=..\dist
OutputBaseFilename=clip-mcp-setup-{#AppVersion}
SetupIconFile=..\src\app\ui\static\icon.ico
UninstallDisplayIcon={app}\icon.ico
UninstallDisplayName={#AppName}
LicenseFile=..\LICENSE
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; tar.exe, which unpacks FFmpeg, comes with Windows 10 1803 and later.
MinVersion=10.0.17763
; An update closes what holds clip-mcp's files — the editor, and any AI client running clip-mcp.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "zh_tw"; MessagesFile: "build\ChineseTraditional.isl"

[CustomMessages]
zh_tw.DesktopIcon=建立桌面捷徑
zh_tw.LaunchApp=開啟 clip-mcp 剪輯
zh_tw.Unpacking=正在解開 FFmpeg（影片處理元件）…
zh_tw.Installing=正在安裝 Python 與 clip-mcp 的元件，第一次需要幾分鐘，請不要關閉…
zh_tw.InstallFailed=元件沒有安裝成功，最常見的原因是網路中斷。請確認網路後再執行一次安裝程式。%n%n詳細記錄：%1
zh_tw.Finished=安裝完成。%n%n開啟後，到左邊的「連接 AI」選擇要讓哪個 AI 程式使用 clip-mcp；安裝程式不會自動修改任何 AI 程式的設定。
zh_tw.AskDeleteData=也要刪除你的專案與素材分析資料嗎？%n%n選「是」會把資料移到資源回收筒；選「否」會保留，之後重新安裝還能接著用。%n輸出到「影片\clip-mcp」的成品影片無論如何都不會被刪除。

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"

[InstallDelete]
; Last version's wheel; only the one being installed is kept.
Type: files; Name: "{app}\wheels\*.whl"

[Files]
Source: "build\uv.exe"; DestDir: "{app}\uv"; Flags: ignoreversion
Source: "..\dist\{#Wheel}"; DestDir: "{app}\wheels"; Flags: ignoreversion
Source: "install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "build\FFMPEG-NOTICE.txt"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "..\src\app\ui\static\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\bin\clip-mcp-editor.exe"; IconFilename: "{app}\icon.ico"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\bin\clip-mcp-editor.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\bin\clip-mcp-editor.exe"; Description: "{cm:LaunchApp}"; Flags: postinstall nowait skipifsilent
; An update started from the editor runs silently and opens the editor again when it is done.
Filename: "{app}\bin\clip-mcp-editor.exe"; Flags: nowait; Check: Relaunch

[UninstallDelete]
; uv, Python and the packages were made by the install step, so the installer does not know
; their files one by one; the folder goes whole.
Type: filesandordirs; Name: "{app}"

[Code]
var
  DeleteData: Boolean;

function Relaunch: Boolean;
begin
  Result := WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

function FfmpegCurrent: Boolean;
var
  Installed: AnsiString;
begin
  { The same FFmpeg as last time is not downloaded again on an update. }
  Result := FileExists(ExpandConstant('{app}\ffmpeg\ffmpeg.exe'))
    and LoadStringFromFile(ExpandConstant('{app}\ffmpeg\build.txt'), Installed)
    and (Trim(String(Installed)) = '{#FfmpegSha256}');
end;

function OnDownloadProgress(const Url, FileName: String; const Progress, ProgressMax: Int64): Boolean;
begin
  Result := True;
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
begin
  Result := '';
  if FfmpegCurrent then
    Exit;
  try
    { Checked against the checksum the build recorded: anything else is refused. }
    DownloadTemporaryFile('{#FfmpegUrl}', 'ffmpeg.zip', '{#FfmpegSha256}', @OnDownloadProgress);
  except
    Result := GetExceptionMessage;
  end;
end;

procedure InstallFfmpeg;
var
  Code: Integer;
  Bin: String;
begin
  if FfmpegCurrent then
    Exit;
  WizardForm.StatusLabel.Caption := CustomMessage('Unpacking');
  Exec(ExpandConstant('{sys}\tar.exe'), '-xf "' + ExpandConstant('{tmp}\ffmpeg.zip') + '" -C "' + ExpandConstant('{tmp}') + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);
  Bin := ExpandConstant('{tmp}\{#FfmpegFolder}\bin\');
  ForceDirectories(ExpandConstant('{app}\ffmpeg'));
  if FileCopy(Bin + 'ffmpeg.exe', ExpandConstant('{app}\ffmpeg\ffmpeg.exe'), False)
    and FileCopy(Bin + 'ffprobe.exe', ExpandConstant('{app}\ffmpeg\ffprobe.exe'), False) then
    SaveStringToFile(ExpandConstant('{app}\ffmpeg\build.txt'), '{#FfmpegSha256}', False);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
begin
  if CurStep <> ssPostInstall then
    Exit;
  InstallFfmpeg;
  WizardForm.StatusLabel.Caption := CustomMessage('Installing');
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Exec('powershell.exe',
    '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '" -App "' + ExpandConstant('{app}')
      + '" -Wheel "' + ExpandConstant('{app}\wheels\{#Wheel}') + '"',
    '', SW_HIDE, ewWaitUntilTerminated, Code);
  WizardForm.ProgressGauge.Style := npbstNormal;
  if Code <> 0 then
    SuppressibleMsgBox(FmtMessage(CustomMessage('InstallFailed'), [ExpandConstant('{app}\install.log')]),
      mbError, MB_OK, IDOK);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then
    WizardForm.FinishedLabel.Caption := CustomMessage('Finished');
end;

function InitializeUninstall: Boolean;
begin
  { Asked once, before anything is removed; a silent uninstall keeps the data. }
  DeleteData := (not UninstallSilent) and
    (MsgBox(CustomMessage('AskDeleteData'), mbConfirmation, MB_YESNO or MB_DEFBUTTON2) = IDYES);
  Result := True;
end;

procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  Code: Integer;
  Params: String;
begin
  if CurUninstallStep <> usUninstall then
    Exit;
  { clip-mcp takes itself out of every AI client, and removes its shortcuts and config, first. }
  Params := 'uninstall --no-prompt';
  if DeleteData then
    Params := Params + ' --delete-data';
  if FileExists(ExpandConstant('{app}\bin\clip-mcp.exe')) then
    Exec(ExpandConstant('{app}\bin\clip-mcp.exe'), Params, '', SW_HIDE, ewWaitUntilTerminated, Code);
end;
