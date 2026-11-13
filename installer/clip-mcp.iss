; The clip-mcp installer. Built by .github/workflows/release.yml, which passes:
;   AppVersion        the version, from pyproject.toml
;   Payload           the folder build_payload.ps1 made: runtime, program, bin, ffmpeg
;   GpuSizeMB         the GPU pack's download, in MB, and
;   GpuInstalledMB    what it takes once installed, in MB and
;   GpuInstalledBytes in bytes, all from app.gpu.PACK
;
; Everything is in the installer, so installing downloads nothing: Python, every package,
; clip-mcp and FFmpeg are copied into one folder of the user's own (no administrator rights).
; The one exception is the GPU pack (cuBLAS 12, for transcribing on an NVIDIA card), offered
; only when there is an NVIDIA card, and downloaded only when chosen.
; Uninstalling takes clip-mcp out of every AI client first, asks before deleting the user's
; projects, and then removes that folder whole. No AI client is connected here: the user
; chooses them in the editor.

#ifndef AppVersion
  #error AppVersion is passed by the build
#endif
#ifndef Payload
  #error Payload is passed by the build
#endif

#define AppName "clip-mcp 剪輯"
#define AppId "{E83AD81B-2150-4ABA-A8BA-27C3B683041B}"

[Setup]
AppId={{#AppId}
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
MinVersion=10.0.17763
; An update closes what holds clip-mcp's files — the editor, and any AI client running clip-mcp.
CloseApplications=yes
RestartApplications=no
; Inno Setup 6.7 makes Setup refuse junctions made by an ordinary user, and every program it
; starts inherits that: clip-mcp's setup step and the editor opened at the end would fail on a
; workspace or Videos folder the user reached through a junction of their own. The guard
; protects installers run as administrator; this one runs as the user, in the user's own folder.
RedirectionGuard=no
; Whether the GPU pack is wanted is decided again from this computer on a first install; on an
; update, the choice made last time stands (see InitializeWizard).
UsePreviousComponents=yes

[Languages]
Name: "zh_tw"; MessagesFile: "build\ChineseTraditional.isl"

[CustomMessages]
zh_tw.DesktopIcon=建立桌面捷徑
zh_tw.LaunchApp=開啟 clip-mcp 剪輯
zh_tw.CompProgram=clip-mcp 剪輯（程式、Python 與 FFmpeg）
zh_tw.CompGpu=GPU 加速轉錄：下載 NVIDIA cuBLAS（約 {#GpuSizeMB} MB，安裝後佔 {#GpuInstalledMB} MB）
zh_tw.CompGpuSystem=GPU 加速轉錄：這台電腦已有 cuBLAS（%1），不勾選就直接用它；勾選則另外下載一份自帶的（約 {#GpuSizeMB} MB）
zh_tw.StepGpu=正在下載 GPU 加速元件（約 {#GpuSizeMB} MB），請不要關閉…
zh_tw.StepSetup=正在完成設定…
zh_tw.InstallFailed=設定沒有完成。最常見的原因是有 AI 程式（例如 Claude Desktop）正在使用 clip-mcp。請關掉 AI 程式後再執行一次安裝程式。%n%n詳細記錄：%1
zh_tw.GpuFailed=GPU 加速元件沒有下載成功，clip-mcp 會先在處理器上轉錄。之後可以在編輯器的「儲存空間」頁重新下載。%n%n詳細記錄：%1
zh_tw.FinishedFailed=安裝沒有完成。請關掉正在使用 clip-mcp 的 AI 程式後，再執行一次安裝程式。%n%n詳細記錄：%1
zh_tw.Finished=安裝完成。%n%n開啟後，到左邊的「連接 AI」選擇要讓哪個 AI 程式使用 clip-mcp；安裝程式不會自動修改任何 AI 程式的設定。
zh_tw.AskDeleteData=也要刪除你的專案與素材分析資料嗎？%n%n選「是」會把資料移到資源回收筒；選「否」會保留，之後重新安裝還能接著用。%n輸出到「影片\clip-mcp」的成品影片無論如何都不會被刪除。

[Types]
Name: "custom"; Description: "custom"; Flags: iscustom

[Components]
Name: "program"; Description: "{cm:CompProgram}"; Types: custom; Flags: fixed
Name: "gpu"; Description: "{cm:CompGpu}"; Types: custom; ExtraDiskSpaceRequired: {#GpuInstalledBytes}; Check: HasNvidiaCard

[Tasks]
Name: "desktopicon"; Description: "{cm:DesktopIcon}"

[InstallDelete]
; The last version's Python, packages and program go whole, so nothing it had and this one
; does not is left behind to be imported. The GPU pack and FFmpeg's folder stay.
Type: filesandordirs; Name: "{app}\runtime"
Type: filesandordirs; Name: "{app}\program"
Type: filesandordirs; Name: "{app}\bin"
; What the installers before 0.8 made at install time with uv.
Type: filesandordirs; Name: "{app}\uv"
Type: filesandordirs; Name: "{app}\python"
Type: filesandordirs; Name: "{app}\tools"
Type: filesandordirs; Name: "{app}\cache"
Type: filesandordirs; Name: "{app}\wheels"
Type: files; Name: "{app}\ffmpeg\build.txt"

[Files]
; Separate folders, so that a later update can replace the program alone (#293).
Source: "{#Payload}\runtime\*"; DestDir: "{app}\runtime"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Payload}\program\*"; DestDir: "{app}\program"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Payload}\bin\*"; DestDir: "{app}\bin"; Flags: ignoreversion
Source: "{#Payload}\ffmpeg\*"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "build\FFMPEG-NOTICE.txt"; DestDir: "{app}\ffmpeg"; Flags: ignoreversion
Source: "install.ps1"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\src\app\ui\static\icon.ico"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{userprograms}\{#AppName}"; Filename: "{app}\bin\clip-mcp-editor.exe"; IconFilename: "{app}\icon.ico"
Name: "{userdesktop}\{#AppName}"; Filename: "{app}\bin\clip-mcp-editor.exe"; IconFilename: "{app}\icon.ico"; Tasks: desktopicon

[Run]
Filename: "{app}\bin\clip-mcp-editor.exe"; Description: "{cm:LaunchApp}"; Flags: postinstall nowait skipifsilent; Check: ComponentsInstalled
; An update started from the editor opens the editor again from CurStepChanged, not from here:
; [Run] entries without postinstall are checked before ssPostInstall, when nothing is installed yet.

[UninstallDelete]
; Bytecode, the GPU pack and the install log were made after the files were copied, so the
; installer does not know them one by one; the folder goes whole.
Type: filesandordirs; Name: "{app}"

[Code]
const
  { CUDA 12 runs on a driver from 528.33 on, through minor version compatibility (NVIDIA's
    CUDA compatibility guide); written as Windows reports it, 5 digits: 52833. }
  MinNvidiaDriver = 52833;

var
  DeleteData: Boolean;
  Installed: Boolean;
  GpuFailed: Boolean;
  NvidiaCard: Boolean;
  SystemCublas: String;

function ComponentsInstalled: Boolean;
begin
  Result := Installed;
end;

function Relaunch: Boolean;
begin
  Result := Installed and WizardSilent and (ExpandConstant('{param:RELAUNCH|0}') = '1');
end;

function HasNvidiaCard: Boolean;
begin
  Result := NvidiaCard;
end;

{ Windows writes an NVIDIA driver's version as 32.0.15.6614 for 566.14: its last five digits. }
function NvidiaDriverVersion(const Windows: String): Integer;
var
  Digits: String;
  I: Integer;
begin
  Digits := '';
  for I := 1 to Length(Windows) do
    if (Windows[I] >= '0') and (Windows[I] <= '9') then
      Digits := Digits + Windows[I];
  if Length(Digits) < 5 then
    Result := 0
  else
    Result := StrToIntDef(Copy(Digits, Length(Digits) - 4, 5), 0);
end;

{ An NVIDIA card whose driver runs CUDA 12, from Windows' own list of display adapters: there
  is no nvidia-smi to rely on before the driver's tools are on PATH. }
function FindNvidiaCard: Boolean;
var
  Locator, Service, Cards, Card: Variant;
  I: Integer;
  Name, Version: String;
begin
  Result := False;
  try
    Locator := CreateOleObject('WbemScripting.SWbemLocator');
    Service := Locator.ConnectServer('.', 'root\CIMV2');
    Cards := Service.ExecQuery('SELECT Name, DriverVersion FROM Win32_VideoController');
    for I := 0 to Cards.Count - 1 do begin
      try
        Card := Cards.ItemIndex(I);
        Name := Card.Name;
        Version := Card.DriverVersion;
        if (Pos('NVIDIA', Uppercase(Name)) > 0) and (NvidiaDriverVersion(Version) >= MinNvidiaDriver) then begin
          Log('NVIDIA card: ' + Name + ', driver ' + Version);
          Result := True;
        end;
      except
        Log('A display adapter could not be read: ' + GetExceptionMessage);
      end;
    end;
  except
    Log('Display adapters could not be listed: ' + GetExceptionMessage);
  end;
end;

function HasCublas(const Folder: String): Boolean;
begin
  Result := (Folder <> '') and FileExists(AddBackslash(Folder) + 'cublas64_12.dll')
    and FileExists(AddBackslash(Folder) + 'cublasLt64_12.dll');
end;

{ The system's cuBLAS 12, looked for where clip-mcp looks (app.gpu): PATH, then CUDA_PATH. }
function FindSystemCublas: String;
var
  Found: String;
begin
  Result := '';
  Found := FileSearch('cublas64_12.dll', GetEnv('PATH'));
  if (Found <> '') and HasCublas(ExtractFileDir(Found)) then
    Result := ExtractFileDir(Found)
  else if (GetEnv('CUDA_PATH') <> '') and HasCublas(AddBackslash(GetEnv('CUDA_PATH')) + 'bin') then
    Result := AddBackslash(GetEnv('CUDA_PATH')) + 'bin';
end;

function InitializeSetup: Boolean;
begin
  NvidiaCard := FindNvidiaCard;
  SystemCublas := FindSystemCublas;
  Log('System cuBLAS: ' + SystemCublas);
  Result := True;
end;

{ What the GPU component starts as. An update keeps last time's choice; a first install, or an
  update from a version without the choice, decides from this computer: checked, unless the
  system's own cuBLAS will do, and then it says so. }
procedure InitializeWizard;
var
  I: Integer;
  Chosen: String;
begin
  if not NvidiaCard then
    Exit;
  if RegQueryStringValue(HKCU, 'Software\Microsoft\Windows\CurrentVersion\Uninstall\{#AppId}_is1',
      'Inno Setup: Selected Components', Chosen) then
    Exit;
  if (SystemCublas <> '') and not HasCublas(ExpandConstant('{localappdata}\Programs\clip-mcp\gpu')) then begin
    WizardSelectComponents('!gpu');
    for I := 0 to WizardForm.ComponentsList.Items.Count - 1 do
      if WizardForm.ComponentsList.ItemCaption[I] = CustomMessage('CompGpu') then
        WizardForm.ComponentsList.ItemCaption[I] := FmtMessage(CustomMessage('CompGpuSystem'), [SystemCublas]);
  end;
end;

{ Without an NVIDIA card there is nothing to choose. }
function ShouldSkipPage(PageID: Integer): Boolean;
begin
  Result := (PageID = wpSelectComponents) and not NvidiaCard;
end;

{ Whatever runs from this folder holds its files open, and Windows will not replace an open
  file: an AI client running clip-mcp, or the editor, is stopped first. The AI client starts
  clip-mcp again the next time it needs it. }
function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  Code: Integer;
begin
  Exec('powershell.exe', '-NoProfile -ExecutionPolicy Bypass -Command "$here = ''' + AddBackslash(ExpandConstant('{app}'))
    + '''; Get-Process | Where-Object { $_.Path -and $_.Path.StartsWith($here, [StringComparison]::OrdinalIgnoreCase) }'
    + ' | Stop-Process -Force -ErrorAction SilentlyContinue"', '', SW_HIDE, ewWaitUntilTerminated, Code);
  Result := '';
end;

{ install.ps1 announces each step as "STEP <name>"; that becomes the sentence on the page, and
  every other line it prints is shown under it, so a slow download is visibly moving. }
procedure OnInstallOutput(const S: String; const Error, FirstLine: Boolean);
var
  Line, Step: String;
begin
  Line := Trim(S);
  if Copy(Line, 1, 5) = 'STEP ' then begin
    Step := Copy(Line, 6, MaxInt);
    if Step = 'gpu' then WizardForm.StatusLabel.Caption := CustomMessage('StepGpu')
    else if Step = 'setup' then WizardForm.StatusLabel.Caption := CustomMessage('StepSetup');
    WizardForm.FilenameLabel.Caption := '';
  end else if Line <> '' then
    WizardForm.FilenameLabel.Caption := Copy(Line, 1, 110);
end;

procedure CurStepChanged(CurStep: TSetupStep);
var
  Code: Integer;
  Params: String;
begin
  if CurStep <> ssPostInstall then
    Exit;
  WizardForm.ProgressGauge.Style := npbstMarquee;
  Params := '-NoProfile -ExecutionPolicy Bypass -File "' + ExpandConstant('{app}\install.ps1') + '"'
    + ' -App "' + ExpandConstant('{app}') + '"';
  if WizardIsComponentSelected('gpu') then
    Params := Params + ' -Gpu';
  Installed := ExecAndLogOutput('powershell.exe', Params, '', SW_HIDE, ewWaitUntilTerminated, Code, @OnInstallOutput)
    and ((Code = 0) or (Code = 4));
  GpuFailed := Code = 4;
  WizardForm.ProgressGauge.Style := npbstNormal;
  if not Installed then
    SuppressibleMsgBox(FmtMessage(CustomMessage('InstallFailed'), [ExpandConstant('{app}\install.log')]),
      mbError, MB_OK, IDOK)
  else if GpuFailed then
    SuppressibleMsgBox(FmtMessage(CustomMessage('GpuFailed'), [ExpandConstant('{app}\install.log')]),
      mbInformation, MB_OK, IDOK);
  { An update started from the editor runs silently and opens the editor again when it is done. }
  if Relaunch then
    Exec(ExpandConstant('{app}\bin\clip-mcp-editor.exe'), '', '', SW_SHOWNORMAL, ewNoWait, Code);
end;

procedure CurPageChanged(CurPageID: Integer);
begin
  if CurPageID = wpFinished then begin
    if Installed then
      WizardForm.FinishedLabel.Caption := CustomMessage('Finished')
    else
      WizardForm.FinishedLabel.Caption := FmtMessage(CustomMessage('FinishedFailed'), [ExpandConstant('{app}\install.log')]);
  end;
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
