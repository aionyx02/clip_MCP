# Run by the installer after it has copied its files: puts Python and clip-mcp's packages
# inside clip-mcp's own folder, so nothing lands anywhere uninstalling would not reach.
#
#   install.ps1 -App <install folder> -Wheel <clip-mcp wheel>
#
# Every step is written to install.log in the install folder. Exit codes: 0 done,
# 1 the packages did not install (most often no network), 2 setup did not run.
param(
    [Parameter(Mandatory = $true)][string]$App,
    [Parameter(Mandatory = $true)][string]$Wheel
)

$log = Join-Path $App "install.log"
"clip-mcp install, $(Get-Date -Format s)" | Out-File -FilePath $log -Encoding utf8

# Runs a command, appending everything it prints to the log as plain UTF-8 text; returns its exit code.
function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments 2>&1 | ForEach-Object { "$_" } | Out-File -FilePath $log -Append -Encoding utf8
    return $LASTEXITCODE
}

# uv keeps its Python, its tools, their commands and its cache here, not in the user's profile.
$env:UV_TOOL_DIR = Join-Path $App "tools"
$env:UV_TOOL_BIN_DIR = Join-Path $App "bin"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $App "python"
$env:UV_CACHE_DIR = Join-Path $App "cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_NO_MODIFY_PATH = "1"
$uv = Join-Path $App "uv\uv.exe"

$code = Run $uv @("tool", "install", "--force", "--python", "3.14", $Wheel)
if ($code -ne 0) { "uv tool install failed: $code" | Out-File -FilePath $log -Append -Encoding utf8; exit 1 }

# Point clip-mcp at its workspace and at the FFmpeg the installer downloaded. It connects no
# AI client: that is the user's choice, made in the editor.
$clip = Join-Path $App "bin\clip-mcp.exe"
$code = Run $clip @("setup", "--no-prompt", "--ffmpeg-dir", (Join-Path $App "ffmpeg"))
if ($code -ne 0) { "clip-mcp setup failed: $code" | Out-File -FilePath $log -Append -Encoding utf8; exit 2 }

# The downloads are installed now; the cache would only take up room.
Run $uv @("cache", "clean") | Out-Null
"This folder was installed by the clip-mcp installer; the editor may update it." |
    Out-File -FilePath (Join-Path $App "installed-by-setup.txt") -Encoding utf8
"done" | Out-File -FilePath $log -Append -Encoding utf8
exit 0
