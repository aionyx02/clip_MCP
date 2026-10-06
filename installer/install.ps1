# Run by the installer after it has copied its files: fetches FFmpeg, then puts Python and
# clip-mcp's packages inside clip-mcp's own folder, so nothing lands anywhere uninstalling would
# not reach.
#
#   install.ps1 -App <install folder> -Wheel <clip-mcp wheel> -Constraints <tested versions>
#               -FfmpegUrl <zip> -FfmpegSha256 <checksum> -FfmpegFolder <folder in the zip>
#
# Each step prints a line "STEP <name>" that the wizard shows in words; everything else it
# prints is shown as the current detail. All of it also goes to install.log in the install folder.
# Exit codes: 0 done, 1 the packages did not install, 2 setup did not run, 3 FFmpeg did not arrive.
param(
    [Parameter(Mandatory = $true)][string]$App,
    [Parameter(Mandatory = $true)][string]$Wheel,
    [Parameter(Mandatory = $true)][string]$Constraints,
    [Parameter(Mandatory = $true)][string]$FfmpegUrl,
    [Parameter(Mandatory = $true)][string]$FfmpegSha256,
    [Parameter(Mandatory = $true)][string]$FfmpegFolder
)

$log = Join-Path $App "install.log"
"clip-mcp install, $(Get-Date -Format s)" | Out-File -FilePath $log -Encoding utf8

# Straight to the console, not the pipeline: in a function, piped output would become part of
# what the function returns, and `$code = Run ...` would hold every line instead of the exit code.
function Say([string]$line) {
    [Console]::Out.WriteLine($line)
    [Console]::Out.Flush()
    Add-Content -Path $log -Value $line -Encoding utf8
}

# Runs a command, passing each line it prints to the wizard and the log; returns its exit code.
function Run([string]$exe, [string[]]$arguments) {
    & $exe @arguments 2>&1 | ForEach-Object { Say "$_" }
    return $LASTEXITCODE
}

# --- FFmpeg: fetched by Windows' own curl, which keeps going over a slow or dropped
# connection where the wizard's downloader started over; refused unless the checksum matches.
$ffmpeg = Join-Path $App "ffmpeg"
$stamp = Join-Path $ffmpeg "build.txt"
$current = (Test-Path (Join-Path $ffmpeg "ffmpeg.exe")) -and (Test-Path $stamp) -and
    ((Get-Content $stamp -Raw).Trim() -eq $FfmpegSha256)
if (-not $current) {
    Say "STEP ffmpeg"
    $zip = Join-Path $env:TEMP "clip-mcp-ffmpeg.zip"
    $unpacked = Join-Path $env:TEMP "clip-mcp-ffmpeg"
    $curl = Join-Path $env:SystemRoot "System32\curl.exe"
    # Retried on any error, picking up where it stopped; a transfer stalled under 10 KB/s for
    # 30 seconds counts as an error rather than hanging there.
    $code = Run $curl @("--location", "--fail", "--silent", "--show-error", "--retry", "10", "--retry-delay", "3",
                        "--retry-all-errors", "--continue-at", "-", "--speed-limit", "10240", "--speed-time", "30",
                        "--output", $zip, $FfmpegUrl)
    if ($code -ne 0) { Say "FFmpeg download failed: $code"; exit 3 }
    $hash = (Get-FileHash $zip -Algorithm SHA256).Hash.ToLower()
    if ($hash -ne $FfmpegSha256.ToLower()) {
        Remove-Item $zip -Force -ErrorAction SilentlyContinue
        Say "FFmpeg checksum does not match: $hash"
        exit 3
    }
    New-Item -ItemType Directory -Force $unpacked, $ffmpeg | Out-Null
    $code = Run (Join-Path $env:SystemRoot "System32\tar.exe") @("-xf", $zip, "-C", $unpacked)
    if ($code -ne 0) { Say "FFmpeg could not be unpacked: $code"; exit 3 }
    Copy-Item (Join-Path $unpacked "$FfmpegFolder\bin\ffmpeg.exe"), (Join-Path $unpacked "$FfmpegFolder\bin\ffprobe.exe") $ffmpeg -Force
    Set-Content -Path $stamp -Value $FfmpegSha256 -Encoding ascii
    Remove-Item $zip, $unpacked -Recurse -Force -ErrorAction SilentlyContinue
}

# --- Python and the packages. uv keeps its Python, its tools, their commands and its cache
# here, not in the user's profile.
$env:UV_TOOL_DIR = Join-Path $App "tools"
$env:UV_TOOL_BIN_DIR = Join-Path $App "bin"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $App "python"
$env:UV_CACHE_DIR = Join-Path $App "cache"
$env:UV_PYTHON_PREFERENCE = "only-managed"
$env:UV_NO_MODIFY_PATH = "1"
$uv = Join-Path $App "uv\uv.exe"

# An AI app that has clip-mcp running keeps its files open, and Windows will not let open files
# be removed: uv would take the old packages half away, stop, and leave nothing that starts. So
# whatever runs from this folder is stopped first; the AI app starts clip-mcp again the next
# time it needs it.
function Stop-Running {
    $here = [IO.Path]::GetFullPath($App).TrimEnd('\') + '\'
    $running = @(Get-Process -ErrorAction SilentlyContinue |
        Where-Object { $_.Id -ne $PID -and $_.Path -and $_.Path.StartsWith($here, [StringComparison]::OrdinalIgnoreCase) })
    if ($running.Count -eq 0) { return }
    Say "Stopping $($running.Count) clip-mcp program(s) still running"
    $running | Stop-Process -Force -ErrorAction SilentlyContinue
    $running | Wait-Process -Timeout 15 -ErrorAction SilentlyContinue
}

Say "STEP packages"
# Every package at the version the release was tested with, not whatever is newest today. Tried
# again when it fails: an AI app may start clip-mcp again between the stop and the install.
$tries = 3
for ($try = 1; $try -le $tries; $try++) {
    Stop-Running
    $code = Run $uv @("tool", "install", "--force", "--python", "3.14", "--constraints", $Constraints, $Wheel)
    if ($code -eq 0) { break }
    Say "uv tool install failed: $code (try $try of $tries)"
    if ($try -eq $tries) { exit 1 }
    Start-Sleep -Seconds 3
}

# Point clip-mcp at its workspace and at the FFmpeg above. It connects no AI client: that is
# the user's choice, made in the editor.
Say "STEP setup"
$code = Run (Join-Path $App "bin\clip-mcp.exe") @("setup", "--no-prompt", "--ffmpeg-dir", $ffmpeg)
if ($code -ne 0) { Say "clip-mcp setup failed: $code"; exit 2 }

# The downloads are installed now; the cache would only take up room.
Say "STEP cleanup"
Run $uv @("cache", "clean") | Out-Null
"This folder was installed by the clip-mcp installer; the editor may update it." |
    Out-File -FilePath (Join-Path $App "installed-by-setup.txt") -Encoding utf8
Say "done"
exit 0
