# Builds everything the installer copies, so installing downloads nothing (except the optional
# GPU pack, which install.ps1 asks clip-mcp to fetch). Run by .github/workflows/release.yml, and by
# hand to try a build:
#
#   build_payload.ps1 -Out <folder> -Wheel <clip-mcp wheel> -Requirements <uv export>
#                     -FfmpegZip <checked FFmpeg zip> -FfmpegFolder <folder in the zip> [-NoSlim]
#
# The payload, in parts that can be replaced one at a time (#293 replaces only `program`):
#
#   runtime\   a relocatable Python (python-build-standalone, as uv installs it) with every
#              third-party package in its own site-packages, held to the lockfile's versions
#   program\   clip-mcp itself, put on the path by runtime\Lib\site-packages\clip-mcp-program.pth
#   bin\       clip-mcp.exe and clip-mcp-editor.exe, which start ..\runtime\python(w).exe
#   ffmpeg\    ffmpeg.exe and ffprobe.exe
#   release.json  the version, and a fingerprint of runtime\, bin\ and ffmpeg\ (payload_tools stamp)
#
# Nothing in it names the folder it was built in, so it runs from wherever it is copied.
param(
    [Parameter(Mandatory = $true)][string]$Out,
    [Parameter(Mandatory = $true)][string]$Wheel,
    [Parameter(Mandatory = $true)][string]$Requirements,
    [Parameter(Mandatory = $true)][string]$FfmpegZip,
    [Parameter(Mandatory = $true)][string]$FfmpegFolder,
    # Pinned rather than "3.14": the payload is what was tested, not whatever is newest that day.
    [string]$Python = "3.14.5",
    [switch]$NoSlim
)
$ErrorActionPreference = "Stop"
$tools = Join-Path $PSScriptRoot "payload_tools.py"
# PowerShell names are case-blind: $python, the payload's python.exe below, replaces $Python.
$pythonVersion = $Python

function Check([string]$what) {
    if ($LASTEXITCODE -ne 0) { throw "$what failed: $LASTEXITCODE" }
}

$Out = [IO.Path]::GetFullPath($Out)
if (Test-Path $Out) { Remove-Item $Out -Recurse -Force }
New-Item -ItemType Directory -Force $Out | Out-Null
$work = Join-Path ([IO.Path]::GetTempPath()) ("clip-mcp-payload-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Force $work | Out-Null

try {
    # --- runtime: the standalone Python uv installs, copied out of uv's folder, and kept out
    # of the registry and PATH of the build machine.
    uv python install $Python --install-dir (Join-Path $work "python") --no-bin --no-registry; Check "uv python install"
    $installed = Get-ChildItem (Join-Path $work "python") -Directory | Where-Object { $_.Name -like "cpython-$Python-*" } | Select-Object -First 1
    if (-not $installed) { throw "uv did not install Python $Python" }
    $runtime = Join-Path $Out "runtime"
    Copy-Item $installed.FullName $runtime -Recurse
    $python = Join-Path $runtime "python.exe"

    # Every package at the lockfile's version, and only those: the export lists the whole
    # closure, so nothing is resolved here. Copied, not linked to uv's cache.
    uv pip install --python $python --break-system-packages --link-mode copy --no-deps -r $Requirements; Check "packages"

    # --- program: clip-mcp alone, beside the runtime rather than inside it.
    $program = Join-Path $Out "program"
    uv pip install --python $python --target $program --no-deps --link-mode copy $Wheel; Check "clip-mcp"
    # --target writes clip-mcp's own launchers too, with this build's paths in them, and a lock file.
    foreach ($leftover in "bin", ".lock") {
        if (Test-Path (Join-Path $program $leftover)) { Remove-Item (Join-Path $program $leftover) -Recurse -Force }
    }
    # A .pth line is read relative to site-packages: runtime\Lib\site-packages -> program.
    Set-Content -Path (Join-Path $runtime "Lib\site-packages\clip-mcp-program.pth") -Value "../../../program" -Encoding ascii

    # --- bin: uv's launchers from a relocatable venv, pointed at ..\runtime instead.
    $venv = Join-Path $work "venv"
    uv venv --relocatable --python $python $venv; Check "uv venv"
    uv pip install --python (Join-Path $venv "Scripts\python.exe") --no-deps $Wheel; Check "launchers"
    $bin = Join-Path $Out "bin"
    & $python -I -B $tools launcher (Join-Path $venv "Scripts\clip-mcp.exe") (Join-Path $bin "clip-mcp.exe") "..\runtime\python.exe"; Check "clip-mcp.exe"
    & $python -I -B $tools launcher (Join-Path $venv "Scripts\clip-mcp-editor.exe") (Join-Path $bin "clip-mcp-editor.exe") "..\runtime\pythonw.exe"; Check "clip-mcp-editor.exe"

    # --- ffmpeg: the two programs from the build whose checksum the workflow checked.
    $unpacked = Join-Path $work "ffmpeg"
    New-Item -ItemType Directory -Force $unpacked, (Join-Path $Out "ffmpeg") | Out-Null
    & (Join-Path $env:SystemRoot "System32\tar.exe") -xf $FfmpegZip -C $unpacked; Check "unpacking FFmpeg"
    Copy-Item (Join-Path $unpacked "$FfmpegFolder\bin\ffmpeg.exe"), (Join-Path $unpacked "$FfmpegFolder\bin\ffprobe.exe") (Join-Path $Out "ffmpeg")

    Write-Output "--- before slimming"
    & $python -I -B $tools sizes $Out (Join-Path $work "sizes-before.json"); Check "sizes"
    if (-not $NoSlim) {
        & $python -I -B $tools slim $Out; Check "slimming"
    }

    # The payload's own Python, isolated from the build machine's: every module of clip-mcp
    # imports, and what the slimming came close to still works. -B, so it writes no bytecode
    # into what is shipped.
    & $python -I -B $tools smoke $Out; Check "smoke test"
    # The launcher finds its Python from wherever it is started.
    Push-Location $env:SystemRoot
    $env:PYTHONDONTWRITEBYTECODE = "1"
    try { & (Join-Path $bin "clip-mcp.exe") --help | Out-Null; Check "bin\clip-mcp.exe --help" }
    finally { Pop-Location; Remove-Item Env:PYTHONDONTWRITEBYTECODE }

    # What this build's runtime, FFmpeg and launchers are, so a later release can tell whether
    # replacing program\ alone is enough (#293). Copied into the install folder by the installer.
    & $python -I -B $tools stamp $Out $Requirements $pythonVersion $(if ($NoSlim) { "0" } else { "1" }); Check "release.json"

    Write-Output "--- payload"
    & $python -I -B $tools sizes $Out (Join-Path (Split-Path $Out) "payload-sizes.json"); Check "sizes"
    Copy-Item (Join-Path $work "sizes-before.json") (Join-Path (Split-Path $Out) "payload-sizes-before.json")
} finally {
    Remove-Item $work -Recurse -Force -ErrorAction SilentlyContinue
}
