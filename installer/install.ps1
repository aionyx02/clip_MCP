# Run by the installer after it has copied its files. Python, the packages, clip-mcp and FFmpeg
# are already in place (built by build_payload.ps1); this only finishes the setup, and fetches
# the GPU pack when the user chose it.
#
#   install.ps1 -App <install folder> [-Gpu]
#
# Each step prints a line "STEP <name>" that the wizard shows in words; everything else it
# prints is shown as the current detail. All of it also goes to install.log in the install folder.
# Exit codes: 0 done, 2 setup did not run, 4 done but the GPU pack did not install (clip-mcp
# works on the processor, and the editor can fetch the pack later).
param(
    [Parameter(Mandatory = $true)][string]$App,
    [switch]$Gpu
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

$clip = Join-Path $App "bin\clip-mcp.exe"
# Written first: clip-mcp finds its install folder, and where the GPU pack goes, by this file.
"This folder was installed by the clip-mcp installer; the editor may update it." |
    Out-File -FilePath (Join-Path $App "installed-by-setup.txt") -Encoding utf8

# The GPU pack: fetched when chosen, kept when it is already the pinned one, and taken out when
# the user unchecked it.
$gpuFailed = $false
if ($Gpu) {
    Say "STEP gpu"
    $code = Run $clip @("gpu", "install")
    if ($code -ne 0) { Say "GPU pack not installed: $code"; $gpuFailed = $true }
} else {
    Run $clip @("gpu", "remove") | Out-Null
}

# Point clip-mcp at its workspace and at the FFmpeg it came with. It connects no AI client:
# that is the user's choice, made in the editor.
Say "STEP setup"
$code = Run $clip @("setup", "--no-prompt", "--ffmpeg-dir", (Join-Path $App "ffmpeg"))
if ($code -ne 0) { Say "clip-mcp setup failed: $code"; exit 2 }

Say "done"
if ($gpuFailed) { exit 4 }
exit 0
