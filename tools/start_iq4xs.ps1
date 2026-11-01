[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Model,
    [string]$Engine,
    [ValidateRange(1, 2147483647)][int]$MaxContext = 32768,
    [ValidateRange(1, 256)][int]$Threads = 16,
    [ValidateSet(2, 3, 4)][int]$DraftWindow = 4,
    [switch]$Native,
    [switch]$NoDraft,
    [switch]$FusedNormQ8,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
if (-not $Engine) {
    $Engine = Join-Path $Root $(if ($FusedNormQ8) { 'build-fused\strata\strata-poly.exe' } else { 'build-optimized\strata\strata-poly.exe' })
}
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$CudaBin = Join-Path $Root '.cuda\bin\x64'
foreach ($item in @(@('Python', $Python), @('engine', $Engine), @('model', $Model))) {
    if (-not (Test-Path -LiteralPath $item[1] -PathType Leaf)) {
        throw "Missing $($item[0]) file: $($item[1]). See docs/IQ4XS_SETUP.md."
    }
}
$ServerArgs = @((Join-Path $Root 'tools\serve_sage.py'), '--engine', (Resolve-Path -LiteralPath $Engine).Path,
    '--model', (Resolve-Path -LiteralPath $Model).Path, '--max-context', "$MaxContext", '--threads', "$Threads", '--cuda-graphs')
if (-not $Native) { $ServerArgs += '--ninfer-expert-iq4xs' }
if (-not $NoDraft) { $ServerArgs += @('--draft-window', "$DraftWindow") }
if ($FusedNormQ8) { $ServerArgs += '--fused-norm-q8' }
if ($DryRun) { $ServerArgs += '--dry-run' }
$OriginalPath = $env:PATH
$OriginalGraphSetting = $env:GGML_CUDA_DISABLE_GRAPHS
Push-Location $Root
try {
    if (Test-Path -LiteralPath $CudaBin -PathType Container) { $env:PATH = "$CudaBin;$OriginalPath" }
    Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    & $Python @ServerArgs
    if ($LASTEXITCODE -ne 0) { throw "Server launcher exited with code $LASTEXITCODE." }
} finally {
    $env:PATH = $OriginalPath
    if ($null -ne $OriginalGraphSetting) { $env:GGML_CUDA_DISABLE_GRAPHS = $OriginalGraphSetting }
    Pop-Location
}
