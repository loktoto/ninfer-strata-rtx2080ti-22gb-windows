[CmdletBinding()]
param(
    [string]$BuildDirectory = "build-windows",
    [ValidateRange(1, 64)][int]$Jobs = 2
)
$ErrorActionPreference = 'Stop'
if ([Environment]::OSVersion.Platform -ne [PlatformID]::Win32NT) {
    throw 'This build entry point requires native Windows.'
}
$Root = Split-Path -Parent $PSScriptRoot
Push-Location $Root
try {
    foreach ($Command in @('python', 'git', 'cmake', 'ninja', 'nvcc', 'cl')) {
        if (-not (Get-Command $Command -ErrorAction SilentlyContinue)) {
            throw "Missing $Command. Use an x64 Visual Studio developer shell with CUDA on PATH."
        }
    }
    & python tools/bootstrap.py
    if ($LASTEXITCODE -ne 0) { throw 'Source setup failed.' }
    & cmake -S . -B $BuildDirectory -G Ninja -DCMAKE_BUILD_TYPE=Release `
        -DSAGE_ENABLE_CUDA=ON -DSAGE_BUILD_STRATA=ON -DCMAKE_CUDA_ARCHITECTURES=75
    if ($LASTEXITCODE -ne 0) { throw 'CMake configuration failed.' }
    & cmake --build $BuildDirectory --target strata-poly --parallel $Jobs
    if ($LASTEXITCODE -ne 0) { throw 'strata-poly build failed.' }
    Write-Host "Built strata-poly. GPU numerical tests and model acceptance are still required."
} finally {
    Pop-Location
}
