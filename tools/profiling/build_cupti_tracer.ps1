[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$CuptiRoot,
    [string]$CudaRoot
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $CudaRoot) { $CudaRoot = Join-Path $Root '.cuda' }
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'Run from an x64 Visual Studio developer PowerShell with cl.exe on PATH.' }
foreach ($path in @((Join-Path $CuptiRoot 'include\cupti.h'), (Join-Path $CuptiRoot 'lib\cupti.lib'), (Join-Path $CudaRoot 'include\cuda.h'))) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing required file: $path" }
}
$OutDir = Join-Path $Root '.profiling'
New-Item -ItemType Directory -Path $OutDir -Force | Out-Null
$source = Join-Path $PSScriptRoot 'cupti_activity_injection.cpp'
$object = Join-Path $OutDir 'cupti_activity_injection.obj'
$dll = Join-Path $OutDir 'sage_cupti_activity.dll'
$importLib = Join-Path $OutDir 'sage_cupti_activity.lib'
& cl.exe /nologo /std:c++17 /EHsc /O2 /LD "/I$(Join-Path $CuptiRoot 'include')" "/I$(Join-Path $CudaRoot 'include')" $source "/Fo:$object" /link "/LIBPATH:$(Join-Path $CuptiRoot 'lib')" cupti.lib "/OUT:$dll" "/IMPLIB:$importLib"
if ($LASTEXITCODE -ne 0) { throw "CUPTI tracer build failed with exit code $LASTEXITCODE" }
Write-Host "Built $dll; it is loaded only when explicitly injected into a child process."
