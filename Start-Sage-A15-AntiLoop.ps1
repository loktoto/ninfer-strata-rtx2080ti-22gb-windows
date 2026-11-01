[CmdletBinding()]
param(
    [Parameter(Mandatory=$true)][string]$Config,
    [ValidateRange(0,2147483647)][int]$MaxContext = 0,
    [ValidateRange(0,64)][int]$PromptCache = 6,
    [ValidateRange(0,65536)][int]$VramReserveMiB = 0,
    [switch]$ExperimentalQ8KV,
    [switch]$ConfigureOnly
)
$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Source = (Resolve-Path -LiteralPath $Config).Path
$c = Get-Content -LiteralPath $Source -Raw | ConvertFrom-Json
foreach ($required in @($c.exe, (Join-Path $c.tokenizer 'tokenizer.json'))) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) { throw "Missing file: $required" }
}
function Set-EngineOption([string]$Flag,[string]$Value) {
    $list = [Collections.Generic.List[string]]::new()
    foreach($v in $c.args) { $list.Add([string]$v) }
    $index = $list.IndexOf($Flag)
    if($index -ge 0) {
        if($index + 1 -ge $list.Count) { throw "Missing value for $Flag" }
        $list[$index+1] = $Value
    } else { $list.Add($Flag); $list.Add($Value) }
    $c.args = $list.ToArray()
}
if($MaxContext -gt 0) { Set-EngineOption '--max-context' "$MaxContext" }
Set-EngineOption '--prompt-cache' "$PromptCache"
if($VramReserveMiB -gt 0) { Set-EngineOption '--vram-reserve-mib' "$VramReserveMiB" }
if($ExperimentalQ8KV) {
    Write-Warning 'Q8 KV changes numerical precision and is not full-model quality qualified.'
    Set-EngineOption '--kv' 'q8_0'
}
$settings = @{
    reasoning_effort='high'; reasoning_budget_tokens=6144
    reasoning_loop_recovery='recover'; reasoning_close_retry=$true; repeat_stop_tokens=256
    sampling=@{temperature=1.0; top_p=0.95; top_k=20; presence_penalty=1.5; repetition_penalty=1.0}
}
foreach($key in $settings.Keys) { $c | Add-Member -NotePropertyName $key -NotePropertyValue $settings[$key] -Force }
$Destination = Join-Path $Root ('sage-antiloop-' + [Guid]::NewGuid().ToString('N') + '.json')
[IO.File]::WriteAllText($Destination, ($c | ConvertTo-Json -Depth 64), [Text.UTF8Encoding]::new($false))
Write-Host "Saved configuration: $Destination"
Write-Host "Engine arguments: $($c.args -join ' ')"
Write-Host 'Set a bounded output limit in your client; the thinking budget is not an overall output limit.'
if($ConfigureOnly) { return }
$Python = Join-Path $Root '.venv\Scripts\python.exe'
$oldPath = $env:PATH
$oldGraph = $env:GGML_CUDA_DISABLE_GRAPHS
Push-Location $c.cwd
try {
    $env:PATH = "$(Join-Path $Root '.cuda\bin\x64');$oldPath"
    Remove-Item Env:GGML_CUDA_DISABLE_GRAPHS -ErrorAction SilentlyContinue
    & $Python -m serve.server --engine strata --config $Destination --host 127.0.0.1 --port 8080
    if($LASTEXITCODE -ne 0) { throw "Sage exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
    $env:PATH = $oldPath
    if($null -ne $oldGraph) { $env:GGML_CUDA_DISABLE_GRAPHS=$oldGraph }
}

