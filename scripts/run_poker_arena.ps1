[CmdletBinding()]
param(
    [int]$Port = 8000,
    [string]$EnvPath = "C:\conda_envs\poker_ai_env",
    [string]$ModelPath = "runs\poker_policy_gpu.pt",
    [ValidateSet("cuda", "cpu", "auto")]
    [string]$Device = "cuda",
    [switch]$Lan,
    [switch]$HideCards
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $EnvPath "python.exe"
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Conda environment Python was not found: $python"
}

$candidateModel = if ([System.IO.Path]::IsPathRooted($ModelPath)) {
    $ModelPath
}
else {
    Join-Path $projectRoot $ModelPath
}
if (-not (Test-Path -LiteralPath $candidateModel -PathType Leaf)) {
    throw "Bot checkpoint was not found: $candidateModel"
}
$resolvedModel = (Resolve-Path -LiteralPath $candidateModel).Path

$arguments = @(
    "-B",
    "-m", "poker_arena.web",
    "--port", $Port,
    "--model", $resolvedModel,
    "--device", $Device
)
$arguments += if ($HideCards) { "--no-reveal-cards" } else { "--reveal-cards" }
if ($Lan) {
    $arguments += "--lan"
}

Push-Location $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Poker Arena exited with code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
