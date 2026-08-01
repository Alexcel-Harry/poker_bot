[CmdletBinding()]
param(
    [ValidateSet("kuhn", "leduc", "holdem")]
    [string]$Game = "holdem",
    [int]$Iterations = 100,
    [int]$TraversalsPerPlayer = 100,
    [ValidateRange(3, 9)]
    [int]$Seats = 3,
    [int]$StartingStack = 200,
    [string]$Device = "auto",
    [string]$EnvPath = "C:\conda_envs\poker_ai_env",
    [string]$SnapshotOut = "runs/deep_cfr_snapshot.pt",
    [string]$PolicyOut = "runs/deep_cfr_average_policy.pt"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $EnvPath "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found at $python"
}

Push-Location $projectRoot
try {
    & $python -B examples/train_deep_cfr.py `
        --game $Game `
        --iterations $Iterations `
        --traversals-per-player $TraversalsPerPlayer `
        --seats $Seats `
        --starting-stack $StartingStack `
        --device $Device `
        --snapshot-out $SnapshotOut `
        --policy-out $PolicyOut `
        --summary-out runs/deep_cfr_summary.json
    if ($LASTEXITCODE -ne 0) {
        throw "Deep CFR training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
