[CmdletBinding()]
param(
    [int]$Iterations = 100,
    [int]$TraversalsPerPlayer = 4096,
    [int]$ParallelTraversals = 256,
    [ValidateRange(3, 9)]
    [int]$Seats = 3,
    [int]$StartingStack = 200,
    [string]$EnvPath = "C:\conda_envs\poker_ai_env",
    [string]$SnapshotOut = "runs/cuda_deep_cfr_snapshot.pt",
    [ValidateRange(0, 1000000)]
    [int]$SnapshotEvery = 0,
    [string]$PolicyOut = "runs/cuda_deep_cfr_average_policy.pt",
    [string]$ResumeSnapshot = ""
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $EnvPath "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
    throw "Python executable not found at $python"
}

Push-Location $projectRoot
try {
    $arguments = @(
        "-B", "examples/train_cuda_deep_cfr.py",
        "--iterations", $Iterations,
        "--traversals-per-player", $TraversalsPerPlayer,
        "--parallel-traversals", $ParallelTraversals,
        "--seats", $Seats,
        "--starting-stack", $StartingStack,
        "--gpus", "0",
        "--snapshot-out", $SnapshotOut,
        "--snapshot-every", $SnapshotEvery,
        "--policy-out", $PolicyOut,
        "--summary-out", "runs/cuda_deep_cfr_summary.json"
    )
    if ($ResumeSnapshot) {
        $arguments += @("--resume-snapshot", $ResumeSnapshot)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "CUDA Deep CFR training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
