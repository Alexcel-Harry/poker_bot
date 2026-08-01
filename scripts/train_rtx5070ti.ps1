[CmdletBinding()]
param(
    [int]$Iterations = 10000,
    [int]$ParallelHands = 1024,
    [int]$BranchWidth = 8,
    [int]$ChanceReplicas = 4,
    [string]$EnvPath = "C:\conda_envs\poker_ai_env",
    [string]$ModelOut = "runs/poker_policy_gpu.pt",
    [string]$TrainingSnapshotOut = "runs/poker_policy_gpu_training.pt",
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
        "-B", "examples/train_gpu_prefix_branch.py",
        "--device", "cuda",
        "--gpus", "0",
        "--iterations", $Iterations,
        "--parallel-hands", $ParallelHands,
        "--branch-width", $BranchWidth,
        "--chance-replicas", $ChanceReplicas,
        "--max-rollout-actions", "128",
        "--batch-size", "8192",
        "--model-out", $ModelOut,
        "--training-snapshot-out", $TrainingSnapshotOut,
        "--summary-out", "runs/training_summary_gpu.json"
    )
    if ($ResumeSnapshot) {
        $arguments += @("--resume-snapshot", $ResumeSnapshot)
    }
    & $python @arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
