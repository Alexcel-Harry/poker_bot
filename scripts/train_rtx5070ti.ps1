[CmdletBinding()]
param(
    [int]$Iterations = 10000,
    [int]$ParallelHands = 1024,
    [int]$BranchWidth = 32,
    [string]$EnvPath = "C:\conda_envs\poker_ai_env",
    [string]$ModelOut = "runs/poker_policy_gpu.pt"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    conda run --prefix $EnvPath python -B examples/train_gpu_prefix_branch.py `
        --device cuda `
        --gpus 0 `
        --iterations $Iterations `
        --parallel-hands $ParallelHands `
        --branch-width $BranchWidth `
        --max-rollout-actions 128 `
        --batch-size 8192 `
        --model-out $ModelOut `
        --summary-out runs/training_summary_gpu.json
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
