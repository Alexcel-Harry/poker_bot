[CmdletBinding()]
param(
    [string]$EnvPath = "C:\conda_envs\poker_ai_env"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $projectRoot
try {
    if (Test-Path -LiteralPath (Join-Path $EnvPath "python.exe")) {
        conda env update --prefix $EnvPath --file environment.yml
    }
    else {
        conda env create --prefix $EnvPath --file environment.yml
    }
    if ($LASTEXITCODE -ne 0) {
        throw "Conda environment setup failed with exit code $LASTEXITCODE"
    }

    conda run --prefix $EnvPath python -c "import torch"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Torch cannot load after the Python update; reinstalling the Python 3.12 CUDA wheels."
        conda run --prefix $EnvPath python -m pip install --force-reinstall `
            torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 `
            --index-url https://download.pytorch.org/whl/cu129
        if ($LASTEXITCODE -ne 0) {
            throw "PyTorch CUDA wheel installation failed with exit code $LASTEXITCODE"
        }
    }

    conda run --prefix $EnvPath python -B scripts/verify_environment.py
    if ($LASTEXITCODE -ne 0) {
        throw "Environment verification failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
