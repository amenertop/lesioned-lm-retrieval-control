# Create or update the 'lesion' conda env on the local Windows machine. Safe to re-run.
# Run from the project root: pwsh env/setup_local.ps1
$ErrorActionPreference = 'Stop'
$PSNativeCommandUseErrorActionPreference = $true

$conda = 'D:\anaconda3\Scripts\conda.exe'

function Get-EnvPython {
    $envDir = (& $conda env list --json | ConvertFrom-Json).envs | Where-Object { (Split-Path $_ -Leaf) -eq 'lesion' } | Select-Object -First 1
    if ($envDir) { Join-Path $envDir 'python.exe' }
}

$py = Get-EnvPython
if (-not $py) { & $conda create -n lesion python=3.12 -y; $py = Get-EnvPython }
Write-Output "lesion env python: $py"

# On Windows, PyPI only ships CPU torch; the CUDA 12.8 build comes from the PyTorch index.
& $py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
& $py -m pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

& $py -c "import torch, transformers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.is_available(), '| transformers', transformers.__version__)"
