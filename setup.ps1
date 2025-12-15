<#
.SYNOPSIS
Deterministic environment setup script for the `bokeh_rendering_and_focus_stacking_suite/` project (Windows PowerShell).

.DESCRIPTION
Creates/updates a single Python venv under `.venv`, installs the pinned legacy stack
(numpy==1.19.5, scikit-image==0.18.3, ...), installs PyTorch CUDA wheels (cu117),
and builds/installs the custom CUDA extension (`scatter_cuda`).

This script is the Windows counterpart of `setup.sh`.

.EXAMPLE
PS> cd ".\bokeh_rendering_and_focus_stacking_suite"
PS> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
PS> .\setup.ps1
PS> .\.venv\Scripts\Activate.ps1
PS> python .\gui\gui.py
#>

$ErrorActionPreference = "Stop"

function Write-Info {
    param([Parameter(Mandatory = $true)][string]$Message)
    Write-Host $Message
}

function Get-PythonLauncher {
    <#
    .SYNOPSIS
    Picks a Python command to use.

    .DESCRIPTION
    Prefers the Windows Python launcher (`py -3.9`) to guarantee Python 3.9, falling back
    to `python` if the launcher isn't available.
    #>
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $ver = & py -3.9 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($ver.Trim() -eq "3.9") {
                return @("py", "-3.9")
            }
        } catch {
            # ignore and fall back
        }
    }
    if (Get-Command python -ErrorAction SilentlyContinue) {
        try {
            $ver = & python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($ver.Trim() -eq "3.9") {
                return @("python")
            }
        } catch {
            # ignore and throw below
        }
    }
    throw "ERROR: Python 3.9 not found. Install Python 3.9 (required for this repo's pinned legacy stack), or ensure `py -3.9` is available."
}

function Invoke-Py {
    param(
        [Parameter(Mandatory = $true)][string[]]$PythonCmd,
        [Parameter(Mandatory = $true)][string[]]$Args
    )
    & $PythonCmd @Args
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($PythonCmd -join ' ') $($Args -join ' ')"
    }
}

$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"

Write-Info "Project root: $ProjectRoot"

$PythonCmd = $null
try {
    $PythonCmd = Get-PythonLauncher
} catch {
    # Optional conda bootstrap (Windows counterpart to setup.sh behavior).
    if (-not $env:BRNFS_RERUN_IN_CONDA -and $env:CONDA_PREFIX -and (Get-Command conda -ErrorAction SilentlyContinue)) {
        $CondaEnvName = if ($env:BRNFS_CONDA_ENV_NAME -and $env:BRNFS_CONDA_ENV_NAME.Trim() -ne "") { $env:BRNFS_CONDA_ENV_NAME } else { "BRnFS" }
        Write-Info "Detected conda shell but Python 3.9 was not found; bootstrapping conda env '$CondaEnvName' with Python 3.9..."
        & conda create -n $CondaEnvName python=3.9 -y
        if ($LASTEXITCODE -ne 0) { throw "conda create failed." }

        Write-Info "Re-running setup inside conda env '$CondaEnvName' (via conda run)..."
        $env:BRNFS_RERUN_IN_CONDA = "1"
        $env:BRNFS_CONDA_ENV_NAME = $CondaEnvName
        & conda run -n $CondaEnvName powershell -ExecutionPolicy Bypass -File $PSCommandPath
        exit $LASTEXITCODE
    }
    throw
}

Write-Info "Using python: $($PythonCmd -join ' ')"

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-Info "Creating venv: $VenvDir"
    Invoke-Py -PythonCmd $PythonCmd -Args @("-m", "venv", "--prompt", "BRnFS", $VenvDir)
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path -LiteralPath $VenvPython)) {
    throw "ERROR: venv python not found at: $VenvPython"
}

# Activate venv for the remainder of this script (note: activation persists only
# if the caller dot-sources this setup script).
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
if (Test-Path -LiteralPath $ActivateScript) {
    . $ActivateScript
}

Write-Info "Upgrading pip tooling..."
# Pin pip for best compatibility with legacy manylinux wheel tags.
& $VenvPython -m pip install --upgrade "pip<24" setuptools wheel

Write-Info "Installing pinned numpy first (build prerequisite for legacy packages)..."
& $VenvPython -m pip install --no-cache-dir --only-binary=:all: "numpy==1.19.5"

Write-Info "Installing PyTorch CUDA 11.7 wheels..."
& $VenvPython -m pip install --upgrade `
    "torch==2.0.0+cu117" `
    "torchvision==0.15.0+cu117" `
    "torchaudio==2.0.0+cu117" `
    --extra-index-url https://download.pytorch.org/whl/cu117

Write-Info "Installing remaining Python dependencies from requirements.txt..."
& $VenvPython -m pip install --no-cache-dir -r (Join-Path $ProjectRoot "requirements.txt")

Write-Info "Installing OpenCV (pinned) WITHOUT upgrading numpy..."
& $VenvPython -m pip install --no-deps --no-cache-dir "opencv-python-headless==4.5.5.64"

Write-Info "Installing minimal PyTorch Lightning bits (for LaMa checkpoint loading) WITHOUT upgrading numpy..."
& $VenvPython -m pip install --no-deps --no-cache-dir `
    "pytorch-lightning==1.9.5" `
    "torchmetrics==0.11.4" `
    "lightning-utilities==0.15.2"
& $VenvPython -m pip install --no-deps --no-cache-dir "lightning-fabric==1.9.5"

Write-Info "Uninstalling TensorFlow if it was pulled in previously (optional dependency)..."
try {
    & $VenvPython -m pip uninstall -y tensorflow tensorflow-io-gcs-filesystem *> $null
} catch {
    # ignore
}

###############################################################################
# Model weights (LDF + LaMa + MiDaS/DPT)
#
# Upstream DrBokeh expects users to download weights manually. In this merged
# project, we make `setup.ps1` ensure the files exist.
#
# To skip (e.g., offline install), run:
#   $env:BRNFS_SKIP_WEIGHTS = "1"; .\setup.ps1
###############################################################################

function Download-File {
    <#
    .SYNOPSIS
    Downloads a file from a URL with retry logic.
    
    .PARAMETER Url
    The URL to download from.
    
    .PARAMETER Destination
    The destination file path.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination
    )
    
    $DestDir = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $DestDir)) {
        New-Item -ItemType Directory -Path $DestDir -Force | Out-Null
    }
    
    $MaxRetries = 3
    $RetryDelay = 2
    
    for ($i = 0; $i -lt $MaxRetries; $i++) {
        try {
            Write-Info "  Downloading (attempt $($i + 1)/$MaxRetries)..."
            $ProgressPreference = 'SilentlyContinue'
            Invoke-WebRequest -Uri $Url -OutFile $Destination -TimeoutSec 30 -ErrorAction Stop
            return $true
        } catch {
            if ($i -eq $MaxRetries - 1) {
                Write-Error "Failed to download after $MaxRetries attempts: $($_.Exception.Message)"
                return $false
            }
            Start-Sleep -Seconds $RetryDelay
        }
    }
    return $false
}

function Ensure-Weight {
    <#
    .SYNOPSIS
    Ensures a model weight file exists, downloading it if necessary.
    
    .PARAMETER Name
    Human-readable name of the weight file.
    
    .PARAMETER Url
    The URL to download from.
    
    .PARAMETER Destination
    The destination file path.
    
    .PARAMETER MinBytes
    Minimum expected file size in bytes.
    #>
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Url,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$MinBytes
    )
    
    if (Test-Path -LiteralPath $Destination) {
        $FileInfo = Get-Item -LiteralPath $Destination
        if ($FileInfo.Length -ge $MinBytes) {
            Write-Info "✓ $Name weight present: $Destination ($($FileInfo.Length) bytes)"
            return $true
        }
        Write-Warning "$Name weight exists but looks too small ($($FileInfo.Length) bytes). Re-downloading..."
        Remove-Item -LiteralPath $Destination -Force
    }
    
    Write-Info "Downloading $Name weight..."
    Write-Info "  -> $Destination"
    
    if (-not (Download-File -Url $Url -Destination $Destination)) {
        Write-Error "Failed to download $Name weight from: $Url"
        return $false
    }
    
    $FileInfo = Get-Item -LiteralPath $Destination
    if ($FileInfo.Length -lt $MinBytes) {
        Write-Error "Downloaded $Name weight looks incomplete ($($FileInfo.Length) bytes): $Destination"
        Write-Error "Check network access and the URL: $Url"
        return $false
    }
    Write-Info "✓ Downloaded $Name weight ($($FileInfo.Length) bytes)"
    return $true
}

if ($env:BRNFS_SKIP_WEIGHTS -ne "1") {
    Write-Info "Ensuring model weights are available (set `$env:BRNFS_SKIP_WEIGHTS = '1' to skip)..."
    
    $WeightsOk = $true
    
    $WeightsOk = (Ensure-Weight `
        -Name "LDF (salient detection backbone)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true" `
        -Destination (Join-Path $ProjectRoot "app\bokeh_rendering\Salient\LDF\res\resnet50-19c8e357.pth") `
        -MinBytes 50000000) -and $WeightsOk
    
    $WeightsOk = (Ensure-Weight `
        -Name "LaMa (RGB inpainting)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true" `
        -Destination (Join-Path $ProjectRoot "app\bokeh_rendering\Inpainting\lama\big-lama\models\best.ckpt") `
        -MinBytes 150000000) -and $WeightsOk
    
    $WeightsOk = (Ensure-Weight `
        -Name "MiDaS/DPT (monocular depth)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true" `
        -Destination (Join-Path $ProjectRoot "app\bokeh_rendering\Depth\DPT\weights\dpt_large-midas-2f21e586.pt") `
        -MinBytes 300000000) -and $WeightsOk
    
    if (-not $WeightsOk) {
        Write-Warning "Some model weights failed to download. The application may not work correctly."
    }
} else {
    Write-Info "Skipping model weight downloads (BRNFS_SKIP_WEIGHTS=1)."
}

Write-Info "Configuring CUDA environment variables (best-effort)..."
if (-not $env:CUDA_HOME -or $env:CUDA_HOME.Trim() -eq "") {
    if ($env:CUDA_PATH -and $env:CUDA_PATH.Trim() -ne "") {
        $env:CUDA_HOME = $env:CUDA_PATH
    } elseif ($env:CUDA_PATH_V11_7 -and $env:CUDA_PATH_V11_7.Trim() -ne "") {
        $env:CUDA_HOME = $env:CUDA_PATH_V11_7
    }
}
if (-not $env:CUDA_HOME -or $env:CUDA_HOME.Trim() -eq "") {
    Write-Warning "CUDA_HOME not set and CUDA_PATH not found. Building scatter_cuda may fail. Install CUDA Toolkit 11.7 and ensure CUDA_PATH is set."
} else {
    $CudaBin = Join-Path $env:CUDA_HOME "bin"
    if (Test-Path -LiteralPath $CudaBin) {
        $env:Path = "$CudaBin;$env:Path"
    }
}

Write-Info "Adding PyTorch DLL directory to PATH for this session..."
$TorchLibDir = (& $VenvPython -c "import pathlib, torch; print(pathlib.Path(torch.__file__).resolve().parent / 'lib')").Trim()
if ($TorchLibDir -and (Test-Path -LiteralPath $TorchLibDir)) {
    $env:Path = "$TorchLibDir;$env:Path"
}

Write-Info "Building the CUDA extension (scatter_cuda) against the CURRENT PyTorch..."
$ScatterBuilt = $false
Push-Location (Join-Path $ProjectRoot "app\cuda-src")
try {
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue "build", "dist"
    Get-ChildItem -Force -Directory -Filter "*.egg-info" -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
    Get-ChildItem -Recurse -Force -Include *.pyd,*.dll,*.so -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue

    & $VenvPython -m pip install --no-build-isolation --force-reinstall --no-cache-dir .
    $ScatterBuilt = $true
} catch {
    Write-Warning ("Failed to build scatter_cuda. Bokeh rendering will not work until this succeeds. " +
        "Make sure you installed: (1) NVIDIA driver, (2) CUDA Toolkit 11.7, (3) Visual Studio Build Tools (C++), and (4) are using a CUDA-enabled PyTorch wheel. " +
        "Error: " + $_.Exception.Message)
} finally {
    Pop-Location
}

Write-Info "Quick sanity checks..."
if ($ScatterBuilt) {
    @'
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib
import scatter_cuda

print("✓ numpy:", np.__version__)
print("✓ torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("✓ cv2:", cv2.__version__)
print("✓ pandas:", pd.__version__)
print("✓ matplotlib:", matplotlib.__version__)
print("✓ scatter_cuda import OK")
'@ | & $VenvPython -
} else {
    @'
import numpy as np
import torch
import cv2
import pandas as pd
import matplotlib

print("✓ numpy:", np.__version__)
print("✓ torch:", torch.__version__, "cuda:", torch.cuda.is_available())
print("✓ cv2:", cv2.__version__)
print("✓ pandas:", pd.__version__)
print("✓ matplotlib:", matplotlib.__version__)
print("! scatter_cuda not built (see warnings above)")
'@ | & $VenvPython -
}

Write-Info "DONE."
Write-Info "Next:"
Write-Info "  .\.venv\Scripts\Activate.ps1"
Write-Info "  python .\gui\gui.py"


