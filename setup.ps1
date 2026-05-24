<#
.SYNOPSIS
Deterministic environment setup script for BRnFS (Windows PowerShell).

.DESCRIPTION
Creates/updates `.venv`, installs the pinned Python 3.9 runtime, ensures model
weights, and builds the optional fast `scatter_cuda` renderer.

Useful overrides:
  $env:PYTHON_BIN = "C:\Path\To\python.exe"
  $env:BRNFS_SKIP_WEIGHTS = "1"
  $env:BRNFS_SKIP_SCATTER_CUDA = "1"
  $env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7"

.EXAMPLE
PS> cd ".\bokeh_rendering_and_focus_stacking_suite"
PS> Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
PS> .\setup.ps1
PS> .\.venv\Scripts\Activate.ps1
PS> python -m brnfs gui
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
    Prefers `$env:PYTHON_BIN`, then the Windows Python launcher (`py -3.9`), then
    `python` if it is Python 3.9.
    #>
    if ($env:PYTHON_BIN -and $env:PYTHON_BIN.Trim() -ne "") {
        try {
            $ver = & $env:PYTHON_BIN -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
            if ($ver.Trim() -eq "3.9") {
                return @($env:PYTHON_BIN)
            }
            throw "ERROR: BRnFS requires Python 3.9, but `$env:PYTHON_BIN is Python $($ver.Trim())."
        } catch {
            throw "ERROR: failed to run `$env:PYTHON_BIN='$env:PYTHON_BIN'. $($_.Exception.Message)"
        }
    }

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
    throw "ERROR: Python 3.9 not found. Install Python 3.9 (required for this repo's pinned dependency stack), or ensure `py -3.9` is available."
}

function Invoke-Py {
    param(
        [Parameter(Mandatory = $true)][string[]]$PythonCmd,
        [Parameter(Mandatory = $true)][string[]]$PyArgs
    )
    $Exe = $PythonCmd[0]
    $CommandArgs = @()
    if ($PythonCmd.Count -gt 1) {
        $CommandArgs += $PythonCmd[1..($PythonCmd.Count - 1)]
    }
    $CommandArgs += $PyArgs

    & $Exe @CommandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed: $($PythonCmd -join ' ') $($PyArgs -join ' ')"
    }
}

$ProjectRoot = $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"

Write-Info "Project root: $ProjectRoot"

$PythonCmd = Get-PythonLauncher

Write-Info "Python for new venv: $($PythonCmd -join ' ')"

if (Test-Path -LiteralPath $VenvDir) {
    $ExistingVenvPython = Join-Path $VenvDir "Scripts\python.exe"
    $ExistingActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"
    if (-not (Test-Path -LiteralPath $ExistingVenvPython) -or -not (Test-Path -LiteralPath $ExistingActivateScript)) {
        Write-Info "Removing broken venv: $VenvDir"
        Remove-Item -LiteralPath $VenvDir -Recurse -Force
    } else {
        $VenvPyMinor = (& $ExistingVenvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").Trim()
        if ($VenvPyMinor -ne "3.9") {
            Write-Info "Removing venv with unsupported Python ${VenvPyMinor}: $VenvDir"
            Remove-Item -LiteralPath $VenvDir -Recurse -Force
        }
    }
}

if (-not (Test-Path -LiteralPath $VenvDir)) {
    Write-Info "Creating venv: $VenvDir"
    Invoke-Py -PythonCmd $PythonCmd -PyArgs @("-m", "venv", "--prompt", "BRnFS", $VenvDir)
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
Write-Info "Using venv python: $(& $VenvPython -c 'import sys; print(sys.executable)')"

Write-Info "Upgrading pip tooling..."
# Pin pip for best compatibility with older manylinux wheel tags.
& $VenvPython -m pip install --upgrade "pip<24" "setuptools<81" wheel

Write-Info "Installing pinned numpy first (build prerequisite for older packages)..."
& $VenvPython -m pip install --no-cache-dir --only-binary=:all: "numpy==1.19.5"

Write-Info "Installing PyTorch CUDA 11.7 wheels..."
& $VenvPython -m pip install --upgrade `
    "torch==2.0.0+cu117" `
    "torchvision==0.15.0+cu117" `
    "torchaudio==2.0.0+cu117" `
    --extra-index-url https://download.pytorch.org/whl/cu117

Write-Info "Installing remaining Python dependencies from requirements.txt..."
# requirements.txt intentionally excludes torch/opencv/lightning packages that
# are installed explicitly above/below to avoid accidental numpy upgrades.
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

Write-Info "Installing this project in editable mode (without dependency resolution)..."
& $VenvPython -m pip install -e $ProjectRoot --no-deps

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

function Ensure-ExistingWeight {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$Destination,
        [Parameter(Mandatory = $true)][long]$MinBytes
    )

    if (-not (Test-Path -LiteralPath $Destination)) {
        Write-Error "$Name is missing: $Destination"
        return $false
    }

    $FileInfo = Get-Item -LiteralPath $Destination
    if ($FileInfo.Length -lt $MinBytes) {
        Write-Error "$Name is incomplete ($($FileInfo.Length) bytes): $Destination"
        return $false
    }

    Write-Info "✓ $Name weight present: $Destination ($($FileInfo.Length) bytes)"
    return $true
}

if ($env:BRNFS_SKIP_WEIGHTS -ne "1") {
    Write-Info "Ensuring model weights are available (set `$env:BRNFS_SKIP_WEIGHTS = '1' to skip)..."
    
    $WeightsOk = $true

    $WeightsOk = (Ensure-ExistingWeight `
        -Name "LDF snapshot" `
        -Destination (Join-Path $ProjectRoot "models\ldf\model-40") `
        -MinBytes 50000000) -and $WeightsOk
    
    $WeightsOk = (Ensure-Weight `
        -Name "LDF (salient detection backbone)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/resnet50-19c8e357.pth?download=true" `
        -Destination (Join-Path $ProjectRoot "models\ldf\resnet50-19c8e357.pth") `
        -MinBytes 50000000) -and $WeightsOk
    
    $WeightsOk = (Ensure-Weight `
        -Name "LaMa (RGB inpainting)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/best.ckpt?download=true" `
        -Destination (Join-Path $ProjectRoot "models\lama\big-lama\models\best.ckpt") `
        -MinBytes 150000000) -and $WeightsOk
    
    $WeightsOk = (Ensure-Weight `
        -Name "MiDaS/DPT (monocular depth)" `
        -Url "https://huggingface.co/ysheng/DrBokeh/resolve/main/dpt_large-midas-2f21e586.pt?download=true" `
        -Destination (Join-Path $ProjectRoot "models\dpt\dpt_large-midas-2f21e586.pt") `
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

$ScatterBuilt = $false
if ($env:BRNFS_SKIP_SCATTER_CUDA -eq "1") {
    Write-Info "Skipping scatter_cuda build (BRNFS_SKIP_SCATTER_CUDA=1)."
    Write-Info "NOTE: bokeh rendering requires scatter_cuda for practical performance; focus stacking does not."
} else {
    Write-Info "Building the CUDA extension (scatter_cuda) against the CURRENT PyTorch..."
    Push-Location (Join-Path $ProjectRoot "brnfs\cuda_src")
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
Write-Info "  python -m brnfs gui"
