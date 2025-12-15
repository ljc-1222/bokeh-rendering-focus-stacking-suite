<#
.SYNOPSIS
Build and install the CUDA/C++ extension that provides the `scatter_cuda` Python module (Windows).

.DESCRIPTION
Reuses the currently-active Python environment (venv/conda) and builds the extension in
`app/cuda-src`. Prefer running `bokeh_rendering_and_focus_stacking_suite/setup.ps1` end-to-end, but this is useful for rebuilds.

.EXAMPLE
PS> cd ".\bokeh_rendering_and_focus_stacking_suite"
PS> .\.venv\Scripts\Activate.ps1
PS> .\scripts\build_scatter_cuda.ps1
#>

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Push-Location (Join-Path $ProjectRoot "app\cuda-src")
try {
    python -m pip install --no-build-isolation --force-reinstall --no-cache-dir .
    Write-Host "Done. You should now be able to import scatter_cuda (used by app/bokeh_rendering/DScatter/GPU_scatter.py)."
} finally {
    Pop-Location
}


