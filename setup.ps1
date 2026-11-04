param([switch]$Cpu, [switch]$SkipFenics)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) {
    & python -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw 'Windows Python 3.11 or newer is required.' }
}
& $pythonExe -c 'import sys; assert sys.version_info >= (3, 11), sys.version'
if ($LASTEXITCODE -ne 0) { throw 'Pinned Windows dependencies require Python 3.11 or newer.' }
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw 'pip installation failed.' }
if ($Cpu) { & $pythonExe -m pip install torch==2.11.0 }
else { & $pythonExe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128 }
if ($LASTEXITCODE -ne 0) { throw 'PyTorch installation failed.' }
& $pythonExe -m pip install -r requirements-core.txt
if ($LASTEXITCODE -ne 0) { throw 'Python dependencies failed to install.' }
& $pythonExe -m pip install -r requirements-notebook.txt
if ($LASTEXITCODE -ne 0) { throw 'Notebook dependencies failed to install.' }
& $pythonExe -m pip install -r requirements-documents.txt
if ($LASTEXITCODE -ne 0) { throw 'Report dependencies failed to install.' }
& $pythonExe -m ipykernel install --sys-prefix --name harandi --display-name 'Harandi CUDA Python 3.11'
if ($LASTEXITCODE -ne 0) { throw 'Notebook kernel setup failed.' }
if (!$SkipFenics) {
    & wsl.exe -d Ubuntu -u root -- bash -lc 'apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y software-properties-common && add-apt-repository -y ppa:fenics-packages/fenics && apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dolfin python3-matplotlib python3-pytest python3-h5py'
    if ($LASTEXITCODE -ne 0) { throw 'FEniCS installation in Ubuntu failed.' }
}
& $pythonExe -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
if ($LASTEXITCODE -ne 0) { throw 'PyTorch import check failed.' }
& wsl.exe -d Ubuntu -- python3 -c 'import dolfin; print(dolfin.__version__)'
if ($LASTEXITCODE -ne 0) { throw 'FEniCS import check failed.' }
