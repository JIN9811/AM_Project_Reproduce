param([string]$Notebook = 'Full_Paper_Reproduction_KO.ipynb')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = Join-Path $PSScriptRoot '..\..\work\pinn-env\Scripts\python.exe'
}
if (!(Test-Path -LiteralPath $pythonExe)) { throw 'Run setup.ps1 first.' }
& $pythonExe -m ipykernel install --sys-prefix --name harandi --display-name 'Harandi CUDA Python 3.11'
if ($LASTEXITCODE -ne 0) { throw 'Notebook kernel setup failed.' }
if (!(Test-Path -LiteralPath $Notebook)) { throw "Notebook does not exist: $Notebook" }
& $pythonExe -m jupyterlab $Notebook
if ($LASTEXITCODE -ne 0) { throw 'JupyterLab failed.' }
