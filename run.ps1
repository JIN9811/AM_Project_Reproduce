param(
    [ValidateSet('full','verified','fem','pinn','test','plots','paper','appendix','parametric','baselines','authors','paper-figures','process')]
    [string]$Task = 'verified',
    [switch]$Force,
    [int]$StudySteps = 10000,
    [string]$DocumentPython = ''
)
$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location -LiteralPath $root
$localPython = Join-Path $root '.venv\Scripts\python.exe'
$sessionPython = Join-Path $root '..\..\work\pinn-env\Scripts\python.exe'
if (Test-Path -LiteralPath $localPython) { $pythonExe = $localPython }
elseif (Test-Path -LiteralPath $sessionPython) { $pythonExe = (Resolve-Path -LiteralPath $sessionPython).Path }
else { throw 'Run .\setup.ps1 once to install the Python and FEniCS environments.' }

function Invoke-Python {
    & $pythonExe @args
    if ($LASTEXITCODE -ne 0) { throw "Python command failed (exit $LASTEXITCODE)." }
}
function Invoke-Fenics {
    & wsl.exe -d Ubuntu --cd $root -- env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 @args
    if ($LASTEXITCODE -ne 0) { throw "FEniCS/WSL command failed (exit $LASTEXITCODE)." }
}
function Run-Fem {
    foreach ($geometry in @('circle','rectangles')) {
        foreach ($n in @(50,100,200)) {
            $output = "results/fem_${geometry}_n${n}"
            if ($Force -or !(Test-Path -LiteralPath "$output/metadata.json")) {
                Invoke-Fenics -m harandi.fem --geometry $geometry --n $n --output $output
            }
        }
        $output = "results/fem_${geometry}_n100_nodal"
        if ($Force -or !(Test-Path -LiteralPath "$output/metadata.json")) {
            Invoke-Fenics -m harandi.fem --geometry $geometry --n 100 --coefficient-mode nodal --output $output
        }
    }
}
function Run-Pinn {
    foreach ($geometry in @('circle','rectangles')) {
        foreach ($mode in @('coupled','sequential')) {
            $output = "results/pinn_${geometry}_${mode}"
            if ($Force -or !(Test-Path -LiteralPath "$output/metrics.json")) {
                $gridSize = if ($geometry -eq 'rectangles') { 71 } else { 51 }
                Invoke-Python -m harandi.train --geometry $geometry --mode $mode --n $gridSize --adam-steps 2000 --lbfgs-steps 3000 --switch 2000 --output $output
            }
        }
    }
}
switch ($Task) {
    'full' {
        if (!$DocumentPython) { $DocumentPython = $pythonExe }
        & $DocumentPython -c 'import reportlab, PIL'
        if ($LASTEXITCODE -ne 0) { throw 'Install requirements-documents.txt or specify -DocumentPython.' }
        Run-Fem
        Invoke-Python -m harandi.full_study --fem-only
        Invoke-Python -m harandi.full_study --groups mixed baseline transfer --workers 1
        Invoke-Python scripts/finish_full_study.py --document-python $DocumentPython
    }
    'test' {
        Invoke-Python -m pytest -q --tb=short
        Invoke-Fenics -m pytest -q --tb=short
    }
    'fem' { Run-Fem }
    'pinn' { Run-Pinn }
    'plots' { Invoke-Python -m harandi.report }
    'paper-figures' { Invoke-Python -m harandi.paper_figures }
    'process' { Invoke-Python -m harandi.process --output results/process_circle_coupled; Invoke-Python -m harandi.paper_figures }
    'authors' { Invoke-Python -m harandi.authors evaluate data/authors_released_weights.npz --output results/authors_released_checkpoint }
    'verified' {
        Run-Fem
        Run-Pinn
        Invoke-Python -m harandi.authors evaluate data/authors_released_weights.npz --output results/authors_released_checkpoint
        Invoke-Python -m harandi.report
    }
    'paper' {
        foreach ($geometry in @('circle','rectangles')) {
            foreach ($mode in @('coupled','sequential')) {
                $output = "results/paper_${geometry}_${mode}"
                if ($Force -or !(Test-Path -LiteralPath "$output/metrics.json")) {
                    Invoke-Python -m harandi.train --geometry $geometry --mode $mode --soft --energy paper --adam-steps 100000 --lbfgs-steps 0 --switch 20000 --output $output
                }
            }
        }
    }
    'appendix' {
        Invoke-Python -m harandi.train --geometry rectangles --mode coupled --energy authors --adam-steps 0 --lbfgs-steps 10000 --output results/appendix_rectangles_lbfgs
    }
    'parametric' {
        foreach ($ratio in @((1..10) + @(15))) {
            $output = "results/parametric_fem/ratio_$ratio"
            if ($Force -or !(Test-Path -LiteralPath "$output/metadata.json")) {
                Invoke-Fenics -m harandi.fem --geometry rectangles --ratio $ratio --n 100 --output $output
            }
        }
        foreach ($weight in @(0.0,0.1)) {
            Invoke-Python -m harandi.transfer --references results/parametric_fem --steps $StudySteps --weight $weight --output "results/transfer_w$weight"
        }
    }
    'baselines' {
        foreach ($geometry in @('circle','rectangles')) {
            foreach ($method in @('standard','dem')) {
                Invoke-Python -m harandi.baselines --method $method --geometry $geometry --steps $StudySteps --output "results/baseline_${geometry}_${method}"
            }
        }
    }
}
