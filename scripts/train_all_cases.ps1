param(
    [string]$Python = "python",
    [string[]]$Configs = @(
        "configs/case_i.yaml",
        "configs/case_ii.yaml",
        "configs/case_iii.yaml"
    )
)

$ErrorActionPreference = "Stop"

foreach ($Config in $Configs) {
    Write-Host "Training with $Config"
    & $Python scripts/train_model.py --config $Config
}
