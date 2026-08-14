param(
    [string]$DatasetRoot = "D:\BaiduNetdiskDownload\TJ4DRadSet_Full",
    [string]$PythonExe = "C:\Users\lth\miniconda3\envs\radar_pseudo\python.exe",
    [int]$Epochs = 20,
    [int]$BatchSize = 8,
    [int[]]$Seeds = @(42, 7, 2026)
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$TrainSplit = Join-Path $ProjectRoot "outputs\protocol_splits_v2\train_core.txt"
$CalibrationSplit = Join-Path $ProjectRoot "outputs\protocol_splits_v2\teacher_calibration.txt"
$ValidationSplit = Join-Path $DatasetRoot "ImageSets\val.txt"
$AssociationRoot = Join-Path $ProjectRoot "outputs\pseudo_mgu_assoc_train_core"
$FinalRoot = Join-Path $ProjectRoot "outputs\pseudo_mgu_final_train_core"
$SelectedRoot = Join-Path $ProjectRoot "outputs\pseudo_mgu_final_selected_train_core"
$BaselineRoot = Join-Path $ProjectRoot "outputs\pseudo_q06_train"
$EqualRoot = Join-Path $ProjectRoot "outputs\gt_equal_mgu_final"
$CompleteRoot = Join-Path $ProjectRoot "outputs\gt_complete_mgu_final"

function Invoke-Checked {
    param([string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed: $PythonExe $($Arguments -join ' ')"
    }
}

Set-Location $ProjectRoot

Invoke-Checked @(
    "-m", "radar_pseudo.batch_generate",
    "--dataset-root", $DatasetRoot,
    "--split-file", $TrainSplit,
    "--output-root", $AssociationRoot,
    "--method", "mgu",
    "--quality-threshold", "0.0",
    "--confidence", "0.05",
    "--image-size", "1280"
)

Invoke-Checked @(
    "-m", "radar_pseudo.refine_geometry",
    "--split-file", $TrainSplit,
    "--input-root", $AssociationRoot,
    "--output-root", $FinalRoot,
    "--mode", "selective_reprojection_direct",
    "--dataset-root", $DatasetRoot,
    "--confidence-threshold", "0.5",
    "--reprojection-classes", "Truck"
)

Invoke-Checked @(
    "-m", "radar_pseudo.filter_pseudo_labels",
    "--split-file", $TrainSplit,
    "--input-root", $FinalRoot,
    "--output-root", $SelectedRoot
)

Invoke-Checked @(
    "-m", "radar_pseudo.make_equal_gt",
    "--dataset-root", $DatasetRoot,
    "--split-file", $TrainSplit,
    "--pseudo-label-dir", (Join-Path $SelectedRoot "label_2"),
    "--output-dir", (Join-Path $EqualRoot "label_2"),
    "--ignore-output-dir", (Join-Path $EqualRoot "ignore_label_2"),
    "--seed", "42"
)

$SelectedManifest = Get-Content -Raw -Encoding UTF8 (Join-Path $SelectedRoot "manifest.json") | ConvertFrom-Json
Invoke-Checked @(
    "-m", "radar_pseudo.make_frame_budget_split",
    "--dataset-root", $DatasetRoot,
    "--split-file", $TrainSplit,
    "--target-objects", ([string]$SelectedManifest.output_total),
    "--output-split", (Join-Path $CompleteRoot "train.txt"),
    "--output-manifest", (Join-Path $CompleteRoot "manifest.json"),
    "--seed", "42"
)

$Variants = @(
    @{
        Name = "b0"
        Split = $TrainSplit
        Extra = @("--label-dir", (Join-Path $BaselineRoot "label_2"), "--metadata-dir", (Join-Path $BaselineRoot "metadata"))
    },
    @{
        Name = "mgu"
        Split = $TrainSplit
        Extra = @("--label-dir", (Join-Path $SelectedRoot "label_2"), "--metadata-dir", (Join-Path $SelectedRoot "metadata"))
    },
    @{
        Name = "gt_equal"
        Split = $TrainSplit
        Extra = @("--label-dir", (Join-Path $EqualRoot "label_2"), "--ignore-label-dir", (Join-Path $EqualRoot "ignore_label_2"))
    },
    @{
        Name = "gt_complete"
        Split = Join-Path $CompleteRoot "train.txt"
        Extra = @()
    },
    @{
        Name = "gt_full"
        Split = $TrainSplit
        Extra = @()
    }
)

foreach ($Seed in $Seeds) {
    foreach ($Variant in $Variants) {
        $RunRoot = Join-Path $ProjectRoot "outputs\students_final\$($Variant.Name)_seed$Seed"
        $TrainArguments = @(
            "-m", "radar_pseudo.train_student",
            "--dataset-root", $DatasetRoot,
            "--split-file", $Variant.Split,
            "--output-dir", $RunRoot,
            "--epochs", ([string]$Epochs),
            "--batch-size", ([string]$BatchSize),
            "--seed", ([string]$Seed)
        ) + $Variant.Extra
        $HistoryPath = Join-Path $RunRoot "history.json"
        $CheckpointPath = Join-Path $RunRoot "last.pt"
        $CompletedEpochs = 0
        if (Test-Path $HistoryPath) {
            $CompletedEpochs = @((Get-Content -Raw -Encoding UTF8 $HistoryPath | ConvertFrom-Json)).Count
        }
        if (-not (Test-Path $CheckpointPath) -or $CompletedEpochs -lt $Epochs) {
            Invoke-Checked $TrainArguments
        }
        Invoke-Checked @(
            "-m", "radar_pseudo.evaluate_student",
            "--dataset-root", $DatasetRoot,
            "--split-file", $ValidationSplit,
            "--checkpoint", $CheckpointPath,
            "--output", (Join-Path $ProjectRoot "outputs\evaluation\$($Variant.Name)_seed$Seed.json"),
            "--batch-size", ([string]$BatchSize)
        )
    }
}

Invoke-Checked @(
    "-m", "radar_pseudo.summarize_experiments",
    "--input-dir", (Join-Path $ProjectRoot "outputs\evaluation"),
    "--output", (Join-Path $ProjectRoot "outputs\evaluation\final_multiseed_summary.json")
)
Invoke-Checked @("-m", "pytest", "-q")
