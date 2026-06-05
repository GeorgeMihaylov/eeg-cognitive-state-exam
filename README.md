# EEG Cognitive State Exam Project

Exam project on EEG-based cognitive state prediction using EEG/POW window features and local temporal context.

## 1. Project idea

The project studies whether EEG-derived features can be used to predict cognitive and affective Performance Metrics.

The main targets are:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

The main research hypothesis:

> Local temporal context improves the prediction of cognitive state metrics compared with a single-window tabular baseline.

In addition to the main EEG/PM pipeline, the project includes an optional WESAD wearable physiology block for external stress-related validation.

---

## 2. Current project status

Current version includes:

```text
- processed EEG/PM dataset
- baseline ML models for one target
- multi-PM baseline pipeline
- temporal context tabular baseline scripts
- Multi-Head Attention / TransformerEncoder script
- experiment comparison script
- optional WESAD block
```

Already tested:

```text
- data/processed/windowed_eeg_pm_dataset_w10.parquet is readable
- src/07_train_baselines.py successfully runs on target_focus
- src/09_train_multi_pm_baselines.py successfully runs in smoke-test mode
```

Current processed dataset:

```text
Rows: 51 308
Columns: 508
POW features: 280
EEG features: 168
Total pow_plus_eeg features: 448
```

---

## 3. Repository structure

```text
eeg-cognitive-state-exam/
│
├── configs/
│   └── exam_config.yaml
│
├── data/
│   └── processed/
│       └── windowed_eeg_pm_dataset_w10.parquet
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── comparison/
│   ├── figures/
│   ├── results/
│   ├── runs/
│   ├── tables/
│   └── wearable_pm_alignment/
│
├── results/
│
├── src/
│   ├── 04_build_windowed_pm_dataset.py
│   ├── 05_analyze_pm_sampling.py
│   ├── 06_eda_windowed_dataset.py
│   ├── 07_train_baselines.py
│   ├── 09_train_multi_pm_baselines.py
│   ├── 10_describe_multi_pm_baseline.py
│   ├── 11_train_multihead_attention_short.py
│   ├── 12_visualize_mha_all_pm_run.py
│   ├── 13_train_context_tabular_baselines.py
│   ├── 14_compare_experiments.py
│   └── wesad/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 4. Environment setup

The project was tested with a Conda environment:

```text
D:\miniconda3\envs\eeg_nir\python.exe
```

Recommended setup:

```powershell
conda create -n eeg_nir python=3.11 -y
conda activate eeg_nir
pip install -r requirements.txt
```

If `python` is not available directly from PowerShell, use the full interpreter path:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe
```

Example:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe --version
```

---

## 5. Data

The main processed dataset must be located here:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

This file is not intended to be committed to GitHub if it is large.

Check that the dataset exists:

```powershell
Test-Path data\processed\windowed_eeg_pm_dataset_w10.parquet
```

Expected result:

```text
True
```

Check that the dataset is readable:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe -c "import pandas as pd; df=pd.read_parquet('data/processed/windowed_eeg_pm_dataset_w10.parquet'); print(df.shape); print(df.columns[:30].tolist())"
```

Expected shape:

```text
(51308, 508)
```

Check available target columns:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe -c "import pandas as pd; df=pd.read_parquet('data/processed/windowed_eeg_pm_dataset_w10.parquet'); print([c for c in df.columns if c.startswith('target_')])"
```

Expected target columns:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

---

## 6. Configuration

Main config file:

```text
configs/exam_config.yaml
```

Current target configuration:

```yaml
targets:
  - target_attention
  - target_engagement
  - target_excitement
  - target_stress
  - target_relaxation
  - target_interest
  - target_focus
```

Main feature set:

```yaml
features:
  default_feature_set: pow_plus_eeg
```

Main validation strategy:

```yaml
validation:
  strategy: group_kfold
  n_splits: 5
```

The main evaluation scheme is:

```text
GroupKFold by subject_id
```

This is used as the primary validation scheme because it estimates transfer to unseen users more honestly than a random split.

---

## 7. Scripts

## 7.1. Dataset preparation

### `src/04_build_windowed_pm_dataset.py`

Builds the windowed EEG/PM dataset.

Main role:

```text
Creates window-level EEG/PM observations.
```

---

### `src/05_analyze_pm_sampling.py`

Analyzes the sampling frequency of Performance Metrics.

Main role:

```text
Supports the choice of the 10-second window size.
```

---

### `src/06_eda_windowed_dataset.py`

Runs exploratory analysis of the processed windowed dataset.

Main role:

```text
Checks dataset size, missing values, target distributions and basic statistics.
```

---

### `src/08_build_windowed_eeg_features.py`

Builds EEG-derived features and combines them with POW features.

Main role:

```text
Creates the main feature space used by baseline and temporal models.
```

---

## 7.2. Baseline models

### `src/07_train_baselines.py`

Trains baseline models for one selected target.

Supports:

```text
- classification
- regression
- random split
- GroupKFold by subject_id
- cross-source validation
```

Supported feature sets:

```text
pow
eeg
pow_plus_eeg
```

Check arguments:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\07_train_baselines.py --help
```

---

### `src/09_train_multi_pm_baselines.py`

Trains regression baselines for all PM targets.

Supported models:

```text
ridge
hgb
lgbm
rf
```

Check arguments:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py --help
```

---

### `src/10_describe_multi_pm_baseline.py`

Summarizes multi-PM baseline results.

Main role:

```text
Aggregates model metrics and prepares report-friendly summaries.
```

---

## 7.3. Temporal context and attention models

### `src/11_train_multihead_attention_short.py`

Trains a Multi-Head Attention / TransformerEncoder model on sequences of neighboring EEG/POW windows.

Main idea:

```text
Instead of using only X_t, the model uses a local sequence:
[X_{t-1}, X_t, X_{t+1}]
or
[X_{t-2}, X_{t-1}, X_t, X_{t+1}, X_{t+2}]
```

---

### `src/12_visualize_mha_all_pm_run.py`

Builds visualizations for MHA experiments.

---

### `src/13_train_context_tabular_baselines.py`

Trains context-tabular baselines.

Main idea:

```text
Concatenate neighboring windows into one tabular vector and train a classical ML model.
```

This helps test whether the improvement comes from temporal context itself or from the attention architecture.

---

### `src/14_compare_experiments.py`

Compares:

```text
- single-window tabular baseline
- context-tabular baseline
- Multi-Head Attention model
```

Main role:

```text
Builds the final comparison table for the exam report.
```

---

## 7.4. WESAD optional block

The directory:

```text
src/wesad/
```

contains additional scripts for a wearable stress-related benchmark.

This block is optional for the exam project and should be treated as an additional validation, not as the main research line.

---

## 8. Commands for running experiments

## 8.1. Smoke-test baseline for `target_focus`

Use this command to quickly verify that the baseline pipeline works:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\07_train_baselines.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --output-prefix results/baseline_focus_smoke `
  --reg-target target_focus `
  --class-target label_q5 `
  --feature-set pow_plus_eeg `
  --feature-mode raw_pow `
  --n-splits 3 `
  --test-size 0.2 `
  --max-rows 5000 `
  --fast `
  --skip-cross-source
```

Expected outputs:

```text
data/processed/results/baseline_focus_smoke_classification_metrics.csv
data/processed/results/baseline_focus_smoke_regression_metrics.csv
data/processed/results/baseline_focus_smoke_classification_metrics_agg.csv
data/processed/results/baseline_focus_smoke_regression_metrics_agg.csv
reports/results/baseline_focus_smoke_report.md
reports/figures/results/baseline_focus_smoke/
```

---

## 8.2. Full baseline for `target_focus`

Use this command for the main single-target baseline:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\07_train_baselines.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --output-prefix results/baseline_focus_full `
  --reg-target target_focus `
  --class-target label_q5 `
  --feature-set pow_plus_eeg `
  --feature-mode raw_pow `
  --n-splits 5 `
  --test-size 0.2 `
  --skip-cross-source
```

Current full baseline result for `target_focus` using GroupKFold by subject:

```text
Best regression model: rf_reg

R² mean:       0.146
MAE mean:      0.0887
RMSE mean:     0.1147
Pearson mean:  0.411
Spearman mean: 0.392
```

Random split gives much higher values:

```text
R²:       0.661
Pearson:  0.830
Spearman: 0.819
```

Interpretation:

```text
Random split overestimates model quality. GroupKFold by subject_id is the main validation scheme for the exam report.
```

---

## 8.3. Smoke-test multi-PM baseline

Use this command to verify the multi-target baseline pipeline:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name multi_pm_smoke `
  --feature-set pow_plus_eeg `
  --feature-mode raw_pow `
  --models hgb,lgbm `
  --validation groupkfold `
  --n-splits 3 `
  --max-rows 5000 `
  --seed 42
```

Current smoke-test best results:

| Target     | Best model |    R² | Spearman |
| ---------- | ---------: | ----: | -------: |
| excitement |    hgb_reg | 0.293 |    0.507 |
| relaxation |   lgbm_reg | 0.226 |    0.467 |
| engagement |    hgb_reg | 0.219 |    0.427 |
| stress     |   lgbm_reg | 0.166 |    0.366 |
| attention  |   lgbm_reg | 0.110 |    0.364 |
| interest   |    hgb_reg | 0.096 |    0.338 |
| focus      |   lgbm_reg | 0.088 |    0.317 |

---

## 8.4. Full multi-PM baseline

Use this command for the main multi-target baseline:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\09_train_multi_pm_baselines.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name multi_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --feature-mode raw_pow `
  --models hgb,lgbm,rf `
  --validation groupkfold `
  --n-splits 5 `
  --seed 42
```

After the run, find the latest output directory:

```powershell
Get-ChildItem reports\runs | Sort-Object LastWriteTime -Descending | Select-Object -First 5 Name, LastWriteTime
```

Print the aggregated result table:

```powershell
$run = Get-ChildItem reports\runs | Where-Object {$_.Name -like "*multi_pm_groupkfold_full*"} | Sort-Object LastWriteTime -Descending | Select-Object -First 1
Import-Csv "$($run.FullName)\target_metrics_agg.csv" | Sort-Object {[double]$_.r2_mean} -Descending | Format-Table target_name,model,mae_mean,rmse_mean,r2_mean,pearson_mean,spearman_mean
```

Expected files:

```text
reports/runs/<timestamp>_multi_pm_groupkfold_full_pow_plus_eeg_raw_pow/target_fold_metrics.csv
reports/runs/<timestamp>_multi_pm_groupkfold_full_pow_plus_eeg_raw_pow/target_metrics_agg.csv
reports/runs/<timestamp>_multi_pm_groupkfold_full_pow_plus_eeg_raw_pow/target_summary.csv
reports/runs/<timestamp>_multi_pm_groupkfold_full_pow_plus_eeg_raw_pow/report.md
```

---

## 8.5. Context-tabular baseline

Check available arguments:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\13_train_context_tabular_baselines.py --help
```

The expected experiment should compare:

```text
seq_len = 3
seq_len = 5
feature_set = pow_plus_eeg
validation = GroupKFold by subject_id
targets = all PM targets
```

The purpose is to test whether local temporal context improves prediction quality.

---

## 8.6. Multi-Head Attention model

Check available arguments:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\11_train_multihead_attention_short.py --help
```

This experiment trains an attention-based model on sequences of neighboring windows.

The purpose is to compare:

```text
simple context-tabular model
vs
MHA / TransformerEncoder model
```

---

## 8.7. Experiment comparison

Check available arguments:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\14_compare_experiments.py --help
```

This script should be used after baseline, context-tabular and MHA experiments are available.

Expected comparison:

```text
single-window baseline
context-tabular seq_len=3
context-tabular seq_len=5
MHA seq_len=3
MHA seq_len=5
```

---

## 9. Git and artifact policy

Do not commit large or intermediate generated files:

```text
data/processed/*.parquet
data/processed/results/
models/*.pt
models/*.pkl
models/*.joblib
```

Commit useful report artifacts:

```text
reports/results/*.md
reports/figures/**/*.png
reports/runs/<final_run>/target_metrics_agg.csv
reports/runs/<final_run>/target_summary.csv
reports/runs/<final_run>/report.md
```

Recommended `.gitignore` entries:

```gitignore
# Data
data/processed/*.parquet
data/processed/results/
data/raw/
data/external/

# Models
models/*.pt
models/*.pth
models/*.pkl
models/*.joblib

# Python
__pycache__/
*.py[cod]
.ipynb_checkpoints/
.venv/
venv/
env/

# Smoke runs
reports/runs/*smoke*/
```

---

## 10. Current conclusions

Current baseline experiments show:

```text
1. The processed EEG/PM dataset is valid and readable.
2. EEG/POW features can predict several PM metrics above trivial baseline.
3. Random split strongly overestimates quality.
4. GroupKFold by subject_id is stricter and should be used as the main validation strategy.
5. target_focus is relatively difficult, which makes it useful for temporal context experiments.
6. Multi-PM smoke-test suggests that excitement, relaxation and engagement are more predictable than focus on a small sample.
```

---

## 11. Next steps

Required next steps:

```text
[ ] Finish full multi-PM GroupKFold baseline.
[ ] Commit final report artifacts from the full multi-PM run.
[ ] Run context-tabular baseline for seq_len=3 and seq_len=5.
[ ] Run or transfer MHA results.
[ ] Compare baseline vs context-tabular vs MHA.
[ ] Add Autoencoder-based synthetic EEG-feature generation.
[ ] Add real vs synthetic comparison.
[ ] Add synthetic augmentation experiment.
[ ] Prepare final report.
[ ] Prepare final demo notebook.
[ ] Prepare defense text.
```

Minimum exam-ready pipeline:

```text
1. Dataset preparation
2. Single-target baseline
3. Multi-PM baseline
4. Context-tabular temporal baseline
5. MHA comparison
6. Final comparison report
```

Additional laboratory-compliance pipeline:

```text
7. Autoencoder generation
8. Real vs synthetic comparison
9. Synthetic augmentation
```
