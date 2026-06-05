# LSTM PM Dynamics Report

## Run configuration

- Dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`
- Feature set: `pow_plus_eeg`
- Sequence length: `5`
- Target position: `center`
- Validation: GroupKFold by `subject_id`
- Folds: `5`
- Device: `cuda`
- Epochs: `60`
- Batch size: `512`
- AMP: `True`

## Dataset info

```json
{
  "rows_raw": 51308,
  "rows_clean": 43174,
  "columns_raw": 508,
  "subjects_clean": 53,
  "records_clean": 117,
  "targets": [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus"
  ]
}
```

## Feature info

```json
{
  "feature_set": "pow_plus_eeg",
  "n_features": 448,
  "n_targets": 7
}
```

## Sequence info

```json
{
  "n_sequences": 42706,
  "seq_len": 5,
  "n_features": 448,
  "target_position": "center",
  "subjects": 53,
  "records": 117
}
```

## Aggregated metrics

| target            |   folds |   mae_mean |    mae_std |   rmse_mean |   rmse_std |   r2_mean |    r2_std |   pearson_mean |   pearson_std |   spearman_mean |   spearman_std |
|:------------------|--------:|-----------:|-----------:|------------:|-----------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:|
| target_excitement |       5 |  0.133891  | 0.00560001 |   0.177177  | 0.0098791  | 0.371572  | 0.0717873 |       0.639532 |     0.0649806 |        0.59566  |      0.0789032 |
| target_relaxation |       5 |  0.112917  | 0.00496836 |   0.14181   | 0.00729525 | 0.260346  | 0.146181  |       0.562732 |     0.0620971 |        0.522464 |      0.0688536 |
| target_interest   |       5 |  0.0644765 | 0.00943983 |   0.0866082 | 0.0104756  | 0.168258  | 0.0459155 |       0.460061 |     0.0304639 |        0.369615 |      0.0260603 |
| target_focus      |       5 |  0.0867943 | 0.00817342 |   0.112441  | 0.0104207  | 0.167697  | 0.134701  |       0.478776 |     0.104883  |        0.446073 |      0.125775  |
| target_stress     |       5 |  0.0904901 | 0.0150089  |   0.128906  | 0.0208857  | 0.137148  | 0.058949  |       0.468422 |     0.0462481 |        0.414547 |      0.0132774 |
| target_engagement |       5 |  0.0928631 | 0.00448245 |   0.118472  | 0.00482858 | 0.128741  | 0.0994827 |       0.448996 |     0.0685405 |        0.368499 |      0.0635415 |
| target_attention  |       5 |  0.0963656 | 0.00842913 |   0.123679  | 0.0109824  | 0.0483079 | 0.117803  |       0.379434 |     0.093424  |        0.374309 |      0.0854133 |

## Interpretation

This experiment adds a sequence-based deep learning model to the project. The model receives a local sequence of neighboring EEG/POW windows and predicts the PM metrics for the selected target position. This directly tests whether local temporal dynamics improve cognitive-state prediction compared with single-window tabular baselines.