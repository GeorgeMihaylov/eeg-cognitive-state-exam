# MLP PM Regressor Report

## Run configuration

- Dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`
- Feature set: `pow_plus_eeg`
- Validation: GroupKFold by `subject_id`
- Folds: `5`
- Device: `cuda`
- Epochs: `80`
- Batch size: `1024`
- AMP: `True`

## Dataset info

```json
{
  "rows_raw": 51308,
  "rows_clean": 43174,
  "columns_raw": 508,
  "subjects": 53,
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

## Aggregated metrics

| target            |   folds |   mae_mean |    mae_std |   rmse_mean |   rmse_std |     r2_mean |   r2_std |   pearson_mean |   pearson_std |   spearman_mean |   spearman_std |
|:------------------|--------:|-----------:|-----------:|------------:|-----------:|------------:|---------:|---------------:|--------------:|----------------:|---------------:|
| target_excitement |       5 |  0.146598  | 0.00930768 |    0.191453 | 0.011609   |  0.266737   | 0.110057 |       0.519571 |     0.0910636 |        0.493298 |      0.0894433 |
| target_attention  |       5 |  0.0946393 | 0.00686199 |    0.123504 | 0.00988899 |  0.0479648  | 0.124638 |       0.327843 |     0.0891174 |        0.335147 |      0.0978545 |
| target_stress     |       5 |  0.0923718 | 0.0136517  |    0.13828  | 0.0201291  | -0.00536945 | 0.304272 |       0.347865 |     0.115383  |        0.355773 |      0.0184308 |
| target_focus      |       5 |  0.0941458 | 0.00769714 |    0.131088 | 0.0283122  | -0.160469   | 0.546875 |       0.295873 |     0.0763608 |        0.290149 |      0.074162  |
| target_engagement |       5 |  0.0950306 | 0.0110364  |    0.139549 | 0.0494997  | -0.22738    | 0.828281 |       0.370225 |     0.0835931 |        0.352387 |      0.0455024 |
| target_relaxation |       5 |  0.124115  | 0.009592   |    0.189925 | 0.0843169  | -0.485615   | 1.4333   |       0.3953   |     0.0988043 |        0.431151 |      0.026555  |
| target_interest   |       5 |  0.0668373 | 0.00899286 |    0.114191 | 0.0580044  | -0.862305   | 2.30899  |       0.389981 |     0.105244  |        0.303741 |      0.0637612 |

## Interpretation

This experiment adds a neural-network baseline to the project. The model is a multi-output MLP regressor trained on EEG/POW window features. The main validation scheme is GroupKFold by subject_id, which estimates transfer to unseen users more strictly than a random split.