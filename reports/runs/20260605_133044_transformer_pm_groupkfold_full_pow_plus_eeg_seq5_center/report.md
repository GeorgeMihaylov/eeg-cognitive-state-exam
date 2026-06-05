# Transformer PM Dynamics Report

## Run configuration

- Dataset: `data/processed/windowed_eeg_pm_dataset_w10.parquet`
- Feature set: `pow_plus_eeg`
- Sequence length: `5`
- Target position: `center`
- Pooling: `center`
- d_model: `128`
- nhead: `4`
- Transformer layers: `2`
- Validation: GroupKFold by `subject_id`
- Folds: `5`
- Device: `cuda`
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
  "pooling": "center",
  "subjects": 53,
  "records": 117
}
```

## Aggregated metrics

| target            |   folds |   mae_mean |    mae_std |   rmse_mean |   rmse_std |   r2_mean |    r2_std |   pearson_mean |   pearson_std |   spearman_mean |   spearman_std |
|:------------------|--------:|-----------:|-----------:|------------:|-----------:|----------:|----------:|---------------:|--------------:|----------------:|---------------:|
| target_excitement |       5 |  0.124946  | 0.0101249  |   0.166256  | 0.0118849  | 0.435636  | 0.151867  |       0.684282 |     0.109025  |        0.62434  |      0.132218  |
| target_relaxation |       5 |  0.106013  | 0.00750142 |   0.134988  | 0.0083818  | 0.329365  | 0.136612  |       0.617705 |     0.0673288 |        0.580842 |      0.0802863 |
| target_interest   |       5 |  0.0613443 | 0.00960676 |   0.0844023 | 0.0123659  | 0.214179  | 0.0583648 |       0.517787 |     0.0699397 |        0.429093 |      0.0697236 |
| target_focus      |       5 |  0.0862681 | 0.00696275 |   0.111848  | 0.00898311 | 0.170508  | 0.17483   |       0.506852 |     0.114608  |        0.472063 |      0.133438  |
| target_stress     |       5 |  0.088632  | 0.012659   |   0.12783   | 0.0197455  | 0.148657  | 0.0786341 |       0.52246  |     0.0490156 |        0.450752 |      0.0334732 |
| target_engagement |       5 |  0.0954786 | 0.0120496  |   0.121465  | 0.0167692  | 0.078029  | 0.227451  |       0.506554 |     0.0917984 |        0.448832 |      0.098939  |
| target_attention  |       5 |  0.0957559 | 0.0091111  |   0.122702  | 0.0112699  | 0.0532111 | 0.195935  |       0.444823 |     0.118698  |        0.435877 |      0.0982544 |

## Interpretation

This experiment uses a TransformerEncoder on local EEG/POW window sequences. It provides an attention-based deep learning comparison to the MLP and LSTM models. The goal is to test whether self-attention over neighboring windows improves prediction of cognitive-state PM metrics.