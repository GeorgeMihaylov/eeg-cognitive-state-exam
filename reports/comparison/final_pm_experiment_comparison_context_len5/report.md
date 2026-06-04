# Final experiment comparison

## Configuration

```json
{
  "root": ".",
  "output_dir": "reports\\comparison\\final_pm_experiment_comparison_context_len5",
  "tabular": "data\\processed\\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv",
  "context": "reports\\runs\\20260512_152527_context_tabular_len5_fast_all_pow_plus_eeg_len5",
  "mha_len3": "reports\\runs\\20260508_172632_mha_all_pm_short_len3_control_all_pow_plus_eeg_len3",
  "mha_len5": "reports\\runs\\20260508_171708_mha_all_pm_short_len5_all_pow_plus_eeg_len5",
  "tabular_models": null,
  "context_models": null,
  "mha_len3_models": null,
  "mha_len5_models": null,
  "tabular_validation": null,
  "context_validation": null,
  "mha_len3_validation": null,
  "mha_len5_validation": null,
  "primary_metric": "r2",
  "secondary_metric": "spearman",
  "no_plots": false
}
```

## Source files

- **tabular**: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\baseline_pow_plus_eeg_w10_log_regression_metrics_agg.csv`
- **context**: `D:\PycharmProjects\eeg-cognitive-state-nir\reports\runs\20260512_152527_context_tabular_len5_fast_all_pow_plus_eeg_len5\all_targets_summary.csv`
- **mha_len3**: `D:\PycharmProjects\eeg-cognitive-state-nir\reports\runs\20260508_172632_mha_all_pm_short_len3_control_all_pow_plus_eeg_len3\all_targets_summary.csv`
- **mha_len5**: `D:\PycharmProjects\eeg-cognitive-state-nir\reports\runs\20260508_171708_mha_all_pm_short_len5_all_pow_plus_eeg_len5\all_targets_summary.csv`

## Best model per target and experiment

| experiment   | target     | model         | validation         |   folds |   n_val_total |   mae_mean |   rmse_mean |   r2_mean |   pearson_mean |   spearman_mean |   seq_len | feature_set   | feature_mode   |
|:-------------|:-----------|:--------------|:-------------------|--------:|--------------:|-----------:|------------:|----------:|---------------:|----------------:|----------:|:--------------|:---------------|
| tabular      | focus      | rf_reg        | groupkfold_subject |       5 |               |     0.0887 |      0.1147 |    0.1455 |         0.4098 |          0.3915 |           | nan           | nan            |
| context      | attention  | hgb_reg       | groupkfold_subject |       5 |         42932 |     0.0887 |      0.1132 |    0.2028 |         0.4718 |          0.4555 |         5 | pow_plus_eeg  | log_pow        |
| context      | engagement | hgb_reg       | groupkfold_subject |       5 |         47994 |     0.0855 |      0.1081 |    0.3057 |         0.5949 |          0.5037 |         5 | pow_plus_eeg  | log_pow        |
| context      | excitement | lgbm_reg      | groupkfold_subject |       5 |         50700 |     0.1118 |      0.1505 |    0.5794 |         0.7754 |          0.7184 |         5 | pow_plus_eeg  | log_pow        |
| context      | focus      | lgbm_reg      | groupkfold_subject |       5 |         45137 |     0.0775 |      0.1    |    0.3447 |         0.6109 |          0.5682 |         5 | pow_plus_eeg  | log_pow        |
| context      | interest   | hgb_reg       | groupkfold_subject |       5 |         45193 |     0.0601 |      0.0821 |    0.274  |         0.5636 |          0.4649 |         5 | pow_plus_eeg  | log_pow        |
| context      | relaxation | hgb_reg       | groupkfold_subject |       5 |         45147 |     0.0985 |      0.1249 |    0.4264 |         0.6728 |          0.6425 |         5 | pow_plus_eeg  | log_pow        |
| context      | stress     | hgb_reg       | groupkfold_subject |       5 |         45137 |     0.0796 |      0.1103 |    0.3473 |         0.6124 |          0.5017 |         5 | pow_plus_eeg  | log_pow        |
| mha_len3     | attention  | unknown_model | groupkfold_subject |       2 |          4013 |     0.0886 |      0.1143 |    0.2094 |         0.4922 |          0.4907 |           | nan           | nan            |
| mha_len3     | engagement | unknown_model | groupkfold_subject |       2 |          3984 |     0.0854 |      0.1087 |    0.2736 |         0.5388 |          0.48   |           | nan           | nan            |
| mha_len3     | excitement | unknown_model | groupkfold_subject |       2 |          4019 |     0.1122 |      0.152  |    0.5815 |         0.7838 |          0.7086 |           | nan           | nan            |
| mha_len3     | focus      | unknown_model | groupkfold_subject |       2 |          3986 |     0.089  |      0.1127 |    0.1688 |         0.4868 |          0.4517 |           | nan           | nan            |
| mha_len3     | interest   | unknown_model | groupkfold_subject |       2 |          3994 |     0.0697 |      0.0976 |    0.1282 |         0.3715 |          0.3564 |           | nan           | nan            |
| mha_len3     | relaxation | unknown_model | groupkfold_subject |       2 |          3974 |     0.1038 |      0.1342 |    0.3814 |         0.6573 |          0.6539 |           | nan           | nan            |
| mha_len3     | stress     | unknown_model | groupkfold_subject |       2 |          3986 |     0.0845 |      0.1163 |    0.1795 |         0.4743 |          0.3646 |           | nan           | nan            |
| mha_len5     | attention  | unknown_model | groupkfold_subject |       2 |          3988 |     0.0931 |      0.1191 |    0.1926 |         0.45   |          0.4433 |           | nan           | nan            |
| mha_len5     | engagement | unknown_model | groupkfold_subject |       2 |          4007 |     0.0869 |      0.1106 |    0.256  |         0.5256 |          0.4323 |           | nan           | nan            |
| mha_len5     | excitement | unknown_model | groupkfold_subject |       2 |          3964 |     0.1333 |      0.1797 |    0.4267 |         0.7036 |          0.6281 |           | nan           | nan            |
| mha_len5     | focus      | unknown_model | groupkfold_subject |       2 |          3977 |     0.0784 |      0.1008 |    0.3068 |         0.5594 |          0.5372 |           | nan           | nan            |
| mha_len5     | interest   | unknown_model | groupkfold_subject |       2 |          4019 |     0.0618 |      0.0842 |    0.1232 |         0.396  |          0.3399 |           | nan           | nan            |
| mha_len5     | relaxation | unknown_model | groupkfold_subject |       2 |          4020 |     0.1066 |      0.1374 |    0.2485 |         0.5531 |          0.5124 |           | nan           | nan            |
| mha_len5     | stress     | unknown_model | groupkfold_subject |       2 |          3977 |     0.0853 |      0.1206 |    0.2455 |         0.5658 |          0.4883 |           | nan           | nan            |

## Final comparison table

| target     | tabular_model   |   tabular_r2 |   tabular_spearman | context_model   |   context_r2 |   context_spearman | mha_len3_model   |   mha_len3_r2 |   mha_len3_spearman | mha_len5_model   |   mha_len5_r2 |   mha_len5_spearman |   delta_context_vs_tabular_r2 |   delta_mha_len3_vs_context_r2 |   delta_mha_len3_vs_tabular_r2 |   delta_mha_len5_vs_mha_len3_r2 |
|:-----------|:----------------|-------------:|-------------------:|:----------------|-------------:|-------------------:|:-----------------|--------------:|--------------------:|:-----------------|--------------:|--------------------:|------------------------------:|-------------------------------:|-------------------------------:|--------------------------------:|
| attention  | nan             |              |                    | hgb_reg         |       0.2028 |             0.4555 | unknown_model    |        0.2094 |              0.4907 | unknown_model    |        0.1926 |              0.4433 |                               |                         0.0066 |                                |                         -0.0168 |
| engagement | nan             |              |                    | hgb_reg         |       0.3057 |             0.5037 | unknown_model    |        0.2736 |              0.48   | unknown_model    |        0.256  |              0.4323 |                               |                        -0.0321 |                                |                         -0.0176 |
| excitement | nan             |              |                    | lgbm_reg        |       0.5794 |             0.7184 | unknown_model    |        0.5815 |              0.7086 | unknown_model    |        0.4267 |              0.6281 |                               |                         0.0021 |                                |                         -0.1548 |
| stress     | nan             |              |                    | hgb_reg         |       0.3473 |             0.5017 | unknown_model    |        0.1795 |              0.3646 | unknown_model    |        0.2455 |              0.4883 |                               |                        -0.1679 |                                |                          0.066  |
| relaxation | nan             |              |                    | hgb_reg         |       0.4264 |             0.6425 | unknown_model    |        0.3814 |              0.6539 | unknown_model    |        0.2485 |              0.5124 |                               |                        -0.045  |                                |                         -0.1329 |
| interest   | nan             |              |                    | hgb_reg         |       0.274  |             0.4649 | unknown_model    |        0.1282 |              0.3564 | unknown_model    |        0.1232 |              0.3399 |                               |                        -0.1458 |                                |                         -0.0051 |
| focus      | rf_reg          |       0.1455 |             0.3915 | lgbm_reg        |       0.3447 |             0.5682 | unknown_model    |        0.1688 |              0.4517 | unknown_model    |        0.3068 |              0.5372 |                        0.1992 |                        -0.1759 |                         0.0234 |                          0.138  |

## Interpretation notes

- Positive `delta_*_r2` means the left experiment has higher R2 than the right experiment.
- Positive `delta_*_spearman` means the left experiment has higher rank correlation.
- For `mae` and `rmse`, negative deltas are better because lower error is better.
- Main question: whether MHA improves over `context-tabular`; if not, the gain is mostly explained by temporal context itself.