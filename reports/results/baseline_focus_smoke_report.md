# Baseline report

Dataset: `D:\PycharmProjects\eeg-cognitive-state-exam\data\processed\windowed_eeg_pm_dataset_w10.parquet`

## Feature policy

- Excluded from features: `PM.*`, `target_*`, `label_*`, `source`, `subject_id`, `record_id`, time/meta columns.
- Reason: `target_main` is derived from `PM.Focus.Scaled`; including PM columns would cause target leakage.

- Feature set: **pow_plus_eeg**
- POW feature mode: **raw_pow**
- Raw POW feature columns available: **280**
- EEG feature columns available: **168**
- Final feature columns used: **448**

POW log transformation is applied only to `POW.*`. EEG features are used in their original scale.

## Classification data

|   rows |   subjects |   records | sources                             | class_distribution                                 |   min_windows_per_subject | feature_set   | feature_mode   |   n_features |
|-------:|-----------:|----------:|:------------------------------------|:---------------------------------------------------|--------------------------:|:--------------|:---------------|-------------:|
|   4367 |         49 |       111 | {"gpn_data": 2253, "Old_EEG": 2114} | {"0": 807, "1": 854, "2": 895, "3": 916, "4": 895} |                        30 | pow_plus_eeg  | raw_pow        |          448 |

## Regression data

|   rows |   subjects |   records | sources                             |   target_mean |   target_std |   target_min |   target_median |   target_max |   min_windows_per_subject | feature_set   | feature_mode   |   n_features |
|-------:|-----------:|----------:|:------------------------------------|--------------:|-------------:|-------------:|----------------:|-------------:|--------------------------:|:--------------|:---------------|-------------:|
|   4367 |         49 |       111 | {"gpn_data": 2253, "Old_EEG": 2114} |      0.435101 |     0.123769 |     0.097024 |        0.419729 |     0.975516 |                        30 | pow_plus_eeg  | raw_pow        |          448 |

## Classification aggregated metrics

| task           | validation         | model         |   folds |   accuracy_mean |   accuracy_std |   accuracy_min |   accuracy_max |   balanced_accuracy_mean |   balanced_accuracy_std |   balanced_accuracy_min |   balanced_accuracy_max |   macro_f1_mean |   macro_f1_std |   macro_f1_min |   macro_f1_max |   weighted_f1_mean |   weighted_f1_std |   weighted_f1_min |   weighted_f1_max |
|:---------------|:-------------------|:--------------|--------:|----------------:|---------------:|---------------:|---------------:|-------------------------:|------------------------:|------------------------:|------------------------:|----------------:|---------------:|---------------:|---------------:|-------------------:|------------------:|------------------:|------------------:|
| classification | groupkfold_subject | hgb_clf       |       3 |        0.258821 |      0.0209411 |       0.24446  |       0.282849 |                 0.259149 |               0.0181789 |                0.242659 |                0.278643 |        0.254223 |      0.0196172 |       0.237304 |       0.275727 |           0.255167 |         0.021471  |          0.241687 |          0.279927 |
| classification | groupkfold_subject | lgbm_clf      |       3 |        0.264239 |      0.0157667 |       0.247922 |       0.279391 |                 0.263063 |               0.0171996 |                0.243532 |                0.27595  |        0.258064 |      0.014413  |       0.242046 |       0.269986 |           0.259963 |         0.0127574 |          0.248325 |          0.273603 |
| classification | groupkfold_subject | logreg_robust |       3 |        0.248559 |      0.0124141 |       0.234259 |       0.25657  |                 0.246467 |               0.014277  |                0.230589 |                0.258247 |        0.244811 |      0.0129651 |       0.229841 |       0.252399 |           0.246955 |         0.0135251 |          0.231538 |          0.256827 |
| classification | random_split       | hgb_clf       |       1 |        0.399314 |      0         |       0.399314 |       0.399314 |                 0.401206 |               0         |                0.401206 |                0.401206 |        0.396197 |      0         |       0.396197 |       0.396197 |           0.39434  |         0         |          0.39434  |          0.39434  |
| classification | random_split       | lgbm_clf      |       1 |        0.392449 |      0         |       0.392449 |       0.392449 |                 0.395238 |               0         |                0.395238 |                0.395238 |        0.390213 |      0         |       0.390213 |       0.390213 |           0.387666 |         0         |          0.387666 |          0.387666 |
| classification | random_split       | logreg_robust |       1 |        0.346682 |      0         |       0.346682 |       0.346682 |                 0.347464 |               0         |                0.347464 |                0.347464 |        0.347572 |      0         |       0.347572 |       0.347572 |           0.346889 |         0         |          0.346889 |          0.346889 |

## Classification fold metrics

| task           | validation         | model         |   fold |   n_train |   n_test |   elapsed_s |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |   n_train_subjects |   n_test_subjects |
|:---------------|:-------------------|:--------------|-------:|----------:|---------:|------------:|-----------:|--------------------:|-----------:|--------------:|-------------------:|------------------:|
| classification | random_split       | logreg_robust |      0 |      3493 |      874 |     9.54475 |   0.346682 |            0.347464 |   0.347572 |      0.346889 |                nan |               nan |
| classification | random_split       | hgb_clf       |      0 |      3493 |      874 |    13.8625  |   0.399314 |            0.401206 |   0.396197 |      0.39434  |                nan |               nan |
| classification | random_split       | lgbm_clf      |      0 |      3493 |      874 |     6.19017 |   0.392449 |            0.395238 |   0.390213 |      0.387666 |                nan |               nan |
| classification | groupkfold_subject | logreg_robust |      1 |      2921 |     1446 |     8.67438 |   0.25657  |            0.258247 |   0.252399 |      0.252499 |                 33 |                16 |
| classification | groupkfold_subject | hgb_clf       |      1 |      2921 |     1446 |    12.189   |   0.282849 |            0.278643 |   0.275727 |      0.279927 |                 33 |                16 |
| classification | groupkfold_subject | lgbm_clf      |      1 |      2921 |     1446 |     9.85056 |   0.279391 |            0.27595  |   0.269986 |      0.273603 |                 33 |                16 |
| classification | groupkfold_subject | logreg_robust |      2 |      2923 |     1444 |    53.7948  |   0.254848 |            0.250565 |   0.252194 |      0.256827 |                 33 |                16 |
| classification | groupkfold_subject | hgb_clf       |      2 |      2923 |     1444 |    15.5029  |   0.24446  |            0.242659 |   0.237304 |      0.241687 |                 33 |                16 |
| classification | groupkfold_subject | lgbm_clf      |      2 |      2923 |     1444 |     9.84232 |   0.247922 |            0.243532 |   0.242046 |      0.248325 |                 33 |                16 |
| classification | groupkfold_subject | logreg_robust |      3 |      2890 |     1477 |    63.8429  |   0.234259 |            0.230589 |   0.229841 |      0.231538 |                 32 |                17 |
| classification | groupkfold_subject | hgb_clf       |      3 |      2890 |     1477 |    17.183   |   0.249154 |            0.256145 |   0.249638 |      0.243888 |                 32 |                17 |
| classification | groupkfold_subject | lgbm_clf      |      3 |      2890 |     1477 |    10.8365  |   0.265403 |            0.269707 |   0.262159 |      0.257962 |                 32 |                17 |

## Regression aggregated metrics

| task       | validation         | model        |   folds |   mae_mean |    mae_std |   mae_min |   mae_max |   rmse_mean |   rmse_std |   rmse_min |   rmse_max |       r2_mean |       r2_std |        r2_min |     r2_max |   pearson_mean |   pearson_std |   pearson_min |   pearson_max |   spearman_mean |   spearman_std |   spearman_min |   spearman_max |
|:-----------|:-------------------|:-------------|--------:|-----------:|-----------:|----------:|----------:|------------:|-----------:|-----------:|-----------:|--------------:|-------------:|--------------:|-----------:|---------------:|--------------:|--------------:|--------------:|----------------:|---------------:|---------------:|---------------:|
| regression | groupkfold_subject | hgb_reg      |       3 |  0.0914392 | 0.00778944 | 0.0825047 | 0.0968043 |   0.11702   |  0.0101567 |  0.10575   |  0.125466  |     0.0886187 |    0.0415979 |     0.0434352 |   0.125325 |      0.327649  |     0.0346381 |      0.289949 |      0.358068 |        0.3092   |      0.0531297 |       0.258673 |       0.364597 |
| regression | groupkfold_subject | lgbm_reg     |       3 |  0.0908334 | 0.00792926 | 0.0818141 | 0.096708  |   0.116641  |  0.0103948 |  0.105192  |  0.125487  |     0.0948844 |    0.0370619 |     0.0535061 |   0.125032 |      0.334383  |     0.0286362 |      0.302412 |      0.357679 |        0.318116 |      0.0479084 |       0.272816 |       0.368264 |
| regression | groupkfold_subject | ridge_robust |       3 |  0.29878   | 0.219829   | 0.127922  | 0.546783  |   4.00768   |  3.81087   |  0.46286   |  8.03805   | -1588.8       | 2166.91      | -4060.76      | -17.3253   |      0.0240582 |     0.0614841 |     -0.040339 |      0.082142 |        0.240914 |      0.0476831 |       0.211616 |       0.295935 |
| regression | random_split       | hgb_reg      |       1 |  0.0782472 | 0          | 0.0782472 | 0.0782472 |   0.0998745 |  0         |  0.0998745 |  0.0998745 |     0.320033  |    0         |     0.320033  |   0.320033 |      0.567998  |     0         |      0.567998 |      0.567998 |        0.542578 |      0         |       0.542578 |       0.542578 |
| regression | random_split       | lgbm_reg     |       1 |  0.0779671 | 0          | 0.0779671 | 0.0779671 |   0.0995138 |  0         |  0.0995138 |  0.0995138 |     0.324935  |    0         |     0.324935  |   0.324935 |      0.572314  |     0         |      0.572314 |      0.572314 |        0.545739 |      0         |       0.545739 |       0.545739 |
| regression | random_split       | ridge_robust |       1 |  0.116593  | 0          | 0.116593  | 0.116593  |   0.364342  |  0         |  0.364342  |  0.364342  |    -8.04891   |    0         |    -8.04891   |  -8.04891  |      0.109803  |     0         |      0.109803 |      0.109803 |        0.394463 |      0         |       0.394463 |       0.394463 |

## Regression fold metrics

| task       | validation         | model        |   fold |   n_train |   n_test |   elapsed_s |       mae |         mse |      rmse |            r2 |    pearson |   spearman |   n_train_subjects |   n_test_subjects |
|:-----------|:-------------------|:-------------|-------:|----------:|---------:|------------:|----------:|------------:|----------:|--------------:|-----------:|-----------:|-------------------:|------------------:|
| regression | random_split       | ridge_robust |      0 |      3493 |      874 |    0.27089  | 0.116593  |  0.132745   | 0.364342  |    -8.04891   |  0.109803  |   0.394463 |                nan |               nan |
| regression | random_split       | hgb_reg      |      0 |      3493 |      874 |    2.89169  | 0.0782472 |  0.00997491 | 0.0998745 |     0.320033  |  0.567998  |   0.542578 |                nan |               nan |
| regression | random_split       | lgbm_reg     |      0 |      3493 |      874 |    1.48845  | 0.0779671 |  0.009903   | 0.0995138 |     0.324935  |  0.572314  |   0.545739 |                nan |               nan |
| regression | groupkfold_subject | ridge_robust |      1 |      2921 |     1446 |    0.28055  | 0.546783  | 64.6103     | 8.03805   | -4060.76      |  0.082142  |   0.215192 |                 33 |                16 |
| regression | groupkfold_subject | hgb_reg      |      1 |      2921 |     1446 |    3.73737  | 0.0950087 |  0.0143625  | 0.119843  |     0.0970961 |  0.33493   |   0.30433  |                 33 |                16 |
| regression | groupkfold_subject | lgbm_reg     |      1 |      2921 |     1446 |    2.57843  | 0.0939779 |  0.014219   | 0.119243  |     0.106115  |  0.343057  |   0.313267 |                 33 |                16 |
| regression | groupkfold_subject | ridge_robust |      2 |      2923 |     1444 |    0.298263 | 0.221634  | 12.4055     | 3.52214   |  -688.3       | -0.040339  |   0.295935 |                 33 |                16 |
| regression | groupkfold_subject | hgb_reg      |      2 |      2923 |     1444 |    3.7507   | 0.0968043 |  0.0157417  | 0.125466  |     0.125325  |  0.358068  |   0.364597 |                 33 |                16 |
| regression | groupkfold_subject | lgbm_reg     |      2 |      2923 |     1444 |    2.64587  | 0.096708  |  0.015747   | 0.125487  |     0.125032  |  0.357679  |   0.368264 |                 33 |                16 |
| regression | groupkfold_subject | ridge_robust |      3 |      2890 |     1477 |    0.295259 | 0.127922  |  0.21424    | 0.46286   |   -17.3253    |  0.0303715 |   0.211616 |                 32 |                17 |
| regression | groupkfold_subject | hgb_reg      |      3 |      2890 |     1477 |    4.47553  | 0.0825047 |  0.0111831  | 0.10575   |     0.0434352 |  0.289949  |   0.258673 |                 32 |                17 |
| regression | groupkfold_subject | lgbm_reg     |      3 |      2890 |     1477 |    2.3826   | 0.0818141 |  0.0110654  | 0.105192  |     0.0535061 |  0.302412  |   0.272816 |                 32 |                17 |

## Figures

- `reports\figures\results\baseline_focus_smoke\classification_macro_f1_groupkfold_subject.png`
- `reports\figures\results\baseline_focus_smoke\classification_macro_f1_random_split.png`
- `reports\figures\results\baseline_focus_smoke\regression_rmse_groupkfold_subject.png`
- `reports\figures\results\baseline_focus_smoke\regression_rmse_random_split.png`
- `reports\figures\results\baseline_focus_smoke\best_confusion_matrix_groupkfold_subject_lgbm_clf.png`
- `reports\figures\results\baseline_focus_smoke\best_regression_scatter_groupkfold_subject_lgbm_reg.png`

## Interpretation

1. Random split is only a sanity check and likely overestimates performance.
2. GroupKFold by `subject_id` is the main baseline validation scheme.
3. Cross-source validation with subject overlap estimates source transfer but may be optimistic.
4. Cross-source validation without subject overlap is stricter and should be used for transfer conclusions.
5. Comparing `pow`, `eeg`, and `pow_plus_eeg` shows whether raw EEG time-domain features add useful signal beyond Emotiv bandpower features.