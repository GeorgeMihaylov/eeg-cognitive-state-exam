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

|   rows |   subjects |   records | sources                               | class_distribution                                      |   min_windows_per_subject | feature_set   | feature_mode   |   n_features |
|-------:|-----------:|----------:|:--------------------------------------|:--------------------------------------------------------|--------------------------:|:--------------|:---------------|-------------:|
|  45378 |         53 |       117 | {"gpn_data": 23823, "Old_EEG": 21555} | {"0": 9080, "1": 9069, "2": 9075, "3": 9078, "4": 9076} |                        30 | pow_plus_eeg  | raw_pow        |          448 |

## Regression data

|   rows |   subjects |   records | sources                               |   target_mean |   target_std |   target_min |   target_median |   target_max |   min_windows_per_subject | feature_set   | feature_mode   |   n_features |
|-------:|-----------:|----------:|:--------------------------------------|--------------:|-------------:|-------------:|----------------:|-------------:|--------------------------:|:--------------|:---------------|-------------:|
|  45378 |         53 |       117 | {"gpn_data": 23823, "Old_EEG": 21555} |      0.430059 |     0.125002 |     0.004077 |        0.414503 |     0.991193 |                        30 | pow_plus_eeg  | raw_pow        |          448 |

## Classification aggregated metrics

| task           | validation         | model         |   folds |   accuracy_mean |   accuracy_std |   accuracy_min |   accuracy_max |   balanced_accuracy_mean |   balanced_accuracy_std |   balanced_accuracy_min |   balanced_accuracy_max |   macro_f1_mean |   macro_f1_std |   macro_f1_min |   macro_f1_max |   weighted_f1_mean |   weighted_f1_std |   weighted_f1_min |   weighted_f1_max |
|:---------------|:-------------------|:--------------|--------:|----------------:|---------------:|---------------:|---------------:|-------------------------:|------------------------:|------------------------:|------------------------:|----------------:|---------------:|---------------:|---------------:|-------------------:|------------------:|------------------:|------------------:|
| classification | groupkfold_subject | hgb_clf       |       5 |        0.308357 |      0.0248051 |       0.274514 |       0.338595 |                 0.311605 |               0.0274275 |                0.270458 |                0.342122 |        0.303937 |      0.023451  |       0.268038 |       0.325939 |           0.303908 |         0.0228275 |          0.273279 |          0.328104 |
| classification | groupkfold_subject | lgbm_clf      |       5 |        0.310978 |      0.0311946 |       0.266938 |       0.346048 |                 0.31386  |               0.033108  |                0.263654 |                0.348853 |        0.306856 |      0.0288246 |       0.262156 |       0.333852 |           0.306758 |         0.0285232 |          0.266617 |          0.335691 |
| classification | groupkfold_subject | logreg_robust |       5 |        0.286996 |      0.0194487 |       0.272208 |       0.318607 |                 0.28637  |               0.0178933 |                0.264101 |                0.312494 |        0.274817 |      0.0113598 |       0.260544 |       0.292316 |           0.277543 |         0.0143282 |          0.264324 |          0.299978 |
| classification | groupkfold_subject | rf_clf        |       5 |        0.319647 |      0.031425  |       0.281651 |       0.358435 |                 0.323346 |               0.0342467 |                0.277178 |                0.363332 |        0.312618 |      0.0294679 |       0.271297 |       0.342988 |           0.312506 |         0.0292385 |          0.27675  |          0.344079 |
| classification | random_split       | hgb_clf       |       1 |        0.617673 |      0         |       0.617673 |       0.617673 |                 0.617661 |               0         |                0.617661 |                0.617661 |        0.614178 |      0         |       0.614178 |       0.614178 |           0.614186 |         0         |          0.614186 |          0.614186 |
| classification | random_split       | lgbm_clf      |       1 |        0.653922 |      0         |       0.653922 |       0.653922 |                 0.653911 |               0         |                0.653911 |                0.653911 |        0.651098 |      0         |       0.651098 |       0.651098 |           0.651105 |         0         |          0.651105 |          0.651105 |
| classification | random_split       | logreg_robust |       1 |        0.359409 |      0         |       0.359409 |       0.359409 |                 0.359404 |               0         |                0.359404 |                0.359404 |        0.34796  |      0         |       0.34796  |       0.34796  |           0.347964 |         0         |          0.347964 |          0.347964 |
| classification | random_split       | rf_clf        |       1 |        0.726752 |      0         |       0.726752 |       0.726752 |                 0.726742 |               0         |                0.726742 |                0.726742 |        0.725054 |      0         |       0.725054 |       0.725054 |           0.725061 |         0         |          0.725061 |          0.725061 |

## Classification fold metrics

| task           | validation         | model         |   fold |   n_train |   n_test |   elapsed_s |   accuracy |   balanced_accuracy |   macro_f1 |   weighted_f1 |   n_train_subjects |   n_test_subjects |
|:---------------|:-------------------|:--------------|-------:|----------:|---------:|------------:|-----------:|--------------------:|-----------:|--------------:|-------------------:|------------------:|
| classification | random_split       | logreg_robust |      0 |     36302 |     9076 |     72.6267 |   0.359409 |            0.359404 |   0.34796  |      0.347964 |                nan |               nan |
| classification | random_split       | hgb_clf       |      0 |     36302 |     9076 |     52.1447 |   0.617673 |            0.617661 |   0.614178 |      0.614186 |                nan |               nan |
| classification | random_split       | rf_clf        |      0 |     36302 |     9076 |     32.0128 |   0.726752 |            0.726742 |   0.725054 |      0.725061 |                nan |               nan |
| classification | random_split       | lgbm_clf      |      0 |     36302 |     9076 |     48.7395 |   0.653922 |            0.653911 |   0.651098 |      0.651105 |                nan |               nan |
| classification | groupkfold_subject | logreg_robust |      1 |     36255 |     9123 |     60.862  |   0.292338 |            0.288508 |   0.274969 |      0.283118 |                 42 |                11 |
| classification | groupkfold_subject | hgb_clf       |      1 |     36255 |     9123 |     50.1613 |   0.338595 |            0.342122 |   0.325939 |      0.328104 |                 42 |                11 |
| classification | groupkfold_subject | rf_clf        |      1 |     36255 |     9123 |     28.9924 |   0.358435 |            0.363332 |   0.342988 |      0.344079 |                 42 |                11 |
| classification | groupkfold_subject | lgbm_clf      |      1 |     36255 |     9123 |     40.7241 |   0.346048 |            0.348853 |   0.333852 |      0.335691 |                 42 |                11 |
| classification | groupkfold_subject | logreg_robust |      2 |     36392 |     8986 |     55.7851 |   0.318607 |            0.312494 |   0.292316 |      0.299978 |                 43 |                10 |
| classification | groupkfold_subject | hgb_clf       |      2 |     36392 |     8986 |     49.5543 |   0.315268 |            0.314096 |   0.312826 |      0.316479 |                 43 |                10 |
| classification | groupkfold_subject | rf_clf        |      2 |     36392 |     8986 |     28.768  |   0.326619 |            0.321986 |   0.319052 |      0.325713 |                 43 |                10 |
| classification | groupkfold_subject | lgbm_clf      |      2 |     36392 |     8986 |     39.4767 |   0.327732 |            0.32558  |   0.32287  |      0.327024 |                 43 |                10 |
| classification | groupkfold_subject | logreg_robust |      3 |     36271 |     9107 |     53.1496 |   0.272208 |            0.264101 |   0.260544 |      0.268996 |                 42 |                11 |
| classification | groupkfold_subject | hgb_clf       |      3 |     36271 |     9107 |     49.6557 |   0.274514 |            0.270458 |   0.268038 |      0.273279 |                 42 |                11 |
| classification | groupkfold_subject | rf_clf        |      3 |     36271 |     9107 |     28.4684 |   0.281651 |            0.277178 |   0.271297 |      0.27675  |                 42 |                11 |
| classification | groupkfold_subject | lgbm_clf      |      3 |     36271 |     9107 |     39.2422 |   0.266938 |            0.263654 |   0.262156 |      0.266617 |                 42 |                11 |
| classification | groupkfold_subject | logreg_robust |      4 |     36322 |     9056 |     52.713  |   0.279152 |            0.289689 |   0.272625 |      0.264324 |                 43 |                10 |
| classification | groupkfold_subject | hgb_clf       |      4 |     36322 |     9056 |     49.8448 |   0.293507 |            0.302532 |   0.29349  |      0.286985 |                 43 |                10 |
| classification | groupkfold_subject | rf_clf        |      4 |     36322 |     9056 |     28.4785 |   0.29428  |            0.305545 |   0.294974 |      0.28644  |                 43 |                10 |
| classification | groupkfold_subject | lgbm_clf      |      4 |     36322 |     9056 |     39.0016 |   0.292734 |            0.299859 |   0.294686 |      0.288922 |                 43 |                10 |
| classification | groupkfold_subject | logreg_robust |      5 |     36272 |     9106 |     54.4008 |   0.272677 |            0.277059 |   0.273629 |      0.271299 |                 42 |                11 |
| classification | groupkfold_subject | hgb_clf       |      5 |     36272 |     9106 |     50.6175 |   0.319899 |            0.328817 |   0.31939  |      0.314695 |                 42 |                11 |
| classification | groupkfold_subject | rf_clf        |      5 |     36272 |     9106 |     28.2631 |   0.33725  |            0.34869  |   0.334777 |      0.329546 |                 42 |                11 |
| classification | groupkfold_subject | lgbm_clf      |      5 |     36272 |     9106 |     39.639  |   0.321436 |            0.331354 |   0.320714 |      0.315533 |                 42 |                11 |

## Regression aggregated metrics

| task       | validation         | model        |   folds |   mae_mean |    mae_std |   mae_min |   mae_max |   rmse_mean |   rmse_std |   rmse_min |   rmse_max |     r2_mean |      r2_std |        r2_min |     r2_max |   pearson_mean |   pearson_std |   pearson_min |   pearson_max |   spearman_mean |   spearman_std |   spearman_min |   spearman_max |
|:-----------|:-------------------|:-------------|--------:|-----------:|-----------:|----------:|----------:|------------:|-----------:|-----------:|-----------:|------------:|------------:|--------------:|-----------:|---------------:|--------------:|--------------:|--------------:|----------------:|---------------:|---------------:|---------------:|
| regression | groupkfold_subject | hgb_reg      |       5 |  0.0901464 | 0.00793572 | 0.0791766 | 0.100196  |   0.115378  | 0.00976302 |  0.101192  |  0.128092  |   0.136091  |   0.0819547 |    0.0235616  |  0.224998  |       0.401492 |     0.0825472 |     0.301409  |      0.499655 |        0.37967  |      0.0906998 |       0.25865  |       0.500756 |
| regression | groupkfold_subject | lgbm_reg     |       5 |  0.0900184 | 0.00795256 | 0.0787327 | 0.10039   |   0.115493  | 0.0100427  |  0.100717  |  0.12878   |   0.133859  |   0.0927929 |    0.00325835 |  0.220644  |       0.399487 |     0.0888236 |     0.288357  |      0.490814 |        0.380569 |      0.0932424 |       0.251727 |       0.493698 |
| regression | groupkfold_subject | rf_reg       |       5 |  0.0886973 | 0.00715419 | 0.0803882 | 0.0991643 |   0.114698  | 0.00869377 |  0.102568  |  0.12673   |   0.146112  |   0.0675968 |    0.0574371  |  0.211647  |       0.410696 |     0.0785303 |     0.331129  |      0.506242 |        0.392287 |      0.0778229 |       0.316948 |       0.475    |
| regression | groupkfold_subject | ridge_robust |       5 |  0.125772  | 0.059613   | 0.0826073 | 0.230259  |   0.615396  | 0.687005   |  0.123024  |  1.76249   | -53.2733    | 100.187     | -230.949      | -0.218098  |       0.128949 |     0.0970773 |     0.0131903 |      0.266203 |        0.33075  |      0.0974667 |       0.180064 |       0.438027 |
| regression | random_split       | hgb_reg      |       1 |  0.0694058 | 0          | 0.0694058 | 0.0694058 |   0.0902864 | 0          |  0.0902864 |  0.0902864 |   0.489316  |   0         |    0.489316   |  0.489316  |       0.709508 |     0         |     0.709508  |      0.709508 |        0.67923  |      0         |       0.67923  |       0.67923  |
| regression | random_split       | lgbm_reg     |       1 |  0.0663957 | 0          | 0.0663957 | 0.0663957 |   0.0866896 | 0          |  0.0866896 |  0.0866896 |   0.529194  |   0         |    0.529194   |  0.529194  |       0.738337 |     0         |     0.738337  |      0.738337 |        0.709056 |      0         |       0.709056 |       0.709056 |
| regression | random_split       | rf_reg       |       1 |  0.0506071 | 0          | 0.0506071 | 0.0506071 |   0.0735531 | 0          |  0.0735531 |  0.0735531 |   0.66107   |   0         |    0.66107    |  0.66107   |       0.829897 |     0         |     0.829897  |      0.829897 |        0.818941 |      0         |       0.818941 |       0.818941 |
| regression | random_split       | ridge_robust |       1 |  0.0878127 | 0          | 0.0878127 | 0.0878127 |   0.131243  | 0          |  0.131243  |  0.131243  |  -0.0790994 |   0         |   -0.0790994  | -0.0790994 |       0.344281 |     0         |     0.344281  |      0.344281 |        0.47393  |      0         |       0.47393  |       0.47393  |

## Regression fold metrics

| task       | validation         | model        |   fold |   n_train |   n_test |   elapsed_s |       mae |        mse |      rmse |            r2 |   pearson |   spearman |   n_train_subjects |   n_test_subjects |
|:-----------|:-------------------|:-------------|-------:|----------:|---------:|------------:|----------:|-----------:|----------:|--------------:|----------:|-----------:|-------------------:|------------------:|
| regression | random_split       | ridge_robust |      0 |     36302 |     9076 |     2.34007 | 0.0878127 | 0.0172248  | 0.131243  |   -0.0790994  | 0.344281  |   0.47393  |                nan |               nan |
| regression | random_split       | hgb_reg      |      0 |     36302 |     9076 |    12.882   | 0.0694058 | 0.00815163 | 0.0902864 |    0.489316   | 0.709508  |   0.67923  |                nan |               nan |
| regression | random_split       | rf_reg       |      0 |     36302 |     9076 |   511.906   | 0.0506071 | 0.00541006 | 0.0735531 |    0.66107    | 0.829897  |   0.818941 |                nan |               nan |
| regression | random_split       | lgbm_reg     |      0 |     36302 |     9076 |    11.9617  | 0.0663957 | 0.00751509 | 0.0866896 |    0.529194   | 0.738337  |   0.709056 |                nan |               nan |
| regression | groupkfold_subject | ridge_robust |      1 |     36255 |     9123 |     2.21705 | 0.112898  | 0.562282   | 0.749855  |  -31.5619     | 0.0131903 |   0.299658 |                 42 |                11 |
| regression | groupkfold_subject | hgb_reg      |      1 |     36255 |     9123 |    12.4072  | 0.0948824 | 0.0142306  | 0.119292  |    0.175903   | 0.4515    |   0.40923  |                 42 |                11 |
| regression | groupkfold_subject | rf_reg       |      1 |     36255 |     9123 |   465.665   | 0.0910306 | 0.0136662  | 0.116903  |    0.208586   | 0.506242  |   0.475    |                 42 |                11 |
| regression | groupkfold_subject | lgbm_reg     |      1 |     36255 |     9123 |    10.132   | 0.0936603 | 0.01396    | 0.118152  |    0.191577   | 0.463685  |   0.424515 |                 42 |                11 |
| regression | groupkfold_subject | ridge_robust |      2 |     36392 |     8986 |     2.15561 | 0.0946619 | 0.0448916  | 0.211876  |   -1.67891    | 0.12951   |   0.438027 |                 43 |                10 |
| regression | groupkfold_subject | hgb_reg      |      2 |     36392 |     8986 |    12.2909  | 0.087584  | 0.0129871  | 0.113961  |    0.224998   | 0.499655  |   0.500756 |                 43 |                10 |
| regression | groupkfold_subject | rf_reg       |      2 |     36392 |     8986 |   485.844   | 0.0887538 | 0.0132108  | 0.114938  |    0.211647   | 0.477833  |   0.473643 |                 43 |                10 |
| regression | groupkfold_subject | lgbm_reg     |      2 |     36392 |     8986 |     9.95521 | 0.0879254 | 0.01306    | 0.11428   |    0.220644   | 0.490814  |   0.493698 |                 43 |                10 |
| regression | groupkfold_subject | ridge_robust |      3 |     36271 |     9107 |     2.12675 | 0.108436  | 0.0527772  | 0.229733  |   -1.95889    | 0.168978  |   0.363392 |                 42 |                11 |
| regression | groupkfold_subject | hgb_reg      |      3 |     36271 |     9107 |    12.2168  | 0.100196  | 0.0164076  | 0.128092  |    0.0801275  | 0.333026  |   0.331052 |                 42 |                11 |
| regression | groupkfold_subject | rf_reg       |      3 |     36271 |     9107 |   445.911   | 0.0991643 | 0.0160605  | 0.12673   |    0.0995865  | 0.331129  |   0.323912 |                 42 |                11 |
| regression | groupkfold_subject | lgbm_reg     |      3 |     36271 |     9107 |    10.1081  | 0.10039   | 0.0165843  | 0.12878   |    0.0702238  | 0.323619  |   0.327411 |                 42 |                11 |
| regression | groupkfold_subject | ridge_robust |      4 |     36322 |     9056 |     2.12957 | 0.0826073 | 0.0151349  | 0.123024  |   -0.218098   | 0.266203  |   0.37261  |                 43 |                10 |
| regression | groupkfold_subject | hgb_reg      |      4 |     36322 |     9056 |    12.4667  | 0.0791766 | 0.0102399  | 0.101192  |    0.175866   | 0.421871  |   0.398663 |                 43 |                10 |
| regression | groupkfold_subject | rf_reg       |      4 |     36322 |     9056 |   469.273   | 0.0803882 | 0.0105202  | 0.102568  |    0.153304   | 0.394104  |   0.371934 |                 43 |                10 |
| regression | groupkfold_subject | lgbm_reg     |      4 |     36322 |     9056 |    14.5978  | 0.0787327 | 0.0101439  | 0.100717  |    0.18359    | 0.430962  |   0.405492 |                 43 |                10 |
| regression | groupkfold_subject | ridge_robust |      5 |     36272 |     9106 |     2.2153  | 0.230259  | 3.10638    | 1.76249   | -230.949      | 0.0668637 |   0.180064 |                 42 |                11 |
| regression | groupkfold_subject | hgb_reg      |      5 |     36272 |     9106 |    12.3616  | 0.0888926 | 0.013077   | 0.114355  |    0.0235616  | 0.301409  |   0.25865  |                 42 |                11 |
| regression | groupkfold_subject | rf_reg       |      5 |     36272 |     9106 |   466.021   | 0.0841495 | 0.0126233  | 0.112353  |    0.0574371  | 0.344172  |   0.316948 |                 42 |                11 |
| regression | groupkfold_subject | lgbm_reg     |      5 |     36272 |     9106 |    10.1583  | 0.0893841 | 0.0133489  | 0.115537  |    0.00325835 | 0.288357  |   0.251727 |                 42 |                11 |

## Figures

- `reports\figures\results\baseline_focus_full\classification_macro_f1_groupkfold_subject.png`
- `reports\figures\results\baseline_focus_full\classification_macro_f1_random_split.png`
- `reports\figures\results\baseline_focus_full\regression_rmse_groupkfold_subject.png`
- `reports\figures\results\baseline_focus_full\regression_rmse_random_split.png`
- `reports\figures\results\baseline_focus_full\best_confusion_matrix_groupkfold_subject_rf_clf.png`
- `reports\figures\results\baseline_focus_full\best_regression_scatter_groupkfold_subject_rf_reg.png`

## Interpretation

1. Random split is only a sanity check and likely overestimates performance.
2. GroupKFold by `subject_id` is the main baseline validation scheme.
3. Cross-source validation with subject overlap estimates source transfer but may be optimistic.
4. Cross-source validation without subject overlap is stricter and should be used for transfer conclusions.
5. Comparing `pow`, `eeg`, and `pow_plus_eeg` shows whether raw EEG time-domain features add useful signal beyond Emotiv bandpower features.