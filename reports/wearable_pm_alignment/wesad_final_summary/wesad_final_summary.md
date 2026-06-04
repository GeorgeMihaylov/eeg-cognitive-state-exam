# WESAD wearable stress benchmark: final summary

Generated: `2026-05-15 14:25:04`

## 1. Purpose

This report summarizes the external wearable benchmark based on WESAD. The goal was to test whether wrist physiological signals can recover a stress-like state under subject-aware validation and to identify whether the result is physiological or partly driven by movement/protocol confounding.

The benchmark is not a direct Emotiv PM prediction experiment. It is an external validation line for the broader hypothesis:

```text
wearable physiology can provide useful proxy signals for stress/arousal-related cognitive-state estimation
```

## 2. Source files

| name                         | path                                                                                                                                                     |
|:-----------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------|
| dataset_report               | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\wesad_windowed_stress_dataset_report.md                                         |
| baseline_run                 | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full                                          |
| baseline_summary             | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full\metrics_summary.csv                      |
| threshold_summary            | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full\analysis\best_thresholds.csv             |
| baseline_per_subject_summary | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_142937_wesad_stress_full\analysis\per_subject_metrics_summary.csv |
| ablation_run                 | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_144608_wesad_feature_group_ablation                               |
| ablation_summary             | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_144608_wesad_feature_group_ablation\ablation_summary.csv          |
| ablation_best_by_group       | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_144608_wesad_feature_group_ablation\best_by_feature_group.csv     |
| protocol_run                 | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_160829_wesad_protocol_control                                     |
| protocol_summary             | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_160829_wesad_protocol_control\protocol_control_summary.csv        |
| protocol_threshold           | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_160829_wesad_protocol_control\best_thresholds.csv                 |
| protocol_comparison          | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\runs\20260514_160829_wesad_protocol_control\protocol_control_comparison.csv     |
| output_dir                   | D:\PycharmProjects\eeg-cognitive-state-nir\reports\wearable_pm_alignment\wesad_final_summary                                                             |

## 3. Dataset preparation status

Excerpt from the windowed-dataset report:

```markdown
# WESAD windowed stress dataset report

## Configuration

```json
{
  "root": ".",
  "wesad_root": "data/external/WESAD",
  "output_name": "wesad_windowed_stress_dataset",
  "output_dir": "data/processed",
  "report_dir": "reports/wearable_pm_alignment",
  "window_size_sec": 60.0,
  "step_size_sec": 10.0,
  "min_valid_label_fraction": 0.8,
  "min_majority_fraction": 0.8,
  "max_subjects": 0,
  "save_csv": false,
  "no_parquet": false
}
```

## Output

- Dataset: `D:\PycharmProjects\eeg-cognitive-state-nir\data\processed\wesad_windowed_stress_dataset.parquet`

## Dataset summary

- Rows: **4214**
- Columns: **138**
- Subjects: **15**
- Feature columns: **116**

## Binary target distribution

|   stress_binary | class_name   |   n_windows |
|----------------:|:-------------|------------:|
|               0 | non_stress   |        3275 |
|               1 | stress       |         939 |

## Original WESAD majority label distribution

| wesad_label_name   |   n_windows |
|:-------------------|------------:|
| baseline           |        1704 |
| meditation         |        1068 |
| stress             |         939 |
| amusement          |         503 |

## Subject summary

| subject_id   | pkl_path                                                                         |   duration_sec |   total_candidate_windows |   kept_windows |   skipped_short_signal |   skipped_low_valid_label_fraction |   skipped_ambiguous_label | binary_label_counts   | multiclass_label_counts               | original_label_counts                                                                                                                                                                                              |
|:-------------|:---------------------------------------------------------------------------------|---------------:|--------------------------:|---------------:|-----------------------:|-----------------------------------:|--------------------------:|:----------------------|:--------------------------------------|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| S2           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S2\S2.pkl   |           6079 |                       602 |            271 |                      0 |                                331 |                         0 | {"0": 213, "1": 58}   | {"0": 111, "1": 58, "2": 32, "3": 70} | {"0:undefined_or_transient": 2142701, "1:baseline": 800800, "2:stress": 430500, "3:amusement": 253400, "4:meditation": 537599, "6:unused_or_other": 45500, "7:unused_or_other": 44800}                             |
| S3           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S3\S3.pkl   |           6493 |                       644 |            275 |                      0 |                                369 |                         0 | {"0": 215, "1": 60}   | {"0": 110, "1": 60, "2": 34, "3": 71} | {"0:undefined_or_transient": 2345699, "1:baseline": 798000, "2:stress": 448000, "3:amusement": 262500, "4:meditation": 546001, "5:unused_or_other": 51100, "6:unused_or_other": 46900, "7:unused_or_other": 46900} |
| S4           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S4\S4.pkl   |           6423 |                       637 |            278 |                      0 |                                359 |                         0 | {"0": 218, "1": 60}   | {"0": 112, "1": 60, "2": 33, "3": 73} | {"0:undefined_or_transient": 2314199, "1:baseline": 810601, "2:stress": 444500, "3:amusement": 260400, "4:meditation": 563500, "5:unused_or_other": 35699, "6:unused_or_other": 30800, "7:unused_or_other": 36401} |
| S5           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S5\S5.pkl   |           6258 |                       620 |            281 |                      0 |                                339 |                         0 | {"0": 221, "1": 60}   | {"0": 116, "1": 60, "2": 33, "3": 72} | {"0:undefined_or_transient": 2142700, "1:baseline": 838600, "2:stress": 451500, "3:amusement": 261800, "4:meditation": 555800, "5:unused_or_other": 50401, "6:unused_or_other": 30799, "7:unused_or_other": 49000} |
| S6           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S6\S6.pkl   |           7071 |                       702 |            281 |                      0 |                                421 |                         0 | {"0": 219, "1": 62}   | {"0": 114, "1": 62, "2": 34, "3": 71} | {"0:undefined_or_transient": 2733499, "1:baseline": 826000, "2:stress": 455000, "3:amusement": 260400, "4:meditation": 550900, "5:unused_or_other": 40600, "6:unused_or_other": 35001, "7:unused_or_other": 48300} |
| S7           | D:\PycharmProjects\eeg-cognitive-state-nir\data\external\WESAD\WESAD\S7\S7.pkl   |           
```

## 4. Key metrics

| metric                                             | value                      | comment                                                           |
|:---------------------------------------------------|:---------------------------|:------------------------------------------------------------------|
| best_baseline_model                                | logistic_robust            | Best default-threshold model in WESAD stress baseline.            |
| best_baseline_balanced_accuracy                    | 0.802947                   |                                                                   |
| best_baseline_macro_f1                             | 0.793048                   |                                                                   |
| best_baseline_roc_auc                              | 0.87844                    |                                                                   |
| best_baseline_average_precision                    | 0.804481                   |                                                                   |
| best_threshold_optimized_model                     | rf_clf                     | Best OOF threshold-optimized model from stress baseline analysis. |
| best_threshold_optimized_threshold                 | 0.25                       |                                                                   |
| best_threshold_optimized_balanced_accuracy         | 0.809151                   |                                                                   |
| best_feature_group_default                         | bvp_only / logistic_robust | Best default-threshold feature group ablation result.             |
| best_feature_group_default_balanced_accuracy       | 0.83491                    |                                                                   |
| best_protocol_default                              | bvp_temp / logistic_robust | Best default-threshold protocol-control result.                   |
| best_protocol_default_balanced_accuracy            | 0.842779                   |                                                                   |
| delta_all_minus_no_acc_balanced_accuracy           | -0.010112                  | Protocol-control delta for model=logistic_robust.                 |
| delta_no_acc_minus_acc_only_balanced_accuracy      | -0.017269                  | Protocol-control delta for model=logistic_robust.                 |
| delta_bvp_only_minus_acc_only_balanced_accuracy    | 0.004581                   | Protocol-control delta for model=logistic_robust.                 |
| delta_all_minus_no_acc_best_balanced_accuracy      | -0.011003                  | Protocol-control delta for model=logistic_robust.                 |
| delta_no_acc_minus_acc_only_best_balanced_accuracy | -0.030412                  | Protocol-control delta for model=logistic_robust.                 |

## 5. Baseline stress classification

The baseline stage trained standard classifiers on the 60s/10s WESAD windowed dataset using subject-aware validation. Primary metrics are balanced accuracy, macro-F1, ROC-AUC, average precision and stress-class F1.

| model           |   folds |   n_val_total |   balanced_accuracy_mean |   macro_f1_mean |   roc_auc_mean |   average_precision_mean |   f1_stress_mean |   recall_stress_mean |   precision_stress_mean |
|:----------------|--------:|--------------:|-------------------------:|----------------:|---------------:|-------------------------:|-----------------:|---------------------:|------------------------:|
| logistic_robust |       5 |          4214 |                 0.802947 |        0.793048 |       0.87844  |                 0.804481 |         0.693838 |             0.729322 |                0.703919 |
| rf_clf          |       5 |          4214 |                 0.756253 |        0.776207 |       0.893803 |                 0.810045 |         0.645407 |             0.579182 |                0.79278  |
| hgb_clf         |       5 |          4214 |                 0.754498 |        0.770327 |       0.898304 |                 0.821694 |         0.648481 |             0.597384 |                0.824347 |
| lgbm_clf        |       5 |          4214 |                 0.750553 |        0.770613 |       0.904093 |                 0.826393 |         0.642562 |             0.57787  |                0.81312  |

## 6. Threshold analysis

Threshold analysis was used because tree-based models often had strong ranking quality but weaker default-threshold metrics. The threshold was selected on out-of-fold predictions for diagnostic comparison.

| model           |   best_threshold_by_balanced_accuracy |   best_balanced_accuracy |   at_balanced_accuracy_precision_stress |   at_balanced_accuracy_recall_stress |   best_threshold_by_f1_stress |   best_f1_stress |
|:----------------|--------------------------------------:|-------------------------:|----------------------------------------:|-------------------------------------:|------------------------------:|-----------------:|
| rf_clf          |                                  0.25 |                 0.809151 |                                0.570988 |                             0.788072 |                          0.45 |         0.664067 |
| logistic_robust |                                  0.45 |                 0.803744 |                                0.620072 |                             0.736954 |                          0.76 |         0.689021 |
| lgbm_clf        |                                  0.05 |                 0.798189 |                                0.614209 |                             0.72737  |                          0.05 |         0.666017 |
| hgb_clf         |                                  0.05 |                 0.790763 |                                0.573554 |                             0.739084 |                          0.12 |         0.646078 |

## 7. Feature-group ablation

Feature-group ablation tested whether stress prediction depends on EDA, BVP, TEMP, ACC, or their combinations. This stage is important because high ACC performance may indicate movement or protocol confounding.

| feature_group    | model           |   n_features |   balanced_accuracy_mean |   macro_f1_mean |   roc_auc_mean |   average_precision_mean |   f1_stress_mean |   recall_stress_mean |   precision_stress_mean |
|:-----------------|:----------------|-------------:|-------------------------:|----------------:|---------------:|-------------------------:|-----------------:|---------------------:|------------------------:|
| bvp_only         | logistic_robust |           18 |                 0.83491  |        0.796024 |       0.899581 |                 0.738881 |         0.699384 |             0.821218 |                0.61845  |
| acc_only         | logistic_robust |           64 |                 0.830328 |        0.807251 |       0.915075 |                 0.844298 |         0.709202 |             0.779785 |                0.682541 |
| eda_bvp_temp     | logistic_robust |           52 |                 0.813059 |        0.788314 |       0.883281 |                 0.767954 |         0.681507 |             0.758571 |                0.631492 |
| eda_bvp_temp     | lgbm_clf        |           52 |                 0.807053 |        0.818179 |       0.919781 |                 0.829836 |         0.713875 |             0.677332 |                0.796615 |
| all              | logistic_robust |          116 |                 0.802947 |        0.793048 |       0.87844  |                 0.804481 |         0.693838 |             0.729322 |                0.703919 |
| eda_bvp_temp_acc | logistic_robust |          116 |                 0.802947 |        0.793048 |       0.87844  |                 0.804481 |         0.693838 |             0.729322 |                0.703919 |
| eda_bvp          | logistic_robust |           36 |                 0.796236 |        0.771121 |       0.869004 |                 0.749589 |         0.66426  |             0.746413 |                0.617985 |
| eda_bvp_temp     | hgb_clf         |           52 |                 0.778248 |        0.802056 |       0.924565 |                 0.831091 |         0.679661 |             0.59803  |                0.831387 |
| acc_only         | lgbm_clf        |           64 |                 0.775266 |        0.776695 |       0.901674 |                 0.809593 |         0.657258 |             0.646586 |                0.778585 |
| bvp_only         | lgbm_clf        |           18 |                 0.773505 |        0.775453 |       0.896575 |                 0.695502 |         0.647966 |             0.63825  |                0.675243 |
| acc_only         | hgb_clf         |           64 |                 0.768949 |        0.77466  |       0.896684 |                 0.808985 |         0.648156 |             0.620783 |                0.785751 |
| bvp_only         | rf_clf          |           18 |                 0.762495 |        0.759332 |       0.884983 |                 0.675507 |         0.621998 |             0.624811 |                0.657705 |
| eda_only         | logistic_robust |           18 |                 0.758262 |        0.73616  |       0.830878 |                 0.722464 |         0.611114 |             0.686982 |                0.564671 |
| all              | rf_clf          |          116 |                 0.756253 |        0.776207 |       0.893803 |                 0.810045 |         0.645407 |             0.579182 |                0.79278  |
| eda_bvp_temp_acc | rf_clf          |          116 |                 0.756253 |        0.776207 |       0.893803 |                 0.810045 |         0.645407 |             0.579182 |                0.79278  |
| all              | hgb_clf         |          116 |                 0.754498 |        0.770327 |       0.898304 |                 0.821694 |         0.648481 |             0.597384 |                0.824347 |
| eda_bvp_temp_acc | hgb_clf         |          116 |                 0.754498 |        0.770327 |       0.898304 |                 0.821694 |         0.648481 |             0.597384 |                0.824347 |
| bvp_only         | hgb_clf         |           18 |                 0.752226 |        0.765119 |       0.898525 |                 0.704668 |         0.623733 |             0.574293 |                0.707424 |
| all              | lgbm_clf        |          116 |                 0.750553 |        0.770613 |       0.904093 |                 0.826393 |         0.642562 |             0.57787  |                0.81312  |
| eda_bvp_temp_acc | lgbm_clf        |          116 |                 0.750553 |        0.770613 |       0.904093 |                 0.826393 |         0.642562 |             0.57787  |                0.81312  |
| acc_only         | rf_clf          |           64 |                 0.745917 |        0.747329 |       0.907372 |                 0.813923 |         0.60082  |             0.578994 |                0.726935 |
| eda_bvp          | lgbm_clf        |           36 |                 0.739708 |        0.736488 |       0.893851 |                 0.735519 |         0.585626 |             0.585298 |                0.673395 |
| eda_bvp          | hgb_clf         |           36 |                 0.735858 |        0.741934 |       0.895864 |                 0.735441 |         0.585434 |             0.55104  |                0.711599 |
| eda_bvp_temp     | rf_clf          |           52 |                 0.732802 |        0.714543 |       0.888491 |                 0.74903  |         0.560794 |             0.608435 |                0.610207 |
| eda_only         | rf_clf          |           18 |                 0.728343 |        0.722737 |       0.849555 |                 0.618856 |         0.571562 |             0.586387 |                0.619981 |

## 8. Protocol-control experiment

The protocol-control stage compared `all`, `no_acc`, `acc_only`, `bvp_only`, `eda_only`, `temp_only`, `eda_bvp`, `eda_bvp_temp`, and `bvp_temp`. The key question was whether `no_acc` remains strong and whether `acc_only` is suspiciously competitive.

| feature_group   | model           |   n_features |   balanced_accuracy_mean |   macro_f1_mean |   roc_auc_mean |   average_precision_mean |   f1_stress_mean |   recall_stress_mean |   precision_stress_mean |
|:----------------|:----------------|-------------:|-------------------------:|----------------:|---------------:|-------------------------:|-----------------:|---------------------:|------------------------:|
| bvp_temp        | logistic_robust |           34 |                 0.842779 |        0.798    |       0.908893 |                 0.767651 |         0.703222 |             0.842802 |                0.606138 |
| bvp_only        | logistic_robust |           18 |                 0.83491  |        0.796024 |       0.899581 |                 0.738881 |         0.699384 |             0.821218 |                0.61845  |
| acc_only        | logistic_robust |           64 |                 0.830328 |        0.807251 |       0.915075 |                 0.844298 |         0.709202 |             0.779785 |                0.682541 |
| eda_bvp_temp    | logistic_robust |           52 |                 0.813059 |        0.788314 |       0.883281 |                 0.767954 |         0.681507 |             0.758571 |                0.631492 |
| no_acc          | logistic_robust |           52 |                 0.813059 |        0.788314 |       0.883281 |                 0.767954 |         0.681507 |             0.758571 |                0.631492 |
| eda_bvp_temp    | lgbm_clf        |           52 |                 0.807053 |        0.818179 |       0.919781 |                 0.829836 |         0.713875 |             0.677332 |                0.796615 |
| no_acc          | lgbm_clf        |           52 |                 0.807053 |        0.818179 |       0.919781 |                 0.829836 |         0.713875 |             0.677332 |                0.796615 |
| all             | logistic_robust |          116 |                 0.802947 |        0.793048 |       0.87844  |                 0.804481 |         0.693838 |             0.729322 |                0.703919 |
| eda_bvp         | logistic_robust |           36 |                 0.796236 |        0.771121 |       0.869004 |                 0.749589 |         0.66426  |             0.746413 |                0.617985 |
| bvp_temp        | lgbm_clf        |           34 |                 0.788477 |        0.789324 |       0.9076   |                 0.765931 |         0.674767 |             0.673233 |                0.705931 |
| acc_only        | lgbm_clf        |           64 |                 0.775266 |        0.776695 |       0.901674 |                 0.809593 |         0.657258 |             0.646586 |                0.778585 |
| bvp_only        | lgbm_clf        |           18 |                 0.773505 |        0.775453 |       0.896575 |                 0.695502 |         0.647966 |             0.63825  |                0.675243 |
| eda_only        | logistic_robust |           18 |                 0.758262 |        0.73616  |       0.830878 |                 0.722464 |         0.611114 |             0.686982 |                0.564671 |
| all             | lgbm_clf        |          116 |                 0.750553 |        0.770613 |       0.904093 |                 0.826393 |         0.642562 |             0.57787  |                0.81312  |
| eda_bvp         | lgbm_clf        |           36 |                 0.739708 |        0.736488 |       0.893851 |                 0.735519 |         0.585626 |             0.585298 |                0.673395 |
| eda_only        | lgbm_clf        |           18 |                 0.718959 |        0.720392 |       0.856941 |                 0.678225 |         0.563361 |             0.554515 |                0.627718 |
| temp_only       | lgbm_clf        |           16 |                 0.709355 |        0.695164 |       0.764591 |                 0.52434  |         0.534521 |             0.58167  |                0.498855 |
| temp_only       | logistic_robust |           16 |                 0.686791 |        0.617123 |       0.749161 |                 0.456961 |         0.491944 |             0.737636 |                0.369897 |

### Protocol-control deltas

| model           |   all_balanced_accuracy |   all_roc_auc |   all_f1_stress |   all_best_threshold |   all_best_balanced_accuracy |   no_acc_balanced_accuracy |   no_acc_roc_auc |   no_acc_f1_stress |   no_acc_best_threshold |   no_acc_best_balanced_accuracy |   acc_only_balanced_accuracy |   acc_only_roc_auc |   acc_only_f1_stress |   acc_only_best_threshold |   acc_only_best_balanced_accuracy |   bvp_only_balanced_accuracy |   bvp_only_roc_auc |   bvp_only_f1_stress |   bvp_only_best_threshold |   bvp_only_best_balanced_accuracy |   eda_only_balanced_accuracy |   eda_only_roc_auc |   eda_only_f1_stress |   eda_only_best_threshold |   eda_only_best_balanced_accuracy |   eda_bvp_temp_balanced_accuracy |   eda_bvp_temp_roc_auc |   eda_bvp_temp_f1_stress |   eda_bvp_temp_best_threshold |   eda_bvp_temp_best_balanced_accuracy |   delta_all_minus_no_acc_balanced_accuracy |   delta_no_acc_minus_acc_only_balanced_accuracy |   delta_bvp_only_minus_acc_only_balanced_accuracy |   delta_all_minus_no_acc_best_balanced_accuracy |   delta_no_acc_minus_acc_only_best_balanced_accuracy |
|:----------------|------------------------:|--------------:|----------------:|---------------------:|-----------------------------:|---------------------------:|-----------------:|-------------------:|------------------------:|--------------------------------:|-----------------------------:|-------------------:|---------------------:|--------------------------:|----------------------------------:|-----------------------------:|-------------------:|---------------------:|--------------------------:|----------------------------------:|-----------------------------:|-------------------:|---------------------:|--------------------------:|----------------------------------:|---------------------------------:|-----------------------:|-------------------------:|------------------------------:|--------------------------------------:|-------------------------------------------:|------------------------------------------------:|--------------------------------------------------:|------------------------------------------------:|-----------------------------------------------------:|
| lgbm_clf        |                0.750553 |      0.904093 |        0.642562 |                 0.05 |                     0.798189 |                   0.807053 |         0.919781 |           0.713875 |                    0.07 |                        0.844548 |                     0.775266 |           0.901674 |             0.657258 |                      0.15 |                          0.79706  |                     0.773505 |           0.896575 |             0.647966 |                      0.12 |                          0.815954 |                     0.718959 |           0.856941 |             0.563361 |                      0.06 |                          0.766958 |                         0.807053 |               0.919781 |                 0.713875 |                          0.07 |                              0.844548 |                                 -0.0564999 |                                       0.0317869 |                                       -0.0017607  |                                      -0.0463595 |                                            0.0474877 |
| logistic_robust |                0.802947 |      0.87844  |        0.693838 |                 0.45 |                     0.803744 |                   0.813059 |         0.883281 |           0.681507 |                    0.47 |                        0.814748 |                     0.830328 |           0.915075 |             0.709202 |                      0.65 |                          0.845159 |                     0.83491  |           0.899581 |             0.699384 |                      0.49 |                          0.836177 |                     0.758262 |           0.830878 |             0.611114 |                      0.55 |                          0.76656  |                         0.813059 |               0.883281 |                 0.681507 |                          0.47 |                              0.814748 |                                 -0.0101122 |                                      -0.0172691 |                                        0.00458119 |                                      -0.0110033 |                                           -0.0304116 |

## 9. Interpretation

Main conclusions:

1. **Wearable stress detection is feasible on WESAD under subject-aware validation.**
2. **BVP/PPG and BVP+TEMP are the strongest compact physiological proxies in the current feature set.**
3. **The `no_acc` group remains strong**, so the result is not entirely explained by accelerometer signals.
4. **`ACC-only` is also strong**, which means WESAD contains movement/protocol information. ACC should be treated as a control channel, not as a pure physiological stress marker.
5. **The full feature set is not necessarily best**, likely because additional channels add subject-specific noise and protocol dependencies.

Recommended wording for the project:

```text
On WESAD, wrist physiological signals predict stress-like state with good subject-aware performance. The strongest compact physiological proxy is based on BVP/PPG and temperature, while EDA alone is weaker with the current simple statistical features. However, high ACC-only performance indicates protocol/movement confounding. Therefore, ACC should be used as a control/artifact channel, and PM.Stress wearable alignment should primarily rely on BVP/PPG, EDA and TEMP.
```

## 10. Figures

- `figures\wesad_baseline_balanced_accuracy.png`
- `figures\wesad_feature_group_balanced_accuracy.png`
- `figures\wesad_protocol_control_balanced_accuracy.png`
- `figures\wesad_protocol_control_best_threshold.png`
- `figures\wesad_key_takeaways.png`
- `figures\baseline_confusion_matrix_hgb_clf.png`
- `figures\baseline_confusion_matrix_lgbm_clf.png`
- `figures\baseline_confusion_matrix_logistic_robust.png`
- `figures\baseline_confusion_matrix_rf_clf.png`
- `figures\baseline_metric_by_model.png`
- `figures\baseline_pr_curve_hgb_clf.png`
- `figures\baseline_pr_curve_lgbm_clf.png`
- `figures\baseline_pr_curve_logistic_robust.png`
- `figures\baseline_pr_curve_rf_clf.png`
- `figures\baseline_roc_curve_hgb_clf.png`
- `figures\baseline_roc_curve_lgbm_clf.png`
- `figures\baseline_roc_curve_logistic_robust.png`
- `figures\baseline_roc_curve_rf_clf.png`
- `figures\baseline_analysis_per_subject_balanced_accuracy.png`
- `figures\baseline_analysis_per_subject_f1_stress.png`
- `figures\baseline_analysis_per_subject_roc_auc.png`
- `figures\baseline_analysis_score_distribution_hgb_clf.png`
- `figures\baseline_analysis_score_distribution_lgbm_clf.png`
- `figures\baseline_analysis_score_distribution_logistic_robust.png`
- `figures\baseline_analysis_score_distribution_rf_clf.png`
- `figures\baseline_analysis_subject_error_rate.png`
- `figures\baseline_analysis_threshold_balanced_accuracy.png`
- `figures\baseline_analysis_threshold_f1_stress.png`
- `figures\baseline_analysis_threshold_macro_f1.png`
- `figures\baseline_analysis_threshold_precision_stress.png`
- `figures\baseline_analysis_threshold_recall_stress.png`
- `figures\baseline_analysis_top_features_lgbm_clf.png`
- `figures\baseline_analysis_top_features_logistic_robust.png`
- `figures\baseline_analysis_top_features_rf_clf.png`
- `figures\ablation_average_precision_mean_by_group.png`
- `figures\ablation_balanced_accuracy_mean_by_group.png`
- `figures\ablation_f1_stress_mean_by_group.png`
- `figures\ablation_heatmap_group_model_balanced_accuracy_mean.png`
- `figures\ablation_heatmap_group_model_f1_stress_mean.png`
- `figures\ablation_heatmap_group_model_roc_auc_mean.png`
- `figures\ablation_macro_f1_mean_by_group.png`
- `figures\ablation_recall_precision_stress_by_group.png`
- `figures\ablation_roc_auc_mean_by_group.png`
- `figures\protocol_average_precision_mean_by_feature_group_model.png`
- `figures\protocol_balanced_accuracy_mean_by_feature_group_model.png`
- `figures\protocol_best_threshold_balanced_accuracy.png`
- `figures\protocol_best_threshold_f1_stress.png`
- `figures\protocol_best_threshold_macro_f1.png`
- `figures\protocol_f1_stress_mean_by_feature_group_model.png`
- `figures\protocol_macro_f1_mean_by_feature_group_model.png`
- `figures\protocol_protocol_control_all_no_acc_acc_only.png`
- `figures\protocol_roc_auc_mean_by_feature_group_model.png`

## 11. Next steps

Recommended next work items:

1. Move back to the main EEG/PM task and run subject-calibration experiments.
2. Add WESAD result summary to README/roadmap.
3. Treat COLET as blocked until MATLAB conversion is available.
4. Optionally improve WESAD EDA features with SCL/SCR decomposition and lag-aware features.
5. Optionally run WESAD window-size ablation: 30s, 60s, 120s.
