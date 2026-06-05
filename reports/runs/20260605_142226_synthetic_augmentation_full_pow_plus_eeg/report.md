# Synthetic Augmentation Report

## 1. Цель

Цель эксперимента — проверить, улучшают ли synthetic EEG/POW feature vectors обучение downstream-модели. Это закрывает требование лабораторной работы о проверке практической полезности синтетических данных как augmentation.

## 2. Метод

Сравниваются два режима:

- `real_only`: обучение только на реальных EEG/POW-признаках;
- `real_plus_synthetic`: обучение на реальных признаках и synthetic vectors из Autoencoder.

Test set остаётся только реальным. Synthetic features получают pseudo-labels через k-nearest neighbors по реальным train-векторам.

## 3. Конфигурация

```json
{
  "root": ".",
  "dataset": "data/processed/windowed_eeg_pm_dataset_w10.parquet",
  "run_name": "synthetic_augmentation_full",
  "feature_set": "pow_plus_eeg",
  "group_col": "subject_id",
  "test_size": 0.2,
  "max_rows": null,
  "seed": 42,
  "ae_pattern": "ae_pow_plus_eeg_full",
  "use_scaled_synthetic": false,
  "max_synthetic": 10000,
  "synthetic_ratio": 1.0,
  "knn_k": 5,
  "ridge_alpha": 1.0
}
```

Autoencoder run: `20260605_133745_ae_pow_plus_eeg_full_pow_plus_eeg_latent32`

## 4. Dataset info

```json
{
  "rows_real_total": 43174,
  "rows_train_real": 34199,
  "rows_test_real": 8975,
  "rows_synthetic_used": 10000,
  "rows_train_augmented": 44199,
  "n_features": 448,
  "n_targets": 7,
  "subjects_total": 53,
  "subjects_train": 42,
  "subjects_test": 11
}
```

## 5. Metrics

| experiment          | target            |       mae |     rmse |          r2 |   pearson |   spearman |
|:--------------------|:------------------|----------:|---------:|------------:|----------:|-----------:|
| real_only           | target_attention  | 0.10259   | 0.232653 |  -2.30691   | 0.102653  |   0.394124 |
| real_only           | target_engagement | 0.105681  | 0.306493 |  -5.26251   | 0.11183   |   0.287689 |
| real_only           | target_excitement | 0.179445  | 0.87901  | -12.5001    | 0.0642752 |   0.461421 |
| real_only           | target_stress     | 0.12567   | 0.413771 |  -5.49706   | 0.0506468 |   0.266799 |
| real_only           | target_relaxation | 0.131953  | 0.300529 |  -2.32866   | 0.16037   |   0.345396 |
| real_only           | target_interest   | 0.0842076 | 0.248846 |  -3.93715   | 0.12106   |   0.325844 |
| real_only           | target_focus      | 0.114087  | 0.706357 | -28.9649    | 0.0112807 |   0.296877 |
| real_plus_synthetic | target_attention  | 0.0972165 | 0.177697 |  -0.929143  | 0.12531   |   0.381469 |
| real_plus_synthetic | target_engagement | 0.105543  | 0.171006 |  -0.949522  | 0.201945  |   0.274234 |
| real_plus_synthetic | target_excitement | 0.165567  | 0.225349 |   0.112716  | 0.437604  |   0.468705 |
| real_plus_synthetic | target_stress     | 0.117971  | 0.189773 |  -0.366669  | 0.222309  |   0.269763 |
| real_plus_synthetic | target_relaxation | 0.133545  | 0.207656 |  -0.589231  | 0.176058  |   0.273035 |
| real_plus_synthetic | target_interest   | 0.0813679 | 0.143388 |  -0.639232  | 0.202525  |   0.291173 |
| real_plus_synthetic | target_focus      | 0.0997059 | 0.134051 |  -0.0792141 | 0.265854  |   0.306883 |

## 6. Delta table

| target            |   r2_real_only |   r2_real_plus_synthetic |   r2_delta |   spearman_real_only |   spearman_real_plus_synthetic |   spearman_delta |   mae_real_only |   mae_real_plus_synthetic |    mae_delta |
|:------------------|---------------:|-------------------------:|-----------:|---------------------:|-------------------------------:|-----------------:|----------------:|--------------------------:|-------------:|
| target_focus      |      -28.9649  |               -0.0792141 |   28.8857  |             0.296877 |                       0.306883 |       0.0100066  |       0.114087  |                 0.0997059 | -0.0143808   |
| target_excitement |      -12.5001  |                0.112716  |   12.6128  |             0.461421 |                       0.468705 |       0.00728436 |       0.179445  |                 0.165567  | -0.0138787   |
| target_stress     |       -5.49706 |               -0.366669  |    5.13039 |             0.266799 |                       0.269763 |       0.00296386 |       0.12567   |                 0.117971  | -0.00769897  |
| target_engagement |       -5.26251 |               -0.949522  |    4.31299 |             0.287689 |                       0.274234 |      -0.0134545  |       0.105681  |                 0.105543  | -0.000138074 |
| target_interest   |       -3.93715 |               -0.639232  |    3.29792 |             0.325844 |                       0.291173 |      -0.0346719  |       0.0842076 |                 0.0813679 | -0.00283971  |
| target_relaxation |       -2.32866 |               -0.589231  |    1.73943 |             0.345396 |                       0.273035 |      -0.0723611  |       0.131953  |                 0.133545  |  0.00159247  |
| target_attention  |       -2.30691 |               -0.929143  |    1.37777 |             0.394124 |                       0.381469 |      -0.0126556  |       0.10259   |                 0.0972165 | -0.0053739   |

## 7. Interpretation

Synthetic augmentation improved R² for 7 out of 7 target metrics. If the effect is small or negative, this means that synthetic vectors reproduce the feature distribution only partially and should not be treated as universally useful augmentation.

This is a lightweight downstream utility test. It does not prove clinical realism of synthetic EEG, but it checks whether generated EEG/POW representations are useful for a predictive model.