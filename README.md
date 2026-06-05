# EEG Cognitive State Exam Project

Экзаменационный проект по глубокому обучению для анализа ЭЭГ-данных, предсказания когнитивно-аффективных состояний, моделирования временной динамики, обнаружения аномальных эпизодов и генерации синтетических EEG/POW-представлений.

## 1. Идея проекта

Цель проекта — проверить, можно ли по оконным EEG/POW-признакам предсказывать когнитивно-аффективные Performance Metrics, а также оценить, помогает ли локальный временной контекст улучшить качество предсказания.

Основная гипотеза:

> Когнитивно-аффективные состояния человека обладают локальной временной инерцией, поэтому модели, использующие последовательность соседних EEG/POW-окон, должны работать лучше, чем модели, использующие только одно окно.

Проект адаптирован под экзамен по глубокому обучению. В нём реализованы:

```text
- классические ML baseline-модели;
- MLP-регрессор;
- LSTM-модель для временной динамики;
- TransformerEncoder-модель для локального временного контекста;
- Autoencoder для восстановления и генерации EEG/POW feature vectors;
- unsupervised anomaly detection;
- synthetic augmentation experiment.
```

---

## 2. Исходные датасеты и адаптация под задачу

В проекте используются два исходных набора данных с EEG/PM-записями:

```text
gpn_data
Old_EEG
```

Оба датасета относятся к задаче анализа когнитивно-аффективных состояний человека по данным ЭЭГ и связанным Performance Metrics, но отличаются структурой хранения, форматом файлов и степенью унификации. Поэтому перед обучением моделей они были приведены к единому оконному представлению.

### 2.1. `gpn_data`

`gpn_data` — основной новый датасет проекта. Он содержит записи ЭЭГ, полученные с wearable EEG-устройства класса Emotiv, а также связанные Performance Metrics и служебные данные.

В исходном виде данные включали:

```text
- многоканальные EEG-сигналы;
- POW / band-power признаки;
- Performance Metrics;
- идентификаторы записей, субъектов, дней и частей эксперимента;
- временную информацию для синхронизации.
```

Основная роль `gpn_data` в проекте:

```text
- основной источник новых EEG/PM-наблюдений;
- база для построения оконных EEG/POW-признаков;
- источник target-переменных Performance Metrics.
```

### 2.2. `Old_EEG`

`Old_EEG` — старый набор EEG-записей, использованный как дополнительный источник данных. Он отличается от `gpn_data` структурой хранения и форматом, поэтому был отдельно обработан и приведён к совместимому виду.

В проекте `Old_EEG` используется для:

```text
- увеличения общего числа EEG/PM-окон;
- проверки устойчивости моделей на данных из другого источника;
- расширения набора субъектов и записей;
- анализа переносимости между старыми и новыми данными.
```

### 2.3. Унификация данных

Для совместного использования `gpn_data` и `Old_EEG` была выполнена унификация:

```text
1. Приведение записей к общей структуре.
2. Согласование идентификаторов субъектов, записей, дней и частей эксперимента.
3. Синхронизация EEG/POW-признаков с Performance Metrics.
4. Разбиение временных рядов на фиксированные окна.
5. Формирование агрегированных PM-метрик внутри каждого окна.
6. Формирование EEG/POW feature vectors.
7. Создание единого parquet-датасета для обучения моделей.
```

Итоговый обработанный датасет:

```text
data/processed/windowed_eeg_pm_dataset_w10.parquet
```

Размер итогового датасета:

```text
Rows: 51 308
Columns: 508
POW features: 280
EEG features: 168
Total pow_plus_eeg features: 448
```

Источники данных в итоговом датасете:

```text
gpn_data
Old_EEG
```

### 2.4. Оконное представление

Данные были преобразованы в оконный формат. Каждая строка итогового датасета соответствует одному временному EEG/PM-окну.

Для каждого окна сохраняются:

```text
- metadata: record_id, source, subject_id, day, part;
- временные признаки: t_start, t_end, t_center;
- POW-признаки;
- EEG-признаки;
- агрегированные Performance Metrics;
- подготовленные target-переменные.
```

Такое представление позволяет использовать данные как для классических tabular-моделей, так и для deep learning моделей, работающих с последовательностями соседних окон.

### 2.5. Входные признаки

В проекте используются три варианта feature set:

```text
pow
eeg
pow_plus_eeg
```

Основной feature set для финальных экспериментов:

```text
pow_plus_eeg
```

Он включает:

```text
POW features: 280
EEG features: 168
Total features: 448
```

Эти признаки используются как вход модели.

### 2.6. Целевые переменные

В качестве целевых переменных используются подготовленные Performance Metrics:

```text
target_attention
target_engagement
target_excitement
target_stress
target_relaxation
target_interest
target_focus
```

Важно: исходные Performance Metrics не используются как входные признаки. Они используются только как targets, которые модель должна предсказать по EEG/POW-признакам.

Соответствие смыслов:

```text
target_attention   — внимание
target_engagement  — вовлечённость
target_excitement  — возбуждение / эмоциональная активация
target_stress      — стресс
target_relaxation  — расслабленность
target_interest    — интерес
target_focus       — фокусировка
```

### 2.7. Адаптация под экзаменационную задачу

Исходная лабораторная постановка предполагает анализ ЭЭГ, классификацию состояний, моделирование динамики, генерацию новых EEG-фрагментов и сравнение real/synthetic данных.

В данном проекте задача адаптирована следующим образом:

```text
Вместо классификации моторных действий или клинических событий решается задача предсказания когнитивно-аффективных Performance Metrics по EEG/POW-признакам.
```

То есть модель получает:

```text
X = EEG/POW feature vector или последовательность соседних EEG/POW-окон
```

и предсказывает:

```text
y = Performance Metrics:
    attention, engagement, excitement, stress, relaxation, interest, focus
```

Для deep learning части используются несколько постановок:

```text
1. MLP:
   single-window EEG/POW vector → 7 PM targets

2. LSTM:
   sequence of neighboring EEG/POW windows → 7 PM targets

3. TransformerEncoder:
   sequence of neighboring EEG/POW windows → 7 PM targets

4. Autoencoder:
   EEG/POW feature vector → latent representation → reconstructed/synthetic EEG/POW vector
```

Таким образом, исходные EEG-данные были преобразованы в задачу:

```text
EEG/POW-based cognitive state prediction and generation
```

---

## 3. Структура проекта

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
│   ├── final_summary/
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
│   ├── 15_train_mlp_pm_regressor.py
│   ├── 16_train_lstm_pm_dynamics.py
│   ├── 17_train_transformer_pm_dynamics.py
│   ├── 18_detect_pm_anomalies.py
│   ├── 22_train_eeg_autoencoder.py
│   ├── 23_summarize_dl_experiments.py
│   ├── 24_train_with_synthetic_augmentation.py
│   └── wesad/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## 4. Настройка окружения

Проект тестировался в Conda-окружении:

```text
D:\miniconda3\envs\eeg_nir\python.exe
```

Создание окружения:

```powershell
conda create -n eeg_nir python=3.11 -y
conda activate eeg_nir
pip install -r requirements.txt
```

Если команда `python` из PowerShell не работает, используйте полный путь к интерпретатору:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe
```

Проверка:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe --version
```

---

## 5. Проверка данных

Проверить наличие датасета:

```powershell
Test-Path data\processed\windowed_eeg_pm_dataset_w10.parquet
```

Ожидаемый результат:

```text
True
```

Проверить чтение parquet-файла:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe -c "import pandas as pd; df=pd.read_parquet('data/processed/windowed_eeg_pm_dataset_w10.parquet'); print(df.shape); print(df.columns[:30].tolist())"
```

Ожидаемый размер:

```text
(51308, 508)
```

Проверить target-колонки:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe -c "import pandas as pd; df=pd.read_parquet('data/processed/windowed_eeg_pm_dataset_w10.parquet'); print([c for c in df.columns if c.startswith('target_')])"
```

Ожидаемые target-переменные:

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

## 6. Описание скриптов

### 6.1. Подготовка данных

#### `src/04_build_windowed_pm_dataset.py`

Строит оконный EEG/PM-датасет.

Назначение:

```text
- синхронизация EEG и Performance Metrics;
- формирование окон;
- агрегация PM-метрик;
- сохранение итогового parquet-датасета.
```

#### `src/05_analyze_pm_sampling.py`

Анализирует частоту обновления Performance Metrics.

Назначение:

```text
- анализ временных интервалов между PM-измерениями;
- обоснование длины окна;
- подготовка статистики для отчёта.
```

#### `src/06_eda_windowed_dataset.py`

Проводит EDA итогового оконного датасета.

Назначение:

```text
- анализ размера датасета;
- проверка пропусков;
- анализ распределения target-переменных;
- подготовка EDA-графиков.
```

#### `src/08_build_windowed_eeg_features.py`

Формирует EEG-derived признаки и объединяет их с POW-признаками.

Назначение:

```text
- построение EEG-признаков;
- объединение EEG и POW feature sets;
- подготовка признакового пространства для моделей.
```

---

### 6.2. Классические baseline-модели

#### `src/07_train_baselines.py`

Обучает baseline-модели для одного выбранного target.

Поддерживает:

```text
- classification;
- regression;
- random split;
- GroupKFold by subject_id;
- cross-source validation.
```

Пример запуска для `target_focus`:

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

#### `src/09_train_multi_pm_baselines.py`

Обучает классические baseline-регрессоры по всем PM-метрикам.

Поддерживаемые модели:

```text
ridge
hgb
lgbm
rf
```

Полный запуск:

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

#### `src/10_describe_multi_pm_baseline.py`

Формирует описание и сводку результатов multi-PM baseline.

---

### 6.3. Deep Learning модели

#### `src/15_train_mlp_pm_regressor.py`

Multi-output MLP-регрессор на PyTorch.

Задача:

```text
Вход: одно EEG/POW-окно, 448 признаков
Выход: 7 PM target-переменных
```

Архитектура:

```text
448 → 512 → 256 → 128 → 7
BatchNorm
ReLU
Dropout
```

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\15_train_mlp_pm_regressor.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name mlp_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --n-splits 5 `
  --epochs 80 `
  --batch-size 1024 `
  --hidden-dims 512 256 128 `
  --dropout 0.25 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 12 `
  --amp
```

Назначение:

```text
MLP используется как нейросетевой baseline без временного контекста.
```

Основной вывод:

```text
MLP показывает положительные корреляции, но в целом уступает классическим tree-based baseline-моделям и sequence DL-моделям.
```

#### `src/16_train_lstm_pm_dynamics.py`

LSTM-модель для анализа локальной временной динамики EEG/POW-окон.

Задача:

```text
Вход: последовательность из 5 соседних EEG/POW-окон
Форма входа: [batch, seq_len, features]
Выход: 7 PM target-переменных
```

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\16_train_lstm_pm_dynamics.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name lstm_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --seq-len 5 `
  --target-position center `
  --n-splits 5 `
  --epochs 60 `
  --batch-size 512 `
  --hidden-dim 128 `
  --num-layers 1 `
  --head-hidden-dim 128 `
  --dropout 0.2 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 10 `
  --amp
```

Назначение:

```text
Проверяет, помогает ли локальная временная динамика соседних окон улучшить предсказание PM-метрик.
```

#### `src/17_train_transformer_pm_dynamics.py`

TransformerEncoder-модель для анализа локального временного контекста.

Задача:

```text
Вход: последовательность из 5 соседних EEG/POW-окон
Механизм: self-attention over neighboring windows
Выход: 7 PM target-переменных
```

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\17_train_transformer_pm_dynamics.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name transformer_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --seq-len 5 `
  --target-position center `
  --pooling center `
  --n-splits 5 `
  --epochs 50 `
  --batch-size 512 `
  --d-model 128 `
  --nhead 4 `
  --num-layers 2 `
  --dim-feedforward 256 `
  --dropout 0.2 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --patience 8 `
  --amp
```

Назначение:

```text
TransformerEncoder является основной attention-based DL-моделью проекта.
```

Основной вывод:

```text
TransformerEncoder оказался лучшей моделью почти по всем PM-метрикам.
```

---

### 6.4. Обнаружение аномалий

#### `src/18_detect_pm_anomalies.py`

Unsupervised-скрипт для поиска аномальных EEG/PM-окон.

Задача:

```text
Найти статистически необычные окна без экспертной разметки аномалий.
```

Метод:

```text
- robust z-score PM-метрик внутри записи;
- robust z-score абсолютного изменения PM-метрик между соседними окнами;
- отклонение PM-метрик от локального rolling mean;
- robust anomaly score по EEG/POW-признакам;
- объединение PM anomaly score и feature anomaly score.
```

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\18_detect_pm_anomalies.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name pm_eeg_anomaly_detection_full `
  --feature-set pow_plus_eeg `
  --rolling-window 7 `
  --threshold-quantile 0.98 `
  --min-score 3.0 `
  --feature-score-weight 0.35 `
  --timeline-target target_focus
```

Полученный результат:

```text
rows_scored: 43 174
anomaly_rows: 864
anomaly_rate: 2.00%
records: 117
subjects: 53
threshold: 20.37
```

Назначение:

```text
Закрывает задачу обнаружения паттернов или аномальных эпизодов в длинных EEG/PM-записях.
```

---

### 6.5. Генерация EEG/POW-представлений

#### `src/22_train_eeg_autoencoder.py`

Autoencoder для восстановления и генерации EEG/POW feature vectors.

Задача:

```text
Вход: EEG/POW feature vector
Encoder: feature vector → latent representation
Decoder: latent representation → reconstructed feature vector
Synthetic generation: latent vector + Gaussian noise → decoder → synthetic feature vector
```

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\22_train_eeg_autoencoder.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name ae_pow_plus_eeg_full `
  --feature-set pow_plus_eeg `
  --latent-dim 32 `
  --hidden-dims 512 256 128 `
  --epochs 80 `
  --batch-size 1024 `
  --lr 1e-3 `
  --weight-decay 1e-5 `
  --patience 10 `
  --n-synthetic 10000 `
  --noise-scale 0.35 `
  --amp
```

Назначение:

```text
Закрывает генеративную часть экзаменационного задания.
```

Важно:

```text
Генерация выполняется не в пространстве raw EEG-сигнала, а в пространстве EEG/POW feature vectors.
```

Основные reconstruction-метрики:

```text
scaled MSE:     0.145
scaled RMSE:    0.381
scaled MAE:     0.104
scaled cosine:  0.778
```

Основные real-vs-synthetic метрики:

```text
scaled feature mean abs diff mean: 0.031
scaled feature std abs diff mean:  0.149
```

Интерпретация:

```text
Autoencoder достаточно хорошо восстанавливает стандартизированное признаковое пространство. Synthetic vectors близки к real vectors по средним значениям признаков, но хуже совпадают по дисперсии.
```

---

### 6.6. Synthetic augmentation

#### `src/24_train_with_synthetic_augmentation.py`

Скрипт проверяет, помогают ли synthetic EEG/POW feature vectors улучшить downstream prediction.

Задача:

```text
Сравнить обучение модели только на real data и обучение на смеси real + synthetic data.
```

Метод:

```text
Experiment A:
train = real only
test = real only

Experiment B:
train = real + synthetic
test = real only
```

Synthetic features берутся из последнего Autoencoder-run. Synthetic targets формируются через pseudo-labeling на основе k-nearest neighbors среди real train-векторов.

Полный запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\24_train_with_synthetic_augmentation.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name synthetic_augmentation_full `
  --feature-set pow_plus_eeg `
  --ae-pattern ae_pow_plus_eeg_full `
  --test-size 0.2 `
  --max-synthetic 10000 `
  --synthetic-ratio 1.0 `
  --knn-k 5 `
  --ridge-alpha 1.0 `
  --seed 42
```

Полученный результат:

```text
rows_train_real: 34 199
rows_test_real: 8 975
rows_synthetic_used: 10 000
rows_train_augmented: 44 199
subjects_train: 42
subjects_test: 11
```

Интерпретация:

```text
Synthetic augmentation снизила MAE для большинства PM-метрик и уменьшила отрицательные значения R² у нестабильной Ridge-модели. Однако Spearman correlation улучшилась только для части targets. Поэтому текущие synthetic vectors можно рассматривать как источник частичного регуляризующего эффекта, но не как универсально полезную аугментацию.
```

---

### 6.7. Финальная сводка экспериментов

#### `src/23_summarize_dl_experiments.py`

Собирает результаты всех основных экспериментов:

```text
Classical baseline
MLP
LSTM
TransformerEncoder
Autoencoder
```

Запуск:

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\23_summarize_dl_experiments.py `
  --root .
```

Создаёт:

```text
reports/final_summary/
├── final_exam_report.md
├── tables/
│   ├── final_model_comparison.csv
│   ├── best_model_by_target.csv
│   ├── detected_runs.json
│   └── autoencoder_summary_metrics.csv
└── figures/
    ├── final_model_comparison_r2.png
    ├── final_model_comparison_spearman.png
    └── best_r2_by_target.png
```

---

## 7. Итоговые результаты

Финальное сравнение моделей показало, что TransformerEncoder является лучшей моделью почти по всем target-переменным.

| Target              | Лучшая модель по R² |    R² | Лучшая модель по Spearman | Spearman |
| ------------------- | ------------------: | ----: | ------------------------: | -------: |
| `target_excitement` |  TransformerEncoder | 0.436 |        TransformerEncoder |    0.624 |
| `target_relaxation` |  TransformerEncoder | 0.329 |        TransformerEncoder |    0.581 |
| `target_engagement` |                LSTM | 0.129 |        TransformerEncoder |    0.449 |
| `target_interest`   |  TransformerEncoder | 0.214 |        TransformerEncoder |    0.429 |
| `target_focus`      |  TransformerEncoder | 0.171 |        TransformerEncoder |    0.472 |
| `target_stress`     |  TransformerEncoder | 0.149 |        TransformerEncoder |    0.451 |
| `target_attention`  |  TransformerEncoder | 0.053 |        TransformerEncoder |    0.436 |

Главный вывод:

```text
Модели с локальным временным контекстом работают лучше, чем простая MLP-модель на одном окне. TransformerEncoder оказался наиболее сильной deep learning моделью, что подтверждает важность self-attention и локальной временной динамики для анализа EEG/POW-представлений.
```

---

## 8. Связь с требованиями экзамена

| Требование                                     |    Статус | Реализация                                                |
| ---------------------------------------------- | --------: | --------------------------------------------------------- |
| Предобработка EEG-данных                       | Выполнено | `04`, `05`, `06`, `08`                                    |
| Классификация / предсказание состояний         | Выполнено | `07`, `09`, `15`                                          |
| Deep learning модель                           | Выполнено | MLP, LSTM, TransformerEncoder                             |
| Моделирование динамики                         | Выполнено | LSTM, TransformerEncoder                                  |
| Обнаружение паттернов / аномалий               | Выполнено | `18_detect_pm_anomalies.py`                               |
| Генерация новых EEG-фрагментов / представлений | Выполнено | Autoencoder                                               |
| Сравнение real/synthetic                       | Выполнено | Autoencoder metrics + PCA/feature plots                   |
| Оценка полезности synthetic data               | Выполнено | `24_train_with_synthetic_augmentation.py`                 |
| Визуализация экспериментов                     | Выполнено | `reports/runs/*/figures`, `reports/final_summary/figures` |
| Анализ результатов                             | Выполнено | `reports/final_summary/final_exam_report.md`, README      |

---

## 9. Что не коммитить

Не следует коммитить:

```text
data/processed/*.parquet
data/processed/results/
reports/runs/**/models/
reports/runs/**/synthetic/
*.npy
*.pt
*.pth
*.pkl
*.joblib
```

Следует коммитить:

```text
src/*.py
configs/*.yaml
reports/runs/*/report.md
reports/runs/*/target_metrics_agg.csv
reports/runs/*/fold_metrics.csv
reports/runs/*/reconstruction_metrics.csv
reports/runs/*/real_vs_synthetic_metrics.csv
reports/runs/*/augmentation_metrics.csv
reports/runs/*/augmentation_delta.csv
reports/runs/*/anomaly_summary.csv
reports/runs/*/record_anomaly_summary.csv
reports/runs/*/figures/
reports/final_summary/
README.md
```

---

## 10. Рекомендуемый порядок воспроизведения

### 1. Проверить датасет

```powershell
Test-Path data\processed\windowed_eeg_pm_dataset_w10.parquet
```

### 2. Запустить классический baseline

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

### 3. Запустить MLP

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\15_train_mlp_pm_regressor.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name mlp_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --n-splits 5 `
  --epochs 80 `
  --batch-size 1024 `
  --hidden-dims 512 256 128 `
  --dropout 0.25 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 12 `
  --amp
```

### 4. Запустить LSTM

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\16_train_lstm_pm_dynamics.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name lstm_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --seq-len 5 `
  --target-position center `
  --n-splits 5 `
  --epochs 60 `
  --batch-size 512 `
  --hidden-dim 128 `
  --num-layers 1 `
  --head-hidden-dim 128 `
  --dropout 0.2 `
  --lr 1e-3 `
  --weight-decay 1e-4 `
  --patience 10 `
  --amp
```

### 5. Запустить TransformerEncoder

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\17_train_transformer_pm_dynamics.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name transformer_pm_groupkfold_full `
  --feature-set pow_plus_eeg `
  --seq-len 5 `
  --target-position center `
  --pooling center `
  --n-splits 5 `
  --epochs 50 `
  --batch-size 512 `
  --d-model 128 `
  --nhead 4 `
  --num-layers 2 `
  --dim-feedforward 256 `
  --dropout 0.2 `
  --lr 5e-4 `
  --weight-decay 1e-4 `
  --patience 8 `
  --amp
```

### 6. Запустить поиск аномалий

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\18_detect_pm_anomalies.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name pm_eeg_anomaly_detection_full `
  --feature-set pow_plus_eeg `
  --rolling-window 7 `
  --threshold-quantile 0.98 `
  --min-score 3.0 `
  --feature-score-weight 0.35 `
  --timeline-target target_focus
```

### 7. Запустить Autoencoder

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\22_train_eeg_autoencoder.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name ae_pow_plus_eeg_full `
  --feature-set pow_plus_eeg `
  --latent-dim 32 `
  --hidden-dims 512 256 128 `
  --epochs 80 `
  --batch-size 1024 `
  --lr 1e-3 `
  --weight-decay 1e-5 `
  --patience 10 `
  --n-synthetic 10000 `
  --noise-scale 0.35 `
  --amp
```

### 8. Запустить synthetic augmentation

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\24_train_with_synthetic_augmentation.py `
  --root . `
  --dataset data/processed/windowed_eeg_pm_dataset_w10.parquet `
  --run-name synthetic_augmentation_full `
  --feature-set pow_plus_eeg `
  --ae-pattern ae_pow_plus_eeg_full `
  --test-size 0.2 `
  --max-synthetic 10000 `
  --synthetic-ratio 1.0 `
  --knn-k 5 `
  --ridge-alpha 1.0 `
  --seed 42
```

### 9. Собрать финальную сводку

```powershell
D:\miniconda3\envs\eeg_nir\python.exe src\23_summarize_dl_experiments.py `
  --root .
```

---

## 11. Основной вывод проекта

В проекте был построен полный pipeline анализа EEG/POW-данных:

```text
предобработка → baseline ML → MLP → LSTM → TransformerEncoder → anomaly detection → Autoencoder → synthetic augmentation → финальное сравнение
```

Главный результат:

> TransformerEncoder с локальным временным контекстом оказался наиболее эффективной deep learning моделью для большинства PM-метрик. Это подтверждает, что для предсказания когнитивно-аффективных состояний по ЭЭГ важны не только признаки одного окна, но и динамика соседних окон.

Генеративный блок:

> Autoencoder показал возможность восстановления и генерации EEG/POW feature vectors в латентном пространстве. Хотя это не полноценная генерация raw EEG-сигнала, этот подход закрывает генеративную часть проекта и позволяет сравнивать реальные и синтетические EEG-представления.

Аномалии:

> Unsupervised anomaly detection выделил около 2% окон как статистически необычные. Эти окна можно рассматривать как кандидаты на резкие изменения состояния, нестабильные эпизоды или артефактные участки.

Synthetic augmentation:

> Synthetic augmentation дала неоднозначный результат: она стабилизировала Ridge-модель и снизила MAE для большинства PM-метрик, но не обеспечила устойчивого улучшения Spearman correlation для всех targets. Это показывает, что synthetic EEG/POW vectors требуют дальнейшего улучшения перед использованием как универсальной аугментации.
