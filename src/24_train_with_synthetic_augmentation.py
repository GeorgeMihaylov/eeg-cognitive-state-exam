from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler


TARGET_COLUMNS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]


def get_feature_columns(df: pd.DataFrame, feature_set: str) -> List[str]:
    pow_cols = [c for c in df.columns if c.startswith("POW.")]
    eeg_cols = [c for c in df.columns if c.startswith("EEG.")]

    if feature_set == "pow":
        return pow_cols
    if feature_set == "eeg":
        return eeg_cols
    if feature_set == "pow_plus_eeg":
        return pow_cols + eeg_cols

    raise ValueError(f"Unknown feature_set: {feature_set}")


def find_latest_run(runs_dir: Path, pattern: str) -> Optional[Path]:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and pattern in p.name]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def load_synthetic_features(ae_run_dir: Path, use_scaled: bool) -> np.ndarray:
    filename = "synthetic_features_scaled.npy" if use_scaled else "synthetic_features_raw.npy"
    path = ae_run_dir / "synthetic" / filename

    if not path.exists():
        raise FileNotFoundError(f"Synthetic features not found: {path}")

    return np.load(path)


def prepare_real_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    group_col: str,
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    required = feature_cols + target_cols + [group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[required].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=target_cols + [group_col])

    for col in feature_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    out = out.dropna().reset_index(drop=True)

    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    return out


def make_synthetic_targets(
    x_synth_raw: np.ndarray,
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    k: int,
    seed: int,
    max_synthetic: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    if len(x_synth_raw) > max_synthetic:
        idx = rng.choice(len(x_synth_raw), size=max_synthetic, replace=False)
        x_synth_raw = x_synth_raw[idx]

    # Lightweight pseudo-labeling:
    # for each synthetic vector, assign the mean target of k nearest real train vectors.
    # This avoids training augmentation on generated X with random labels.
    x_scaler = StandardScaler()
    x_train_scaled = x_scaler.fit_transform(x_train_raw)
    x_synth_scaled = x_scaler.transform(x_synth_raw)

    y_synth = np.zeros((len(x_synth_scaled), y_train.shape[1]), dtype=np.float32)

    batch_size = 512
    train_norm = np.sum(x_train_scaled ** 2, axis=1, keepdims=True).T

    for start in range(0, len(x_synth_scaled), batch_size):
        chunk = x_synth_scaled[start : start + batch_size]
        chunk_norm = np.sum(chunk ** 2, axis=1, keepdims=True)
        distances = chunk_norm + train_norm - 2.0 * chunk @ x_train_scaled.T
        nn_idx = np.argpartition(distances, kth=min(k, len(x_train_scaled) - 1), axis=1)[:, :k]
        y_synth[start : start + len(chunk)] = y_train[nn_idx].mean(axis=1)

    return x_synth_raw.astype(np.float32), y_synth.astype(np.float32)


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    try:
        return float(func(y_true, y_pred)[0])
    except Exception:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    experiment: str,
) -> List[Dict[str, float]]:
    rows = []

    for i, target in enumerate(target_cols):
        yt = y_true[:, i]
        yp = y_pred[:, i]

        mse = mean_squared_error(yt, yp)

        rows.append(
            {
                "experiment": experiment,
                "target": target,
                "mae": float(mean_absolute_error(yt, yp)),
                "rmse": float(math.sqrt(mse)),
                "r2": float(r2_score(yt, yp)),
                "pearson": safe_corr(pearsonr, yt, yp),
                "spearman": safe_corr(spearmanr, yt, yp),
            }
        )

    return rows


def train_and_evaluate(
    x_train_raw: np.ndarray,
    y_train: np.ndarray,
    x_test_raw: np.ndarray,
    y_test: np.ndarray,
    target_cols: List[str],
    experiment: str,
    alpha: float,
) -> Tuple[pd.DataFrame, np.ndarray]:
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    x_train = x_scaler.fit_transform(x_train_raw)
    x_test = x_scaler.transform(x_test_raw)

    y_train_scaled = y_scaler.fit_transform(y_train)

    model = MultiOutputRegressor(Ridge(alpha=alpha, random_state=42))
    model.fit(x_train, y_train_scaled)

    y_pred_scaled = model.predict(x_test)
    y_pred = y_scaler.inverse_transform(y_pred_scaled)

    metrics = compute_metrics(y_test, y_pred, target_cols, experiment)
    return pd.DataFrame(metrics), y_pred


def plot_metric_comparison(df: pd.DataFrame, metric: str, output_path: Path) -> None:
    pivot = df.pivot_table(index="target", columns="experiment", values=metric, aggfunc="mean")

    target_order = [t for t in TARGET_COLUMNS if t in pivot.index]
    pivot = pivot.loc[target_order]

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(f"Synthetic augmentation comparison by {metric}")
    plt.xlabel("Target")
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def make_delta_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    real = metrics_df[metrics_df["experiment"] == "real_only"].set_index("target")
    aug = metrics_df[metrics_df["experiment"] == "real_plus_synthetic"].set_index("target")

    rows = []

    for target in real.index:
        rows.append(
            {
                "target": target,
                "r2_real_only": real.loc[target, "r2"],
                "r2_real_plus_synthetic": aug.loc[target, "r2"],
                "r2_delta": aug.loc[target, "r2"] - real.loc[target, "r2"],
                "spearman_real_only": real.loc[target, "spearman"],
                "spearman_real_plus_synthetic": aug.loc[target, "spearman"],
                "spearman_delta": aug.loc[target, "spearman"] - real.loc[target, "spearman"],
                "mae_real_only": real.loc[target, "mae"],
                "mae_real_plus_synthetic": aug.loc[target, "mae"],
                "mae_delta": aug.loc[target, "mae"] - real.loc[target, "mae"],
            }
        )

    return pd.DataFrame(rows).sort_values("r2_delta", ascending=False)


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    ae_run_dir: Path,
    dataset_info: Dict[str, object],
    metrics_df: pd.DataFrame,
    delta_df: pd.DataFrame,
) -> None:
    lines = []

    lines.append("# Synthetic Augmentation Report")
    lines.append("")
    lines.append("## 1. Цель")
    lines.append("")
    lines.append(
        "Цель эксперимента — проверить, улучшают ли synthetic EEG/POW feature vectors "
        "обучение downstream-модели. Это закрывает требование лабораторной работы о проверке "
        "практической полезности синтетических данных как augmentation."
    )
    lines.append("")
    lines.append("## 2. Метод")
    lines.append("")
    lines.append("Сравниваются два режима:")
    lines.append("")
    lines.append("- `real_only`: обучение только на реальных EEG/POW-признаках;")
    lines.append("- `real_plus_synthetic`: обучение на реальных признаках и synthetic vectors из Autoencoder.")
    lines.append("")
    lines.append(
        "Test set остаётся только реальным. Synthetic features получают pseudo-labels "
        "через k-nearest neighbors по реальным train-векторам."
    )
    lines.append("")
    lines.append("## 3. Конфигурация")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(f"Autoencoder run: `{ae_run_dir.name}`")
    lines.append("")
    lines.append("## 4. Dataset info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 5. Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Delta table")
    lines.append("")
    lines.append(delta_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 7. Interpretation")
    lines.append("")
    improved = int((delta_df["r2_delta"] > 0).sum())
    total = int(len(delta_df))
    lines.append(
        f"Synthetic augmentation improved R² for {improved} out of {total} target metrics. "
        "If the effect is small or negative, this means that synthetic vectors reproduce "
        "the feature distribution only partially and should not be treated as universally useful augmentation."
    )
    lines.append("")
    lines.append(
        "This is a lightweight downstream utility test. It does not prove clinical realism of synthetic EEG, "
        "but it checks whether generated EEG/POW representations are useful for a predictive model."
    )

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".", type=str)
    parser.add_argument(
        "--dataset",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
        type=str,
    )
    parser.add_argument("--run-name", default="synthetic_augmentation", type=str)
    parser.add_argument("--feature-set", default="pow_plus_eeg", choices=["pow", "eeg", "pow_plus_eeg"])
    parser.add_argument("--group-col", default="subject_id", type=str)
    parser.add_argument("--test-size", default=0.2, type=float)
    parser.add_argument("--max-rows", default=None, type=int)
    parser.add_argument("--seed", default=42, type=int)

    parser.add_argument("--ae-pattern", default="ae_pow_plus_eeg_full", type=str)
    parser.add_argument("--use-scaled-synthetic", action="store_true")
    parser.add_argument("--max-synthetic", default=10000, type=int)
    parser.add_argument("--synthetic-ratio", default=1.0, type=float)
    parser.add_argument("--knn-k", default=5, type=int)
    parser.add_argument("--ridge-alpha", default=1.0, type=float)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    dataset_path = root / args.dataset
    runs_dir = root / "reports" / "runs"

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    ae_run_dir = find_latest_run(runs_dir, args.ae_pattern)
    if ae_run_dir is None:
        raise FileNotFoundError(f"No Autoencoder run found with pattern: {args.ae_pattern}")

    print("=" * 80)
    print("Train with synthetic augmentation")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")
    print(f"Autoencoder run: {ae_run_dir}")

    df = pd.read_parquet(dataset_path)

    feature_cols = get_feature_columns(df, args.feature_set)
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    real_df = prepare_real_data(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col=args.group_col,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    x_real = real_df[feature_cols].to_numpy(dtype=np.float32)
    y_real = real_df[target_cols].to_numpy(dtype=np.float32)
    groups = real_df[args.group_col].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_size, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(x_real, y_real, groups=groups))

    x_train_real = x_real[train_idx]
    y_train_real = y_real[train_idx]
    x_test_real = x_real[test_idx]
    y_test_real = y_real[test_idx]

    x_synth_all = load_synthetic_features(ae_run_dir, use_scaled=args.use_scaled_synthetic)

    if args.use_scaled_synthetic:
        raise ValueError(
            "use_scaled_synthetic=True is not supported for downstream augmentation in raw feature space. "
            "Use raw synthetic features from Autoencoder instead."
        )

    n_synthetic = int(min(len(x_synth_all), len(x_train_real) * args.synthetic_ratio, args.max_synthetic))

    x_synth, y_synth = make_synthetic_targets(
        x_synth_raw=x_synth_all,
        x_train_raw=x_train_real,
        y_train=y_train_real,
        k=args.knn_k,
        seed=args.seed,
        max_synthetic=n_synthetic,
    )

    x_aug = np.vstack([x_train_real, x_synth])
    y_aug = np.vstack([y_train_real, y_synth])

    dataset_info = {
        "rows_real_total": int(len(x_real)),
        "rows_train_real": int(len(x_train_real)),
        "rows_test_real": int(len(x_test_real)),
        "rows_synthetic_used": int(len(x_synth)),
        "rows_train_augmented": int(len(x_aug)),
        "n_features": int(len(feature_cols)),
        "n_targets": int(len(target_cols)),
        "subjects_total": int(pd.Series(groups).nunique()),
        "subjects_train": int(pd.Series(groups[train_idx]).nunique()),
        "subjects_test": int(pd.Series(groups[test_idx]).nunique()),
    }

    print(f"Dataset info: {dataset_info}")

    metrics_real, _ = train_and_evaluate(
        x_train_raw=x_train_real,
        y_train=y_train_real,
        x_test_raw=x_test_real,
        y_test=y_test_real,
        target_cols=target_cols,
        experiment="real_only",
        alpha=args.ridge_alpha,
    )

    metrics_aug, _ = train_and_evaluate(
        x_train_raw=x_aug,
        y_train=y_aug,
        x_test_raw=x_test_real,
        y_test=y_test_real,
        target_cols=target_cols,
        experiment="real_plus_synthetic",
        alpha=args.ridge_alpha,
    )

    metrics_df = pd.concat([metrics_real, metrics_aug], ignore_index=True)
    delta_df = make_delta_table(metrics_df)

    run_dir = (
        root
        / "reports"
        / "runs"
        / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}_{args.feature_set}"
    )
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    metrics_df.to_csv(run_dir / "augmentation_metrics.csv", index=False, encoding="utf-8-sig")
    delta_df.to_csv(run_dir / "augmentation_delta.csv", index=False, encoding="utf-8-sig")

    with (run_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)

    plot_metric_comparison(metrics_df, "r2", figures_dir / "augmentation_r2_comparison.png")
    plot_metric_comparison(metrics_df, "spearman", figures_dir / "augmentation_spearman_comparison.png")
    plot_metric_comparison(metrics_df, "mae", figures_dir / "augmentation_mae_comparison.png")

    write_report(
        output_path=run_dir / "report.md",
        args=args,
        ae_run_dir=ae_run_dir,
        dataset_info=dataset_info,
        metrics_df=metrics_df,
        delta_df=delta_df,
    )

    print("=" * 80)
    print("Saved outputs")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Metrics: {run_dir / 'augmentation_metrics.csv'}")
    print(f"Delta: {run_dir / 'augmentation_delta.csv'}")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Figures: {figures_dir}")
    print("=" * 80)
    print("Delta table")
    print("=" * 80)
    print(delta_df.to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()