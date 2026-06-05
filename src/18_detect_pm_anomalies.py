from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]

TIME_COLUMNS_CANDIDATES = [
    "t_center",
    "t_start",
    "datetime_from_name",
]


def infer_time_column(df: pd.DataFrame) -> str | None:
    for col in TIME_COLUMNS_CANDIDATES:
        if col in df.columns:
            return col
    return None


def robust_zscore(values: pd.Series, eps: float = 1e-9) -> pd.Series:
    median = values.median()
    mad = (values - median).abs().median()
    scale = 1.4826 * mad + eps
    return (values - median) / scale


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


def prepare_dataframe(
    df: pd.DataFrame,
    target_cols: List[str],
    feature_cols: List[str],
    record_col: str,
    group_col: str,
) -> pd.DataFrame:
    time_col = infer_time_column(df)

    keep_cols = [record_col, group_col] + target_cols + feature_cols
    if time_col is not None and time_col not in keep_cols:
        keep_cols.append(time_col)

    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    out = df[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=[record_col, group_col] + target_cols).reset_index(drop=True)

    for col in feature_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    sort_cols = [record_col]
    if time_col is not None:
        sort_cols.append(time_col)

    out = out.sort_values(sort_cols).reset_index(drop=True)
    return out


def compute_target_anomaly_scores(
    df: pd.DataFrame,
    target_cols: List[str],
    record_col: str,
    rolling_window: int,
) -> pd.DataFrame:
    frames = []

    for record_id, g in df.groupby(record_col, sort=False):
        g = g.copy().reset_index(drop=False).rename(columns={"index": "original_index"})

        if len(g) < max(5, rolling_window):
            continue

        score_parts = []

        for target in target_cols:
            value_col = f"{target}_value"
            diff_col = f"{target}_abs_diff"
            rz_col = f"{target}_robust_z"
            diff_rz_col = f"{target}_diff_robust_z"
            roll_col = f"{target}_rolling_dev"
            roll_rz_col = f"{target}_rolling_dev_robust_z"

            g[value_col] = g[target]
            g[diff_col] = g[target].diff().abs().fillna(0.0)
            g[rz_col] = robust_zscore(g[target]).abs()
            g[diff_rz_col] = robust_zscore(g[diff_col]).abs()

            rolling_mean = (
                g[target]
                .rolling(window=rolling_window, center=True, min_periods=max(3, rolling_window // 2))
                .mean()
            )
            rolling_mean = rolling_mean.fillna(g[target].median())
            g[roll_col] = (g[target] - rolling_mean).abs()
            g[roll_rz_col] = robust_zscore(g[roll_col]).abs()

            score_parts.extend([rz_col, diff_rz_col, roll_rz_col])

        g["pm_anomaly_score"] = g[score_parts].mean(axis=1)
        g["pm_anomaly_score_max"] = g[score_parts].max(axis=1)

        frames.append(g)

    if not frames:
        raise ValueError("No records were long enough for anomaly scoring.")

    scored = pd.concat(frames, ignore_index=True)
    return scored


def compute_feature_anomaly_score(
    df: pd.DataFrame,
    feature_cols: List[str],
    batch_size: int = 64,
) -> pd.Series:
    if not feature_cols:
        return pd.Series(np.zeros(len(df)), index=df.index)

    feature_df = df[feature_cols].copy()
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        if feature_df[col].isna().any():
            feature_df[col] = feature_df[col].fillna(feature_df[col].median())

    med = feature_df.median(axis=0)
    mad = (feature_df - med).abs().median(axis=0)
    scale = 1.4826 * mad + 1e-9

    scores = []

    values = feature_df.to_numpy(dtype=np.float32)
    med_np = med.to_numpy(dtype=np.float32)
    scale_np = scale.to_numpy(dtype=np.float32)

    for start in range(0, len(values), batch_size):
        chunk = values[start : start + batch_size]
        z = np.abs((chunk - med_np) / scale_np)
        z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        score = np.percentile(z, 95, axis=1)
        scores.append(score)

    return pd.Series(np.concatenate(scores), index=df.index)


def mark_anomalies(
    scored: pd.DataFrame,
    threshold_quantile: float,
    min_score: float,
) -> pd.DataFrame:
    out = scored.copy()

    threshold = float(out["combined_anomaly_score"].quantile(threshold_quantile))
    threshold = max(threshold, min_score)

    out["is_anomaly"] = out["combined_anomaly_score"] >= threshold
    out["anomaly_threshold"] = threshold

    return out


def make_summary(
    anomalies: pd.DataFrame,
    target_cols: List[str],
    record_col: str,
    group_col: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    summary = {
        "rows_scored": int(len(anomalies)),
        "anomaly_rows": int(anomalies["is_anomaly"].sum()),
        "anomaly_rate": float(anomalies["is_anomaly"].mean()),
        "records": int(anomalies[record_col].nunique()),
        "subjects": int(anomalies[group_col].nunique()),
        "threshold": float(anomalies["anomaly_threshold"].iloc[0]),
        "mean_score": float(anomalies["combined_anomaly_score"].mean()),
        "median_score": float(anomalies["combined_anomaly_score"].median()),
        "max_score": float(anomalies["combined_anomaly_score"].max()),
    }

    summary_df = pd.DataFrame([summary])

    record_summary = (
        anomalies.groupby(record_col)
        .agg(
            rows=(record_col, "size"),
            subject_id=(group_col, "first"),
            anomaly_rows=("is_anomaly", "sum"),
            anomaly_rate=("is_anomaly", "mean"),
            mean_score=("combined_anomaly_score", "mean"),
            max_score=("combined_anomaly_score", "max"),
        )
        .reset_index()
        .sort_values(["anomaly_rows", "max_score"], ascending=False)
    )

    return summary_df, record_summary


def plot_score_distribution(anomalies: pd.DataFrame, output_path: Path) -> None:
    threshold = float(anomalies["anomaly_threshold"].iloc[0])

    plt.figure(figsize=(9, 5))
    plt.hist(anomalies["combined_anomaly_score"], bins=80, alpha=0.85)
    plt.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    plt.xlabel("Combined anomaly score")
    plt.ylabel("Count")
    plt.title("Anomaly score distribution")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top_records(record_summary: pd.DataFrame, output_path: Path, top_n: int) -> None:
    top = record_summary.head(top_n).copy()
    top["record_id"] = top["record_id"].astype(str)

    plt.figure(figsize=(10, 6))
    plt.barh(top["record_id"], top["anomaly_rows"])
    plt.xlabel("Number of anomalous windows")
    plt.ylabel("Record ID")
    plt.title(f"Top {top_n} records by anomalous windows")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_timeline(
    anomalies: pd.DataFrame,
    target: str,
    record_col: str,
    output_path: Path,
) -> None:
    record_counts = (
        anomalies.groupby(record_col)["is_anomaly"]
        .sum()
        .sort_values(ascending=False)
    )

    if record_counts.empty:
        return

    top_record = record_counts.index[0]
    g = anomalies[anomalies[record_col] == top_record].copy().reset_index(drop=True)

    x = np.arange(len(g))
    anomaly_mask = g["is_anomaly"].to_numpy(dtype=bool)

    plt.figure(figsize=(12, 6))
    plt.plot(x, g[target], label=target, linewidth=1.2)
    plt.scatter(
        x[anomaly_mask],
        g.loc[anomaly_mask, target],
        s=25,
        label="anomaly",
        zorder=3,
    )
    plt.xlabel("Window index")
    plt.ylabel(target)
    plt.title(f"Anomalous windows in record {top_record}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_score_timeline(
    anomalies: pd.DataFrame,
    record_col: str,
    output_path: Path,
) -> None:
    record_counts = (
        anomalies.groupby(record_col)["is_anomaly"]
        .sum()
        .sort_values(ascending=False)
    )

    if record_counts.empty:
        return

    top_record = record_counts.index[0]
    g = anomalies[anomalies[record_col] == top_record].copy().reset_index(drop=True)

    x = np.arange(len(g))
    threshold = float(g["anomaly_threshold"].iloc[0])

    plt.figure(figsize=(12, 5))
    plt.plot(x, g["combined_anomaly_score"], label="combined anomaly score", linewidth=1.2)
    plt.axhline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.3f}")
    plt.xlabel("Window index")
    plt.ylabel("Anomaly score")
    plt.title(f"Anomaly score timeline in record {top_record}")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    args: argparse.Namespace,
    dataset_info: Dict[str, object],
    summary_df: pd.DataFrame,
    record_summary: pd.DataFrame,
    target_cols: List[str],
    feature_cols: List[str],
) -> None:
    summary = summary_df.iloc[0].to_dict()

    lines = []
    lines.append("# PM/EEG Anomaly Detection Report")
    lines.append("")
    lines.append("## 1. Цель")
    lines.append("")
    lines.append(
        "Цель этого блока — закрыть задачу обнаружения паттернов или аномальных эпизодов "
        "в длинных EEG/PM-записях. Аномалии ищутся без ручной разметки, по резким изменениям "
        "PM-метрик и статистически необычным EEG/POW-признакам."
    )
    lines.append("")
    lines.append("## 2. Метод")
    lines.append("")
    lines.append("Для каждого окна рассчитываются несколько компонент anomaly score:")
    lines.append("")
    lines.append("- robust z-score значения PM-метрики внутри записи;")
    lines.append("- robust z-score абсолютного изменения PM-метрики между соседними окнами;")
    lines.append("- отклонение PM-метрики от локального rolling mean;")
    lines.append("- robust anomaly score по EEG/POW-признакам.")
    lines.append("")
    lines.append(
        "Итоговый `combined_anomaly_score` — это смесь PM anomaly score и feature anomaly score. "
        "Окна выше заданного квантильного порога помечаются как аномальные."
    )
    lines.append("")
    lines.append("## 3. Конфигурация")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(vars(args), ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 4. Dataset info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 5. Summary")
    lines.append("")
    lines.append(summary_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Top records by anomaly count")
    lines.append("")
    lines.append(record_summary.head(15).to_markdown(index=False))
    lines.append("")
    lines.append("## 7. Использованные PM targets")
    lines.append("")
    for target in target_cols:
        lines.append(f"- `{target}`")
    lines.append("")
    lines.append("## 8. Feature set")
    lines.append("")
    lines.append(f"Number of feature columns: `{len(feature_cols)}`")
    lines.append("")
    lines.append("## 9. Интерпретация")
    lines.append("")
    lines.append(
        f"Было оценено {int(summary['rows_scored'])} окон. "
        f"Аномальными отмечено {int(summary['anomaly_rows'])} окон "
        f"({summary['anomaly_rate']:.3%})."
    )
    lines.append("")
    lines.append(
        "Такие окна можно интерпретировать как участки с резкой сменой состояния, "
        "нестабильной динамикой PM-метрик или необычным EEG/POW-профилем. "
        "Это не клиническая диагностика, а unsupervised способ выделения кандидатов "
        "на нестабильные или артефактные эпизоды для дальнейшего анализа."
    )
    lines.append("")
    lines.append("## 10. Ограничения")
    lines.append("")
    lines.append("- Метод не использует экспертную разметку аномалий.")
    lines.append("- Аномальность определяется статистически, а не клинически.")
    lines.append("- Feature anomaly score зависит от выбранного признакового пространства.")
    lines.append("- Порог выбирается по квантилю и может быть изменён.")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".", type=str)
    parser.add_argument(
        "--dataset",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
        type=str,
    )
    parser.add_argument("--run-name", default="pm_anomaly_detection", type=str)
    parser.add_argument(
        "--feature-set",
        default="pow_plus_eeg",
        choices=["pow", "eeg", "pow_plus_eeg"],
    )
    parser.add_argument("--record-col", default="record_id", type=str)
    parser.add_argument("--group-col", default="subject_id", type=str)
    parser.add_argument("--rolling-window", default=7, type=int)
    parser.add_argument("--threshold-quantile", default=0.98, type=float)
    parser.add_argument("--min-score", default=3.0, type=float)
    parser.add_argument("--feature-score-weight", default=0.35, type=float)
    parser.add_argument("--max-rows", default=None, type=int)
    parser.add_argument("--top-n-records", default=15, type=int)
    parser.add_argument("--timeline-target", default="target_focus", type=str)
    parser.add_argument("--seed", default=42, type=int)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    dataset_path = root / args.dataset

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print("=" * 80)
    print("Detect PM/EEG anomalies")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")

    df = pd.read_parquet(dataset_path)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled max_rows={args.max_rows}")

    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    if not target_cols:
        raise ValueError("No target_* columns found.")

    if args.timeline_target not in target_cols:
        raise ValueError(
            f"timeline_target={args.timeline_target} is not available. "
            f"Available targets: {target_cols}"
        )

    feature_cols = get_feature_columns(df, args.feature_set)

    prepared = prepare_dataframe(
        df=df,
        target_cols=target_cols,
        feature_cols=feature_cols,
        record_col=args.record_col,
        group_col=args.group_col,
    )

    dataset_info = {
        "rows_raw": int(len(df)),
        "rows_prepared": int(len(prepared)),
        "columns_raw": int(df.shape[1]),
        "records": int(prepared[args.record_col].nunique()),
        "subjects": int(prepared[args.group_col].nunique()),
        "targets": target_cols,
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
    }

    print(f"Dataset info: {dataset_info}")

    scored = compute_target_anomaly_scores(
        df=prepared,
        target_cols=target_cols,
        record_col=args.record_col,
        rolling_window=args.rolling_window,
    )

    feature_score = compute_feature_anomaly_score(
        df=scored,
        feature_cols=feature_cols,
    )

    scored["feature_anomaly_score"] = feature_score.to_numpy()
    scored["pm_anomaly_score_norm"] = robust_zscore(scored["pm_anomaly_score"]).abs()
    scored["feature_anomaly_score_norm"] = robust_zscore(scored["feature_anomaly_score"]).abs()

    feature_weight = float(args.feature_score_weight)
    feature_weight = min(max(feature_weight, 0.0), 1.0)
    pm_weight = 1.0 - feature_weight

    scored["combined_anomaly_score"] = (
        pm_weight * scored["pm_anomaly_score_norm"]
        + feature_weight * scored["feature_anomaly_score_norm"]
    )

    anomalies = mark_anomalies(
        scored=scored,
        threshold_quantile=args.threshold_quantile,
        min_score=args.min_score,
    )

    run_dir = (
        root
        / "reports"
        / "runs"
        / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}_{args.feature_set}"
    )
    figures_dir = run_dir / "figures"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    summary_df, record_summary = make_summary(
        anomalies=anomalies,
        target_cols=target_cols,
        record_col=args.record_col,
        group_col=args.group_col,
    )

    anomaly_columns = [
        args.record_col,
        args.group_col,
        "original_index",
        "pm_anomaly_score",
        "pm_anomaly_score_max",
        "feature_anomaly_score",
        "combined_anomaly_score",
        "anomaly_threshold",
        "is_anomaly",
    ]

    time_col = infer_time_column(anomalies)
    if time_col is not None and time_col not in anomaly_columns:
        anomaly_columns.insert(2, time_col)

    anomaly_columns += target_cols

    anomalies[anomaly_columns].to_csv(
        run_dir / "anomaly_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    anomalies[anomalies["is_anomaly"]][anomaly_columns].to_csv(
        run_dir / "top_anomaly_windows.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary_df.to_csv(run_dir / "anomaly_summary.csv", index=False, encoding="utf-8-sig")
    record_summary.to_csv(run_dir / "record_anomaly_summary.csv", index=False, encoding="utf-8-sig")

    plot_score_distribution(
        anomalies=anomalies,
        output_path=figures_dir / "anomaly_score_distribution.png",
    )

    plot_top_records(
        record_summary=record_summary,
        output_path=figures_dir / "top_records_by_anomaly_count.png",
        top_n=args.top_n_records,
    )

    plot_timeline(
        anomalies=anomalies,
        target=args.timeline_target,
        record_col=args.record_col,
        output_path=figures_dir / f"timeline_{args.timeline_target}.png",
    )

    plot_score_timeline(
        anomalies=anomalies,
        record_col=args.record_col,
        output_path=figures_dir / "anomaly_score_timeline.png",
    )

    write_report(
        output_path=run_dir / "report.md",
        args=args,
        dataset_info=dataset_info,
        summary_df=summary_df,
        record_summary=record_summary,
        target_cols=target_cols,
        feature_cols=feature_cols,
    )

    print("=" * 80)
    print("Saved outputs")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Anomaly windows: {run_dir / 'anomaly_windows.csv'}")
    print(f"Top anomaly windows: {run_dir / 'top_anomaly_windows.csv'}")
    print(f"Summary: {run_dir / 'anomaly_summary.csv'}")
    print(f"Record summary: {run_dir / 'record_anomaly_summary.csv'}")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Figures: {figures_dir}")
    print("=" * 80)
    print("Summary")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)
    print("Top records")
    print("=" * 80)
    print(record_summary.head(args.top_n_records).to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()