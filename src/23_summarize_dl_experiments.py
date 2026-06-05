from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd


TARGET_ORDER = [
    "target_excitement",
    "target_relaxation",
    "target_engagement",
    "target_interest",
    "target_focus",
    "target_stress",
    "target_attention",
]


def find_latest_run(runs_dir: Path, pattern: str) -> Optional[Path]:
    candidates = [p for p in runs_dir.iterdir() if p.is_dir() and pattern in p.name]
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def read_metrics(run_dir: Path, model_name: str) -> pd.DataFrame:
    path = run_dir / "target_metrics_agg.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")

    df = pd.read_csv(path)

    if "target_name" in df.columns and "target" not in df.columns:
        df = df.rename(columns={"target_name": "target"})

    required = [
        "target",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
        "pearson_mean",
        "spearman_mean",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    out = df[required].copy()
    out.insert(0, "model_family", model_name)
    out.insert(1, "run_dir", run_dir.name)

    return out


def read_autoencoder_metrics(run_dir: Path) -> pd.DataFrame:
    reconstruction_path = run_dir / "reconstruction_metrics.csv"
    real_synth_path = run_dir / "real_vs_synthetic_metrics.csv"

    if not reconstruction_path.exists():
        raise FileNotFoundError(f"Missing reconstruction metrics: {reconstruction_path}")

    if not real_synth_path.exists():
        raise FileNotFoundError(f"Missing real-vs-synthetic metrics: {real_synth_path}")

    reconstruction = pd.read_csv(reconstruction_path)
    real_synth = pd.read_csv(real_synth_path)

    combined = pd.concat([reconstruction, real_synth], axis=1)
    combined.insert(0, "run_dir", run_dir.name)
    combined.insert(0, "model_family", "Autoencoder")

    return combined


def make_best_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for target, g in comparison_df.groupby("target"):
        best_r2 = g.sort_values("r2_mean", ascending=False).iloc[0]
        best_spearman = g.sort_values("spearman_mean", ascending=False).iloc[0]

        rows.append(
            {
                "target": target,
                "best_by_r2": best_r2["model_family"],
                "best_r2": best_r2["r2_mean"],
                "best_by_spearman": best_spearman["model_family"],
                "best_spearman": best_spearman["spearman_mean"],
            }
        )

    out = pd.DataFrame(rows)
    out["target"] = pd.Categorical(out["target"], categories=TARGET_ORDER, ordered=True)
    out = out.sort_values("target").reset_index(drop=True)

    return out


def plot_metric_grouped(
    comparison_df: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    pivot = comparison_df.pivot_table(
        index="target",
        columns="model_family",
        values=metric,
        aggfunc="mean",
    )

    ordered_targets = [t for t in TARGET_ORDER if t in pivot.index]
    pivot = pivot.loc[ordered_targets]

    plt.figure(figsize=(12, 6))
    pivot.plot(kind="bar", figsize=(12, 6))
    plt.title(title)
    plt.xlabel("Target")
    plt.ylabel(metric)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_best_r2(best_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = best_df.copy()
    plot_df["target_short"] = plot_df["target"].str.replace("target_", "", regex=False)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["target_short"], plot_df["best_r2"])
    plt.xlabel("Best R²")
    plt.title("Best model per PM target by R²")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    output_path: Path,
    run_map: Dict[str, Optional[Path]],
    comparison_df: pd.DataFrame,
    best_df: pd.DataFrame,
    ae_df: Optional[pd.DataFrame],
) -> None:
    lines: List[str] = []

    lines.append("# Final Exam Report Draft")
    lines.append("")
    lines.append("## 1. Project goal")
    lines.append("")
    lines.append(
        "The project studies EEG-based prediction of cognitive and affective Performance Metrics "
        "using EEG/POW feature windows and deep learning models with local temporal context."
    )
    lines.append("")
    lines.append("The main PM targets are:")
    lines.append("")
    for target in TARGET_ORDER:
        if target in set(comparison_df["target"]):
            lines.append(f"- `{target}`")
    lines.append("")
    lines.append("## 2. Compared models")
    lines.append("")
    lines.append("The current version compares the following model families:")
    lines.append("")
    lines.append("- Classical ML baseline: tree-based and linear tabular models.")
    lines.append("- MLP: multi-output neural baseline on single-window EEG/POW features.")
    lines.append("- LSTM: sequence model using local temporal context.")
    lines.append("- TransformerEncoder: attention-based sequence model using local temporal context.")
    lines.append("- Autoencoder: reconstruction and synthetic generation of EEG/POW feature vectors.")
    lines.append("")
    lines.append("## 3. Run directories")
    lines.append("")
    for name, path in run_map.items():
        if path is None:
            lines.append(f"- {name}: not found")
        else:
            lines.append(f"- {name}: `{path.name}`")
    lines.append("")
    lines.append("## 4. Main model comparison")
    lines.append("")
    cols = [
        "model_family",
        "target",
        "mae_mean",
        "rmse_mean",
        "r2_mean",
        "pearson_mean",
        "spearman_mean",
    ]
    report_df = comparison_df[cols].copy()
    report_df["target"] = pd.Categorical(report_df["target"], categories=TARGET_ORDER, ordered=True)
    report_df = report_df.sort_values(["target", "model_family"])
    lines.append(report_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 5. Best model per target")
    lines.append("")
    lines.append(best_df.to_markdown(index=False))
    lines.append("")
    lines.append("## 6. Main findings")
    lines.append("")
    lines.append(
        "The MLP model serves as a neural tabular baseline. It confirms that simply replacing "
        "classical models with a fully connected network is not sufficient for this task."
    )
    lines.append("")
    lines.append(
        "The LSTM model improves the results for several targets by using neighboring EEG/POW windows. "
        "This supports the hypothesis that local temporal dynamics are informative for PM prediction."
    )
    lines.append("")
    lines.append(
        "The TransformerEncoder provides the strongest DL result in the current project. "
        "It improves several targets compared with the single-window classical baseline, especially "
        "`target_excitement`, `target_relaxation`, `target_interest`, and `target_focus`."
    )
    lines.append("")
    lines.append(
        "Some targets, such as `target_engagement` and `target_attention`, are still better predicted "
        "by classical models. This suggests that the benefit of temporal attention is target-dependent."
    )
    lines.append("")
    lines.append("## 7. Autoencoder generation block")
    lines.append("")
    if ae_df is None or ae_df.empty:
        lines.append("Autoencoder metrics were not found.")
    else:
        ae_row = ae_df.iloc[0].to_dict()
        important_keys = [
            "scaled_reconstruction_mse",
            "scaled_reconstruction_rmse",
            "scaled_reconstruction_mae",
            "scaled_reconstruction_cosine_mean",
            "scaled_feature_mean_abs_diff_mean",
            "scaled_feature_std_abs_diff_mean",
            "scaled_real_synthetic_pairwise_cosine_mean",
        ]

        lines.append("The Autoencoder was trained on standardized EEG/POW feature vectors.")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|---|---:|")
        for key in important_keys:
            if key in ae_row:
                lines.append(f"| `{key}` | {ae_row[key]:.6f} |")

        lines.append("")
        lines.append(
            "The generation is performed in EEG/POW feature space rather than raw EEG waveform space. "
            "Synthetic vectors are obtained by perturbing latent representations and decoding them. "
            "This is a lightweight approximation suitable for the current exam project."
        )

    lines.append("")
    lines.append("## 8. Limitations")
    lines.append("")
    lines.append("- The project uses device-derived PM targets rather than independent expert labels.")
    lines.append("- Synthetic generation is performed in feature space, not directly in raw EEG signal space.")
    lines.append("- GroupKFold by subject is much stricter than random split and exposes inter-subject variability.")
    lines.append("- The current models do not explicitly model electrode topology.")
    lines.append("")
    lines.append("## 9. Next steps")
    lines.append("")
    lines.append("- Add a cleaner final notebook for demonstration.")
    lines.append("- Add a short defense text.")
    lines.append("- Optionally add error-based anomaly detection from model residuals.")
    lines.append("- Optionally test synthetic augmentation in a downstream predictor.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".", type=str)
    parser.add_argument("--reports-dir", default="reports", type=str)
    parser.add_argument("--output-dir", default="reports/final_summary", type=str)

    parser.add_argument(
        "--classical-pattern",
        default="multi_pm_groupkfold_full",
        type=str,
    )
    parser.add_argument(
        "--mlp-pattern",
        default="mlp_pm_groupkfold_full",
        type=str,
    )
    parser.add_argument(
        "--lstm-pattern",
        default="lstm_pm_groupkfold_full",
        type=str,
    )
    parser.add_argument(
        "--transformer-pattern",
        default="transformer_pm_groupkfold_full",
        type=str,
    )
    parser.add_argument(
        "--autoencoder-pattern",
        default="ae_pow_plus_eeg_full",
        type=str,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = Path(args.root).resolve()
    reports_dir = root / args.reports_dir
    runs_dir = reports_dir / "runs"
    output_dir = root / args.output_dir
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"

    if not runs_dir.exists():
        raise FileNotFoundError(f"Runs directory not found: {runs_dir}")

    run_map = {
        "Classical baseline": find_latest_run(runs_dir, args.classical_pattern),
        "MLP": find_latest_run(runs_dir, args.mlp_pattern),
        "LSTM": find_latest_run(runs_dir, args.lstm_pattern),
        "TransformerEncoder": find_latest_run(runs_dir, args.transformer_pattern),
        "Autoencoder": find_latest_run(runs_dir, args.autoencoder_pattern),
    }

    print("=" * 80)
    print("Detected runs")
    print("=" * 80)
    for name, path in run_map.items():
        print(f"{name}: {path}")

    metric_frames = []

    model_metric_names = [
        "Classical baseline",
        "MLP",
        "LSTM",
        "TransformerEncoder",
    ]

    for model_name in model_metric_names:
        run_dir = run_map[model_name]
        if run_dir is None:
            print(f"WARNING: missing run for {model_name}")
            continue

        metric_frames.append(read_metrics(run_dir, model_name))

    if not metric_frames:
        raise RuntimeError("No model metrics found.")

    comparison_df = pd.concat(metric_frames, ignore_index=True)
    comparison_df["target"] = pd.Categorical(
        comparison_df["target"],
        categories=TARGET_ORDER,
        ordered=True,
    )
    comparison_df = comparison_df.sort_values(["target", "model_family"]).reset_index(drop=True)

    best_df = make_best_table(comparison_df)

    ae_df = None
    if run_map["Autoencoder"] is not None:
        ae_df = read_autoencoder_metrics(run_map["Autoencoder"])

    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = tables_dir / "final_model_comparison.csv"
    best_path = tables_dir / "best_model_by_target.csv"
    run_map_path = tables_dir / "detected_runs.json"

    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_path, index=False, encoding="utf-8-sig")

    with run_map_path.open("w", encoding="utf-8") as f:
        json.dump(
            {k: (str(v) if v is not None else None) for k, v in run_map.items()},
            f,
            ensure_ascii=False,
            indent=2,
        )

    if ae_df is not None:
        ae_df.to_csv(tables_dir / "autoencoder_summary_metrics.csv", index=False, encoding="utf-8-sig")

    plot_metric_grouped(
        comparison_df=comparison_df,
        metric="r2_mean",
        output_path=figures_dir / "final_model_comparison_r2.png",
        title="Model comparison by R²",
    )

    plot_metric_grouped(
        comparison_df=comparison_df,
        metric="spearman_mean",
        output_path=figures_dir / "final_model_comparison_spearman.png",
        title="Model comparison by Spearman correlation",
    )

    plot_best_r2(
        best_df=best_df,
        output_path=figures_dir / "best_r2_by_target.png",
    )

    write_report(
        output_path=output_dir / "final_exam_report.md",
        run_map=run_map,
        comparison_df=comparison_df,
        best_df=best_df,
        ae_df=ae_df,
    )

    print("=" * 80)
    print("Saved final summary")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(f"Comparison table: {comparison_path}")
    print(f"Best model table: {best_path}")
    print(f"Report: {output_dir / 'final_exam_report.md'}")
    print(f"Figures: {figures_dir}")
    print("=" * 80)
    print("Best model by target")
    print("=" * 80)
    print(best_df.to_string(index=False))


if __name__ == "__main__":
    main()