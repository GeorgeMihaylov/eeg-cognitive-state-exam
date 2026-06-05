from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TARGET_COLUMNS = [
    "target_attention",
    "target_engagement",
    "target_excitement",
    "target_stress",
    "target_relaxation",
    "target_interest",
    "target_focus",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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


def clean_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    group_col: str,
    min_windows_per_subject: int,
) -> pd.DataFrame:
    required = feature_cols + target_cols + [group_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    out = df[required].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=target_cols + [group_col])

    for c in feature_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    counts = out[group_col].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    out = out[out[group_col].isin(keep_subjects)].reset_index(drop=True)

    return out


class MLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Tuple[int, ...] = (512, 256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[MLPRegressor, Dict[str, List[float]], np.ndarray]:
    model = MLPRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(x_train, y_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, y_val, args.batch_size, shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    use_amp = args.amp and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    history = {
        "train_loss": [],
        "val_loss": [],
    }

    best_val = math.inf
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(xb)
                loss = criterion(pred, yb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []
        preds = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    pred = model(xb)
                    loss = criterion(pred, yb)

                val_losses.append(float(loss.detach().cpu().item()))
                preds.append(pred.detach().cpu().numpy())

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"epoch={epoch:03d} | train_loss={train_loss:.6f} | "
                f"val_loss={val_loss:.6f}"
            )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch={epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    final_preds = []

    with torch.no_grad():
        for xb, _ in val_loader:
            xb = xb.to(device, non_blocking=True)
            pred = model(xb)
            final_preds.append(pred.detach().cpu().numpy())

    y_pred = np.vstack(final_preds)
    return model, history, y_pred


def safe_corr(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.std(y_true) < 1e-12 or np.std(y_pred) < 1e-12:
        return float("nan")
    try:
        value = func(y_true, y_pred)[0]
        return float(value)
    except Exception:
        return float("nan")


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_cols: List[str],
    fold: int,
) -> List[Dict[str, float]]:
    rows = []

    for idx, target in enumerate(target_cols):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        mse = mean_squared_error(yt, yp)
        rmse = math.sqrt(mse)
        mae = mean_absolute_error(yt, yp)
        r2 = r2_score(yt, yp)
        pearson = safe_corr(pearsonr, yt, yp)
        spearman = safe_corr(spearmanr, yt, yp)

        rows.append(
            {
                "fold": fold,
                "target": target,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "pearson": pearson,
                "spearman": spearman,
            }
        )

    return rows


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    agg = (
        metrics_df.groupby("target")
        .agg(
            folds=("fold", "nunique"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            pearson_mean=("pearson", "mean"),
            pearson_std=("pearson", "std"),
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
        )
        .reset_index()
        .sort_values("r2_mean", ascending=False)
    )
    return agg


def plot_loss(histories: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for fold, hist in histories.items():
        plt.plot(hist["train_loss"], linestyle="--", alpha=0.7, label=f"fold {fold} train")
        plt.plot(hist["val_loss"], alpha=0.9, label=f"fold {fold} val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("MLP PM regressor training curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_r2(agg_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = agg_df.sort_values("r2_mean", ascending=True)

    plt.figure(figsize=(9, 5))
    plt.barh(plot_df["target"], plot_df["r2_mean"])
    plt.xlabel("Mean R²")
    plt.title("MLP PM regressor: GroupKFold R² by target")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_info: Dict[str, object],
    feature_info: Dict[str, object],
    agg_df: pd.DataFrame,
    device: torch.device,
) -> None:
    lines = []
    lines.append("# MLP PM Regressor Report")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Feature set: `{args.feature_set}`")
    lines.append(f"- Validation: GroupKFold by `{args.group_col}`")
    lines.append(f"- Folds: `{args.n_splits}`")
    lines.append(f"- Device: `{device}`")
    lines.append(f"- Epochs: `{args.epochs}`")
    lines.append(f"- Batch size: `{args.batch_size}`")
    lines.append(f"- AMP: `{args.amp}`")
    lines.append("")
    lines.append("## Dataset info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(dataset_info, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Feature info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(feature_info, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Aggregated metrics")
    lines.append("")
    lines.append(agg_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This experiment adds a neural-network baseline to the project. "
        "The model is a multi-output MLP regressor trained on EEG/POW window features. "
        "The main validation scheme is GroupKFold by subject_id, which estimates transfer "
        "to unseen users more strictly than a random split."
    )

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--root", default=".", type=str)
    parser.add_argument(
        "--dataset",
        default="data/processed/windowed_eeg_pm_dataset_w10.parquet",
        type=str,
    )
    parser.add_argument("--run-name", default="mlp_pm_regressor", type=str)
    parser.add_argument(
        "--feature-set",
        default="pow_plus_eeg",
        choices=["pow", "eeg", "pow_plus_eeg"],
    )
    parser.add_argument("--group-col", default="subject_id", type=str)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--min-windows-per-subject", default=30, type=int)
    parser.add_argument("--max-rows", default=None, type=int)

    parser.add_argument("--hidden-dims", nargs="+", default=[512, 256, 128], type=int)
    parser.add_argument("--dropout", default=0.2, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--log-every", default=5, type=int)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    root = Path(args.root).resolve()
    dataset_path = root / args.dataset

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    device = torch.device(
        "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    )

    print("=" * 80)
    print("Train MLP PM regressor")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    df = pd.read_parquet(dataset_path)

    if args.max_rows is not None and len(df) > args.max_rows:
        df = df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)
        print(f"Sampled max_rows={args.max_rows}")

    feature_cols = get_feature_columns(df, args.feature_set)
    target_cols = [c for c in TARGET_COLUMNS if c in df.columns]

    if len(target_cols) != len(TARGET_COLUMNS):
        missing = [c for c in TARGET_COLUMNS if c not in df.columns]
        raise ValueError(f"Missing target columns: {missing}")

    clean_df = clean_dataframe(
        df=df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col=args.group_col,
        min_windows_per_subject=args.min_windows_per_subject,
    )

    dataset_info = {
        "rows_raw": int(len(df)),
        "rows_clean": int(len(clean_df)),
        "columns_raw": int(df.shape[1]),
        "subjects": int(clean_df[args.group_col].nunique()),
        "targets": target_cols,
    }

    feature_info = {
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }

    print(f"Dataset info: {dataset_info}")
    print(f"Feature info: {feature_info}")

    x = clean_df[feature_cols].to_numpy(dtype=np.float32)
    y = clean_df[target_cols].to_numpy(dtype=np.float32)
    groups = clean_df[args.group_col].to_numpy()

    run_dir = (
        root
        / "reports"
        / "runs"
        / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}_{args.feature_set}"
    )
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    gkf = GroupKFold(n_splits=args.n_splits)

    all_metrics = []
    histories: Dict[int, Dict[str, List[float]]] = {}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        print("-" * 80)
        print(f"Fold {fold}/{args.n_splits}")
        print(f"Train rows: {len(train_idx)} | Val rows: {len(val_idx)}")

        x_train_raw, x_val_raw = x[train_idx], x[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]

        x_scaler = StandardScaler()
        y_scaler = StandardScaler()

        x_train = x_scaler.fit_transform(x_train_raw).astype(np.float32)
        x_val = x_scaler.transform(x_val_raw).astype(np.float32)

        y_train = y_scaler.fit_transform(y_train_raw).astype(np.float32)
        y_val = y_scaler.transform(y_val_raw).astype(np.float32)

        model, history, y_pred_scaled = train_one_fold(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            input_dim=x_train.shape[1],
            output_dim=y_train.shape[1],
            device=device,
            args=args,
        )

        y_pred = y_scaler.inverse_transform(y_pred_scaled)
        fold_metrics = compute_metrics(y_val_raw, y_pred, target_cols, fold)
        all_metrics.extend(fold_metrics)
        histories[fold] = history

        fold_metrics_df = pd.DataFrame(fold_metrics)
        print(
            fold_metrics_df.sort_values("r2", ascending=False)[
                ["target", "mae", "rmse", "r2", "pearson", "spearman"]
            ].to_string(index=False)
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "feature_cols": feature_cols,
                "target_cols": target_cols,
                "args": vars(args),
            },
            models_dir / f"fold_{fold}_model.pt",
        )

    metrics_df = pd.DataFrame(all_metrics)
    agg_df = aggregate_metrics(metrics_df)

    metrics_path = run_dir / "fold_metrics.csv"
    agg_path = run_dir / "target_metrics_agg.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    agg_df.to_csv(agg_path, index=False, encoding="utf-8-sig")

    plot_loss(histories, figures_dir / "loss_curves.png")
    plot_r2(agg_df, figures_dir / "r2_by_target.png")

    write_report(
        args=args,
        run_dir=run_dir,
        dataset_info=dataset_info,
        feature_info=feature_info,
        agg_df=agg_df,
        device=device,
    )

    print("=" * 80)
    print("Saved outputs")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Fold metrics: {metrics_path}")
    print(f"Aggregated metrics: {agg_path}")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Figures: {figures_dir}")
    print("=" * 80)
    print("Aggregated metrics")
    print("=" * 80)
    print(
        agg_df[
            [
                "target",
                "mae_mean",
                "rmse_mean",
                "r2_mean",
                "pearson_mean",
                "spearman_mean",
            ]
        ].to_string(index=False)
    )
    print("Done.")


if __name__ == "__main__":
    main()