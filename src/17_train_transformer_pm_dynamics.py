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

TIME_COLUMNS_CANDIDATES = [
    "t_center",
    "t_start",
    "datetime_from_name",
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


def infer_time_column(df: pd.DataFrame) -> str | None:
    for col in TIME_COLUMNS_CANDIDATES:
        if col in df.columns:
            return col
    return None


def clean_dataframe(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    group_col: str,
    record_col: str,
    min_windows_per_subject: int,
) -> pd.DataFrame:
    required = feature_cols + target_cols + [group_col, record_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    time_col = infer_time_column(df)
    keep_cols = required + ([time_col] if time_col is not None else [])

    out = df[keep_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=target_cols + [group_col, record_col])

    for c in feature_cols:
        if out[c].isna().any():
            out[c] = out[c].fillna(out[c].median())

    counts = out[group_col].value_counts()
    keep_subjects = counts[counts >= min_windows_per_subject].index
    out = out[out[group_col].isin(keep_subjects)].reset_index(drop=True)

    return out


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    group_col: str,
    record_col: str,
    seq_len: int,
    target_position: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if seq_len < 2:
        raise ValueError("seq_len must be >= 2")

    if target_position not in {"center", "last", "next"}:
        raise ValueError("target_position must be one of: center, last, next")

    time_col = infer_time_column(df)
    sort_cols = [record_col, time_col] if time_col is not None else [record_col]

    x_list = []
    y_list = []
    group_list = []
    meta_rows = []

    if target_position == "center":
        target_offset = seq_len // 2
    elif target_position == "last":
        target_offset = seq_len - 1
    else:
        target_offset = seq_len

    for record_id, g in df.sort_values(sort_cols).groupby(record_col, sort=False):
        g = g.reset_index(drop=True)

        if target_position == "next":
            max_start = len(g) - seq_len - 1
        else:
            max_start = len(g) - seq_len

        if max_start < 0:
            continue

        features = g[feature_cols].to_numpy(dtype=np.float32)
        targets = g[target_cols].to_numpy(dtype=np.float32)
        groups = g[group_col].to_numpy()

        for start in range(max_start + 1):
            end = start + seq_len
            target_idx = start + target_offset

            x_seq = features[start:end]
            y = targets[target_idx]

            if np.isnan(x_seq).any() or np.isnan(y).any():
                continue

            x_list.append(x_seq)
            y_list.append(y)
            group_list.append(groups[target_idx])

            meta = {
                "record_id": record_id,
                "subject_id": groups[target_idx],
                "start_index": start,
                "target_index": target_idx,
            }
            if time_col is not None:
                meta["target_time"] = g.loc[target_idx, time_col]
            meta_rows.append(meta)

    if not x_list:
        raise ValueError("No valid sequences were created. Check seq_len and dataset ordering.")

    return (
        np.stack(x_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.asarray(group_list),
        pd.DataFrame(meta_rows),
    )


class TransformerPMRegressor(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        seq_len: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()

        if d_model % nhead != 0:
            raise ValueError("d_model must be divisible by nhead")

        self.seq_len = seq_len
        self.pooling = pooling

        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Parameter(torch.zeros(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, output_dim),
        )

        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x = x + self.pos_embedding[:, : x.size(1), :]
        x = self.encoder(x)

        if self.pooling == "mean":
            pooled = x.mean(dim=1)
        elif self.pooling == "last":
            pooled = x[:, -1, :]
        elif self.pooling == "center":
            pooled = x[:, x.size(1) // 2, :]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        return self.head(pooled)


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


def fit_scalers_3d(x_train: np.ndarray, y_train: np.ndarray) -> Tuple[StandardScaler, StandardScaler]:
    n, t, f = x_train.shape
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    x_scaler.fit(x_train.reshape(n * t, f))
    y_scaler.fit(y_train)
    return x_scaler, y_scaler


def transform_x_3d(x: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    n, t, f = x.shape
    out = scaler.transform(x.reshape(n * t, f)).reshape(n, t, f)
    return out.astype(np.float32)


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
    fold: int,
) -> List[Dict[str, float]]:
    rows = []

    for idx, target in enumerate(target_cols):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]

        mse = mean_squared_error(yt, yp)

        rows.append(
            {
                "fold": fold,
                "target": target,
                "mae": mean_absolute_error(yt, yp),
                "rmse": math.sqrt(mse),
                "r2": r2_score(yt, yp),
                "pearson": safe_corr(pearsonr, yt, yp),
                "spearman": safe_corr(spearmanr, yt, yp),
            }
        )

    return rows


def aggregate_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return (
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


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    input_dim: int,
    output_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[TransformerPMRegressor, Dict[str, List[float]], np.ndarray]:
    model = TransformerPMRegressor(
        input_dim=input_dim,
        output_dim=output_dim,
        seq_len=args.seq_len,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        pooling=args.pooling,
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
            print(f"epoch={epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

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

    return model, history, np.vstack(final_preds)


def plot_loss(histories: Dict[int, Dict[str, List[float]]], output_path: Path) -> None:
    plt.figure(figsize=(10, 6))

    for fold, hist in histories.items():
        plt.plot(hist["train_loss"], linestyle="--", alpha=0.7, label=f"fold {fold} train")
        plt.plot(hist["val_loss"], alpha=0.9, label=f"fold {fold} val")

    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Transformer PM dynamics training curves")
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
    plt.title("Transformer PM dynamics: GroupKFold R² by target")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_info: Dict[str, object],
    feature_info: Dict[str, object],
    sequence_info: Dict[str, object],
    agg_df: pd.DataFrame,
    device: torch.device,
) -> None:
    lines = []
    lines.append("# Transformer PM Dynamics Report")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Feature set: `{args.feature_set}`")
    lines.append(f"- Sequence length: `{args.seq_len}`")
    lines.append(f"- Target position: `{args.target_position}`")
    lines.append(f"- Pooling: `{args.pooling}`")
    lines.append(f"- d_model: `{args.d_model}`")
    lines.append(f"- nhead: `{args.nhead}`")
    lines.append(f"- Transformer layers: `{args.num_layers}`")
    lines.append(f"- Validation: GroupKFold by `{args.group_col}`")
    lines.append(f"- Folds: `{args.n_splits}`")
    lines.append(f"- Device: `{device}`")
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
    lines.append("## Sequence info")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(sequence_info, ensure_ascii=False, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Aggregated metrics")
    lines.append("")
    lines.append(agg_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This experiment uses a TransformerEncoder on local EEG/POW window sequences. "
        "It provides an attention-based deep learning comparison to the MLP and LSTM models. "
        "The goal is to test whether self-attention over neighboring windows improves "
        "prediction of cognitive-state PM metrics."
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
    parser.add_argument("--run-name", default="transformer_pm_dynamics", type=str)
    parser.add_argument(
        "--feature-set",
        default="pow_plus_eeg",
        choices=["pow", "eeg", "pow_plus_eeg"],
    )
    parser.add_argument("--group-col", default="subject_id", type=str)
    parser.add_argument("--record-col", default="record_id", type=str)
    parser.add_argument("--n-splits", default=5, type=int)
    parser.add_argument("--min-windows-per-subject", default=30, type=int)
    parser.add_argument("--max-rows", default=None, type=int)

    parser.add_argument("--seq-len", default=5, type=int)
    parser.add_argument(
        "--target-position",
        default="center",
        choices=["center", "last", "next"],
    )
    parser.add_argument(
        "--pooling",
        default="center",
        choices=["center", "last", "mean"],
    )

    parser.add_argument("--d-model", default=128, type=int)
    parser.add_argument("--nhead", default=4, type=int)
    parser.add_argument("--num-layers", default=2, type=int)
    parser.add_argument("--dim-feedforward", default=256, type=int)
    parser.add_argument("--dropout", default=0.2, type=float)

    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=5e-4, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--patience", default=8, type=int)
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

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    print("=" * 80)
    print("Train Transformer PM dynamics model")
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
        record_col=args.record_col,
        min_windows_per_subject=args.min_windows_per_subject,
    )

    x, y, groups, meta_df = build_sequences(
        df=clean_df,
        feature_cols=feature_cols,
        target_cols=target_cols,
        group_col=args.group_col,
        record_col=args.record_col,
        seq_len=args.seq_len,
        target_position=args.target_position,
    )

    dataset_info = {
        "rows_raw": int(len(df)),
        "rows_clean": int(len(clean_df)),
        "columns_raw": int(df.shape[1]),
        "subjects_clean": int(clean_df[args.group_col].nunique()),
        "records_clean": int(clean_df[args.record_col].nunique()),
        "targets": target_cols,
    }

    feature_info = {
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
        "n_targets": len(target_cols),
    }

    sequence_info = {
        "n_sequences": int(x.shape[0]),
        "seq_len": int(x.shape[1]),
        "n_features": int(x.shape[2]),
        "target_position": args.target_position,
        "pooling": args.pooling,
        "subjects": int(pd.Series(groups).nunique()),
        "records": int(meta_df["record_id"].nunique()) if "record_id" in meta_df else None,
    }

    print(f"Dataset info: {dataset_info}")
    print(f"Feature info: {feature_info}")
    print(f"Sequence info: {sequence_info}")

    run_dir = (
        root
        / "reports"
        / "runs"
        / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}_{args.feature_set}_seq{args.seq_len}_{args.target_position}"
    )
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    meta_df.to_csv(run_dir / "sequence_metadata.csv", index=False, encoding="utf-8-sig")

    gkf = GroupKFold(n_splits=args.n_splits)

    all_metrics = []
    histories: Dict[int, Dict[str, List[float]]] = {}

    for fold, (train_idx, val_idx) in enumerate(gkf.split(x, y, groups=groups), start=1):
        print("-" * 80)
        print(f"Fold {fold}/{args.n_splits}")
        print(f"Train sequences: {len(train_idx)} | Val sequences: {len(val_idx)}")

        x_train_raw, x_val_raw = x[train_idx], x[val_idx]
        y_train_raw, y_val_raw = y[train_idx], y[val_idx]

        x_scaler, y_scaler = fit_scalers_3d(x_train_raw, y_train_raw)

        x_train = transform_x_3d(x_train_raw, x_scaler)
        x_val = transform_x_3d(x_val_raw, x_scaler)

        y_train = y_scaler.transform(y_train_raw).astype(np.float32)
        y_val = y_scaler.transform(y_val_raw).astype(np.float32)

        model, history, y_pred_scaled = train_one_fold(
            x_train=x_train,
            y_train=y_train,
            x_val=x_val,
            y_val=y_val,
            input_dim=x_train.shape[2],
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
        sequence_info=sequence_info,
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