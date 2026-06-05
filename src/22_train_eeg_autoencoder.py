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
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


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


def prepare_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    max_rows: int | None,
    seed: int,
) -> pd.DataFrame:
    out = df[feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    for col in feature_cols:
        if out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())

    out = out.dropna().reset_index(drop=True)

    if max_rows is not None and len(out) > max_rows:
        out = out.sample(n=max_rows, random_state=seed).reset_index(drop=True)

    return out


class EEGAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        latent_dim: int,
        hidden_dims: Tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()

        encoder_layers: List[nn.Module] = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            encoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        encoder_layers.append(nn.Linear(prev_dim, latent_dim))

        decoder_layers: List[nn.Module] = []
        prev_dim = latent_dim

        for hidden_dim in reversed(hidden_dims):
            decoder_layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z


def make_loader(x: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensor = torch.tensor(x, dtype=torch.float32)
    dataset = TensorDataset(tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_autoencoder(
    x_train: np.ndarray,
    x_val: np.ndarray,
    input_dim: int,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[EEGAutoencoder, Dict[str, List[float]]]:
    model = EEGAutoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)

    train_loader = make_loader(x_train, args.batch_size, shuffle=True)
    val_loader = make_loader(x_val, args.batch_size, shuffle=False)

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

        for (xb,) in train_loader:
            xb = xb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                recon, _ = model(xb)
                loss = criterion(recon, xb)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            train_losses.append(float(loss.detach().cpu().item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    recon, _ = model(xb)
                    loss = criterion(recon, xb)

                val_losses.append(float(loss.detach().cpu().item()))

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

    return model, history


def reconstruct(
    model: EEGAutoencoder,
    x: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = make_loader(x, batch_size, shuffle=False)
    model.eval()

    recon_list = []
    z_list = []

    with torch.no_grad():
        for (xb,) in loader:
            xb = xb.to(device, non_blocking=True)
            recon, z = model(xb)

            recon_list.append(recon.detach().cpu().numpy())
            z_list.append(z.detach().cpu().numpy())

    return np.vstack(recon_list), np.vstack(z_list)


def generate_synthetic(
    model: EEGAutoencoder,
    z_real: np.ndarray,
    device: torch.device,
    n_samples: int,
    noise_scale: float,
    seed: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    idx = rng.choice(len(z_real), size=n_samples, replace=True)
    z_base = z_real[idx]

    latent_std = z_real.std(axis=0, keepdims=True)
    latent_std = np.where(latent_std < 1e-8, 1.0, latent_std)

    noise = rng.normal(loc=0.0, scale=noise_scale, size=z_base.shape)
    z_synth = z_base + noise * latent_std

    z_tensor = torch.tensor(z_synth, dtype=torch.float32)
    loader = DataLoader(TensorDataset(z_tensor), batch_size=batch_size, shuffle=False)

    synth_list = []

    model.eval()
    with torch.no_grad():
        for (zb,) in loader:
            zb = zb.to(device, non_blocking=True)
            xb = model.decode(zb)
            synth_list.append(xb.detach().cpu().numpy())

    return np.vstack(synth_list), z_synth


def compute_reconstruction_metrics(x_true: np.ndarray, x_recon: np.ndarray) -> Dict[str, float]:
    mse = mean_squared_error(x_true.reshape(-1), x_recon.reshape(-1))
    mae = mean_absolute_error(x_true.reshape(-1), x_recon.reshape(-1))

    cos = []
    for i in range(len(x_true)):
        cos.append(cosine_similarity(x_true[i : i + 1], x_recon[i : i + 1])[0, 0])

    return {
        "reconstruction_mse": float(mse),
        "reconstruction_rmse": float(math.sqrt(mse)),
        "reconstruction_mae": float(mae),
        "reconstruction_cosine_mean": float(np.mean(cos)),
        "reconstruction_cosine_std": float(np.std(cos)),
    }


def compute_real_synthetic_metrics(x_real: np.ndarray, x_synth: np.ndarray) -> Dict[str, float]:
    n = min(len(x_real), len(x_synth))
    real = x_real[:n]
    synth = x_synth[:n]

    mean_diff = np.abs(real.mean(axis=0) - synth.mean(axis=0))
    std_diff = np.abs(real.std(axis=0) - synth.std(axis=0))

    cos = []
    for i in range(n):
        cos.append(cosine_similarity(real[i : i + 1], synth[i : i + 1])[0, 0])

    mse = mean_squared_error(real.reshape(-1), synth.reshape(-1))
    mae = mean_absolute_error(real.reshape(-1), synth.reshape(-1))

    return {
        "real_synthetic_pairwise_mse": float(mse),
        "real_synthetic_pairwise_rmse": float(math.sqrt(mse)),
        "real_synthetic_pairwise_mae": float(mae),
        "real_synthetic_pairwise_cosine_mean": float(np.mean(cos)),
        "real_synthetic_pairwise_cosine_std": float(np.std(cos)),
        "feature_mean_abs_diff_mean": float(mean_diff.mean()),
        "feature_mean_abs_diff_median": float(np.median(mean_diff)),
        "feature_std_abs_diff_mean": float(std_diff.mean()),
        "feature_std_abs_diff_median": float(np.median(std_diff)),
    }


def plot_loss(history: Dict[str, List[float]], output_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    plt.plot(history["train_loss"], label="train")
    plt.plot(history["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder reconstruction loss")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_reconstruction_examples(
    x_true: np.ndarray,
    x_recon: np.ndarray,
    feature_cols: List[str],
    output_path: Path,
    n_examples: int = 5,
) -> None:
    n = min(n_examples, len(x_true))
    feature_idx = np.arange(min(80, x_true.shape[1]))

    plt.figure(figsize=(12, 2.6 * n))

    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(feature_idx, x_true[i, feature_idx], label="real", linewidth=1)
        plt.plot(feature_idx, x_recon[i, feature_idx], label="reconstructed", linewidth=1, alpha=0.8)
        plt.ylabel(f"sample {i}")
        if i == 0:
            plt.title("Real vs reconstructed feature vectors")
        if i == n - 1:
            plt.xlabel("Feature index")
        plt.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_pca_real_synthetic(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    output_path: Path,
    seed: int,
    max_points: int = 3000,
) -> None:
    rng = np.random.default_rng(seed)

    n_real = min(max_points, len(x_real))
    n_synth = min(max_points, len(x_synth))

    real_idx = rng.choice(len(x_real), size=n_real, replace=False)
    synth_idx = rng.choice(len(x_synth), size=n_synth, replace=False)

    real = x_real[real_idx]
    synth = x_synth[synth_idx]

    combined = np.vstack([real, synth])
    labels = np.array(["real"] * len(real) + ["synthetic"] * len(synth))

    pca = PCA(n_components=2, random_state=seed)
    points = pca.fit_transform(combined)

    plt.figure(figsize=(8, 6))

    real_mask = labels == "real"
    synth_mask = labels == "synthetic"

    plt.scatter(points[real_mask, 0], points[real_mask, 1], s=8, alpha=0.45, label="real")
    plt.scatter(points[synth_mask, 0], points[synth_mask, 1], s=8, alpha=0.45, label="synthetic")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA: real vs synthetic EEG feature vectors")
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_feature_distribution_comparison(
    x_real: np.ndarray,
    x_synth: np.ndarray,
    output_path: Path,
    seed: int,
    n_features: int = 8,
) -> None:
    rng = np.random.default_rng(seed)

    n_features = min(n_features, x_real.shape[1])
    feature_indices = rng.choice(x_real.shape[1], size=n_features, replace=False)

    plt.figure(figsize=(12, 2.5 * n_features))

    for i, idx in enumerate(feature_indices):
        plt.subplot(n_features, 1, i + 1)
        plt.hist(x_real[:, idx], bins=40, alpha=0.55, density=True, label="real")
        plt.hist(x_synth[:, idx], bins=40, alpha=0.55, density=True, label="synthetic")
        plt.ylabel(f"f{idx}")
        if i == 0:
            plt.title("Feature distribution comparison")
        if i == n_features - 1:
            plt.xlabel("Standardized feature value")
        plt.legend(fontsize=8)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_report(
    args: argparse.Namespace,
    run_dir: Path,
    dataset_info: Dict[str, object],
    feature_info: Dict[str, object],
    reconstruction_metrics: Dict[str, float],
    real_synthetic_metrics: Dict[str, float],
    device: torch.device,
) -> None:
    lines = []
    lines.append("# EEG Feature Autoencoder Report")
    lines.append("")
    lines.append("## Run configuration")
    lines.append("")
    lines.append(f"- Dataset: `{args.dataset}`")
    lines.append(f"- Feature set: `{args.feature_set}`")
    lines.append(f"- Latent dim: `{args.latent_dim}`")
    lines.append(f"- Hidden dims: `{args.hidden_dims}`")
    lines.append(f"- Noise scale: `{args.noise_scale}`")
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
    lines.append("## Reconstruction metrics")
    lines.append("")
    lines.append(pd.DataFrame([reconstruction_metrics]).to_markdown(index=False))
    lines.append("")
    lines.append("## Real vs synthetic metrics")
    lines.append("")
    lines.append(pd.DataFrame([real_synthetic_metrics]).to_markdown(index=False))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "This experiment adds a generative deep learning block to the project. "
        "The autoencoder is trained to reconstruct EEG/POW feature vectors. "
        "Synthetic vectors are generated by adding Gaussian noise in the latent space "
        "and decoding the perturbed latent representations. "
        "This is not raw EEG waveform generation; it is generation in the prepared EEG/POW feature space. "
        "The approach is used as a practical lightweight approximation for the exam project."
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
    parser.add_argument("--run-name", default="eeg_autoencoder", type=str)
    parser.add_argument(
        "--feature-set",
        default="pow_plus_eeg",
        choices=["pow", "eeg", "pow_plus_eeg"],
    )
    parser.add_argument("--max-rows", default=None, type=int)
    parser.add_argument("--test-size", default=0.2, type=float)

    parser.add_argument("--latent-dim", default=32, type=int)
    parser.add_argument("--hidden-dims", nargs="+", default=[512, 256, 128], type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--epochs", default=80, type=int)
    parser.add_argument("--batch-size", default=512, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-5, type=float)
    parser.add_argument("--patience", default=10, type=int)
    parser.add_argument("--grad-clip", default=1.0, type=float)

    parser.add_argument("--n-synthetic", default=10000, type=int)
    parser.add_argument("--noise-scale", default=0.35, type=float)

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
    print("Train EEG feature autoencoder")
    print("=" * 80)
    print(f"Root: {root}")
    print(f"Dataset: {dataset_path}")
    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

    df = pd.read_parquet(dataset_path)
    feature_cols = get_feature_columns(df, args.feature_set)
    features_df = prepare_features(
        df=df,
        feature_cols=feature_cols,
        max_rows=args.max_rows,
        seed=args.seed,
    )

    dataset_info = {
        "rows_raw": int(len(df)),
        "rows_used": int(len(features_df)),
        "columns_raw": int(df.shape[1]),
    }

    feature_info = {
        "feature_set": args.feature_set,
        "n_features": len(feature_cols),
    }

    print(f"Dataset info: {dataset_info}")
    print(f"Feature info: {feature_info}")

    x_raw = features_df.to_numpy(dtype=np.float32)

    x_train_raw, x_val_raw = train_test_split(
        x_raw,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train_raw).astype(np.float32)
    x_val = scaler.transform(x_val_raw).astype(np.float32)

    run_dir = (
        root
        / "reports"
        / "runs"
        / f"{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}_{args.run_name}_{args.feature_set}_latent{args.latent_dim}"
    )
    figures_dir = run_dir / "figures"
    models_dir = run_dir / "models"
    synthetic_dir = run_dir / "synthetic"

    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir.mkdir(parents=True, exist_ok=True)

    model, history = train_autoencoder(
        x_train=x_train,
        x_val=x_val,
        input_dim=x_train.shape[1],
        device=device,
        args=args,
    )

    x_val_recon_scaled, z_val = reconstruct(
        model=model,
        x=x_val,
        device=device,
        batch_size=args.batch_size,
    )

    x_val_recon_raw = scaler.inverse_transform(x_val_recon_scaled)

    reconstruction_metrics_scaled = compute_reconstruction_metrics(x_val, x_val_recon_scaled)
    reconstruction_metrics_raw = compute_reconstruction_metrics(x_val_raw, x_val_recon_raw)

    reconstruction_metrics = {
        **{f"scaled_{k}": v for k, v in reconstruction_metrics_scaled.items()},
        **{f"raw_{k}": v for k, v in reconstruction_metrics_raw.items()},
    }

    x_synth_scaled, z_synth = generate_synthetic(
        model=model,
        z_real=z_val,
        device=device,
        n_samples=args.n_synthetic,
        noise_scale=args.noise_scale,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    x_synth_raw = scaler.inverse_transform(x_synth_scaled)

    real_synthetic_metrics_scaled = compute_real_synthetic_metrics(x_val, x_synth_scaled)
    real_synthetic_metrics_raw = compute_real_synthetic_metrics(x_val_raw, x_synth_raw)

    real_synthetic_metrics = {
        **{f"scaled_{k}": v for k, v in real_synthetic_metrics_scaled.items()},
        **{f"raw_{k}": v for k, v in real_synthetic_metrics_raw.items()},
    }

    pd.DataFrame([reconstruction_metrics]).to_csv(
        run_dir / "reconstruction_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame([real_synthetic_metrics]).to_csv(
        run_dir / "real_vs_synthetic_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        {
            "feature": feature_cols,
            "real_mean": x_val_raw.mean(axis=0),
            "synthetic_mean": x_synth_raw.mean(axis=0),
            "real_std": x_val_raw.std(axis=0),
            "synthetic_std": x_synth_raw.std(axis=0),
            "mean_abs_diff": np.abs(x_val_raw.mean(axis=0) - x_synth_raw.mean(axis=0)),
            "std_abs_diff": np.abs(x_val_raw.std(axis=0) - x_synth_raw.std(axis=0)),
        }
    ).to_csv(run_dir / "feature_distribution_summary.csv", index=False, encoding="utf-8-sig")

    np.save(synthetic_dir / "synthetic_features_scaled.npy", x_synth_scaled)
    np.save(synthetic_dir / "synthetic_features_raw.npy", x_synth_raw)
    np.save(synthetic_dir / "synthetic_latents.npy", z_synth)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "feature_cols": feature_cols,
            "args": vars(args),
        },
        models_dir / "autoencoder.pt",
    )

    plot_loss(history, figures_dir / "loss_curve.png")
    plot_reconstruction_examples(
        x_true=x_val,
        x_recon=x_val_recon_scaled,
        feature_cols=feature_cols,
        output_path=figures_dir / "reconstruction_examples_scaled.png",
    )
    plot_pca_real_synthetic(
        x_real=x_val,
        x_synth=x_synth_scaled,
        output_path=figures_dir / "pca_real_vs_synthetic_scaled.png",
        seed=args.seed,
    )
    plot_feature_distribution_comparison(
        x_real=x_val,
        x_synth=x_synth_scaled,
        output_path=figures_dir / "feature_distribution_comparison_scaled.png",
        seed=args.seed,
    )

    write_report(
        args=args,
        run_dir=run_dir,
        dataset_info=dataset_info,
        feature_info=feature_info,
        reconstruction_metrics=reconstruction_metrics,
        real_synthetic_metrics=real_synthetic_metrics,
        device=device,
    )

    print("=" * 80)
    print("Saved outputs")
    print("=" * 80)
    print(f"Run directory: {run_dir}")
    print(f"Reconstruction metrics: {run_dir / 'reconstruction_metrics.csv'}")
    print(f"Real vs synthetic metrics: {run_dir / 'real_vs_synthetic_metrics.csv'}")
    print(f"Feature distribution summary: {run_dir / 'feature_distribution_summary.csv'}")
    print(f"Report: {run_dir / 'report.md'}")
    print(f"Figures: {figures_dir}")
    print(f"Synthetic vectors: {synthetic_dir}")
    print("=" * 80)
    print("Reconstruction metrics")
    print("=" * 80)
    print(json.dumps(reconstruction_metrics, indent=2))
    print("=" * 80)
    print("Real vs synthetic metrics")
    print("=" * 80)
    print(json.dumps(real_synthetic_metrics, indent=2))
    print("Done.")


if __name__ == "__main__":
    main()