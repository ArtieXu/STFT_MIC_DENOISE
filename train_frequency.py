#!/usr/bin/env python3
"""Train CardioSpecNet. One flag switches the training set, nothing else changes.

This script exists to run the two arms of one experiment:

    arm A (baseline)   --no_circor    train on device clean windows only
    arm B (expanded)   default        train on device + CirCor clean windows

Noise is device-only in both arms -- CirCor has no exterior microphone.

Validation is device-only in both arms by default, which is what makes the two
runs comparable: same held-out subject, same noise, same seed, so the mixtures
are identical and only the training set differs. `--val_circor` puts CirCor into
validation too, but then the two arms are scored on different sets and the
comparison no longer means anything.

    # once: build the CirCor pool
    PYTHONPATH=. python scripts/build_circor_pool.py --root circor-heart-sound-1.0.3

    # check what each arm will see
    PYTHONPATH=. python scripts/audit_pools.py

    # the two arms
    PYTHONPATH=. python train_frequency.py --no_circor --output_dir checkpoints/device_only
    PYTHONPATH=. python train_frequency.py            --output_dir checkpoints/combined

    # score both on one fixed evaluation set
    PYTHONPATH=. python scripts/compare_datasets.py \
        --device_only checkpoints/device_only/best.pt \
        --combined    checkpoints/combined/best.pt
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.frequency_loss import FrequencyDenoiseLoss, FrequencyLossConfig
from src.frequency_metrics import (
    MetricAccumulator,
    log_spectral_distance,
    pearson_correlation,
    si_sdr_db,
    snr_db,
)
from src.frequency_model import CardioSpecNet, FrequencyModelConfig, STFTConfig
from src.pools import DEFAULT_TARGET_RMS, build_clean_pool, build_noise_pool, describe

DEFAULT_CIRCOR_POOL = Path("data/circor/circor_pool_4khz_2s.npz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    data = parser.add_argument_group("data")
    data.add_argument("--data_root", type=Path, default=Path("data"))
    data.add_argument("--step", type=str, default="1s",
                      help="Window-hop folder under data/clean and data/noise (this folder ships step_1s).")
    data.add_argument("--source_stride", type=int, default=1,
                      help="Keep every Nth stored window. step_1s windows are already 1 s apart, so 1.")
    data.add_argument("--circor_pool", type=Path, default=DEFAULT_CIRCOR_POOL,
                      help="Pool npz from scripts/build_circor_pool.py.")
    data.add_argument("--no_circor", action="store_true",
                      help="Arm A: train on device clean windows only.")
    data.add_argument("--val_circor", action="store_true",
                      help="Put CirCor into validation as well. Off by default so both arms "
                           "are scored on the same device-only held-out set.")
    data.add_argument("--circor_heart_rate_max", type=float, default=None,
                      help="Drop CirCor windows above this bpm. CirCor is pediatric; this device "
                           "sits at 61-79 bpm. Off by default; audit_pools.py prints the cost of "
                           "each threshold before you commit to one.")
    data.add_argument("--circor_heart_rate_min", type=float, default=None)
    data.add_argument("--pool_target_rms", type=float, default=DEFAULT_TARGET_RMS,
                      help="Per-window RMS every pool is normalised to. 0 disables normalisation, "
                           "which reintroduces the ~30-100x level gap between device and CirCor.")
    data.add_argument("--max_clean_windows", type=int, default=None, help="Per source, not in total.")
    data.add_argument("--max_noise_windows", type=int, default=None)
    data.add_argument("--samples_per_epoch", type=int, default=20_000)
    data.add_argument("--val_samples", type=int, default=2_000)

    mixing = parser.add_argument_group("mixing")
    mixing.add_argument("--snr_min_db", type=float, default=-10.0)
    mixing.add_argument("--snr_max_db", type=float, default=20.0)
    mixing.add_argument("--reference_dropout", type=float, default=0.20)
    mixing.add_argument("--chest_only_max", type=float, default=0.35,
                        help="Training-time extra noise on the chest channel, as a fraction of the "
                             "noise window. Above 0 the realised SNR is below the drawn SNR.")
    mixing.add_argument("--val_chest_only_max", type=float, default=0.0,
                        help="0 keeps the validation SNR axis exact (upstream used 0.35).")
    mixing.add_argument("--sensor_noise_fraction", type=float, default=0.01,
                        help="Reference-mic self-noise as a fraction of reference RMS (0.01 = -40 dB).")

    train = parser.add_argument_group("optimisation")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--num_workers", type=int, default=0,
                       help="Keep at 0 unless RAM is ample; each worker owns its window pools.")
    train.add_argument("--learning_rate", type=float, default=3e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--grad_clip", type=float, default=5.0)
    train.add_argument("--base_channels", type=int, default=12)
    train.add_argument("--grid_blocks", type=int, default=2)
    train.add_argument("--device", type=str, default="auto")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--output_dir", type=Path, default=Path("checkpoints/frequency"))
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--use_wandb", action="store_true")
    train.add_argument("--wandb_project", type=str, default="stft-pcg-denoise")
    train.add_argument("--smoke", action="store_true",
                       help="Small deterministic CPU run that validates the full train/eval/checkpoint path.")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def move_batch(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def resolve_circor_pool(args: argparse.Namespace) -> Path | None:
    """None for a device-only run; otherwise an existing pool path."""
    if args.no_circor:
        return None
    path = args.circor_pool
    if path is None:
        return None
    if not Path(path).is_file():
        raise SystemExit(
            f"CirCor pool not found: {path}\n\n"
            "This folder trains on device + CirCor by default. Build the pool once:\n"
            "  wget -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/\n"
            "  unzip -q circor.zip\n"
            "  PYTHONPATH=. python scripts/build_circor_pool.py \\\n"
            f"      --root circor-heart-sound-1.0.3 --out {path}\n\n"
            "Or run device-only on purpose with --no_circor."
        )
    return Path(path)


@torch.no_grad()
def validate(
    model: CardioSpecNet,
    criterion: FrequencyDenoiseLoss,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    accumulator = MetricAccumulator()
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch["chest"], batch["reference"], batch["ref_available"])
        loss, terms = criterion(output, batch["clean"])
        input_sisdr = si_sdr_db(batch["chest"], batch["clean"])
        output_sisdr = si_sdr_db(output, batch["clean"])
        input_snr = snr_db(batch["chest"], batch["clean"])
        output_snr = snr_db(output, batch["clean"])
        accumulator.update(
            loss=loss,
            input_si_sdr_db=input_sisdr,
            output_si_sdr_db=output_sisdr,
            si_sdr_improvement_db=output_sisdr - input_sisdr,
            input_snr_db=input_snr,
            output_snr_db=output_snr,
            snr_improvement_db=output_snr - input_snr,
            correlation=pearson_correlation(output, batch["clean"]),
            log_spectral_distance_db=log_spectral_distance(output, batch["clean"]),
            **{f"loss_{key}": value for key, value in terms.items() if key != "loss"},
        )
    return accumulator.summary()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 256)
        args.val_samples = min(args.val_samples, 128)
        args.batch_size = min(args.batch_size, 8)
        args.base_channels = min(args.base_channels, 8)
        args.grid_blocks = min(args.grid_blocks, 1)
        args.max_clean_windows = args.max_clean_windows or 128
        args.max_noise_windows = args.max_noise_windows or 96
        args.output_dir = args.output_dir / "smoke"

    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circor_pool = resolve_circor_pool(args)
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None

    clean_dir = args.data_root / "clean" / f"step_{args.step}"
    noise_dir = args.data_root / "noise" / f"step_{args.step}"
    pools = {}
    for split in ("train", "val"):
        # Validation stays device-only unless asked otherwise, so the two arms of
        # the experiment are scored on one identical set.
        split_circor = circor_pool if (split == "train" or args.val_circor) else None
        pools[f"clean_{split}"] = build_clean_pool(
            clean_dir / split,
            split,
            circor_pool=split_circor,
            source_stride=args.source_stride,
            max_windows=args.max_clean_windows,
            target_rms=target_rms,
            circor_heart_rate_max=args.circor_heart_rate_max,
            circor_heart_rate_min=args.circor_heart_rate_min,
        )
        pools[f"noise_{split}"] = build_noise_pool(
            noise_dir / split,
            split,
            source_stride=args.source_stride,
            max_windows=args.max_noise_windows,
            target_rms=target_rms,
        )
    arm = "A (device-only)" if circor_pool is None else "B (device + CirCor)"
    print(f"Arm {arm}; validation is {'device + CirCor' if args.val_circor else 'device-only'}")
    print("Window pools")
    for pool in pools.values():
        print(describe(pool))

    mixing_config = MixingConfig(
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
        reference_dropout_probability=args.reference_dropout,
        chest_only_noise_max=args.chest_only_max,
        sensor_noise_fraction=args.sensor_noise_fraction,
    )
    train_dataset = SyntheticFrequencyDataset(
        pools["clean_train"],
        pools["noise_train"],
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        config=mixing_config,
    )
    # Validation drops the identity cases and the reference dropout so methods
    # are compared at controlled SNR with a usable exterior channel.
    val_config = replace(
        mixing_config,
        identity_probability=0.0,
        reference_dropout_probability=0.0,
        chest_only_noise_max=args.val_chest_only_max,
    )
    val_dataset = SyntheticFrequencyDataset(
        pools["clean_val"],
        pools["noise_val"],
        samples_per_epoch=args.val_samples,
        seed=args.seed + 10_000,
        config=val_config,
    )

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    stft_config = STFTConfig()
    model_config = FrequencyModelConfig(
        base_channels=args.base_channels,
        grid_blocks=args.grid_blocks,
    )
    model = CardioSpecNet(stft_config, model_config).to(device)
    loss_config = FrequencyLossConfig()
    criterion = FrequencyDenoiseLoss(loss_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
    )

    start_epoch = 0
    best_score = -float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_score = float(checkpoint.get("best_score", best_score))

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"Device={device}; parameters={parameter_count:,}; "
        f"clean train={len(train_dataset.clean):,} / val={len(val_dataset.clean):,}; "
        f"noise train={len(train_dataset.noise):,} / val={len(val_dataset.noise):,}"
    )

    pool_stats = {
        "arm": "device_only" if circor_pool is None else "combined",
        "train": train_dataset.pool_stats(),
        "val": val_dataset.pool_stats(),
        "circor_pool": str(circor_pool) if circor_pool else None,
        "val_includes_circor": bool(args.val_circor),
    }
    (args.output_dir / "pools.json").write_text(
        json.dumps(pool_stats, indent=2, default=str) + "\n", encoding="utf-8"
    )

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={**vars(args), "parameters": parameter_count},
        )

    history_path = args.output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        start_time = perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        train_metrics = MetricAccumulator()
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                output = model(batch["chest"], batch["reference"], batch["ref_available"])
                loss, terms = criterion(output, batch["clean"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_metrics.update(loss=loss, **{key: value for key, value in terms.items() if key != "loss"})
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        validation = validate(model, criterion, val_loader, device)
        score = 0.5 * (
            validation["si_sdr_improvement_db_mean"]
            + validation["snr_improvement_db_mean"]
        )
        scheduler.step(score)
        elapsed = perf_counter() - start_time
        record = {
            "epoch": epoch,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics.summary(),
            "validation": validation,
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        checkpoint = {
            "epoch": epoch,
            "best_score": max(best_score, score),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model.config_dict,
            "mixing_config": mixing_config.to_dict(),
            "val_mixing_config": val_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "pool_stats": pool_stats,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameter_count": parameter_count,
            "validation": validation,
        }
        torch.save(checkpoint, args.output_dir / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, args.output_dir / "best.pt")

        print(
            f"epoch={epoch + 1:03d} loss={record['train']['loss_mean']:.4f} "
            f"val_SI-SDRi={validation['si_sdr_improvement_db_mean']:.3f} dB "
            f"val_SNRi={validation['snr_improvement_db_mean']:.3f} dB "
            f"score={score:.3f} corr={validation['correlation_mean']:.3f} time={elapsed:.1f}s"
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in record["train"].items()},
                **{f"val/{key}": value for key, value in validation.items()},
            })

    if wandb_run is not None:
        wandb_run.finish()
    print(f"Best checkpoint: {args.output_dir / 'best.pt'} (balanced improvement score={best_score:.3f} dB)")


if __name__ == "__main__":
    # Avoid excessive thread oversubscription on shared CPU hosts.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
