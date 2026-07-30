#!/usr/bin/env python3
"""One single-microphone training entry point for all three denoising arms.

    arm 1  device      --no_circor                   simple STFT U-Net
    arm 2  expanded    (default)                     same STFT U-Net
    arm 3  waveform    --arch cleanunet --no_circor  previous waveform system

All arms receive exactly one waveform: ``chest = clean + scaled_noise``.  There
is no exterior/reference microphone.  The STFT model sees only the real and
imaginary planes of that waveform's complex STFT.

The two STFT runs use the same total samples per epoch.  In the combined run,
clean targets are scheduled exactly 50:50 between device and CirCor, so the
larger CirCor pool cannot dominate by accident.

The final fit uses device clean subjects 1/2/4/5 and four independently mixed
noise recordings (1/2/3/5). Subject/session 6 is test-only. There is no
validation-driven scheduler or best-epoch selection: every arm is compared
after the same fixed number of optimiser steps.

    # once: build the CirCor pool
    PYTHONPATH=. python scripts/build_circor_pool.py --root circor-heart-sound-1.0.3

    # check what each arm will see
    PYTHONPATH=. python scripts/audit_pools.py

    # the three arms -- identical apart from the flags shown
    PYTHONPATH=. python train_frequency.py --no_circor --output_dir checkpoints/device_only
    PYTHONPATH=. python train_frequency.py            --output_dir checkpoints/combined
    PYTHONPATH=. python train_frequency.py --no_circor --arch cleanunet \
        --output_dir checkpoints/waveform

    # score them on one fixed evaluation set, then draw the spectrograms
    PYTHONPATH=. python scripts/compare_datasets.py \
        --device_only checkpoints/device_only/final.pt \
        --combined    checkpoints/combined/final.pt \
        --waveform    checkpoints/waveform/final.pt
    PYTHONPATH=. python scripts/make_demo_figures.py --all
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.frequency_loss import FrequencyDenoiseLoss, FrequencyLossConfig
from src.frequency_metrics import MetricAccumulator
from src.arms import (
    AMP_MAX_RETRIES,
    ARCH_LABEL,
    ARCHS,
    OPTIMIZER_UPDATE_POLICY,
    build_arm,
    parameter_count,
)
from src.pools import (
    DEFAULT_TARGET_RMS,
    DEVICE_FILE_SPLITS,
    build_clean_pool,
    build_noise_pool,
    describe,
    device_protocol_metadata,
)

DEFAULT_CIRCOR_POOL = Path("data/circor/circor_pool_4khz_2s_v2.npz")


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

    mixing = parser.add_argument_group("mixing")
    mixing.add_argument("--snr_min_db", type=float, default=-10.0)
    mixing.add_argument("--snr_max_db", type=float, default=20.0)

    train = parser.add_argument_group("optimisation")
    train.add_argument("--epochs", type=int, default=60)
    train.add_argument("--batch_size", type=int, default=16)
    train.add_argument("--num_workers", type=int, default=2,
                       help="Mixtures are synthesised on the CPU (~1 ms each). At 0 that work "
                            "runs in the training loop and the GPU waits for it; 2 workers cover "
                            "a T4 with room to spare. Each worker holds its own copy of the "
                            "window pools (~0.5 GB with CirCor), so drop to 1 if RAM is tight.")
    train.add_argument("--learning_rate", type=float, default=3e-4)
    train.add_argument("--weight_decay", type=float, default=1e-4)
    train.add_argument("--grad_clip", type=float, default=5.0)
    model = parser.add_argument_group("model")
    model.add_argument("--arch", choices=ARCHS, default="cardiospecnet",
                       help="cardiospecnet = complex-STFT 2-D U-Net; cleanunet = waveform baseline.")
    model.add_argument("--base_channels", type=int, default=12, help="Simple STFT U-Net width.")
    model.add_argument("--wave_channels_H", type=int, default=None)
    model.add_argument("--wave_max_H", type=int, default=None)
    model.add_argument("--wave_encoder_layers", type=int, default=None)
    model.add_argument("--wave_tsfm_layers", type=int, default=None)
    model.add_argument("--wave_tsfm_d_model", type=int, default=None)
    model.add_argument("--wave_tsfm_d_inner", type=int, default=None)
    train.add_argument("--device", type=str, default="auto")
    train.add_argument("--seed", type=int, default=2026)
    train.add_argument("--output_dir", type=Path, default=Path("checkpoints/frequency"))
    train.add_argument("--resume", type=Path, default=None)
    train.add_argument("--use_wandb", action="store_true")
    train.add_argument("--wandb_project", type=str, default="stft-pcg-denoise")
    train.add_argument("--smoke", action="store_true",
                       help="Small deterministic CPU run that validates train/resume/final checkpointing.")
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


def optimizer_update(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    batch: dict[str, Tensor],
    optimizer: torch.optim.Optimizer,
    scaler,
    *,
    device: torch.device,
    grad_clip: float,
    use_amp: bool,
    max_amp_retries: int = AMP_MAX_RETRIES,
) -> tuple[Tensor, dict[str, Tensor], int, bool, bool]:
    """Complete exactly one real optimizer update for one planned batch.

    CUDA GradScaler is allowed to skip an overflowing attempt and lower its
    scale. The same batch is retried, so fixed-budget metadata counts successful
    parameter updates rather than merely counting calls to ``scaler.step``.
    After bounded retries (or a non-finite FP16 forward), that batch is completed
    once in FP32. FP32 remains strictly fail-fast.
    """

    if max_amp_retries <= 0:
        raise ValueError("max_amp_retries must be positive")

    overflow_retries = 0
    amp_forward_fallback = False
    if use_amp:
        while overflow_retries < max_amp_retries:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=True,
            ):
                output = model(batch["chest"])
                loss, terms = criterion(output, batch["clean"])
            if not torch.isfinite(loss):
                amp_forward_fallback = True
                break

            scale_before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip,
                error_if_nonfinite=False,
            )
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            if scale_after < scale_before:
                overflow_retries += 1
                continue
            return loss, terms, overflow_retries, False, amp_forward_fallback

    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type=device.type, enabled=False):
        output = model(batch["chest"])
        loss, terms = criterion(output, batch["clean"])
    if not torch.isfinite(loss):
        raise RuntimeError(
            "non-finite FP32 training loss; refusing to count an optimizer update"
        )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        grad_clip,
        error_if_nonfinite=True,
    )
    optimizer.step()
    return loss, terms, overflow_retries, use_amp, amp_forward_fallback


def save_checkpoint_atomic(checkpoint: dict, path: Path) -> None:
    """Never leave a partially written checkpoint at the public path."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


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
            "  wget -c -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/\n"
            "  unzip -q circor.zip\n"
            "  PYTHONPATH=. python scripts/build_circor_pool.py \\\n"
            f"      --root circor-heart-sound-1.0.3 --out {path}\n\n"
            "Or run device-only on purpose with --no_circor."
        )
    return Path(path)


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.epochs = min(args.epochs, 2)
        args.samples_per_epoch = min(args.samples_per_epoch, 256)
        args.batch_size = min(args.batch_size, 8)
        args.base_channels = min(args.base_channels, 8)
        args.max_clean_windows = args.max_clean_windows or 128
        args.max_noise_windows = args.max_noise_windows or 96
        args.output_dir = args.output_dir / "smoke"

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch_size must be positive")
    if args.samples_per_epoch % args.batch_size:
        raise SystemExit(
            "--samples_per_epoch must be divisible by --batch_size so every arm "
            "receives the declared number of inputs"
        )

    set_seed(args.seed)
    device = choose_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    circor_pool = resolve_circor_pool(args)
    if args.arch == "cleanunet" and circor_pool is not None:
        raise SystemExit(
            "The waveform comparison arm is device-only. "
            "Run it with --arch cleanunet --no_circor."
        )
    role = (
        "waveform"
        if args.arch == "cleanunet"
        else ("device_only" if circor_pool is None else "combined")
    )
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None

    clean_dir = args.data_root / "clean" / f"step_{args.step}"
    noise_dir = args.data_root / "noise" / f"step_{args.step}"
    pools = {
        "clean_train": build_clean_pool(
            clean_dir / "train",
            "train",
            circor_pool=circor_pool,
            source_stride=args.source_stride,
            max_windows=args.max_clean_windows,
            target_rms=target_rms,
            circor_heart_rate_max=args.circor_heart_rate_max,
            circor_heart_rate_min=args.circor_heart_rate_min,
        ),
        "noise_train": build_noise_pool(
            noise_dir / "train",
            "train",
            source_stride=args.source_stride,
            max_windows=args.max_noise_windows,
            target_rms=target_rms,
        ),
    }
    training_set = "device-only" if circor_pool is None else "device + CirCor"
    print(
        f"Arch {ARCH_LABEL[args.arch]}; training set {training_set}; "
        "fixed-budget final fit (no validation selection)"
    )
    print("Window pools")
    for pool in pools.values():
        print(describe(pool))

    mixing_config = MixingConfig(
        snr_min_db=args.snr_min_db,
        snr_max_db=args.snr_max_db,
    )
    train_dataset = SyntheticFrequencyDataset(
        pools["clean_train"],
        pools["noise_train"],
        samples_per_epoch=args.samples_per_epoch,
        seed=args.seed,
        config=mixing_config,
    )
    train_pool_stats = train_dataset.pool_stats()
    expected_device_clean_origins = {
        f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["clean"]["train"]
    }
    actual_device_clean_origins = set(
        train_pool_stats["device_clean_origin_counts"]
    )
    expected_noise_origins = {
        f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["noise"]["train"]
    }
    actual_noise_origins = set(train_pool_stats["noise_origin_counts"])
    if actual_device_clean_origins != expected_device_clean_origins:
        raise SystemExit(
            "The training clean pool does not contain the complete fixed device manifest: "
            f"expected {sorted(expected_device_clean_origins)}, "
            f"got {sorted(actual_device_clean_origins)}"
        )
    if actual_noise_origins != expected_noise_origins:
        raise SystemExit(
            "The training noise pool does not contain the complete four-recording manifest: "
            f"expected {sorted(expected_noise_origins)}, got {sorted(actual_noise_origins)}"
        )
    if args.batch_size % train_dataset.sampling_cycle:
        raise SystemExit(
            "--batch_size must be divisible by the clean/noise sampling cycle "
            f"({train_dataset.sampling_cycle}) so every batch is source-balanced"
        )

    pin_memory = device.type == "cuda"
    effective_num_workers = args.num_workers
    if effective_num_workers > 0 and not train_dataset.epoch_is_shared:
        print(
            "Shared epoch state is unavailable; forcing num_workers=0 so every "
            "epoch uses the intended deterministic sampling recipe."
        )
        effective_num_workers = 0
    loader_extra = (
        {"persistent_workers": True, "prefetch_factor": 4}
        if effective_num_workers > 0
        else {}
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=effective_num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **loader_extra,
    )
    model = build_arm(
        args.arch,
        base_channels=args.base_channels,
        wave_overrides={
            "channels_H": args.wave_channels_H,
            "max_H": args.wave_max_H,
            "encoder_n_layers": args.wave_encoder_layers,
            "tsfm_n_layers": args.wave_tsfm_layers,
            "tsfm_d_model": args.wave_tsfm_d_model,
            "tsfm_d_inner": args.wave_tsfm_d_inner,
            "input_channels": 1,
        },
    ).to(device)
    # Same objective for both architectures: it takes waveforms and does its own
    # STFT internally, so it does not favour either representation.
    loss_config = FrequencyLossConfig()
    criterion = FrequencyDenoiseLoss(loss_config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    use_amp = device.type == "cuda"
    precision_policy = (
        "cuda_fp16_retry_then_fp32_fallback_v1"
        if use_amp
        else "fp32_fail_fast_v1"
    )

    start_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if (
            checkpoint.get("schema_version") != 3
            or checkpoint.get("input_mode") != "single_mic"
            or checkpoint.get("role") != role
            or checkpoint.get("arch") != args.arch
            or checkpoint.get("checkpoint_kind") != "resume"
            or checkpoint.get("data_protocol") != device_protocol_metadata()
            or checkpoint.get("mixing_config") != mixing_config.to_dict()
            or checkpoint.get("loss_config") != loss_config.to_dict()
            or (checkpoint.get("pool_stats") or {}).get("train") != train_pool_stats
        ):
            raise SystemExit(
                f"{args.resume} is not a matching schema-v3 fixed-budget "
                f"{role}/{args.arch} resume checkpoint"
            )
        saved_args = checkpoint.get("training_args") or {}
        resume_keys = (
            "epochs", "samples_per_epoch", "batch_size", "learning_rate",
            "weight_decay", "grad_clip", "seed", "snr_min_db", "snr_max_db",
            "pool_target_rms", "source_stride", "max_clean_windows",
            "max_noise_windows",
        )
        mismatches = {
            key: (saved_args.get(key), vars(args).get(key))
            for key in resume_keys
            if saved_args.get(key) != vars(args).get(key)
        }
        if mismatches:
            raise SystemExit(
                f"{args.resume} was created with a different training protocol: "
                f"{mismatches}"
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        policy = checkpoint.get("training_policy") or {}
        expected_steps = start_epoch * len(train_loader)
        expected_inputs = start_epoch * args.samples_per_epoch
        runtime_stats = checkpoint.get("runtime_stats") or {}
        if (
            int(policy.get("completed_epochs", -1)) != start_epoch
            or int(policy.get("optimizer_steps", -1)) != expected_steps
            or int(policy.get("steps_per_epoch", -1)) != len(train_loader)
            or int(policy.get("batch_size", -1)) != args.batch_size
            or int(policy.get("inputs_per_epoch", -1)) != args.samples_per_epoch
            or int(policy.get("trained_inputs", -1)) != expected_inputs
            or policy.get("optimizer_update_policy") != OPTIMIZER_UPDATE_POLICY
            or policy.get("precision_policy") != precision_policy
            or int(policy.get("amp_max_retries_per_batch", -1))
            != AMP_MAX_RETRIES
            or int(runtime_stats.get("successful_optimizer_updates", -1))
            != expected_steps
        ):
            raise SystemExit(
                f"{args.resume} has inconsistent fixed-budget step/input metadata"
            )
        if start_epoch > args.epochs:
            raise SystemExit(
                f"{args.resume} reports more than the requested {args.epochs} epochs"
            )
        if start_epoch == args.epochs:
            recovered = {**checkpoint, "checkpoint_kind": "final"}
            save_checkpoint_atomic(recovered, args.output_dir / "final.pt")
            print(
                f"Recovered completed final checkpoint from {args.resume}: "
                f"{args.output_dir / 'final.pt'}"
            )
            return

    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if args.resume and checkpoint.get("scaler_state_dict"):
        scaler.load_state_dict(checkpoint["scaler_state_dict"])
    if args.resume:
        if not checkpoint.get("rng_state"):
            raise SystemExit(f"{args.resume} lacks RNG state for an exact resume")
        restore_rng_state(checkpoint["rng_state"])
        saved_runtime = checkpoint.get("runtime_stats") or {}
        successful_optimizer_updates = int(
            saved_runtime["successful_optimizer_updates"]
        )
        amp_overflow_retries = int(saved_runtime.get("amp_overflow_retries", 0))
        amp_forward_fallbacks = int(saved_runtime.get("amp_forward_fallbacks", 0))
        fp32_fallback_updates = int(saved_runtime.get("fp32_fallback_updates", 0))
    else:
        successful_optimizer_updates = 0
        amp_overflow_retries = 0
        amp_forward_fallbacks = 0
        fp32_fallback_updates = 0
    parameters = parameter_count(model)
    print(
        f"Device={device}; arch={ARCH_LABEL[args.arch]}; parameters={parameters:,}; "
        f"clean train={len(train_dataset.clean):,}; "
        f"noise train={len(train_dataset.noise):,}"
    )
    print(
        "Training input clean-source schedule: "
        f"{train_pool_stats['clean_source_sampling']}"
    )
    print(
        "Training noise-recording schedule: "
        f"{train_pool_stats['noise_origin_sampling']}"
    )
    print(
        f"Fixed budget: {args.epochs} epochs x {len(train_loader)} steps x "
        f"{args.batch_size} inputs = {args.epochs * args.samples_per_epoch:,} inputs"
    )

    pool_stats = {
        "arm": role,
        "train": train_pool_stats,
        "circor_pool": str(circor_pool) if circor_pool else None,
        "selection_uses_validation": False,
        "data_protocol": device_protocol_metadata(),
    }
    (args.output_dir / "pools.json").write_text(
        json.dumps(pool_stats, indent=2, default=str) + "\n", encoding="utf-8"
    )

    wandb_run = None
    if args.use_wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            config={**vars(args), "parameters": parameters},
        )

    history_path = args.output_dir / "history.jsonl"
    if start_epoch == 0:
        history_path.write_text("", encoding="utf-8")
        # A new non-resume run must not expose any stale checkpoint from an
        # older budget while its first epoch is still in progress.
        for stale_name in ("last.pt", "final.pt", "best.pt"):
            (args.output_dir / stale_name).unlink(missing_ok=True)
    for epoch in range(start_epoch, args.epochs):
        start_time = perf_counter()
        train_dataset.set_epoch(epoch)
        model.train()
        train_metrics = MetricAccumulator()
        epoch_amp_retries = 0
        epoch_fp32_fallbacks = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch + 1}/{args.epochs}", leave=False)
        for batch in progress:
            batch = move_batch(batch, device)
            loss, terms, retries, used_fp32_fallback, forward_fallback = (
                optimizer_update(
                    model,
                    criterion,
                    batch,
                    optimizer,
                    scaler,
                    device=device,
                    grad_clip=args.grad_clip,
                    use_amp=use_amp,
                )
            )
            successful_optimizer_updates += 1
            amp_overflow_retries += retries
            amp_forward_fallbacks += int(forward_fallback)
            fp32_fallback_updates += int(used_fp32_fallback)
            epoch_amp_retries += retries
            epoch_fp32_fallbacks += int(used_fp32_fallback)
            train_metrics.update(loss=loss, **{key: value for key, value in terms.items() if key != "loss"})
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        expected_successful_updates = (epoch + 1) * len(train_loader)
        if successful_optimizer_updates != expected_successful_updates:
            raise RuntimeError(
                "successful optimizer-update count does not match the fixed "
                f"budget: {successful_optimizer_updates} != "
                f"{expected_successful_updates}"
            )
        elapsed = perf_counter() - start_time
        record = {
            "epoch": epoch,
            "completed_epochs": epoch + 1,
            "optimizer_steps": successful_optimizer_updates,
            "amp_overflow_retries": epoch_amp_retries,
            "fp32_fallback_updates": epoch_fp32_fallbacks,
            "elapsed_seconds": elapsed,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "train": train_metrics.summary(),
        }
        with history_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        training_policy = {
            "selection_policy": "fixed_budget_final_epoch",
            "uses_validation": False,
            "optimizer": "AdamW",
            "learning_rate_schedule": "constant",
            "optimizer_update_policy": OPTIMIZER_UPDATE_POLICY,
            "precision_policy": precision_policy,
            "amp_max_retries_per_batch": AMP_MAX_RETRIES,
            "target_epochs": args.epochs,
            "completed_epochs": epoch + 1,
            "steps_per_epoch": len(train_loader),
            "batch_size": args.batch_size,
            "num_workers": effective_num_workers,
            "inputs_per_epoch": args.samples_per_epoch,
            "optimizer_steps": successful_optimizer_updates,
            "trained_inputs": successful_optimizer_updates * args.batch_size,
        }
        runtime_stats = {
            "successful_optimizer_updates": successful_optimizer_updates,
            "amp_overflow_retries": amp_overflow_retries,
            "amp_forward_fallbacks": amp_forward_fallbacks,
            "fp32_fallback_updates": fp32_fallback_updates,
            "final_loss_scale": float(scaler.get_scale()) if use_amp else None,
        }
        checkpoint = {
            "epoch": epoch,
            "checkpoint_kind": "resume",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "rng_state": capture_rng_state(),
            "schema_version": 3,
            "input_mode": "single_mic",
            "role": role,
            "arch": args.arch,
            "model_config": model.config_dict,
            "mixing_config": mixing_config.to_dict(),
            "loss_config": loss_config.to_dict(),
            "pool_stats": pool_stats,
            "data_protocol": device_protocol_metadata(),
            "training_policy": training_policy,
            "runtime_stats": runtime_stats,
            "training_args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "parameter_count": parameters,
        }
        if epoch + 1 == args.epochs:
            final_checkpoint = {**checkpoint, "checkpoint_kind": "final"}
            save_checkpoint_atomic(final_checkpoint, args.output_dir / "final.pt")
        save_checkpoint_atomic(checkpoint, args.output_dir / "last.pt")

        print(
            f"epoch={epoch + 1:03d} loss={record['train']['loss_mean']:.4f} "
            f"steps={successful_optimizer_updates:,} "
            f"inputs={successful_optimizer_updates * args.batch_size:,} "
            f"amp_retries={epoch_amp_retries} time={elapsed:.1f}s"
        )
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch,
                "learning_rate": optimizer.param_groups[0]["lr"],
                **{f"train/{key}": value for key, value in record["train"].items()},
            })

    if wandb_run is not None:
        wandb_run.finish()
    print(
        f"Final checkpoint: {args.output_dir / 'final.pt'} "
        f"({successful_optimizer_updates:,} optimiser steps, "
        f"{successful_optimizer_updates * args.batch_size:,} inputs)"
    )


if __name__ == "__main__":
    # Avoid excessive thread oversubscription on shared CPU hosts.
    torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
    main()
