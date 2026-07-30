#!/usr/bin/env python3
"""Measure the real cost of one training step before committing hours to it.

    PYTHONPATH=. python scripts/benchmark.py
    PYTHONPATH=. python scripts/benchmark.py --epochs 30 --samples_per_epoch 8000 --batch_size 32

Times the data pipeline and the optimiser step separately, then prints the
projected wall clock for all three arms. Two numbers matter:

    data     ms per batch of synthesised mixtures, per worker
    step     ms per forward+backward+update on this device

If ``data / num_workers`` exceeds ``step``, the GPU is waiting on the CPU and
more workers (or a bigger batch) is the cheapest win. If ``step`` dominates,
only less work helps: fewer epochs, fewer samples per epoch, or a smaller model.
The projection covers the single training stage only; there is no validation
pass and the subject-6 final test is never touched.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arms import ARCH_LABEL, build_arm, parameter_count  # noqa: E402
from src.frequency_data import MixingConfig, SyntheticFrequencyDataset  # noqa: E402
from src.frequency_loss import FrequencyDenoiseLoss  # noqa: E402
from src.pools import build_clean_pool, build_noise_pool  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--step_dir", type=str, default="1s")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--samples_per_epoch", type=int, default=20_000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--steps", type=int, default=12, help="Timed steps per architecture.")
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.samples_per_epoch <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs, --samples_per_epoch, and --batch_size must be positive")
    if args.num_workers < 0 or args.steps <= 0:
        raise SystemExit("--num_workers must be non-negative and --steps must be positive")
    if args.samples_per_epoch % args.batch_size:
        raise SystemExit("--samples_per_epoch must be divisible by --batch_size")
    # The largest training schedule is combined: two clean sources x four
    # independently mixed noise recordings.
    if args.samples_per_epoch % 8 or args.batch_size % 8:
        raise SystemExit(
            "--samples_per_epoch and --batch_size must be divisible by the "
            "combined arm's 2x4 sampling cycle (8)"
        )
    device = choose_device(args.device)
    print(f"device: {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    clean = build_clean_pool(
        args.data_root / "clean" / f"step_{args.step_dir}" / "train", "train",
        circor_pool=None, max_windows=400,
    )
    noise = build_noise_pool(
        args.data_root / "noise" / f"step_{args.step_dir}" / "train", "train", max_windows=400,
    )
    dataset = SyntheticFrequencyDataset(
        clean, noise, samples_per_epoch=512, seed=0, config=MixingConfig()
    )

    warm = 8
    for index in range(warm):
        dataset[index]
    start = time.perf_counter()
    for index in range(warm, warm + 96):
        dataset[index]
    per_sample_ms = (time.perf_counter() - start) / 96 * 1_000
    per_batch_ms = per_sample_ms * args.batch_size
    effective_ms = per_batch_ms / max(args.num_workers, 1)
    print(f"\ndata    {per_sample_ms:6.2f} ms/sample -> {per_batch_ms:7.1f} ms/batch"
          f" -> {effective_ms:7.1f} ms/batch with {args.num_workers} worker(s)")

    criterion = FrequencyDenoiseLoss().to(device)
    chest = torch.randn(args.batch_size, 8_000, device=device) * 0.02
    target = torch.randn(args.batch_size, 8_000, device=device) * 0.02
    use_amp = device.type == "cuda"

    steps_per_epoch = args.samples_per_epoch // args.batch_size
    span = f"{args.epochs} ep"
    print(f"\n{'arch':<22}{'params':>10}{'step ms':>10}{'epoch':>10}{span:>10}")
    per_arch: dict[str, float] = {}
    step_ms_by_arch: dict[str, float] = {}
    for arch in ("cardiospecnet", "cleanunet"):
        model = build_arm(arch).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        def one_step(
            active_model: torch.nn.Module,
            active_optimizer: torch.optim.Optimizer,
            active_scaler: torch.amp.GradScaler,
        ) -> None:
            active_optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, _ = criterion(active_model(chest), target)
            active_scaler.scale(loss).backward()
            active_scaler.step(active_optimizer)
            active_scaler.update()

        for _ in range(3):
            one_step(model, optimizer, scaler)
        sync(device)
        start = time.perf_counter()
        for _ in range(args.steps):
            one_step(model, optimizer, scaler)
        sync(device)
        step_ms = (time.perf_counter() - start) / args.steps * 1_000

        # Data and compute overlap when workers > 0; otherwise they add up.
        batch_ms = max(step_ms, effective_ms) if args.num_workers > 0 else step_ms + per_batch_ms
        epoch_s = steps_per_epoch * batch_ms / 1_000
        per_arch[arch] = epoch_s
        step_ms_by_arch[arch] = step_ms
        print(f"{ARCH_LABEL[arch]:<22}{parameter_count(model):>10,}{step_ms:>10.1f}"
              f"{epoch_s:>9.0f}s{epoch_s * args.epochs / 60:>9.0f}m")
        del model, optimizer

    # Two CardioSpecNet arms (device / expanded) plus one CleanUNet arm.
    projected = (2 * per_arch["cardiospecnet"] + per_arch["cleanunet"]) * args.epochs / 3_600
    print(f"\nthree arms at --epochs {args.epochs} --samples_per_epoch {args.samples_per_epoch} "
          f"--batch_size {args.batch_size}: {projected:.1f} hours total")

    print("\nWhere the time goes")
    slowest = step_ms_by_arch["cardiospecnet"]
    if args.num_workers == 0:
        print(f"  --num_workers 0: the {per_batch_ms:.0f} ms of mixing runs in the training loop, "
              f"on top of a {slowest:.0f} ms step. Set --num_workers 2 first.")
    elif effective_ms > slowest:
        print(f"  data-bound: {effective_ms:.0f} ms/batch of mixing vs a {slowest:.0f} ms step. "
              "Raise --num_workers, or --batch_size (more samples per step, same mixing rate).")
    else:
        print(f"  compute-bound: {effective_ms:.0f} ms/batch of mixing hides behind a "
              f"{slowest:.0f} ms step, so the GPU is busy. Only less work helps from here: "
              "fewer --epochs or a smaller --samples_per_epoch.")
    print("  Whatever you pick, use the same value for all three arms -- otherwise the "
          "comparison measures training budget instead of the thing you meant to test.")


if __name__ == "__main__":
    main()
