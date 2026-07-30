#!/usr/bin/env python3
"""Descriptively compare single-microphone denoising systems on fixed mixtures.

    PYTHONPATH=. python scripts/compare_datasets.py \
        --device_only checkpoints/device_only/final.pt \
        --combined    checkpoints/combined/final.pt \
        --waveform    checkpoints/waveform/final.pt

Every system receives exactly the same noisy waveform. No exterior/reference
microphone is supplied. Results are denoising effects over the raw noisy input
and descriptive system differences on this synthetic evaluation set.

The source windows overlap and come from test-only device subject 6 and one
noise recording. They are not independent experimental units, so this script
deliberately reports no p-values or confidence intervals.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.arms import describe_arm, load_arm, validate_shared_protocol  # noqa: E402
from src.frequency_data import MixingConfig, SyntheticFrequencyDataset  # noqa: E402
from src.frequency_metrics import (  # noqa: E402
    log_spectral_distance,
    pearson_correlation,
    si_sdr_db,
    snr_db,
)
from src.pools import (  # noqa: E402
    DEFAULT_TARGET_RMS,
    DEVICE_FILE_SPLITS,
    build_clean_pool,
    build_noise_pool,
    describe,
)

DEFAULT_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
IMPROVEMENT_METRICS = ("snr_db", "si_sdr_db")
ABSOLUTE_METRICS = ("correlation", "lsd_db")
BASELINE = "device_only"
ARM_LABEL = {
    "device_only": "STFT / device",
    "combined": "STFT / expanded",
    "waveform": "waveform / device",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device_only", type=Path, required=True,
                        help="Baseline: CardioSpecNet trained with --no_circor.")
    parser.add_argument("--combined", type=Path, default=None,
                        help="CardioSpecNet trained on device + CirCor.")
    parser.add_argument("--waveform", type=Path, default=None,
                        help="CleanUNet trained on device clean only.")
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--step", type=str, default="1s")
    parser.add_argument("--snrs", type=float, nargs="+", default=list(DEFAULT_SNRS))
    parser.add_argument("--samples_per_snr", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--pool_target_rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--eval_seed", type=int, default=20260729,
                        help="Independent of the training seed; fixes the evaluation mixtures.")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "comparison")
    parser.add_argument("--no_plot", action="store_true")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def per_sample_metrics(estimate: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "snr_db": snr_db(estimate, target).detach().cpu().numpy(),
        "si_sdr_db": si_sdr_db(estimate, target).detach().cpu().numpy(),
        "correlation": pearson_correlation(estimate, target).detach().cpu().numpy(),
        "lsd_db": log_spectral_distance(estimate, target).detach().cpu().numpy(),
    }


def mean_within_mixture_delta(values: np.ndarray, baseline: np.ndarray) -> float:
    """Mean within-mixture difference; no inferential statistics."""
    delta = np.asarray(values, dtype=np.float64) - np.asarray(baseline, dtype=np.float64)
    if delta.shape != np.asarray(values).shape:
        raise ValueError("system and baseline metric arrays must have identical shape")
    return float(delta.mean())


def build_eval_pools(args: argparse.Namespace):
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None
    clean = build_clean_pool(
        args.data_root / "clean" / f"step_{args.step}" / "test", "test",
        circor_pool=None, target_rms=target_rms,
    )
    noise = build_noise_pool(
        args.data_root / "noise" / f"step_{args.step}" / "test", "test",
        target_rms=target_rms,
    )
    return clean, noise, target_rms


def validate_final_test_pools(clean, noise) -> dict[str, list[str]]:
    """Require the exact subject-6 recordings declared by the protocol."""

    observed = {
        "clean_subjects": sorted(set(clean.subject.tolist())),
        "noise_subjects": sorted(set(noise.subject.tolist())),
        "clean_origins": sorted(set(clean.origin.tolist())),
        "noise_origins": sorted(set(noise.origin.tolist())),
    }
    expected = {
        "clean_subjects": ["device_6"],
        "noise_subjects": ["device_6"],
        "clean_origins": sorted(
            f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["clean"]["test"]
        ),
        "noise_origins": sorted(
            f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["noise"]["test"]
        ),
    }
    mismatches = [
        f"{key}: expected {expected[key]}, got {observed[key]}"
        for key in expected
        if observed[key] != expected[key]
    ]
    if mismatches:
        raise RuntimeError(
            "controlled final test does not match the exact subject-6 manifest: "
            + "; ".join(mismatches)
        )
    return observed


def eval_config(target_snr: float) -> MixingConfig:
    return MixingConfig(
        snr_min_db=target_snr,
        snr_max_db=target_snr,
    )


def evaluate(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    clean, noise, target_rms = build_eval_pools(args)
    print("Test pools (device subject 6, identical for every checkpoint)")
    print(describe(clean))
    print(describe(noise))
    final_test_manifest = validate_final_test_pools(clean, noise)

    requested = [(BASELINE, args.device_only), ("combined", args.combined), ("waveform", args.waveform)]
    models: dict[str, object] = {}
    infos: dict[str, object] = {}
    for label, path in requested:
        if path is None:
            continue
        models[label], infos[label] = load_arm(path, device, expected_role=label)
        print(f"\n{ARM_LABEL[label]:<20} {path}")
        print(f"{'':20} {describe_arm(infos[label])}")
    if len(models) < 2:   # baseline system + at least one comparison system
        raise SystemExit("Nothing to compare: pass --combined and/or --waveform.")
    validate_shared_protocol(infos)

    keys = ("input", *models)
    results: dict[float, dict] = {}
    for target_snr in args.snrs:
        dataset = SyntheticFrequencyDataset(
            clean, noise,
            samples_per_epoch=args.samples_per_snr,
            seed=args.eval_seed,
            config=eval_config(target_snr),
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        collected = {
            key: {metric: [] for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)}
            for key in keys
        }
        with torch.inference_mode():
            for batch in loader:
                chest = batch["chest"].to(device)
                target = batch["clean"].to(device)
                for metric, values in per_sample_metrics(chest, target).items():
                    collected["input"][metric].append(values)
                for label, model in models.items():
                    output = model(chest)
                    for metric, values in per_sample_metrics(output, target).items():
                        collected[label][metric].append(values)

        stacked = {
            key: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
            for key, metrics in collected.items()
        }
        bin_result: dict[str, object] = {
            "n": int(args.samples_per_snr),
            "input": {
                metric: float(stacked["input"][metric].mean())
                for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)
            },
            "arms": {},
            "system_delta_descriptive": {},
        }
        for label in models:
            entry = {
                metric: float(stacked[label][metric].mean())
                for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)
            }
            for metric in IMPROVEMENT_METRICS:
                entry[f"{metric}_improvement"] = float(
                    (stacked[label][metric] - stacked["input"][metric]).mean()
                )
            bin_result["arms"][label] = entry
        for label in models:
            if label == BASELINE:
                continue
            bin_result["system_delta_descriptive"][label] = {
                metric: mean_within_mixture_delta(
                    stacked[label][metric], stacked[BASELINE][metric]
                )
                for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)
            }
        results[float(target_snr)] = bin_result
        print(f"  SNR {target_snr:+6.1f} dB done")

    return {
        "eval_seed": args.eval_seed,
        "samples_per_snr": args.samples_per_snr,
        "pool_target_rms": target_rms,
        "eval_clean_windows": len(clean),
        "eval_noise_windows": len(noise),
        "eval_subjects": ["device_6"],
        "eval_clean_subjects": final_test_manifest["clean_subjects"],
        "eval_noise_subjects": final_test_manifest["noise_subjects"],
        "eval_clean_origins": final_test_manifest["clean_origins"],
        "eval_noise_origins": final_test_manifest["noise_origins"],
        "denoising_baseline": "raw_noisy_input",
        "system_difference_baseline": BASELINE,
        "arms": list(models),
        "input_mode": "single_mic",
        "target": "raw_clean",
        "snr_definition": "full_band",
        "comparison_scope": "descriptive denoising effect and system comparison",
        "statistical_inference": {
            "enabled": False,
            "reason": (
                "overlapping windows from one held-out device subject and one noise "
                "recording are not independent experimental units"
            ),
        },
        "checkpoints": infos,
        "bins": results,
    }


def macro_mean_delta(bins: dict, label: str, metric: str) -> float:
    """Equal-weight mean of the per-SNR descriptive differences."""
    return float(np.mean([
        bins[snr]["system_delta_descriptive"][label][metric] for snr in sorted(bins)
    ]))


def print_tables(report: dict) -> None:
    bins = report["bins"]
    arms = report["arms"]
    order = sorted(bins)

    for metric, unit in (("snr_db", "SNRi"), ("si_sdr_db", "SI-SDRi")):
        print("\n" + "=" * 92)
        print(f"{unit} dB -- gain over the raw noisy input, identical mixtures")
        print("=" * 92)
        header = f"{'input SNR':>10} " + " ".join(f"{ARM_LABEL[a]:>20}" for a in arms)
        print(header)
        for snr in order:
            row = f"{snr:>+10.1f} "
            for label in arms:
                row += f"{bins[snr]['arms'][label][metric + '_improvement']:>20.2f} "
            print(row)

        for label in arms:
            if label == BASELINE:
                continue
            print(f"\n  {ARM_LABEL[label]} minus {ARM_LABEL[BASELINE]} "
                  "(descriptive, same mixtures)")
            print(f"  {'input SNR':>10}{'mean delta':>14}")
            for snr in order:
                delta = bins[snr]["system_delta_descriptive"][label][metric]
                print(f"  {snr:>+10.1f}{delta:>+14.2f}")
            print(f"  {'macro mean':>10}{macro_mean_delta(bins, label, metric):>+14.2f}")

    print("\n" + "=" * 92)
    print("correlation (higher better) and log-spectral distance dB (lower better)")
    print("=" * 92)
    print(f"{'input SNR':>10} " + " ".join(f"{ARM_LABEL[a] + ' corr':>25}" for a in arms))
    for snr in order:
        row = f"{snr:>+10.1f} "
        for label in arms:
            row += f"{bins[snr]['arms'][label]['correlation']:>25.3f} "
        print(row)
    print(f"{'input SNR':>10} " + " ".join(f"{ARM_LABEL[a] + ' LSD':>25}" for a in arms))
    for snr in order:
        row = f"{snr:>+10.1f} "
        for label in arms:
            row += f"{bins[snr]['arms'][label]['lsd_db']:>25.2f} "
        print(row)


def make_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = report["bins"]
    arms = report["arms"]
    order = sorted(bins)
    styles = {"device_only": ("o--", "tab:blue"),
              "combined": ("s-", "tab:red"), "waveform": ("^-.", "tab:green")}

    figure, axes = plt.subplots(1, 3, figsize=(16.5, 4.4))
    for metric, axis, title in (
        ("snr_db", axes[0], "SNR improvement (dB)"),
        ("si_sdr_db", axes[1], "SI-SDR improvement (dB)"),
    ):
        for label in arms:
            style, colour = styles.get(label, ("o-", None))
            axis.plot(
                order,
                [bins[snr]["arms"][label][metric + "_improvement"] for snr in order],
                style, color=colour, label=ARM_LABEL[label],
            )
        axis.axhline(0.0, color="k", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("input SNR (dB)")
        axis.legend(fontsize=8)

    axis = axes[2]
    for label in arms:
        if label == BASELINE:
            continue
        _, colour = styles.get(label, ("o-", None))
        means = [
            bins[snr]["system_delta_descriptive"][label]["snr_db"] for snr in order
        ]
        axis.plot(order, means, "-o", color=colour, label=f"{ARM_LABEL[label]} − baseline")
    axis.axhline(0.0, color="k", linewidth=0.9)
    axis.set_title("mean SNR system difference (descriptive)")
    axis.set_xlabel("input SNR (dB)")
    axis.set_ylabel("dB")
    axis.legend(fontsize=8)

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    report = evaluate(args)
    print_tables(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_plot:
        plot_path = args.output_dir / "comparison.png"
        make_plot(report, plot_path)
        print(f"\nwrote {plot_path}")

    json_path = args.output_dir / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print("\nDescriptive only: no confidence interval or p-value is reported because "
          "overlapping windows from one subject/noise recording are not independent. "
          "Repeat training seeds and evaluate independent subjects before making a "
          "general system claim.")


if __name__ == "__main__":
    main()
