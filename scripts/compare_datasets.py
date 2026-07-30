#!/usr/bin/env python3
"""Score the two training-set arms on one identical evaluation set.

    PYTHONPATH=. python scripts/compare_datasets.py \
        --device_only checkpoints/device_only/best.pt \
        --combined    checkpoints/combined/best.pt

The evaluation set is built here, not read from either run: device clean val
(held-out subject 4) mixed with device val noise at fixed SNR bins, one seed.
Every checkpoint sees byte-identical mixtures, so the comparison is *paired* --
per-sample differences, which is far more sensitive than comparing two means.

Outputs a per-SNR table, `comparison.json`, and `comparison.png`.

What this does and does not tell you:

  does      whether these two checkpoints differ on this evaluation set, and by
            how much, with a paired confidence interval
  does not  whether the difference survives training-seed variance. Two runs is
            two samples of one noisy process. Re-run both arms with --seed
            changed before believing a small gap.
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

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset  # noqa: E402
from src.frequency_metrics import (  # noqa: E402
    log_spectral_distance,
    pearson_correlation,
    si_sdr_db,
    snr_db,
)
from src.frequency_model import build_frequency_model  # noqa: E402
from src.pools import DEFAULT_TARGET_RMS, build_clean_pool, build_noise_pool, describe  # noqa: E402

DEFAULT_SNRS = (-10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)
#: Reported per SNR bin. Improvements are output minus input on the same sample.
IMPROVEMENT_METRICS = ("snr_db", "si_sdr_db")
ABSOLUTE_METRICS = ("correlation", "lsd_db")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device_only", type=Path, required=True,
                        help="Arm A checkpoint (trained with --no_circor).")
    parser.add_argument("--combined", type=Path, required=True,
                        help="Arm B checkpoint (trained on device + CirCor).")
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


def load_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_frequency_model(checkpoint.get("model_config")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    info = {
        "path": str(path),
        "epoch": checkpoint.get("epoch"),
        "parameter_count": checkpoint.get("parameter_count"),
        "arm_recorded_in_checkpoint": (checkpoint.get("pool_stats") or {}).get("arm"),
        "train_clean_windows": (
            ((checkpoint.get("pool_stats") or {}).get("train") or {}).get("clean", {}) or {}
        ).get("n_windows"),
        "train_clean_sources": (
            ((checkpoint.get("pool_stats") or {}).get("train") or {}).get("clean", {}) or {}
        ).get("source_counts"),
    }
    return model, info


def per_sample_metrics(estimate: torch.Tensor, target: torch.Tensor) -> dict[str, np.ndarray]:
    return {
        "snr_db": snr_db(estimate, target).detach().cpu().numpy(),
        "si_sdr_db": si_sdr_db(estimate, target).detach().cpu().numpy(),
        "correlation": pearson_correlation(estimate, target).detach().cpu().numpy(),
        "lsd_db": log_spectral_distance(estimate, target).detach().cpu().numpy(),
    }


def paired_delta(values_b: np.ndarray, values_a: np.ndarray) -> dict[str, float]:
    """Mean paired difference (B - A) with a 95% interval and a paired p-value."""
    delta = np.asarray(values_b, dtype=np.float64) - np.asarray(values_a, dtype=np.float64)
    n = len(delta)
    mean = float(delta.mean())
    if n < 2:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "p_value": float("nan"), "n": n}
    sem = float(delta.std(ddof=1) / np.sqrt(n))
    try:
        from scipy import stats

        critical = float(stats.t.ppf(0.975, n - 1))
        p_value = float(stats.ttest_rel(values_b, values_a).pvalue)
    except Exception:  # scipy absent: normal approximation, no p-value
        critical, p_value = 1.96, float("nan")
    return {
        "mean": mean,
        "ci_low": mean - critical * sem,
        "ci_high": mean + critical * sem,
        "p_value": p_value,
        "n": n,
    }


def evaluate(args: argparse.Namespace) -> dict:
    device = choose_device(args.device)
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None

    # Evaluation pools: device only, held-out subject, in both arms.
    clean = build_clean_pool(
        args.data_root / "clean" / f"step_{args.step}" / "val", "val",
        circor_pool=None, target_rms=target_rms,
    )
    noise = build_noise_pool(
        args.data_root / "noise" / f"step_{args.step}" / "val", "val",
        target_rms=target_rms,
    )
    print("Evaluation pools (device-only, identical for every checkpoint)")
    print(describe(clean))
    print(describe(noise))

    models = {}
    infos = {}
    for label, path in (("device_only", args.device_only), ("combined", args.combined)):
        models[label], infos[label] = load_model(path, device)
        info = infos[label]
        print(f"\n{label:12} {path}")
        print(f"{'':12} epoch {info['epoch']}, {info['parameter_count']} params, "
              f"trained on {info['train_clean_windows']} clean windows "
              f"{info['train_clean_sources']}")

    results: dict[float, dict] = {}
    for target_snr in args.snrs:
        config = MixingConfig(
            snr_min_db=target_snr,
            snr_max_db=target_snr,
            identity_probability=0.0,
            reference_dropout_probability=0.0,
            chest_only_noise_max=0.0,   # keep the SNR axis exact
        )
        dataset = SyntheticFrequencyDataset(
            clean, noise,
            samples_per_epoch=args.samples_per_snr,
            seed=args.eval_seed,
            config=config,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        collected: dict[str, dict[str, list[np.ndarray]]] = {
            key: {metric: [] for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)}
            for key in ("input", *models)
        }
        with torch.inference_mode():
            for batch in loader:
                chest = batch["chest"].to(device)
                clean_target = batch["clean"].to(device)
                reference = batch["reference"].to(device)
                available = batch["ref_available"].to(device)

                for metric, values in per_sample_metrics(chest, clean_target).items():
                    collected["input"][metric].append(values)
                for label, model in models.items():
                    output = model(chest, reference, available)
                    for metric, values in per_sample_metrics(output, clean_target).items():
                        collected[label][metric].append(values)

        stacked = {
            key: {metric: np.concatenate(chunks) for metric, chunks in metrics.items()}
            for key, metrics in collected.items()
        }
        bin_result: dict[str, object] = {"n": int(args.samples_per_snr), "arms": {}}
        for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS):
            bin_result.setdefault("input", {})[metric] = float(stacked["input"][metric].mean())
        for label in models:
            entry: dict[str, float] = {}
            for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS):
                entry[metric] = float(stacked[label][metric].mean())
            for metric in IMPROVEMENT_METRICS:
                entry[f"{metric}_improvement"] = float(
                    (stacked[label][metric] - stacked["input"][metric]).mean()
                )
            bin_result["arms"][label] = entry

        bin_result["paired_delta"] = {
            metric: paired_delta(stacked["combined"][metric], stacked["device_only"][metric])
            for metric in (*IMPROVEMENT_METRICS, *ABSOLUTE_METRICS)
        }
        bin_result["_samples"] = stacked  # kept for the plot, dropped before JSON
        results[float(target_snr)] = bin_result
        print(f"  SNR {target_snr:+6.1f} dB done")

    return {
        "eval_seed": args.eval_seed,
        "samples_per_snr": args.samples_per_snr,
        "pool_target_rms": target_rms,
        "eval_clean_windows": len(clean),
        "eval_noise_windows": len(noise),
        "eval_subjects": sorted(set(clean.subject.tolist())),
        "checkpoints": infos,
        "bins": results,
    }


def print_tables(report: dict) -> None:
    bins = report["bins"]
    for metric, unit in (("snr_db", "SNRi dB"), ("si_sdr_db", "SI-SDRi dB")):
        print("\n" + "=" * 84)
        print(f"{unit}   (output minus input, on identical mixtures)")
        print("=" * 84)
        print(f"{'input SNR':>10}{'A device-only':>16}{'B combined':>14}"
              f"{'B - A':>10}{'95% CI':>20}{'p':>10}")
        for snr in sorted(bins):
            arms = bins[snr]["arms"]
            delta = bins[snr]["paired_delta"][metric]
            print(f"{snr:>+10.1f}"
                  f"{arms['device_only'][metric + '_improvement']:>16.2f}"
                  f"{arms['combined'][metric + '_improvement']:>14.2f}"
                  f"{delta['mean']:>+10.2f}"
                  f"{f'[{delta_ci(delta)}]':>20}"
                  f"{format_p(delta['p_value']):>10}")
        overall = pooled_delta(bins, metric)
        print("-" * 84)
        print(f"{'pooled':>10}{'':16}{'':14}{overall['mean']:>+10.2f}"
              f"{f'[{delta_ci(overall)}]':>20}{format_p(overall['p_value']):>10}")

    print("\n" + "=" * 84)
    print("correlation and log-spectral distance (absolute, not improvements)")
    print("=" * 84)
    print(f"{'input SNR':>10}{'corr A':>10}{'corr B':>10}{'d corr':>10}"
          f"{'LSD A':>10}{'LSD B':>10}{'d LSD':>10}")
    for snr in sorted(bins):
        arms = bins[snr]["arms"]
        print(f"{snr:>+10.1f}"
              f"{arms['device_only']['correlation']:>10.3f}"
              f"{arms['combined']['correlation']:>10.3f}"
              f"{bins[snr]['paired_delta']['correlation']['mean']:>+10.3f}"
              f"{arms['device_only']['lsd_db']:>10.2f}"
              f"{arms['combined']['lsd_db']:>10.2f}"
              f"{bins[snr]['paired_delta']['lsd_db']['mean']:>+10.2f}")
    print("\nLSD: lower is better. corr and SNRi/SI-SDRi: higher is better.")


def delta_ci(delta: dict) -> str:
    return f"{delta['ci_low']:+.2f}, {delta['ci_high']:+.2f}"


def format_p(value: float) -> str:
    if value != value:  # NaN
        return "n/a"
    return "<0.001" if value < 0.001 else f"{value:.3f}"


def pooled_delta(bins: dict, metric: str) -> dict[str, float]:
    combined = np.concatenate([bins[snr]["_samples"]["combined"][metric] for snr in sorted(bins)])
    device_only = np.concatenate(
        [bins[snr]["_samples"]["device_only"][metric] for snr in sorted(bins)]
    )
    return paired_delta(combined, device_only)


def make_plot(report: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bins = report["bins"]
    snrs = sorted(bins)
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for metric, axis, title in (
        ("snr_db", axes[0], "SNR improvement (dB)"),
        ("si_sdr_db", axes[1], "SI-SDR improvement (dB)"),
    ):
        for label, style in (("device_only", "o--"), ("combined", "s-")):
            axis.plot(
                snrs,
                [bins[snr]["arms"][label][metric + "_improvement"] for snr in snrs],
                style, label=label,
            )
        axis.axhline(0.0, color="k", linewidth=0.6)
        axis.set_title(title)
        axis.set_xlabel("input SNR (dB)")
        axis.legend()

    axis = axes[2]
    for metric, colour in (("snr_db", "tab:blue"), ("si_sdr_db", "tab:orange")):
        means = [bins[snr]["paired_delta"][metric]["mean"] for snr in snrs]
        low = [bins[snr]["paired_delta"][metric]["ci_low"] for snr in snrs]
        high = [bins[snr]["paired_delta"][metric]["ci_high"] for snr in snrs]
        axis.plot(snrs, means, "-o", color=colour, label=metric)
        axis.fill_between(snrs, low, high, color=colour, alpha=0.2)
    axis.axhline(0.0, color="k", linewidth=0.8)
    axis.set_title("combined − device-only (paired, 95% CI)")
    axis.set_xlabel("input SNR (dB)")
    axis.set_ylabel("dB")
    axis.legend()

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

    for bin_result in report["bins"].values():
        bin_result.pop("_samples", None)
    json_path = args.output_dir / "comparison.json"
    json_path.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {json_path}")
    print("\nReminder: this is two checkpoints on one evaluation set. A gap inside the "
          "confidence interval, or of a size comparable to it, is not yet evidence "
          "for or against the expanded dataset -- re-run both arms with a different "
          "--seed to see whether it survives training variance.")


if __name__ == "__main__":
    main()
