#!/usr/bin/env python3
"""Fair spectrogram views of the single-microphone denoising systems.

    PYTHONPATH=. python scripts/make_demo_figures.py --all \
        --device_only checkpoints/device_only/final.pt \
        --combined    checkpoints/combined/final.pt \
        --waveform    checkpoints/waveform/final.pt

Two figures, because they answer different objections.

``demo_synthetic_grid.png`` uses one fixed clean/noise example at several SNRs.
Every system receives the same noisy waveform, with no reference microphone.
Every panel shares one dB colour scale and the same 0--800 Hz frequency range.

``demo_real_walking.png`` is a fixed 0--10 s excerpt from the designated
test-only subject-6 walking recording. There is no clean target, so it is
explicitly qualitative only and carries no denoising metric or efficacy claim.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.arms import describe_arm, load_arm, validate_shared_protocol  # noqa: E402
from src.frequency_data import MixingConfig, SyntheticFrequencyDataset  # noqa: E402
from src.frequency_metrics import si_sdr_db, snr_db  # noqa: E402
from src.pools import (  # noqa: E402
    DEFAULT_TARGET_RMS,
    DEVICE_FILE_SPLITS,
    build_clean_pool,
    build_noise_pool,
    to_float,
)
from src.window_alignment import load_window_archive, overlap_add_windows  # noqa: E402

ARM_LABEL = {
    "device_only": "STFT / device",
    "combined": "STFT / expanded",
    "waveform": "waveform / device",
}
SAMPLE_RATE = 4_000
SPECTROGRAM_MAX_HZ = 800.0
REAL_TEST_SUBJECT = "6"
REAL_EXCERPT_SECONDS = (0.0, 10.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device_only", type=Path,
                        default=REPO_ROOT / "checkpoints" / "device_only" / "final.pt")
    parser.add_argument("--combined", type=Path,
                        default=REPO_ROOT / "checkpoints" / "combined" / "final.pt")
    parser.add_argument("--waveform", type=Path,
                        default=REPO_ROOT / "checkpoints" / "waveform" / "final.pt")
    parser.add_argument("--all", action="store_true", help="Both figures (default if neither flag given).")
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--real", action="store_true")
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--step", type=str, default="1s")
    parser.add_argument("--snrs", type=float, nargs="+", default=[-5.0, 0.0, 5.0, 10.0],
                        help="Columns of the synthetic figure.")
    parser.add_argument("--sample_index", type=int, default=0,
                        help="Which mixture from each SNR bin to draw.")
    parser.add_argument("--eval_seed", type=int, default=20260729,
                        help="Same default as compare_datasets.py, so the picture matches the table.")
    parser.add_argument("--walking", type=Path,
                        default=REPO_ROOT / "data" / "test_real" / "walking" / "step_1s"
                        / "heart_w6_windows.npz",
                        help="Path to the designated subject-6 test-only walking archive. "
                             "The recording identity and 0--10 s excerpt are fixed.")
    parser.add_argument("--dynamic_range_db", type=float, default=45.0)
    parser.add_argument("--pool_target_rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--wav", action="store_true",
                        help="Also export WAVs (needs soundfile). Listening beats looking.")
    parser.add_argument("--output_dir", type=Path, default=REPO_ROOT / "outputs" / "demo")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def load_arms(args: argparse.Namespace, device: torch.device):
    models: dict = {}
    infos: dict = {}
    for label, path in (("device_only", args.device_only), ("combined", args.combined),
                        ("waveform", args.waveform)):
        if path is None or not Path(path).is_file():
            raise SystemExit(f"required {label} checkpoint not found: {path}")
        models[label], infos[label] = load_arm(path, device, expected_role=label)
        print(f"{ARM_LABEL[label]:<26} {describe_arm(infos[label])}")
    validate_shared_protocol(infos)
    return models, infos


def spectrogram(x: np.ndarray):
    waveform = torch.as_tensor(np.asarray(x), dtype=torch.float32)
    spectrum = torch.stft(
        waveform,
        n_fft=256,
        hop_length=64,
        win_length=256,
        window=torch.hann_window(256),
        center=True,
        return_complex=True,
    )
    freqs = torch.fft.rfftfreq(256, d=1.0 / SAMPLE_RATE)
    keep = freqs <= SPECTROGRAM_MAX_HZ
    times = torch.arange(spectrum.shape[-1], dtype=torch.float32) * (64.0 / SAMPLE_RATE)
    values = 20.0 * torch.log10(spectrum.abs().clamp_min(1e-10))
    return (
        freqs[keep].numpy(),
        times.numpy(),
        values[keep].numpy(),
    )


def draw_panel(axis, signal, vmin, vmax, title):
    freqs, times, values = spectrogram(signal)
    axis.pcolormesh(times, freqs, values, vmin=vmin, vmax=vmax, shading="nearest", cmap="magma")
    axis.set_title(title, fontsize=8)
    axis.tick_params(labelsize=7)


# --------------------------------------------------------------------------- #
# figure 1: synthetic mixtures, ground truth available
# --------------------------------------------------------------------------- #
def validate_final_test_pools(clean_pool, noise_pool) -> dict[str, list[str]]:
    """Require the exact subject-6 recordings declared by the protocol."""

    observed = {
        "clean_subjects": sorted(set(clean_pool.subject.tolist())),
        "noise_subjects": sorted(set(noise_pool.subject.tolist())),
        "clean_origins": sorted(set(clean_pool.origin.tolist())),
        "noise_origins": sorted(set(noise_pool.origin.tolist())),
    }
    expected = {
        "clean_subjects": [f"device_{REAL_TEST_SUBJECT}"],
        "noise_subjects": [f"device_{REAL_TEST_SUBJECT}"],
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
            "synthetic final test does not match the exact subject-6 manifest: "
            + "; ".join(mismatches)
        )
    return observed


def synthetic_figure(args, models, device) -> dict:
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None
    clean_pool = build_clean_pool(
        args.data_root / "clean" / f"step_{args.step}" / "test", "test",
        circor_pool=None, target_rms=target_rms,
    )
    noise_pool = build_noise_pool(
        args.data_root / "noise" / f"step_{args.step}" / "test", "test",
        target_rms=target_rms,
    )
    final_test_manifest = validate_final_test_pools(clean_pool, noise_pool)

    rows = ["clean", "noisy", *models]
    figure, axes = plt.subplots(
        len(rows), len(args.snrs),
        figsize=(3.6 * len(args.snrs), 2.1 * len(rows)),
        squeeze=False, sharex=True, sharey=True,
    )
    numbers: dict[str, dict] = {}
    audio: dict[str, np.ndarray] = {}
    common_clean: np.ndarray | None = None
    common_vmin: float | None = None
    common_vmax: float | None = None

    for column, target_snr in enumerate(args.snrs):
        dataset = SyntheticFrequencyDataset(
            clean_pool, noise_pool,
            samples_per_epoch=args.sample_index + 1,
            seed=args.eval_seed,
            config=MixingConfig(
                snr_min_db=target_snr, snr_max_db=target_snr,
            ),
        )
        sample = dataset[args.sample_index]
        chest = sample["chest"][None].to(device)
        clean = sample["clean"][None].to(device)

        clean_np = clean[0].cpu().numpy()
        chest_np = chest[0].cpu().numpy()
        if common_clean is None:
            common_clean = clean_np.copy()
            _, _, clean_db = spectrogram(common_clean)
            common_vmax = float(clean_db.max())
            common_vmin = common_vmax - args.dynamic_range_db
        elif not np.array_equal(clean_np, common_clean):
            raise RuntimeError(
                "SNR columns resolved to different clean samples; the visual comparison "
                "requires one fixed sample"
            )
        assert common_vmin is not None and common_vmax is not None

        raw_snr = float(snr_db(chest, clean))
        panels = [("clean", clean_np, "clean target"),
                  ("noisy", chest_np, f"noisy input  (raw {raw_snr:+.1f} dB)")]
        column_numbers = {"nominal_snr_db": float(target_snr), "raw_input_snr_db": raw_snr}
        with torch.inference_mode():
            for label, model in models.items():
                estimate = model(chest)
                estimate_np = estimate[0].cpu().numpy()
                out_snr = float(snr_db(estimate, clean))
                out_sisdr = float(si_sdr_db(estimate, clean))
                delta = out_snr - raw_snr
                title = (
                    f"{ARM_LABEL[label]}   {out_snr:+.1f} dB  "
                    f"(vs noisy {delta:+.1f})"
                )
                panels.append((label, estimate_np, title))
                column_numbers[label] = {
                    "output_snr_db": out_snr,
                    "gain_over_noisy_db": delta,
                    "output_si_sdr_db": out_sisdr,
                }
                audio[f"snr{target_snr:+.0f}_{label}"] = estimate_np
        audio[f"snr{target_snr:+.0f}_clean"] = clean_np
        audio[f"snr{target_snr:+.0f}_noisy"] = chest_np

        for row, (_, signal, title) in enumerate(panels):
            draw_panel(axes[row][column], signal, common_vmin, common_vmax, title)
        numbers[f"{target_snr:+.0f} dB"] = column_numbers

    for row in range(len(rows)):
        axes[row][0].set_ylabel("Hz", fontsize=8)
    for column in range(len(args.snrs)):
        axes[-1][column].set_xlabel("s", fontsize=8)
    figure.suptitle(
        "Synthetic mixtures, held-out device subject\n"
        "one fixed sample across SNRs; identical noisy waveform for every system; "
        "single shared dB scale; 0–800 Hz; gains over raw noisy input",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.955))
    path = args.output_dir / "demo_synthetic_grid.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")
    if args.wav:
        export_wavs(audio, args.output_dir / "wav_synthetic")
    return {
        "subject": f"device_{REAL_TEST_SUBJECT}",
        "clean_origins": final_test_manifest["clean_origins"],
        "noise_origins": final_test_manifest["noise_origins"],
        "sample_index": args.sample_index,
        "snrs_db": [float(value) for value in args.snrs],
        "shared_db_limits": [common_vmin, common_vmax],
        "columns": numbers,
    }


# --------------------------------------------------------------------------- #
# figure 2: fixed real walking excerpt, no ground truth
# --------------------------------------------------------------------------- #
def real_walking_figure(args, models, device) -> dict:
    if not Path(args.walking).is_file():
        raise SystemExit(
            f"required qualitative subject-6 recording not found: {args.walking}\n"
            "run: python scripts/fetch_device_data.py --include-test-real"
        )
    expected_stem = f"heart_w{REAL_TEST_SUBJECT}_windows"
    if Path(args.walking).stem != expected_stem:
        raise SystemExit(
            f"real demo is fixed to designated test-only subject {REAL_TEST_SUBJECT}: "
            f"expected {expected_stem}.npz, got {Path(args.walking).name}"
        )

    chest_archive = load_window_archive(args.walking)

    segment = int(np.unique(chest_archive.segment_id)[0])
    mask = chest_archive.segment_id == segment
    origin = int(chest_archive.start_idx[mask].min())

    start, end = REAL_EXCERPT_SECONDS
    start_sample = origin + int(round(start * SAMPLE_RATE))
    end_sample = origin + int(round(end * SAMPLE_RATE))
    length = chest_archive.x.shape[1]
    select = mask & (chest_archive.start_idx < end_sample) & (
        chest_archive.start_idx + length > start_sample
    )
    indices = np.flatnonzero(select)
    if len(indices) == 0:
        raise SystemExit("No walking windows overlap the requested interval")

    chest_windows = to_float(chest_archive.x[indices])
    chest_windows = chest_windows - chest_windows.mean(axis=1, keepdims=True)
    starts = chest_archive.start_idx[indices]

    estimates: dict[str, np.ndarray] = {}
    with torch.inference_mode():
        for label, model in models.items():
            chunks = []
            for begin in range(0, len(chest_windows), args.batch_size):
                stop = begin + args.batch_size
                noisy = torch.from_numpy(chest_windows[begin:stop]).to(device)
                chunks.append(model(noisy).cpu().numpy())
            estimates[label] = np.concatenate(chunks, axis=0)

    def reconstruct(windows: np.ndarray) -> np.ndarray:
        # overlap_add_windows indexes by the absolute start_idx of each window --
        # it allocates up to starts.max() + window_length and writes at those
        # offsets -- so the crop is absolute too. (Upstream's inference script
        # subtracts starts.min() here as well, which shifts the crop earlier by a
        # whole excerpt and pads the front of every figure with silence.)
        audio = overlap_add_windows(windows, starts)
        return audio[start_sample:end_sample]

    chest_audio = reconstruct(chest_windows)
    panels = [("chest", chest_audio, "chest microphone (walking, unprocessed)")]
    audio_export = {"chest": chest_audio}
    for label in models:
        signal = reconstruct(estimates[label])
        panels.append((label, signal, ARM_LABEL[label]))
        audio_export[label] = signal

    _, _, chest_db = spectrogram(chest_audio)
    vmax = float(chest_db.max())
    vmin = vmax - args.dynamic_range_db

    figure, axes = plt.subplots(len(panels), 1, figsize=(11, 2.1 * len(panels)),
                                sharex=True, sharey=True)
    for axis, (_, signal, title) in zip(np.atleast_1d(axes), panels):
        draw_panel(axis, signal, vmin, vmax, title)
        axis.set_ylabel("Hz", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("s", fontsize=8)
    figure.suptitle(
        f"QUALITATIVE ONLY — subject {REAL_TEST_SUBJECT} test-only walking, "
        f"fixed {start:g}–{end:g} s excerpt\n"
        "single microphone; no clean target and no efficacy metric; shared dB scale; 0–800 Hz",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.965))
    path = args.output_dir / "demo_real_walking.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    print(f"wrote {path}")
    if args.wav:
        export_wavs(audio_export, args.output_dir / "wav_real")
    return {
        "recording": Path(args.walking).stem,
        "subject": REAL_TEST_SUBJECT,
        "test_only": True,
        "qualitative_only": True,
        "input_mode": "single_mic",
        "excerpt_selection": "fixed before inference",
        "seconds": [start, end],
        "metric": None,
    }


def export_wavs(signals: dict[str, np.ndarray], directory: Path) -> None:
    try:
        import soundfile as sf
    except ImportError:
        print("skip --wav: pip install soundfile")
        return
    directory.mkdir(parents=True, exist_ok=True)
    shared_peak = max((float(np.abs(signal).max()) for signal in signals.values()), default=0.0)
    gain = 0.9 / shared_peak if shared_peak > 0 else 1.0
    for name, signal in signals.items():
        scaled = signal * gain
        sf.write(directory / f"{name}.wav", scaled.astype(np.float32), SAMPLE_RATE)
    print(f"wrote {len(signals)} WAVs to {directory} (one shared gain preserves levels)")


def main() -> None:
    args = parse_args()
    if not (args.synthetic or args.real):
        args.all = True
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    models, infos = load_arms(args, device)

    summary: dict[str, object] = {
        "checkpoints": infos,
        "arms": list(models),
        "eval_seed": args.eval_seed,
        "input_mode": "single_mic",
        "target": "raw_clean",
        "snr_definition": "full_band",
        "spectrogram_hz": [0.0, SPECTROGRAM_MAX_HZ],
        "shared_colour_scale_within_each_figure": True,
    }
    if args.all or args.synthetic:
        summary["synthetic"] = synthetic_figure(args, models, device)
    if args.all or args.real:
        summary["real_walking"] = real_walking_figure(args, models, device)

    path = args.output_dir / "demo_summary.json"
    path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
