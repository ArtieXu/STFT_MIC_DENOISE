#!/usr/bin/env python3
"""Denoise a test_real window archive and write one continuous waveform per arm.

    PYTHONPATH=. python scripts/denoise_real_windows.py \
        --input data/test_real/walking/step_1s/heart_w1_windows.npz \
        --device_only checkpoints/device_only/final.pt \
        --combined    checkpoints/combined/final.pt \
        --waveform    checkpoints/waveform/final.pt

Each 2 s window (8000 samples @ 4 kHz) is processed independently. Windows with a
1 s hop are merged with Hann overlap-add so overlapping regions are averaged, not
duplicated. There is no clean target; output is qualitative only.
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

from src.arms import load_arm  # noqa: E402
from src.pools import SAMPLE_RATE, WINDOW_SAMPLES, remove_dc, to_float  # noqa: E402
from src.window_alignment import (  # noqa: E402
    hop_samples_from_step,
    load_window_archive,
    overlap_add_runs,
    resolve_start_samples,
    split_contiguous_runs,
)

ARM_LABEL = {
    "device_only": "STFT / device",
    "combined": "STFT / expanded",
    "waveform": "waveform / device",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--input", type=Path, required=True,
                        help="test_real npz, e.g. heart_w1_windows.npz")
    parser.add_argument("--device_only", type=Path, default=None)
    parser.add_argument("--combined", type=Path, default=None)
    parser.add_argument("--waveform", type=Path, default=None)
    parser.add_argument("--step", type=str, default="1s",
                        help="Window hop encoded in the archive path (step_1s -> 1 s hop).")
    parser.add_argument("--segment", type=int, default=None,
                        help="Keep one segment_id only (default: all windows).")
    parser.add_argument("--start_seconds", type=float, default=None,
                        help="Crop reconstructed audio after overlap-add.")
    parser.add_argument("--end_seconds", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Default: outputs/real_denoise/<npz stem>")
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def prepare_windows(
    archive_path: Path,
    segment: int | None,
    hop_samples: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    archive = load_window_archive(archive_path)
    if archive is None:
        raise SystemExit(f"archive not found: {archive_path}")
    if archive.x.shape[1] != WINDOW_SAMPLES:
        raise SystemExit(
            f"{archive_path}: expected window length {WINDOW_SAMPLES}, "
            f"got {archive.x.shape[1]}"
        )

    indices = np.arange(len(archive.x))
    if segment is not None:
        indices = indices[archive.segment_id == segment]
    elif len(np.unique(archive.segment_id[indices])) > 1:
        counts = {
            int(value): int(np.sum(archive.segment_id[indices] == value))
            for value in np.unique(archive.segment_id[indices])
        }
        best = max(counts, key=counts.get)
        print(
            "multiple segment_id values found; using the largest segment "
            f"{best} ({counts[best]} windows). Pass --segment to override."
        )
        indices = indices[archive.segment_id == best]

    if len(indices) == 0:
        raise SystemExit("no windows selected")

    order = np.argsort(archive.start_idx[indices], kind="stable")
    indices = indices[order]
    windows = remove_dc(to_float(archive.x[indices]))
    raw_starts = archive.start_idx[indices].astype(np.int64)
    starts = resolve_start_samples(
        raw_starts,
        window_samples=WINDOW_SAMPLES,
        hop_samples=hop_samples,
    )
    runs = split_contiguous_runs(
        starts,
        window_samples=WINDOW_SAMPLES,
        hop_samples=hop_samples,
    )
    meta = {
        "segment_ids": sorted(int(value) for value in np.unique(archive.segment_id[indices])),
        "raw_start_min": int(raw_starts.min()),
        "raw_start_max": int(raw_starts.max()),
        "resolved_runs": [len(run) for run in runs],
        "hop_samples": hop_samples,
    }
    return windows, starts, meta


def run_model(
    model: torch.nn.Module,
    windows: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for begin in range(0, len(windows), batch_size):
            batch = torch.from_numpy(windows[begin : begin + batch_size]).to(device)
            outputs.append(model(batch).cpu().numpy())
    return np.concatenate(outputs, axis=0)


def reconstruct(windows: np.ndarray, starts: np.ndarray, hop_samples: int) -> np.ndarray:
    return overlap_add_runs(
        windows,
        starts,
        hop_samples=hop_samples,
        synthesis_window="hann",
    )


def crop_seconds(
    audio: np.ndarray,
    start_seconds: float | None,
    end_seconds: float | None,
) -> np.ndarray:
    start = 0 if start_seconds is None else int(round(start_seconds * SAMPLE_RATE))
    end = len(audio) if end_seconds is None else int(round(end_seconds * SAMPLE_RATE))
    start = max(0, min(start, len(audio)))
    end = max(start, min(end, len(audio)))
    return audio[start:end]


def write_wav(path: Path, audio: np.ndarray, *, peak: float = 0.9) -> None:
    try:
        import soundfile as sf
    except ImportError as exc:
        raise SystemExit("pip install soundfile to write WAV output") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    maximum = float(np.abs(audio).max())
    scaled = audio if maximum <= 0 else audio * (peak / maximum)
    sf.write(path, scaled.astype(np.float32), SAMPLE_RATE)


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"input not found: {input_path}")

    checkpoints = {
        label: path
        for label, path in (
            ("device_only", args.device_only),
            ("combined", args.combined),
            ("waveform", args.waveform),
        )
        if path is not None
    }
    if not checkpoints:
        raise SystemExit("pass at least one checkpoint: --device_only / --combined / --waveform")

    output_dir = args.output_dir or (REPO_ROOT / "outputs" / "real_denoise" / input_path.stem)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    hop_samples = hop_samples_from_step(args.step, SAMPLE_RATE)
    windows, starts, prep_meta = prepare_windows(input_path, args.segment, hop_samples)
    runs = prep_meta["resolved_runs"]
    run_duration_s = sum(
        (run_len - 1) * hop_samples + WINDOW_SAMPLES for run_len in runs
    ) / SAMPLE_RATE
    hop_msg = "n/a"
    if len(starts) > 1:
        hop_msg = f"{(np.diff(starts).mean() / SAMPLE_RATE):.2f}s mean raw spacing"
    print(
        f"input={input_path.name}; windows={len(windows):,}; "
        f"contiguous_runs={len(runs)} sizes={runs}; "
        f"hop={hop_samples / SAMPLE_RATE:g}s; "
        f"output≈{run_duration_s:.1f}s continuous"
    )
    if len(runs) > 1:
        print(
            "timeline gaps in start_idx were removed by concatenating each contiguous "
            "run back-to-back (no silent holes between valid windows)."
        )

    chest = crop_seconds(
        reconstruct(windows, starts, hop_samples),
        args.start_seconds,
        args.end_seconds,
    )
    write_wav(output_dir / "chest.wav", chest)
    np.save(output_dir / "chest.npy", chest.astype(np.float32))

    manifest = {
        "input": str(input_path),
        "sample_rate_hz": SAMPLE_RATE,
        "window_samples": WINDOW_SAMPLES,
        "n_windows": int(len(windows)),
        "overlap_add": "hann",
        "step": args.step,
        "hop_samples": hop_samples,
        "reconstruction": "contiguous_runs_concatenated",
        "prep": prep_meta,
        "crop_seconds": [args.start_seconds, args.end_seconds],
        "qualitative_only": True,
        "outputs": {"chest": str(output_dir / "chest.wav")},
        "checkpoints": {},
    }

    for label, path in checkpoints.items():
        if not Path(path).is_file():
            raise SystemExit(f"{label} checkpoint not found: {path}")
        model, info = load_arm(path, device, expected_role=label)
        denoised_windows = run_model(model, windows, device, args.batch_size)
        audio = crop_seconds(
            reconstruct(denoised_windows, starts, hop_samples),
            args.start_seconds,
            args.end_seconds,
        )
        wav_path = output_dir / f"{label}.wav"
        npy_path = output_dir / f"{label}.npy"
        write_wav(wav_path, audio)
        np.save(npy_path, audio.astype(np.float32))
        manifest["outputs"][label] = str(wav_path)
        manifest["checkpoints"][label] = {
            "path": str(path),
            "arch_label": ARM_LABEL[label],
            "duration_seconds": len(audio) / SAMPLE_RATE,
            "sha256": info["sha256"],
        }
        print(f"{ARM_LABEL[label]:<22} wrote {wav_path} ({len(audio) / SAMPLE_RATE:.1f}s)")

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
