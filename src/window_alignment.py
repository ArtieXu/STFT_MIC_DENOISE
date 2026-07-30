"""Load and overlap-add windowed PCG archives."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WindowArchive:
    x: np.ndarray
    start_idx: np.ndarray
    segment_id: np.ndarray


def load_window_archive(path: Path | str | None) -> WindowArchive | None:
    if path is None:
        return None
    archive_path = Path(path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Window archive not found: {archive_path}")
    payload = np.load(archive_path)
    if "x" not in payload:
        raise KeyError(f"{archive_path} must contain key 'x'")
    start_idx = payload["start_idx"] if "start_idx" in payload else np.arange(len(payload["x"]), dtype=np.int64)
    segment_id = payload["segment_id"] if "segment_id" in payload else np.zeros(len(payload["x"]), dtype=np.int32)
    return WindowArchive(
        x=payload["x"],
        start_idx=start_idx.astype(np.int64),
        segment_id=segment_id.astype(np.int32),
    )


def hop_samples_from_step(step: str, sample_rate: int) -> int:
    """Convert a folder step label such as ``1s`` into a hop length in samples."""

    token = step.strip().lower()
    if not token.endswith("s"):
        raise ValueError(f"step must look like '1s' or '0.1s', got {step!r}")
    seconds = float(token[:-1])
    if seconds <= 0:
        raise ValueError(f"step must be positive, got {step!r}")
    return int(round(seconds * sample_rate))


def resolve_start_samples(
    starts: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
) -> np.ndarray:
    """Return monotonic sample offsets, accepting either sample or hop indices."""

    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        return starts
    if len(starts) == 1:
        return np.zeros(1, dtype=np.int64)

    span = int(starts.max() - starts.min())
    expected = (len(starts) - 1) * hop_samples
    # Some archives store 0, 1, 2, ... window indices instead of sample offsets.
    if span < max(hop_samples // 2, expected // max(len(starts), 1)):
        starts = starts * hop_samples
    return starts - int(starts.min())


def split_contiguous_runs(
    starts: np.ndarray,
    *,
    window_samples: int,
    hop_samples: int,
    max_gap_samples: int | None = None,
) -> list[np.ndarray]:
    """Split sorted start offsets wherever the archive skips part of the timeline."""

    starts = np.asarray(starts, dtype=np.int64)
    if len(starts) == 0:
        return []
    if len(starts) == 1:
        return [np.array([0], dtype=np.int64)]

    diffs = np.diff(starts)
    if max_gap_samples is None:
        # Anything much larger than the nominal hop is treated as a break between
        # contiguous recording segments, not as silence we should preserve.
        max_gap_samples = max(hop_samples + window_samples // 4, window_samples)

    breaks = np.concatenate(
        ([0], np.flatnonzero(diffs > max_gap_samples) + 1, [len(starts)])
    )
    return [
        np.arange(breaks[i], breaks[i + 1], dtype=np.int64)
        for i in range(len(breaks) - 1)
    ]


def overlap_add_runs(
    windows: np.ndarray,
    starts: np.ndarray,
    *,
    hop_samples: int,
    synthesis_window: str = "hann",
    max_gap_samples: int | None = None,
) -> np.ndarray:
    """Overlap-add each contiguous run, then concatenate runs without timeline gaps."""

    windows = np.asarray(windows, dtype=np.float32)
    starts = resolve_start_samples(
        starts,
        window_samples=windows.shape[1],
        hop_samples=hop_samples,
    )
    runs = split_contiguous_runs(
        starts,
        window_samples=windows.shape[1],
        hop_samples=hop_samples,
        max_gap_samples=max_gap_samples,
    )
    parts: list[np.ndarray] = []
    for run in runs:
        run_starts = starts[run] - int(starts[run[0]])
        parts.append(
            overlap_add_windows(windows[run], run_starts, synthesis_window=synthesis_window)
        )
    if not parts:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(parts)


def overlap_add_windows(
    windows: np.ndarray,
    starts: np.ndarray,
    synthesis_window: str = "hann",
) -> np.ndarray:
    """Reconstruct a continuous waveform from possibly overlapping windows."""
    windows = np.asarray(windows, dtype=np.float32)
    starts = np.asarray(starts, dtype=np.int64)
    if windows.ndim != 2:
        raise ValueError(f"windows must be [N, T], got {windows.shape}")
    if len(starts) != len(windows):
        raise ValueError("starts length must match number of windows")

    window_length = windows.shape[1]
    end = int(starts.max()) + window_length
    output = np.zeros(end, dtype=np.float32)
    weights = np.zeros(end, dtype=np.float32)

    if synthesis_window == "ones":
        weight = np.ones(window_length, dtype=np.float32)
    elif synthesis_window == "hann":
        weight = np.hanning(window_length).astype(np.float32)
    else:
        raise ValueError(f"Unsupported synthesis_window: {synthesis_window}")

    for window, start in zip(windows, starts):
        stop = int(start) + window_length
        output[int(start):stop] += window * weight
        weights[int(start):stop] += weight

    nonzero = weights > 0
    output[nonzero] /= weights[nonzero]
    return output
