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
