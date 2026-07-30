"""Window pools: one place where every clean/noise source is made commensurable.

Two clean sources feed this pipeline and they arrive in different shapes:

    device   data/clean/step_1s/{train,val,test}/*.npz
                                                    int16, full scale 32768,
                                                    2 s windows @ 4 kHz, 1 s hop
    CirCor   data/circor/circor_pool_*.npz          float32 in +-1, already
                                                    DC-removed and quality
                                                    filtered by src/circor.py

They are *not* interchangeable as stored. Measured on this project's data:

    pool                     median window RMS
    device clean                   0.00025-0.00072
    device noise                   0.00476-0.00750
    CirCor (digital stethoscope)   ~0.02        <-- 30-100x louder than device

The mono mixing code in frequency_data.py scales noise relative to clean RMS,
and the loss has scale-dependent terms (waveform L1, log-magnitude).
Concatenating raw pools would therefore mean CirCor windows silently train at a
different operating point than device windows. So every window that leaves this
module is:

    1. converted to float32 in +-1 (int16 divided by 32768, matching the
       convention of the source npz files -- nothing is re-quantised),
    2. DC-removed per window,
    3. dropped if it is non-finite, silent, or clipped,
    4. normalised to a single target RMS per window.

Step 4 is the one that changes numbers relative to the upstream repo. It is a
deliberate choice, recorded in the checkpoint, and can be turned off with
``target_rms=None``. Per-window RMS normalisation is safe here because a window
is 2 s -- two to three cardiac cycles -- so it does not flatten within-cycle
dynamics.

Provenance travels with the samples (``source``/``subject``/``origin``) so an
audit can always answer "how much of this pool is CirCor, and from whom".
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

FULL_SCALE = 32_768.0
WINDOW_SAMPLES = 8_000
SAMPLE_RATE = 4_000
#: Below this RMS (in +-1 units) a window carries no signal worth normalising.
SILENCE_RMS = 1e-6
#: int16 saturation, expressed in +-1 units.
CLIP_LEVEL = 32_767.0 / 32_768.0
#: Everything is scaled to this RMS. The loudest windows in this dataset have a
#: crest factor near 46 (a transient-heavy noise window), so 0.02 keeps every
#: source window inside +-1 while staying far from float32 denormals. Mixtures
#: may still exceed +-1; the model normalises its input by RMS anyway.
DEFAULT_TARGET_RMS = 0.02

# The experiment is defined by exact recording names, not by whichever legacy
# folder happens to contain each archive. Clean and noise are unpaired pools, so
# their available participant IDs are intentionally asymmetric. Subject/session
# 6 is the only final test source.
DEVICE_PROTOCOL_ID = "single_stage_fixed_budget_v1"
DEVICE_UPSTREAM_COMMIT = "c38092d286e03fea81f72fa66c7731100d9266ec"
DEVICE_FILE_SPLITS: dict[str, dict[str, tuple[str, ...]]] = {
    "clean": {
        "train": (
            "heart_aw1", "heart_aw2", "heart_aw4",
            "heart_bw1", "heart_bw2", "heart_bw4", "heart_bw5",
        ),
        "test": ("heart_aw6", "heart_bw6"),
    },
    "noise": {
        "train": ("noise1", "noise2", "noise3", "noise5"),
        "test": ("noise6",),
    },
}


def device_protocol_metadata() -> dict[str, Any]:
    """JSON-safe copy of the fixed device data contract stored in checkpoints."""

    return {
        "id": DEVICE_PROTOCOL_ID,
        "device_files": {
            modality: {
                split: list(stems)
                for split, stems in split_table.items()
            }
            for modality, split_table in DEVICE_FILE_SPLITS.items()
        },
        "device_upstream_commit": DEVICE_UPSTREAM_COMMIT,
        "selection_policy": "fixed training budget; no validation",
        "circor_policy": "combined arm uses only the subject-disjoint CirCor train split",
        "circor_subject_identity": "Patient ID aliases merged through Additional ID",
        "final_test_subject": "device_6",
    }


@dataclass(frozen=True)
class WindowPool:
    """Float32 windows plus per-window provenance and a summary."""

    x: np.ndarray            # [N, WINDOW_SAMPLES] float32, +-1
    source: np.ndarray       # "device" | "circor"
    subject: np.ndarray      # "device_1" ... / "circor_50186"
    origin: np.ndarray       # file stem or CirCor record name
    stats: dict[str, Any]

    def __len__(self) -> int:
        return len(self.x)

    @property
    def source_counts(self) -> dict[str, int]:
        return {
            str(name): int((self.source == name).sum())
            for name in sorted(set(self.source.tolist()))
        }


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #
def to_float(x: np.ndarray) -> np.ndarray:
    """int16 (or any integer) -> float32 in +-1 using the dtype's full scale."""
    x = np.asarray(x)
    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        scale = float(max(abs(int(info.min)), int(info.max)))
        return x.astype(np.float32) / scale
    return x.astype(np.float32)


def remove_dc(x: np.ndarray) -> np.ndarray:
    """Per-window mean removal, accumulated in float64."""
    return (x - x.mean(axis=1, dtype=np.float64, keepdims=True)).astype(np.float32)


def window_rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=1))


def clip_mask(x: np.ndarray) -> np.ndarray:
    """Windows touching int16 saturation. Must be evaluated *before* DC removal:
    removing a DC offset of a fraction of an LSB pushes saturated samples just
    under the threshold and hides them."""
    return (np.abs(x) >= CLIP_LEVEL).any(axis=1)


def purity_mask(
    x: np.ndarray, clipped: np.ndarray | None = None
) -> tuple[np.ndarray, dict[str, int]]:
    """Keep finite, non-silent, non-clipped windows. Returns (mask, reasons)."""
    finite = np.isfinite(x).all(axis=1)
    rms = window_rms(x)
    silent = rms <= SILENCE_RMS
    clipped = clip_mask(x) if clipped is None else np.asarray(clipped, dtype=bool)
    keep = finite & ~silent & ~clipped
    reasons = {
        "dropped_non_finite": int((~finite).sum()),
        "dropped_silent": int((silent & finite).sum()),
        "dropped_clipped": int((clipped & finite & ~silent).sum()),
    }
    return keep, reasons


def normalise_rms(x: np.ndarray, target_rms: float) -> np.ndarray:
    rms = window_rms(x)[:, None]
    return (x * (target_rms / np.maximum(rms, SILENCE_RMS))).astype(np.float32)


def thin(n: int, source_stride: int, max_windows: int | None) -> np.ndarray:
    """Indices after overlap thinning and an optional cap.

    ``source_stride`` drops overlapping neighbours (step_1s windows already sit
    1 s apart, so the default of 1 is a no-op). The cap is taken as evenly
    spaced indices rather than the first N, so truncating never silently
    reduces the pool to whichever file happened to sort first.
    """
    if source_stride < 1:
        raise ValueError(f"source_stride must be >= 1, got {source_stride}")
    index = np.arange(n)[::source_stride]
    if max_windows is not None and len(index) > max_windows:
        index = index[np.linspace(0, len(index) - 1, max_windows).round().astype(int)]
    return index


def _finalise(
    x: np.ndarray,
    source: Sequence[str],
    subject: Sequence[str],
    origin: Sequence[str],
    *,
    source_stride: int,
    max_windows: int | None,
    target_rms: float | None,
    label: str,
    extra: dict[str, Any] | None = None,
) -> WindowPool:
    x = to_float(x)
    if x.ndim != 2:
        raise ValueError(f"{label}: expected [N, T] windows, got shape {x.shape}")
    if x.shape[1] != WINDOW_SAMPLES:
        raise ValueError(
            f"{label}: window length {x.shape[1]} != {WINDOW_SAMPLES} "
            f"({WINDOW_SAMPLES / SAMPLE_RATE:g} s @ {SAMPLE_RATE} Hz). "
            "All sources must share one window length."
        )
    source = np.asarray(source, dtype=object).astype(str)
    subject = np.asarray(subject, dtype=object).astype(str)
    origin = np.asarray(origin, dtype=object).astype(str)

    clipped = clip_mask(x)
    x = remove_dc(x)
    keep, reasons = purity_mask(x, clipped=clipped)
    x, source, subject, origin = x[keep], source[keep], subject[keep], origin[keep]
    if len(x) == 0:
        raise ValueError(f"{label}: no windows survived the purity filter ({reasons})")

    index = thin(len(x), source_stride, max_windows)
    x, source, subject, origin = x[index], source[index], subject[index], origin[index]

    rms_before = window_rms(x)
    if target_rms is not None:
        x = normalise_rms(x, target_rms)

    stats = {
        "label": label,
        "n_windows": int(len(x)),
        "n_subjects": int(len(set(subject.tolist()))),
        "source_stride": source_stride,
        "max_windows": max_windows,
        "target_rms": target_rms,
        "rms_before_normalisation": {
            "median": float(np.median(rms_before)),
            "p5": float(np.percentile(rms_before, 5)),
            "p95": float(np.percentile(rms_before, 95)),
        },
        **reasons,
    }
    if extra:
        stats.update(extra)
    return WindowPool(x=x, source=source, subject=subject, origin=origin, stats=stats)


# --------------------------------------------------------------------------- #
# device recordings
# --------------------------------------------------------------------------- #
def canonical_device_subject(stem: str) -> str:
    """Return one human ID across clean, noise, and walking archive names.

    Examples:
        ``heart_aw6_windows`` -> ``device_6``
        ``heart_bw6_windows`` -> ``device_6``
        ``heart_w6_windows``  -> ``device_6``
        ``noise6_windows``    -> ``device_6``
    """

    name = stem.removesuffix("_windows").removesuffix("_preprocessed")
    match = re.fullmatch(r"heart_(?:aw|bw|w)(\d+)", name)
    if match is None:
        match = re.fullmatch(r"noise(\d+)", name)
    if match is None:
        # Direct load_device_pool callers may use descriptive fixture names.
        # Keep those auditable without pretending they are a known human.
        return f"device_{name}"
    return f"device_{match.group(1)}"


def file_sha256(path: str | Path) -> str:
    """Stream a file fingerprint for checkpoint provenance."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_device_stem(stem: str) -> str:
    """Remove archive suffixes while preserving the recording identity."""

    return stem.removesuffix(".npz").removesuffix("_windows").removesuffix("_preprocessed")


def device_split_for_recording(stem: str, modality: str) -> str:
    """Return the logical train/test role for one exact clean or noise recording."""

    if modality not in DEVICE_FILE_SPLITS:
        raise ValueError(
            f"modality must be one of {tuple(DEVICE_FILE_SPLITS)}, got {modality!r}"
        )
    recording = canonical_device_stem(stem)
    matches = [
        split
        for split, recordings in DEVICE_FILE_SPLITS[modality].items()
        if recording in recordings
    ]
    if len(matches) != 1:
        raise ValueError(
            f"{modality} recording {recording!r} is not assigned to exactly one split"
        )
    return matches[0]


def _split_device_files(
    directory: Path,
    split: str,
    *,
    modality: str,
    skip_preprocessed: bool,
) -> list[Path]:
    """Discover legacy/current sibling folders, de-duplicate, then filter."""

    if modality not in DEVICE_FILE_SPLITS:
        raise ValueError(
            f"modality must be one of {tuple(DEVICE_FILE_SPLITS)}, got {modality!r}"
        )
    if split not in DEVICE_FILE_SPLITS[modality]:
        raise ValueError(
            f"split must be one of {tuple(DEVICE_FILE_SPLITS[modality])} "
            f"for {modality}, got {split!r}"
        )
    step_root = directory.parent
    candidates = sorted(step_root.glob("*/*.npz"))
    if directory.is_dir():
        candidates.extend(sorted(directory.glob("*.npz")))
    if skip_preprocessed:
        candidates = [path for path in candidates if "preprocessed" not in path.name]

    allowed = set(DEVICE_FILE_SPLITS[modality][split])
    by_name: dict[str, Path] = {}
    for path in candidates:
        if canonical_device_stem(path.stem) not in allowed:
            continue
        previous = by_name.get(path.name)
        # Prefer the canonical destination if both a legacy and newly fetched
        # copy exist; otherwise either path contains the same upstream archive.
        if previous is None or (path.parent.name == split and previous.parent.name != split):
            by_name[path.name] = path
    return sorted(by_name.values())


def load_device_pool(
    directory: str | Path,
    *,
    split: str | None = None,
    modality: str | None = None,
    source_stride: int = 1,
    max_windows: int | None = None,
    target_rms: float | None = DEFAULT_TARGET_RMS,
    skip_preprocessed: bool = True,
    label: str | None = None,
) -> WindowPool:
    """Load device windows, optionally enforcing the exact recording manifest.

    With ``split=None`` only ``directory`` is read (useful for a standalone
    archive or fixture). With a split, all sibling legacy/current directories
    are searched and filtered by the declared clean/noise recording names.
    """

    directory = Path(directory)
    if split is None:
        files = sorted(directory.glob("*.npz"))
        if skip_preprocessed:
            # Raw is this project's default target; the *_preprocessed variants are
            # the same windows bandpassed, so mixing both would duplicate windows.
            files = [path for path in files if "preprocessed" not in path.name]
    else:
        if modality is None:
            raise ValueError("modality='clean' or 'noise' is required when split is set")
        files = _split_device_files(
            directory,
            split,
            modality=modality,
            skip_preprocessed=skip_preprocessed,
        )
    if not files:
        suffix = f" for split={split!r}" if split is not None else ""
        raise FileNotFoundError(f"No .npz files found in {directory}{suffix}")

    chunks, source, subject, origin = [], [], [], []
    for path in files:
        with np.load(path) as payload:
            if "x" not in payload:
                raise KeyError(f"{path} has no 'x' array")
            x = np.asarray(payload["x"])
        chunks.append(x)
        source.extend(["device"] * len(x))
        subject.extend([canonical_device_subject(path.stem)] * len(x))
        origin.extend([path.stem] * len(x))
    return _finalise(
        np.concatenate(chunks, axis=0),
        source, subject, origin,
        source_stride=source_stride,
        max_windows=max_windows,
        target_rms=target_rms,
        label=label or f"device:{directory.parent.parent.name}/{directory.name}",
        extra={
            "files": [path.name for path in files],
            "split": split,
            "modality": modality,
            "device_protocol_id": DEVICE_PROTOCOL_ID,
        },
    )


# --------------------------------------------------------------------------- #
# CirCor pool
# --------------------------------------------------------------------------- #
def load_circor_pool(
    pool_path: str | Path,
    split: str,
    *,
    source_stride: int = 1,
    max_windows: int | None = None,
    target_rms: float | None = DEFAULT_TARGET_RMS,
    heart_rate_max: float | None = None,
    heart_rate_min: float | None = None,
) -> WindowPool:
    """Load one split of a pool written by ``scripts/build_circor_pool.py``.

    The split column is subject-disjoint by construction, so a CirCor subject
    never appears in both train and val. ``heart_rate_max`` re-filters an
    already built pool: CirCor is pediatric (0-21 y) while this project's device
    recordings are adult at 61-79 bpm, and rhythm is the main cue available to
    the model when heart sound and motion artifact overlap in frequency.
    """
    pool_path = Path(pool_path)
    if not pool_path.is_file():
        raise FileNotFoundError(
            f"CirCor pool not found: {pool_path}\n"
            "Build it once with:\n"
            "  PYTHONPATH=. python scripts/build_circor_pool.py "
            "--root circor-heart-sound-1.0.3 "
            "--out data/circor/circor_pool_4khz_2s_v2.npz"
        )
    if split not in ("train", "val"):
        raise ValueError(f"split must be 'train' or 'val', got {split!r}")

    from src.circor import POOL_SCHEMA_VERSION

    with np.load(pool_path, allow_pickle=False) as payload:
        missing = [
            key
            for key in (
                "pool_schema_version", "x", "split", "subject",
                "participant_id", "record",
            )
            if key not in payload
        ]
        if missing:
            raise KeyError(
                f"{pool_path} is missing {missing}; rebuild with build_circor_pool.py "
                "so Additional ID aliases are merged"
            )
        stored_schema = int(payload["pool_schema_version"])
        if stored_schema != POOL_SCHEMA_VERSION:
            raise ValueError(
                f"{pool_path}: CirCor pool schema {stored_schema} != "
                f"{POOL_SCHEMA_VERSION}; rebuild the pool"
            )
        x = np.asarray(payload["x"])
        pool_split = np.asarray(payload["split"]).astype(str)
        subject = np.asarray(payload["subject"]).astype(str)
        record = np.asarray(payload["record"]).astype(str)
        bpm = np.asarray(payload["bpm"], dtype=np.float64) if "bpm" in payload else np.full(len(x), np.nan)
        stored_sr = int(payload["sample_rate"]) if "sample_rate" in payload else SAMPLE_RATE

    if stored_sr != SAMPLE_RATE:
        raise ValueError(f"{pool_path} is {stored_sr} Hz; this pipeline is {SAMPLE_RATE} Hz")

    keep = pool_split == split
    dropped_hr = 0
    if heart_rate_max is not None or heart_rate_min is not None:
        known = np.isfinite(bpm)
        inside = np.ones(len(x), dtype=bool)
        if heart_rate_max is not None:
            inside &= bpm <= heart_rate_max
        if heart_rate_min is not None:
            inside &= bpm >= heart_rate_min
        # Windows with no usable S1 onsets keep the benefit of the doubt.
        hr_keep = inside | ~known
        dropped_hr = int((keep & ~hr_keep).sum())
        keep &= hr_keep
    if not keep.any():
        raise ValueError(f"{pool_path}: no windows left for split={split!r} after filtering")

    finite_bpm = bpm[keep][np.isfinite(bpm[keep])]
    return _finalise(
        x[keep],
        ["circor"] * int(keep.sum()),
        [f"circor_{value}" for value in subject[keep]],
        record[keep],
        source_stride=source_stride,
        max_windows=max_windows,
        target_rms=target_rms,
        label=f"circor:{split}",
        extra={
            "pool_path": str(pool_path),
            "pool_sha256": file_sha256(pool_path),
            "split": split,
            "dropped_by_heart_rate": dropped_hr,
            "heart_rate_max": heart_rate_max,
            "heart_rate_min": heart_rate_min,
            "bpm_median": float(np.median(finite_bpm)) if len(finite_bpm) else None,
            "bpm_p10": float(np.percentile(finite_bpm, 10)) if len(finite_bpm) else None,
            "bpm_p90": float(np.percentile(finite_bpm, 90)) if len(finite_bpm) else None,
        },
    )


# --------------------------------------------------------------------------- #
# combination
# --------------------------------------------------------------------------- #
def concat_pools(pools: Iterable[WindowPool], *, label: str) -> WindowPool:
    """Concatenate pools that have already been filtered and normalised."""
    pools = [pool for pool in pools if pool is not None and len(pool)]
    if not pools:
        raise ValueError(f"{label}: nothing to concatenate")
    if len(pools) == 1:
        single = pools[0]
        stats = dict(single.stats)
        stats["label"] = label
        stats["parts"] = [single.stats]
        stats["source_counts"] = single.source_counts
        return WindowPool(single.x, single.source, single.subject, single.origin, stats)

    targets = {pool.stats.get("target_rms") for pool in pools}
    if len(targets) > 1:
        raise ValueError(
            f"{label}: pools were normalised to different target RMS values {targets}; "
            "that is exactly the mismatch this module exists to prevent."
        )
    combined = WindowPool(
        x=np.concatenate([pool.x for pool in pools], axis=0),
        source=np.concatenate([pool.source for pool in pools], axis=0),
        subject=np.concatenate([pool.subject for pool in pools], axis=0),
        origin=np.concatenate([pool.origin for pool in pools], axis=0),
        stats={},
    )
    stats = {
        "label": label,
        "n_windows": len(combined),
        "n_subjects": int(len(set(combined.subject.tolist()))),
        "target_rms": next(iter(targets)),
        "source_counts": combined.source_counts,
        "parts": [pool.stats for pool in pools],
    }
    return WindowPool(combined.x, combined.source, combined.subject, combined.origin, stats)


def build_clean_pool(
    device_dir: str | Path,
    split: str,
    *,
    circor_pool: str | Path | None = None,
    source_stride: int = 1,
    max_windows: int | None = None,
    target_rms: float | None = DEFAULT_TARGET_RMS,
    circor_heart_rate_max: float | None = None,
    circor_heart_rate_min: float | None = None,
) -> WindowPool:
    """Build a logical device split, optionally adding CirCor to training.

    ``max_windows`` caps each source separately, so it thins the pool without
    changing the device:CirCor balance.
    """
    if circor_pool is not None and split != "train":
        raise ValueError("CirCor is allowed only in the combined training pool")
    parts = [
        load_device_pool(
            device_dir,
            split=split,
            modality="clean",
            source_stride=source_stride,
            max_windows=max_windows,
            target_rms=target_rms,
            label=f"device_clean:{split}",
        )
    ]
    if circor_pool is not None:
        parts.append(
            load_circor_pool(
                circor_pool,
                split,
                source_stride=1,  # CirCor offsets are already >= 2 s apart
                max_windows=max_windows,
                target_rms=target_rms,
                heart_rate_max=circor_heart_rate_max,
                heart_rate_min=circor_heart_rate_min,
            )
        )
    return concat_pools(parts, label=f"clean:{split}")


def build_noise_pool(
    device_dir: str | Path,
    split: str,
    *,
    source_stride: int = 1,
    max_windows: int | None = None,
    target_rms: float | None = DEFAULT_TARGET_RMS,
) -> WindowPool:
    """Motion/ambient noise windows. Device only -- CirCor has no noise channel."""
    return concat_pools(
        [
            load_device_pool(
                device_dir,
                split=split,
                modality="noise",
                source_stride=source_stride,
                max_windows=max_windows,
                target_rms=target_rms,
                label=f"device_noise:{split}",
            )
        ],
        label=f"noise:{split}",
    )


def describe(pool: WindowPool) -> str:
    """One compact human-readable block per pool, for logs and audits."""
    stats = pool.stats
    lines = [
        f"{stats.get('label', '?'):<18} {len(pool):>6} windows  "
        f"{len(pool) * WINDOW_SAMPLES / SAMPLE_RATE / 60:>6.1f} min  "
        f"{stats.get('n_subjects', '?')} subjects"
    ]
    counts = stats.get("source_counts") or pool.source_counts
    if len(counts) > 1 or "device" not in counts:
        share = ", ".join(
            f"{name} {count} ({count / len(pool):.0%})" for name, count in counts.items()
        )
        lines.append(f"{'':18} sources: {share}")
    for part in stats.get("parts", [stats]):
        rms = part.get("rms_before_normalisation", {})
        dropped = sum(
            int(part.get(key, 0))
            for key in ("dropped_non_finite", "dropped_silent", "dropped_clipped")
        )
        if rms:
            lines.append(
                f"{'':18} {part.get('label', '?'):<20} raw RMS median {rms['median']:.5f} "
                f"(p5 {rms['p5']:.5f} / p95 {rms['p95']:.5f}), dropped {dropped}"
            )
        if part.get("bpm_median") is not None:
            lines.append(
                f"{'':18} {'':20} bpm median {part['bpm_median']:.0f} "
                f"(p10-p90 {part['bpm_p10']:.0f}-{part['bpm_p90']:.0f}); device is 73"
            )
    target = stats.get("target_rms")
    lines.append(
        f"{'':18} normalised to RMS {target}" if target is not None
        else f"{'':18} NOT rms-normalised (target_rms=None)"
    )
    return "\n".join(lines)
