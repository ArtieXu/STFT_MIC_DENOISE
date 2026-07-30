"""CirCor clean-heart-sound pool: TSV-filtered, cross-location, spread sampling.

Importable from both the CLI (scripts/build_circor_pool.py) and the Colab
notebooks, so the sampling rules live in exactly one place.

Layout (https://physionet.org/content/circor-heart-sound/1.0.3/):

    training_data/ABCDE_XY[_n].wav   16-bit PCM, 4000 Hz
    training_data/ABCDE_XY[_n].tsv   start_s, end_s, state
    training_data.csv                one row per subject

    XY in {AV, MV, PV, TV, Phc};  state 1=S1 2=systole 3=S2 4=diastole 0=unannotated

State 0 is the dataset's own signal-quality label, not merely "unlabelled":

    "Segmentation labels were retained for sections ... considered high quality
     and representative by the cardiac physiologists. The remainder of the signal
     may include both low and high quality data."

    "Different noisy sources have been observed in our dataset, including
     stethoscope rubbing noise, speaking, crying, or laughing sounds."

For a denoiser those regions are mislabelled targets. Everything here samples
only inside contiguous nonzero-state runs.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 4000
LOCATIONS = ("AV", "MV", "PV", "TV", "Phc")
ANNOTATED_STATES = (1, 2, 3, 4)

# Measured on this project's device clean recordings (adult).
DEVICE_BPM_MEDIAN = 73.0
DEVICE_BPM_RANGE = (61.0, 79.0)


def parse_record_name(stem: str) -> tuple[str, str]:
    """'50186_MV_2' -> ('50186', 'MV')."""
    parts = stem.split("_")
    return parts[0], (parts[1] if len(parts) > 1 else "unknown")


def read_tsv_rows(path: Path) -> list[tuple[float, float, int]]:
    rows = []
    with Path(path).open() as handle:
        for line in handle:
            fields = line.split()
            if len(fields) != 3:
                continue
            try:
                start, end = float(fields[0]), float(fields[1])
                state = int(float(fields[2]))
            except ValueError:
                continue
            if end > start:
                rows.append((start, end, state))
    rows.sort()
    return rows


def annotated_runs(rows, min_seconds: float) -> list[tuple[float, float]]:
    """Contiguous nonzero-state runs, merging touching S1/systole/S2/diastole rows."""
    spans = [(s, e) for s, e, state in rows if state in ANNOTATED_STATES]
    if not spans:
        return []
    runs = [list(spans[0])]
    for start, end in spans[1:]:
        if start <= runs[-1][1] + 1e-6:
            runs[-1][1] = max(runs[-1][1], end)
        else:
            runs.append([start, end])
    return [(a, b) for a, b in runs if b - a >= min_seconds]


def heart_rate_bpm(rows) -> float:
    """Heart rate from consecutive S1 onsets -- free, and exact.

    CirCor is pediatric (0-21 y, mean 6.1 y); this project's device recordings are
    adult at 61-79 bpm. Because clean heart sound and motion artifact overlap
    almost entirely in frequency here, the cue the model must learn is rhythm, so
    a large heart-rate mismatch is not cosmetic.
    """
    onsets = [s for s, _, state in rows if state == 1]
    if len(onsets) < 3:
        return float("nan")
    intervals = np.diff(np.asarray(onsets))
    intervals = intervals[(intervals > 0.25) & (intervals < 2.0)]
    if len(intervals) < 2:
        return float("nan")
    return float(60.0 / np.median(intervals))


def candidate_offsets(runs, window_seconds: float, hop_seconds: float) -> list[int]:
    """Offsets (samples) whose full window sits inside a single annotated run."""
    offsets = []
    for start, end in runs:
        position, last = start, end - window_seconds
        while position <= last + 1e-9:
            offsets.append(int(round(position * SR)))
            position += hop_seconds
    return offsets


def load_metadata(root: Path) -> dict[str, dict]:
    path = Path(root) / "training_data.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found. Point root at the extracted CirCor directory."
        )
    meta = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            subject = str(row.get("Patient ID", "")).strip()
            if subject:
                meta[subject] = row
    return meta


def spread_pick(offsets, count: int, min_gap: int, rng) -> list[int]:
    """Up to `count` offsets, each at least `min_gap` samples from the others."""
    pool = list(offsets)
    rng.shuffle(pool)
    chosen: list[int] = []
    for offset in pool:
        if len(chosen) >= count:
            break
        if all(abs(offset - c) >= min_gap for c in chosen):
            chosen.append(offset)
    return sorted(chosen)


def build_pool(
    root,
    *,
    windows_per_patient: int = 12,
    max_patients: int = 300,
    window_seconds: float = 2.0,
    hop_seconds: float = 0.5,
    min_gap_seconds: float = 2.0,
    murmur: str = "all",
    heart_rate_max: float | None = None,
    heart_rate_min: float = 0.0,
    val_fraction: float = 0.2,
    seed: int = 20260727,
    progress=None,
) -> dict:
    """Sample a clean-window pool. Returns arrays plus a `stats` dict.

    `murmur`: 'all' keeps Present and Absent and always drops Unknown (the 119
    subjects whose recordings "did not meet the required signal quality
    standards"). 'absent' / 'present' restrict further.
    """
    from scipy.io import wavfile

    root = Path(root)
    data_dir = root / "training_data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"{data_dir} not found")

    metadata = load_metadata(root)
    rng = np.random.default_rng(seed)

    by_subject: dict[str, list[Path]] = defaultdict(list)
    for wav in sorted(data_dir.glob("*.wav")):
        by_subject[parse_record_name(wav.stem)[0]].append(wav)

    eligible, dropped = [], {"unknown_murmur": 0, "no_metadata": 0, "murmur_filter": 0}
    for subject in sorted(by_subject):
        row = metadata.get(subject)
        if row is None:
            dropped["no_metadata"] += 1
            continue
        value = str(row.get("Murmur", "")).strip()
        if value == "Unknown":
            dropped["unknown_murmur"] += 1
            continue
        if murmur == "absent" and value != "Absent":
            dropped["murmur_filter"] += 1
            continue
        if murmur == "present" and value != "Present":
            dropped["murmur_filter"] += 1
            continue
        eligible.append(subject)
    rng.shuffle(eligible)

    window_samples = int(round(window_seconds * SR))
    min_gap = int(round(min_gap_seconds * SR))
    windows, provenance = [], []
    used, dropped_hr = 0, 0

    for subject in eligible:
        if used >= max_patients:
            break

        per_record, record_bpm = {}, {}
        for wav in by_subject[subject]:
            tsv = wav.with_suffix(".tsv")
            if not tsv.is_file():
                continue
            rows = read_tsv_rows(tsv)
            bpm = heart_rate_bpm(rows)
            record_bpm[wav.stem] = bpm
            if heart_rate_max is not None and bpm == bpm:  # not NaN
                if bpm > heart_rate_max or bpm < heart_rate_min:
                    dropped_hr += 1
                    continue
            offsets = candidate_offsets(
                annotated_runs(rows, window_seconds), window_seconds, hop_seconds
            )
            if offsets:
                per_record[wav] = offsets
        if not per_record:
            continue

        # Round-robin over locations so one record cannot dominate the subject.
        records = sorted(per_record, key=lambda p: p.name)
        rng.shuffle(records)
        base, extra = divmod(windows_per_patient, len(records))
        picks = {
            wav: spread_pick(per_record[wav], base + (1 if i < extra else 0), min_gap, rng)
            for i, wav in enumerate(records)
        }
        shortfall = windows_per_patient - sum(len(v) for v in picks.values())
        for wav in records:
            if shortfall <= 0:
                break
            remaining = [o for o in per_record[wav] if o not in picks[wav]]
            more = [
                o for o in spread_pick(remaining, shortfall, min_gap, rng)
                if all(abs(o - c) >= min_gap for c in picks[wav])
            ]
            picks[wav].extend(more)
            shortfall -= len(more)

        count = 0
        for wav, offsets in picks.items():
            if not offsets:
                continue
            sr, audio = wavfile.read(wav)
            if sr != SR:
                continue
            audio = np.asarray(audio)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32) / 32768.0
            location = parse_record_name(wav.stem)[1]
            row = metadata[subject]
            for offset in offsets:
                if offset + window_samples > len(audio):
                    continue
                window = audio[offset: offset + window_samples].copy()
                window -= np.float32(window.mean(dtype=np.float64))
                rms = float(np.sqrt(np.mean(window.astype(np.float64) ** 2)))
                if not np.isfinite(window).all() or rms <= 1e-5:
                    continue
                if np.abs(window).max() >= 0.999:  # clipped
                    continue
                windows.append(window)
                provenance.append({
                    "subject": subject,
                    "location": location,
                    "record": wav.stem,
                    "offset": offset,
                    "murmur": str(row.get("Murmur", "")).strip(),
                    "outcome": str(row.get("Outcome", "")).strip(),
                    "age": str(row.get("Age", "")).strip(),
                    "campaign": str(row.get("Campaign", "")).strip(),
                    "bpm": record_bpm.get(wav.stem, float("nan")),
                })
                count += 1
        if count:
            used += 1
            if progress and used % 25 == 0:
                progress(used, len(windows))

    if not windows:
        raise RuntimeError("no windows collected -- check the root path and TSV files")

    subjects = np.asarray([p["subject"] for p in provenance])
    unique = sorted(set(subjects.tolist()))
    split_rng = np.random.default_rng(seed + 1)
    permuted = list(unique)
    split_rng.shuffle(permuted)
    val_subjects = set(permuted[: int(round(len(permuted) * val_fraction))])
    split = np.asarray(["val" if s in val_subjects else "train" for s in subjects])

    bpm = np.asarray([p["bpm"] for p in provenance], dtype=np.float64)
    finite = bpm[np.isfinite(bpm)]
    pool = {
        "x": np.stack(windows).astype(np.float32),
        "subject": subjects,
        "location": np.asarray([p["location"] for p in provenance]),
        "record": np.asarray([p["record"] for p in provenance]),
        "offset": np.asarray([p["offset"] for p in provenance], dtype=np.int64),
        "murmur": np.asarray([p["murmur"] for p in provenance]),
        "outcome": np.asarray([p["outcome"] for p in provenance]),
        "age": np.asarray([p["age"] for p in provenance]),
        "campaign": np.asarray([p["campaign"] for p in provenance]),
        "bpm": bpm,
        "split": split,
    }
    pool["stats"] = {
        "n_windows": len(windows),
        "n_subjects": len(unique),
        "n_records": len(set(pool["record"].tolist())),
        "dropped_subjects": dropped,
        "dropped_records_by_heart_rate": dropped_hr,
        "bpm_median": float(np.median(finite)) if len(finite) else None,
        "bpm_p10": float(np.percentile(finite, 10)) if len(finite) else None,
        "bpm_p90": float(np.percentile(finite, 90)) if len(finite) else None,
        "device_bpm_median": DEVICE_BPM_MEDIAN,
        "murmur_counts": {
            m: int((pool["murmur"] == m).sum())
            for m in sorted(set(pool["murmur"].tolist()))
        },
        "location_counts": {
            loc: int((pool["location"] == loc).sum())
            for loc in LOCATIONS if (pool["location"] == loc).sum()
        },
    }
    return pool
