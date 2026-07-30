#!/usr/bin/env python3
"""Build the CirCor clean-heart-sound pool. Run once, locally or in Colab.

    # get the data (449 MB, PhysioNet must be reachable)
    wget -c -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/
    unzip -q circor.zip

    # all eligible subjects, 12 windows each -- the default
    PYTHONPATH=. python scripts/build_circor_pool.py --root circor-heart-sound-1.0.3

Writes ``data/circor/circor_pool_4khz_2s_v2.npz`` plus a ``.manifest.json`` beside
it. Nothing else in this folder needs the raw CirCor tree afterwards.

CirCor is 4 kHz 16-bit like this project's device recordings, so no resampling
and no requantisation happen anywhere. The sampling rules live in src/circor.py;
the two that matter most:

  * only inside contiguous nonzero-state TSV runs. State 0 is CirCor's own
    signal-quality label, and the documented noise sources in those regions are
    stethoscope rubbing, speech, crying and laughing -- i.e. exactly the content
    a denoiser must not be handed as a clean target;
  * subjects whose Murmur is "Unknown" are dropped: those are the 119 subjects
    whose recordings did not meet the signal-quality standard.

The split column is subject-disjoint, so train/val never share a subject.
Patient ID aliases connected by the official ``Additional ID`` field are
merged before per-subject limiting and splitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.circor import POOL_SCHEMA_VERSION, build_pool  # noqa: E402
from src.pools import SAMPLE_RATE, WINDOW_SAMPLES  # noqa: E402

DEFAULT_OUT = REPO_ROOT / "data" / "circor" / "circor_pool_4khz_2s_v2.npz"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True,
                        help="Extracted CirCor directory (contains training_data/ and training_data.csv)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--windows-per-patient", type=int, default=12)
    parser.add_argument("--max-patients", type=int, default=0,
                        help="0 means every eligible subject (~940 after dropping Unknown murmur).")
    parser.add_argument("--hop-seconds", type=float, default=0.5,
                        help="Candidate-offset grid inside each annotated run.")
    parser.add_argument("--min-gap-seconds", type=float, default=2.0,
                        help="Minimum spacing between chosen offsets; 2.0 means non-overlapping windows.")
    parser.add_argument("--murmur", choices=("all", "absent", "present"), default="all")
    parser.add_argument("--heart-rate-max", type=float, default=None,
                        help="Drop records above this bpm. Off by default -- the report below shows "
                             "what each threshold would cost, and train_frequency.py can apply one "
                             "to an already built pool.")
    parser.add_argument("--heart-rate-min", type=float, default=0.0)
    parser.add_argument("--val-fraction", type=float, default=0.2,
                        help="Subject-disjoint validation share.")
    parser.add_argument("--seed", type=int, default=20260727)
    args = parser.parse_args()

    window_seconds = WINDOW_SAMPLES / SAMPLE_RATE  # 2.0 s, fixed by the pipeline
    if not (args.root / "training_data").is_dir():
        raise SystemExit(
            f"{args.root}/training_data not found. Download CirCor first:\n"
            "  wget -c -O circor.zip https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/\n"
            "  unzip -q circor.zip"
        )

    pool = build_pool(
        args.root,
        windows_per_patient=args.windows_per_patient,
        max_patients=args.max_patients if args.max_patients > 0 else 10**9,
        window_seconds=window_seconds,
        hop_seconds=args.hop_seconds,
        min_gap_seconds=args.min_gap_seconds,
        murmur=args.murmur,
        heart_rate_max=args.heart_rate_max,
        heart_rate_min=args.heart_rate_min,
        val_fraction=args.val_fraction,
        seed=args.seed,
        progress=lambda used, n: print(f"  {used} subjects, {n} windows"),
    )
    stats = pool.pop("stats")
    if pool["x"].shape[1] != WINDOW_SAMPLES:
        raise SystemExit(f"window length {pool['x'].shape[1]} != {WINDOW_SAMPLES}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        pool_schema_version=POOL_SCHEMA_VERSION,
        sample_rate=SAMPLE_RATE,
        window_samples=WINDOW_SAMPLES,
        **pool,
    )

    split = pool["split"]
    print("\n" + "=" * 58)
    print(f"windows                 {stats['n_windows']}")
    print(f"subjects                {stats['n_subjects']} "
          f"(from {stats['n_participant_ids']} Patient IDs; "
          f"{stats['linked_id_groups']} linked-ID groups)")
    print(f"records                 {stats['n_records']} "
          f"({stats['n_records'] / stats['n_subjects']:.2f} per subject)")
    print(f"dropped subjects        {stats['dropped_subjects']}")
    if stats["dropped_records_by_heart_rate"]:
        print(f"dropped by heart rate   {stats['dropped_records_by_heart_rate']} records")
    print(f"locations               {stats['location_counts']}")
    print(f"murmur                  {stats['murmur_counts']}")
    print(f"split                   train:{int((split == 'train').sum())} "
          f"val:{int((split == 'val').sum())} (subject-disjoint)")

    rms = np.sqrt(np.mean(pool["x"].astype(np.float64) ** 2, axis=1))
    print(f"window RMS              median {np.median(rms):.5f} "
          f"(device clean is ~0.0007 -- train_frequency.py normalises both)")

    finite = pool["bpm"][np.isfinite(pool["bpm"])]
    if len(finite):
        print("\nHeart rate (from TSV S1 onsets)")
        print(f"  CirCor pool           median {stats['bpm_median']:.0f} bpm, "
              f"p10-p90 {stats['bpm_p10']:.0f}-{stats['bpm_p90']:.0f}")
        print("  this device dataset   median 73 bpm, range 61-79 (adult)")
        print("  windows surviving --circor_heart_rate_max of:")
        for threshold in (90, 100, 110, 130):
            keep = int((finite <= threshold).sum())
            print(f"    <= {threshold:3d} bpm   {keep:5d}  ({keep / len(finite):5.1%})")

    manifest = args.out.with_suffix(".manifest.json")
    manifest.write_text(json.dumps(
        {"source": "CirCor DigiScope 1.0.3",
         "pool_schema_version": POOL_SCHEMA_VERSION,
         "annotated_states_only": True,
         "sample_rate": SAMPLE_RATE, "window_samples": WINDOW_SAMPLES,
         "args": {k: str(v) for k, v in vars(args).items()}, "stats": stats},
        indent=2) + "\n")
    print(f"\nwrote {args.out}  ({args.out.stat().st_size / 2**20:.1f} MiB)")
    print(f"wrote {manifest}")


if __name__ == "__main__":
    main()
