#!/usr/bin/env python3
"""Report exactly what each arm of the experiment will see, before spending GPU hours.

    PYTHONPATH=. python scripts/audit_pools.py                 # arm B: device + CirCor
    PYTHONPATH=. python scripts/audit_pools.py --no_circor     # arm A: device only
    PYTHONPATH=. python scripts/audit_pools.py --json outputs/pool_audit.json

Answers four questions the combined training set makes easy to get wrong:

  format     is every source 2 s @ 4 kHz, float in +-1, DC-free?
  purity     how many windows are silent, clipped, or non-finite?
  scale      how far apart are the sources before normalisation?
  leakage    does any subject appear in both train and val?

Validation mirrors training: device-only unless --val_circor, because the two
arms have to be scored on one identical set for the comparison to mean anything.

Exit code is 1 if a hard check fails, so it can gate a training job.
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

from src.pools import (  # noqa: E402
    DEFAULT_TARGET_RMS,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WindowPool,
    build_clean_pool,
    build_noise_pool,
    describe,
    window_rms,
)

DEFAULT_CIRCOR_POOL = REPO_ROOT / "data" / "circor" / "circor_pool_4khz_2s.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data_root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--step", type=str, default="1s")
    parser.add_argument("--source_stride", type=int, default=1)
    parser.add_argument("--circor_pool", type=Path, default=DEFAULT_CIRCOR_POOL)
    parser.add_argument("--no_circor", action="store_true")
    parser.add_argument("--val_circor", action="store_true",
                        help="Mirror train_frequency.py --val_circor.")
    parser.add_argument("--circor_heart_rate_max", type=float, default=None)
    parser.add_argument("--pool_target_rms", type=float, default=DEFAULT_TARGET_RMS)
    parser.add_argument("--json", type=Path, default=None, help="Also write the report as JSON.")
    return parser.parse_args()


def check_format(
    pool: WindowPool, failures: list[str], warnings: list[str]
) -> dict[str, object]:
    x = pool.x
    dc = np.abs(x.mean(axis=1))
    rms = window_rms(x)
    peak = np.abs(x).max(axis=1)
    crest = peak / np.maximum(rms, 1e-12)
    label = pool.stats.get("label", "?")
    if x.dtype != np.float32:
        failures.append(f"{label}: dtype is {x.dtype}, expected float32")
    if x.shape[1] != WINDOW_SAMPLES:
        failures.append(f"{label}: window length {x.shape[1]} != {WINDOW_SAMPLES}")
    if not np.isfinite(x).all():
        failures.append(f"{label}: contains non-finite samples")
    if dc.max() > 1e-4 * max(rms.max(), 1e-9):
        failures.append(f"{label}: residual DC offset up to {dc.max():.2e}")
    # Not a failure: peaks above +-1 only mean a spiky window, and everything
    # downstream is RMS-relative. Worth surfacing, since a crest factor far above
    # the pool median usually is a thump or a dropout rather than heart sound.
    if peak.max() >= 1.0:
        warnings.append(
            f"{label}: {int((peak >= 1.0).sum())} window(s) peak above +-1 "
            f"(max crest factor {crest.max():.0f}); lower --pool_target_rms if that matters"
        )
    return {
        "windows": int(len(x)),
        "minutes": round(len(x) * WINDOW_SAMPLES / SAMPLE_RATE / 60, 1),
        "dtype": str(x.dtype),
        "rms_median": float(np.median(rms)),
        "rms_min": float(rms.min()),
        "rms_max": float(rms.max()),
        "peak_median": float(np.median(peak)),
        "peak_max": float(peak.max()),
        "crest_median": float(np.median(crest)),
        "crest_max": float(crest.max()),
        "abs_dc_max": float(dc.max()),
        "source_counts": pool.source_counts,
        "subjects": sorted(set(pool.subject.tolist()))[:12],
        "n_subjects": int(len(set(pool.subject.tolist()))),
    }


def main() -> int:
    args = parse_args()
    circor_pool = None if args.no_circor else args.circor_pool
    if circor_pool is not None and not Path(circor_pool).is_file():
        print(f"note: {circor_pool} not present -- auditing device-only.\n"
              "      build it with scripts/build_circor_pool.py to audit the combined pool.\n")
        circor_pool = None
    target_rms = args.pool_target_rms if args.pool_target_rms > 0 else None

    clean_dir = args.data_root / "clean" / f"step_{args.step}"
    noise_dir = args.data_root / "noise" / f"step_{args.step}"
    pools: dict[str, WindowPool] = {}
    for split in ("train", "val"):
        split_circor = circor_pool if (split == "train" or args.val_circor) else None
        pools[f"clean_{split}"] = build_clean_pool(
            clean_dir / split, split,
            circor_pool=split_circor,
            source_stride=args.source_stride,
            target_rms=target_rms,
            circor_heart_rate_max=args.circor_heart_rate_max,
        )
        pools[f"noise_{split}"] = build_noise_pool(
            noise_dir / split, split,
            source_stride=args.source_stride,
            target_rms=target_rms,
        )

    failures: list[str] = []
    warnings: list[str] = []
    report: dict[str, object] = {
        "step": args.step,
        "source_stride": args.source_stride,
        "target_rms": target_rms,
        "circor_pool": str(circor_pool) if circor_pool else None,
        "pools": {},
        "raw_pool_stats": {name: pool.stats for name, pool in pools.items()},
    }

    print("=" * 78)
    arm = "A (device-only)" if circor_pool is None else "B (device + CirCor)"
    print(f"POOLS -- arm {arm}, validation "
          f"{'device + CirCor' if args.val_circor else 'device-only'}")
    print("=" * 78)
    for name, pool in pools.items():
        print(describe(pool))
        report["pools"][name] = check_format(pool, failures, warnings)

    print("\n" + "=" * 78)
    print("SCALE BEFORE NORMALISATION  (why normalisation is on by default)")
    print("=" * 78)
    print(f"{'pool':<22}{'source':<10}{'median RMS':>12}{'p5':>12}{'p95':>12}")
    for name, pool in pools.items():
        for part in pool.stats.get("parts", [pool.stats]):
            rms = part.get("rms_before_normalisation")
            if not rms:
                continue
            source = "circor" if "circor" in str(part.get("label", "")) else "device"
            print(f"{name:<22}{source:<10}{rms['median']:>12.5f}{rms['p5']:>12.5f}{rms['p95']:>12.5f}")
    ratios = []
    for pool in pools.values():
        medians = [
            part["rms_before_normalisation"]["median"]
            for part in pool.stats.get("parts", [pool.stats])
            if part.get("rms_before_normalisation")
        ]
        if len(medians) > 1:
            ratios.append(max(medians) / max(min(medians), 1e-12))
    if ratios:
        print(f"\nlargest device/CirCor level gap inside one pool: {max(ratios):.0f}x")
    report["max_intra_pool_level_ratio"] = max(ratios) if ratios else None

    print("\n" + "=" * 78)
    print("PURITY  (windows removed before training)")
    print("=" * 78)
    print(f"{'pool':<22}{'part':<24}{'non-finite':>12}{'silent':>10}{'clipped':>10}")
    for name, pool in pools.items():
        for part in pool.stats.get("parts", [pool.stats]):
            print(f"{name:<22}{str(part.get('label', '?')):<24}"
                  f"{part.get('dropped_non_finite', 0):>12}"
                  f"{part.get('dropped_silent', 0):>10}"
                  f"{part.get('dropped_clipped', 0):>10}")

    print("\n" + "=" * 78)
    print("SPLIT HYGIENE")
    print("=" * 78)
    overlaps = {}
    for role in ("clean", "noise"):
        train_subjects = set(pools[f"{role}_train"].subject.tolist())
        val_subjects = set(pools[f"{role}_val"].subject.tolist())
        shared = sorted(train_subjects & val_subjects)
        overlaps[role] = shared
        verdict = "OK (disjoint)" if not shared else f"LEAK: {shared[:8]}"
        print(f"{role:<8} train {len(train_subjects):>4} subjects, "
              f"val {len(val_subjects):>4} subjects -> {verdict}")
        if shared:
            failures.append(f"{role}: subjects in both train and val: {shared[:8]}")
    report["subject_overlap"] = overlaps

    device_clean_val = [
        subject for subject in set(pools["clean_val"].subject.tolist())
        if not subject.startswith("circor_")
    ]
    print(f"\ndevice val subjects: {sorted(device_clean_val)}")
    if args.val_circor:
        print("CirCor val rows come from the pool's subject-disjoint val split. Note that "
              "with --val_circor the two arms are no longer scored on the same set.")
    else:
        print("Validation is device-only, so both arms are scored on the same held-out subject.")

    print("\n" + "=" * 78)
    if warnings:
        print("WARNINGS")
        for warning in warnings:
            print(f"  - {warning}")
    if failures:
        print("FAILED")
        for failure in failures:
            print(f"  - {failure}")
    else:
        print("All format, purity and split checks passed.")
    print("=" * 78)
    report["failures"] = failures
    report["warnings"] = warnings

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
        print(f"wrote {args.json}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
