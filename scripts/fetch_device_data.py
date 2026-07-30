#!/usr/bin/env python3
"""Re-fetch the device recordings this folder trains on, from the upstream repo.

    python scripts/fetch_device_data.py                       # clean + noise, step_1s
    python scripts/fetch_device_data.py --include-test-real    # + held-out subject-6 walking
    python scripts/fetch_device_data.py --step 0.1s           # a different window hop

The npz files are already committed under ``data/``; this script exists so their
provenance is reproducible and so another hop can be pulled without hunting
through the source repo by hand.

Source: https://github.com/jiayimaggieshao/denoise_stft
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pools import DEVICE_FILE_SPLITS, DEVICE_UPSTREAM_COMMIT  # noqa: E402

REPO = "https://github.com/jiayimaggieshao/denoise_stft.git"
UPSTREAM_COMMIT = DEVICE_UPSTREAM_COMMIT
RAW = (
    "https://raw.githubusercontent.com/jiayimaggieshao/denoise_stft/"
    f"{UPSTREAM_COMMIT}"
)

# The upstream repository still stores these three archives under its legacy
# ``val`` directories. Locally they are placed by DEVICE_FILE_SPLITS: clean 4
# is training data, while noise 6 is final-test data.
UPSTREAM_LEGACY_VAL = {
    ("clean", "heart_aw4"),
    ("clean", "heart_bw4"),
    ("noise", "noise6"),
}
# Subject 6 is absent from training and is the only real walking example
# fetched for qualitative final-test inspection.
TEST_REAL = ("heart_w6",)


def existing_archive(
    data_root: Path, role: str, step: str, name: str
) -> Path | None:
    """Find a canonical or legacy copy so a split change needs no re-download."""

    matches = sorted((data_root / role / f"step_{step}").glob(f"*/{name}"))
    return matches[0] if matches else None


def clone_fallback(step: str, data_root: Path, jobs: list[tuple[str, Path]]) -> None:
    """Some networks allow git but block raw.githubusercontent. Try a shallow clone."""
    import shutil
    import subprocess
    import tempfile

    print("raw download blocked; falling back to a shallow git clone")
    with tempfile.TemporaryDirectory() as temporary:
        clone = Path(temporary) / "repo"
        subprocess.run(
            ["git", "init", str(clone)],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "remote", "add", "origin", REPO],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "fetch", "--depth", "1", "origin", UPSTREAM_COMMIT],
            check=True, capture_output=True,
        )
        subprocess.run(
            ["git", "-C", str(clone), "checkout", "--detach", "FETCH_HEAD"],
            check=True, capture_output=True,
        )
        source_root = clone / "data"
        for url, destination in jobs:
            relative = url.split(f"/{UPSTREAM_COMMIT}/data/", 1)[1]
            source = source_root / relative
            if not source.is_file():
                print(f"  missing in clone: {relative}")
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source, destination)
            print(f"{'cloned':>10}  {destination.relative_to(REPO_ROOT)}")


def download(url: str, destination: Path, force: bool) -> str:
    if destination.exists() and not force:
        return "skip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return f"{len(payload) / 2**20:.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default="1s", choices=("0.01s", "0.1s", "1s"))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--include-test-real", action="store_true",
                        help="Also fetch data/test_real/walking/step_*/heart_w*.npz, needed by "
                             "scripts/make_demo_figures.py for the real-recording figure.")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    jobs: list[tuple[str, Path]] = []
    for role, split_table in DEVICE_FILE_SPLITS.items():
        for logical_split, stems in split_table.items():
            for stem in stems:
                name = f"{stem}_windows.npz"
                # The URL follows the upstream legacy layout; the destination
                # follows the exact, validation-free experiment manifest.
                upstream_split = (
                    "val" if (role, stem) in UPSTREAM_LEGACY_VAL else "train"
                )
                jobs.append((
                    f"{RAW}/data/{role}/step_{args.step}/{upstream_split}/{name}",
                    args.data_root / role / f"step_{args.step}" / logical_split / name,
                ))

    if args.include_test_real:
        for stem in TEST_REAL:
            name = f"{stem}_windows.npz"
            jobs.append((
                f"{RAW}/data/test_real/walking/step_{args.step}/{name}",
                args.data_root / "test_real" / "walking" / f"step_{args.step}" / name,
            ))

    pending = []
    for url, destination in jobs:
        role = destination.parents[2].name
        legacy = (
            destination if role == "test_real" and destination.is_file()
            else existing_archive(args.data_root, role, args.step, destination.name)
        )
        if args.force or legacy is None:
            pending.append((url, destination))
        else:
            print(f"{'skip':>10}  {legacy.relative_to(REPO_ROOT)}")
    try:
        for url, destination in pending:
            status = download(url, destination, args.force)
            print(f"{status:>10}  {destination.relative_to(REPO_ROOT)}")
    except SystemExit:
        raise
    except Exception as error:  # network shape varies; the clone path is the safety net
        print(f"direct download failed ({error})")
        clone_fallback(args.step, args.data_root, pending)
    print(f"\n{len(jobs)} files under {args.data_root}")
    return


if __name__ == "__main__":
    sys.exit(main())
