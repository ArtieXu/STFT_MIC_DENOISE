#!/usr/bin/env python3
"""Re-fetch the device recordings this folder trains on, from the upstream repo.

    python scripts/fetch_device_data.py                # clean + noise, step_1s
    python scripts/fetch_device_data.py --step 0.1s    # a different window hop

The npz files are already committed under ``data/``; this script exists so their
provenance is reproducible and so another hop can be pulled without hunting
through the source repo by hand.

Source: https://github.com/jiayimaggieshao/denoise_stft
"""
from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW = "https://raw.githubusercontent.com/jiayimaggieshao/denoise_stft/main"

CLEAN = {
    "train": ["heart_aw1", "heart_aw2", "heart_aw6", "heart_bw1", "heart_bw2", "heart_bw5", "heart_bw6"],
    "val": ["heart_aw4", "heart_bw4"],
}
NOISE = {
    "train": ["noise1", "noise2", "noise3", "noise5"],
    "val": ["noise6"],
}


def download(url: str, destination: Path, force: bool) -> str:
    if destination.exists() and not force:
        return "skip"
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            payload = response.read()
    except urllib.error.HTTPError as error:
        raise SystemExit(f"HTTP {error.code} for {url}") from error
    destination.write_bytes(payload)
    return f"{len(payload) / 2**20:.1f} MiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--step", default="1s", choices=("0.01s", "0.1s", "1s"))
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist.")
    args = parser.parse_args()

    jobs: list[tuple[str, Path]] = []
    for role, table in (("clean", CLEAN), ("noise", NOISE)):
        for split, stems in table.items():
            for stem in stems:
                name = f"{stem}_windows.npz"
                jobs.append((
                    f"{RAW}/data/{role}/step_{args.step}/{split}/{name}",
                    args.data_root / role / f"step_{args.step}" / split / name,
                ))

    for url, destination in jobs:
        status = download(url, destination, args.force)
        print(f"{status:>10}  {destination.relative_to(REPO_ROOT)}")
    print(f"\n{len(jobs)} files under {args.data_root}")
    return


if __name__ == "__main__":
    sys.exit(main())
