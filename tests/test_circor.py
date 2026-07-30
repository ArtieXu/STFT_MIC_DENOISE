"""CirCor subject identity tests."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from src.circor import build_pool, canonical_subject_map


class CirCorIdentityTests(unittest.TestCase):
    def test_additional_ids_share_one_canonical_subject(self) -> None:
        metadata = {
            "49729": {"Additional ID": "69125"},
            "69125": {"Additional ID": "49729.0"},
            "12345": {"Additional ID": "nan"},
        }
        mapping = canonical_subject_map(metadata)
        self.assertEqual(mapping["49729"], mapping["69125"])
        self.assertEqual(mapping["49729"], "49729")
        self.assertEqual(mapping["12345"], "12345")

    def test_linked_visits_are_limited_and_split_as_one_human(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "training_data"
            data.mkdir()
            rows = [
                {
                    "Patient ID": "100",
                    "Additional ID": "200",
                    "Murmur": "Absent",
                },
                {
                    "Patient ID": "200",
                    "Additional ID": "100",
                    "Murmur": "Absent",
                },
                {
                    "Patient ID": "300",
                    "Additional ID": "",
                    "Murmur": "Absent",
                },
            ]
            with (root / "training_data.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("Patient ID", "Additional ID", "Murmur"),
                )
                writer.writeheader()
                writer.writerows(rows)

            time = np.arange(5 * 4_000, dtype=np.float64) / 4_000
            audio = (2_000 * np.sin(2 * np.pi * 70 * time)).astype(np.int16)
            tsv = "0 1 1\n1 2 2\n2 3 3\n3 4 4\n4 5 1\n"
            for participant_id in ("100", "200", "300"):
                stem = data / f"{participant_id}_AV"
                wavfile.write(stem.with_suffix(".wav"), 4_000, audio)
                stem.with_suffix(".tsv").write_text(tsv, encoding="utf-8")

            pool = build_pool(
                root,
                windows_per_patient=2,
                max_patients=10,
                val_fraction=0.5,
                seed=7,
            )

        self.assertEqual(set(pool["subject"]), {"100", "300"})
        self.assertEqual(int((pool["subject"] == "100").sum()), 2)
        self.assertEqual(
            set(pool["participant_id"][pool["subject"] == "100"]),
            {"100", "200"},
        )
        for subject in set(pool["subject"]):
            self.assertEqual(len(set(pool["split"][pool["subject"] == subject])), 1)
        self.assertEqual(pool["stats"]["n_subjects"], 2)
        self.assertEqual(pool["stats"]["n_participant_ids"], 3)
        self.assertEqual(pool["stats"]["linked_id_groups"], 1)


if __name__ == "__main__":
    unittest.main()
