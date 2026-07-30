"""Tests for overlap-add reconstruction helpers."""
from __future__ import annotations

import unittest

import numpy as np

from src.window_alignment import (
    hop_samples_from_step,
    overlap_add_runs,
    resolve_start_samples,
    split_contiguous_runs,
)


class WindowAlignmentTests(unittest.TestCase):
    def test_resolve_start_samples_accepts_hop_indices(self) -> None:
        starts = resolve_start_samples(
            np.array([0, 1, 2, 3], dtype=np.int64),
            window_samples=8000,
            hop_samples=4000,
        )
        np.testing.assert_array_equal(starts, [0, 4000, 8000, 12000])

    def test_split_contiguous_runs_breaks_on_timeline_gap(self) -> None:
        starts = np.array([0, 4000, 8000, 50000, 54000], dtype=np.int64)
        runs = split_contiguous_runs(
            starts,
            window_samples=8000,
            hop_samples=4000,
        )
        self.assertEqual(len(runs), 2)
        self.assertEqual(len(runs[0]), 3)
        self.assertEqual(len(runs[1]), 2)

    def test_overlap_add_runs_concatenates_without_silent_holes(self) -> None:
        hop = 4000
        window = np.ones(8000, dtype=np.float32)
        windows = np.stack([window, window, window, window], axis=0)
        starts = np.array([0, 4000, 8000, 50000], dtype=np.int64)
        audio = overlap_add_runs(windows, starts, hop_samples=hop)
        expected_len = (2 - 1) * hop + 8000 + ((2 - 1) * hop + 8000)
        self.assertEqual(len(audio), expected_len)
        self.assertGreater(float(np.abs(audio).max()), 0.0)
        self.assertLess(float(np.abs(audio).mean()), 1.0)

    def test_hop_samples_from_step(self) -> None:
        self.assertEqual(hop_samples_from_step("1s", 4000), 4000)
        self.assertEqual(hop_samples_from_step("0.1s", 4000), 400)


if __name__ == "__main__":
    unittest.main()
