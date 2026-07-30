"""Tests for the device+CirCor pool layer. No torch required."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.pools import (
    CLIP_LEVEL,
    DEFAULT_TARGET_RMS,
    FULL_SCALE,
    WINDOW_SAMPLES,
    build_clean_pool,
    build_noise_pool,
    concat_pools,
    load_circor_pool,
    load_device_pool,
    purity_mask,
    thin,
    to_float,
    window_rms,
)

RNG = np.random.default_rng(0)


def device_npz(path: Path, count: int, amplitude: int = 500) -> None:
    x = RNG.normal(0.0, amplitude, (count, WINDOW_SAMPLES)).astype(np.int16)
    np.savez_compressed(
        path,
        x=x,
        start_idx=np.arange(count, dtype=np.int64) * 4_000,
        segment_id=np.zeros(count, dtype=np.int32),
        start_wall_epoch_us=np.arange(count, dtype=np.int64) * 1_000_000,
    )


def circor_npz(path: Path, per_split: int = 10) -> None:
    count = per_split * 2
    x = (RNG.normal(0.0, 0.02, (count, WINDOW_SAMPLES))).astype(np.float32)
    np.savez_compressed(
        path,
        x=x,
        sample_rate=4_000,
        window_samples=WINDOW_SAMPLES,
        split=np.array(["train"] * per_split + ["val"] * per_split),
        subject=np.array([f"{1000 + i // 2}" for i in range(count)]),
        record=np.array([f"{1000 + i // 2}_MV" for i in range(count)]),
        bpm=np.array([70.0] * per_split + [140.0] * per_split),
    )


class PrimitiveTests(unittest.TestCase):
    def test_int16_uses_full_scale(self) -> None:
        x = np.array([[-32768, 0, 32767]], dtype=np.int16)
        got = to_float(x)
        self.assertEqual(got.dtype, np.float32)
        self.assertAlmostEqual(float(got[0, 0]), -1.0, places=6)
        self.assertAlmostEqual(float(got[0, 2]), 32767 / FULL_SCALE, places=6)

    def test_purity_mask_flags_each_defect_once(self) -> None:
        x = np.zeros((4, 16), dtype=np.float32)
        x[0] = RNG.normal(0, 0.1, 16)          # good
        x[1] = 0.0                             # silent
        x[2] = RNG.normal(0, 0.1, 16)
        x[2, 3] = CLIP_LEVEL                   # clipped
        x[3] = np.nan                          # non-finite
        keep, reasons = purity_mask(x)
        np.testing.assert_array_equal(keep, [True, False, False, False])
        self.assertEqual(reasons["dropped_silent"], 1)
        self.assertEqual(reasons["dropped_clipped"], 1)
        self.assertEqual(reasons["dropped_non_finite"], 1)

    def test_thin_spans_the_pool_instead_of_truncating(self) -> None:
        index = thin(100, source_stride=1, max_windows=5)
        self.assertEqual(len(index), 5)
        self.assertEqual(index[0], 0)
        self.assertEqual(index[-1], 99)
        np.testing.assert_array_equal(thin(10, source_stride=3, max_windows=None), [0, 3, 6, 9])
        with self.assertRaises(ValueError):
            thin(10, source_stride=0, max_windows=None)


class DevicePoolTests(unittest.TestCase):
    def test_loads_normalises_and_labels(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device_npz(root / "heart_bw1_windows.npz", 6)
            device_npz(root / "heart_aw2_windows.npz", 4)
            pool = load_device_pool(root)
        self.assertEqual(len(pool), 10)
        self.assertEqual(sorted(set(pool.subject.tolist())), ["aw2", "bw1"])
        self.assertEqual(pool.source_counts, {"device": 10})
        np.testing.assert_allclose(window_rms(pool.x), DEFAULT_TARGET_RMS, rtol=1e-4)
        # DC removed
        self.assertLess(float(np.abs(pool.x.mean(axis=1)).max()), 1e-6)

    def test_skips_preprocessed_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device_npz(root / "heart_bw1_windows.npz", 5)
            device_npz(root / "heart_bw1_preprocessed_windows.npz", 5)
            self.assertEqual(len(load_device_pool(root)), 5)
            self.assertEqual(len(load_device_pool(root, skip_preprocessed=False)), 10)

    def test_target_rms_none_preserves_level(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            device_npz(root / "heart_bw1_windows.npz", 4, amplitude=500)
            pool = load_device_pool(root, target_rms=None)
        self.assertLess(float(np.median(window_rms(pool.x))), 0.03)

    def test_wrong_window_length_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(root / "short_windows.npz", x=np.ones((3, 4_000), dtype=np.int16))
            with self.assertRaisesRegex(ValueError, "window length"):
                load_device_pool(root)

    def test_missing_directory_and_empty_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                load_device_pool(Path(temporary) / "nope")
            with self.assertRaises(FileNotFoundError):
                load_device_pool(temporary)


class CirCorPoolTests(unittest.TestCase):
    def test_split_filter_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.npz"
            circor_npz(path, per_split=8)
            train = load_circor_pool(path, "train")
            val = load_circor_pool(path, "val")
        self.assertEqual(len(train), 8)
        self.assertEqual(len(val), 8)
        self.assertTrue(all(name.startswith("circor_") for name in train.subject))
        self.assertEqual(train.source_counts, {"circor": 8})
        # subject-disjoint by construction of the pool
        self.assertFalse(set(train.subject.tolist()) & set(val.subject.tolist()))
        np.testing.assert_allclose(window_rms(train.x), DEFAULT_TARGET_RMS, rtol=1e-4)

    def test_heart_rate_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.npz"
            circor_npz(path, per_split=6)  # train rows are 70 bpm, val rows 140 bpm
            kept = load_circor_pool(path, "train", heart_rate_max=100.0)
            self.assertEqual(len(kept), 6)
            with self.assertRaisesRegex(ValueError, "no windows left"):
                load_circor_pool(path, "val", heart_rate_max=100.0)

    def test_bad_split_and_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.npz"
            circor_npz(path)
            with self.assertRaises(ValueError):
                load_circor_pool(path, "test")
            with self.assertRaisesRegex(FileNotFoundError, "build_circor_pool"):
                load_circor_pool(Path(temporary) / "absent.npz", "train")


class CombinationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        clean = root / "clean" / "step_1s"
        noise = root / "noise" / "step_1s"
        for split, count in (("train", 12), ("val", 6)):
            (clean / split).mkdir(parents=True)
            (noise / split).mkdir(parents=True)
            device_npz(clean / split / f"heart_bw{split[0]}_windows.npz", count)
            device_npz(noise / split / f"noise{split[0]}_windows.npz", count)
        circor = root / "circor.npz"
        circor_npz(circor, per_split=9)
        return clean, noise, circor

    def test_clean_pool_combines_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, noise, circor = self._fixture(Path(temporary))
            train = build_clean_pool(clean / "train", "train", circor_pool=circor)
            val = build_clean_pool(clean / "val", "val", circor_pool=circor)
            noise_train = build_noise_pool(noise / "train", "train")
        self.assertEqual(train.source_counts, {"circor": 9, "device": 12})
        self.assertEqual(val.source_counts, {"circor": 9, "device": 6})
        self.assertEqual(len(train), 21)
        # one common level for every source, so the mixing constants mean one thing
        np.testing.assert_allclose(window_rms(train.x), DEFAULT_TARGET_RMS, rtol=1e-4)
        np.testing.assert_allclose(window_rms(noise_train.x), DEFAULT_TARGET_RMS, rtol=1e-4)
        self.assertEqual(noise_train.source_counts, {"device": 12})
        self.assertEqual(train.stats["parts"][0]["label"], "device_clean:train")
        self.assertEqual(train.stats["parts"][1]["label"], "circor:train")

    def test_device_only_when_no_circor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, _, _ = self._fixture(Path(temporary))
            train = build_clean_pool(clean / "train", "train", circor_pool=None)
        self.assertEqual(train.source_counts, {"device": 12})

    def test_max_windows_caps_each_source_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, _, circor = self._fixture(Path(temporary))
            train = build_clean_pool(clean / "train", "train", circor_pool=circor, max_windows=4)
        self.assertEqual(train.source_counts, {"circor": 4, "device": 4})

    def test_mismatched_normalisation_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").mkdir()
            (root / "b").mkdir()
            device_npz(root / "a" / "one_windows.npz", 3)
            device_npz(root / "b" / "two_windows.npz", 3)
            first = load_device_pool(root / "a", target_rms=0.05)
            second = load_device_pool(root / "b", target_rms=None)
            with self.assertRaisesRegex(ValueError, "different target RMS"):
                concat_pools([first, second], label="mixed")


class ShippedDataTests(unittest.TestCase):
    """Runs against the real device npz committed under data/, when present."""

    data_root = Path(__file__).resolve().parents[1] / "data"

    def setUp(self) -> None:
        if not (self.data_root / "clean" / "step_1s" / "train").is_dir():
            self.skipTest("device data not present")

    def test_expected_counts_and_disjoint_subjects(self) -> None:
        clean_train = build_clean_pool(self.data_root / "clean" / "step_1s" / "train", "train")
        clean_val = build_clean_pool(self.data_root / "clean" / "step_1s" / "val", "val")
        noise_train = build_noise_pool(self.data_root / "noise" / "step_1s" / "train", "train")
        noise_val = build_noise_pool(self.data_root / "noise" / "step_1s" / "val", "val")
        self.assertEqual(len(clean_train), 3_936)
        self.assertEqual(len(clean_val), 1_024)
        self.assertEqual(len(noise_train), 1_345)  # 1367 stored, 22 clipped windows dropped
        self.assertEqual(len(noise_val), 377)
        self.assertFalse(
            set(clean_train.subject.tolist()) & set(clean_val.subject.tolist()),
            "device clean subjects leak between train and val",
        )
        for pool in (clean_train, clean_val, noise_train, noise_val):
            np.testing.assert_allclose(window_rms(pool.x), DEFAULT_TARGET_RMS, rtol=1e-4)
            self.assertLess(float(np.abs(pool.x).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
