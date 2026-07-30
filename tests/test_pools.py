"""Tests for leakage-safe pools and minimal mono mixture sampling."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.circor import POOL_SCHEMA_VERSION
from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.pools import (
    CLIP_LEVEL,
    DEFAULT_TARGET_RMS,
    DEVICE_FILE_SPLITS,
    DEVICE_PROTOCOL_ID,
    FULL_SCALE,
    WINDOW_SAMPLES,
    build_clean_pool,
    build_noise_pool,
    canonical_device_subject,
    concat_pools,
    device_protocol_metadata,
    device_split_for_recording,
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
        pool_schema_version=POOL_SCHEMA_VERSION,
        sample_rate=4_000,
        window_samples=WINDOW_SAMPLES,
        split=np.array(["train"] * per_split + ["val"] * per_split),
        subject=np.array([f"{1000 + i // 2}" for i in range(count)]),
        participant_id=np.array([f"{1000 + i // 2}" for i in range(count)]),
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
        self.assertEqual(sorted(set(pool.subject.tolist())), ["device_1", "device_2"])
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

    def test_canonical_subject_is_shared_across_modalities(self) -> None:
        names = (
            "heart_aw6_windows", "heart_bw6_windows",
            "heart_w6_windows", "noise6_windows",
        )
        self.assertEqual(
            {canonical_device_subject(name) for name in names},
            {"device_6"},
        )

    def test_exact_recording_manifest_is_independent_of_legacy_storage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "clean" / "step_1s"
            (step / "train").mkdir(parents=True)
            (step / "val").mkdir()
            # Deliberately store logical train/test files in the "wrong"
            # legacy folders. The declared recording manifest, not the folder
            # name, must decide which pool receives them.
            device_npz(step / "val" / "heart_bw1_windows.npz", 3)
            device_npz(step / "train" / "heart_aw2_windows.npz", 4)
            device_npz(step / "val" / "heart_aw4_windows.npz", 5)
            device_npz(step / "train" / "heart_bw5_windows.npz", 6)
            device_npz(step / "train" / "heart_aw6_windows.npz", 7)

            train = build_clean_pool(step / "train", "train")
            test = build_clean_pool(step / "test", "test")

        self.assertEqual(len(train), 18)
        self.assertEqual(
            set(train.subject),
            {"device_1", "device_2", "device_4", "device_5"},
        )
        self.assertEqual(
            set(train.origin),
            {
                "heart_bw1_windows",
                "heart_aw2_windows",
                "heart_aw4_windows",
                "heart_bw5_windows",
            },
        )
        self.assertEqual(len(test), 7)
        self.assertEqual(set(test.subject), {"device_6"})
        self.assertEqual(set(test.origin), {"heart_aw6_windows"})

    def test_clean_and_noise_have_separate_exact_manifests(self) -> None:
        self.assertEqual(
            DEVICE_FILE_SPLITS,
            {
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
            },
        )
        for stem in ("heart_aw1", "heart_bw2", "heart_aw4", "heart_bw5"):
            self.assertEqual(device_split_for_recording(stem, "clean"), "train")
        for stem in ("noise1", "noise2", "noise3", "noise5"):
            self.assertEqual(device_split_for_recording(stem, "noise"), "train")
        self.assertEqual(device_split_for_recording("heart_bw6", "clean"), "test")
        self.assertEqual(device_split_for_recording("noise6", "noise"), "test")
        metadata = device_protocol_metadata()
        self.assertEqual(metadata["id"], DEVICE_PROTOCOL_ID)
        self.assertEqual(metadata["selection_policy"], "fixed training budget; no validation")

    def test_noise_manifest_ignores_legacy_folder_names(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            step = Path(temporary) / "noise" / "step_1s"
            (step / "train").mkdir(parents=True)
            (step / "val").mkdir()
            device_npz(step / "val" / "noise1_windows.npz", 2)
            device_npz(step / "train" / "noise2_windows.npz", 3)
            device_npz(step / "val" / "noise3_windows.npz", 4)
            device_npz(step / "train" / "noise5_windows.npz", 5)
            device_npz(step / "train" / "noise6_windows.npz", 6)

            train = build_noise_pool(step / "train", "train")
            test = build_noise_pool(step / "test", "test")

        self.assertEqual(len(train), 14)
        self.assertEqual(
            set(train.origin),
            {
                "noise1_windows",
                "noise2_windows",
                "noise3_windows",
                "noise5_windows",
            },
        )
        self.assertEqual(len(test), 6)
        self.assertEqual(set(test.origin), {"noise6_windows"})


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

    def test_legacy_pool_without_linked_id_schema_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "legacy_pool.npz"
            np.savez_compressed(
                path,
                x=RNG.normal(0.0, 0.02, (2, WINDOW_SAMPLES)).astype(np.float32),
                split=np.array(["train", "val"]),
                subject=np.array(["100", "200"]),
                record=np.array(["100_AV", "200_AV"]),
            )
            with self.assertRaisesRegex(KeyError, "Additional ID aliases"):
                load_circor_pool(path, "train")


class CombinationTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        clean = root / "clean" / "step_1s"
        noise = root / "noise" / "step_1s"
        for directory in (clean / "train", clean / "val", noise / "train", noise / "val"):
            directory.mkdir(parents=True)
        device_npz(clean / "train" / "heart_bw1_windows.npz", 12)
        # Subject 4 is part of the final training fit even though the upstream
        # archive stores it under val/.
        device_npz(clean / "val" / "heart_bw4_windows.npz", 6)
        for stem in ("noise1", "noise2", "noise3", "noise5"):
            device_npz(noise / "train" / f"{stem}_windows.npz", 6)
        circor = root / "circor.npz"
        circor_npz(circor, per_split=9)
        return clean, noise, circor

    def test_clean_pool_combines_both_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, noise, circor = self._fixture(Path(temporary))
            train = build_clean_pool(clean / "train", "train", circor_pool=circor)
            noise_train = build_noise_pool(noise / "train", "train")
        self.assertEqual(train.source_counts, {"circor": 9, "device": 18})
        self.assertEqual(len(train), 27)
        # one common level for every source, so the mixing constants mean one thing
        np.testing.assert_allclose(window_rms(train.x), DEFAULT_TARGET_RMS, rtol=1e-4)
        np.testing.assert_allclose(window_rms(noise_train.x), DEFAULT_TARGET_RMS, rtol=1e-4)
        self.assertEqual(noise_train.source_counts, {"device": 24})
        self.assertEqual(
            set(noise_train.origin),
            {
                "noise1_windows",
                "noise2_windows",
                "noise3_windows",
                "noise5_windows",
            },
        )
        self.assertEqual(train.stats["parts"][0]["label"], "device_clean:train")
        self.assertEqual(train.stats["parts"][1]["label"], "circor:train")

    def test_device_only_when_no_circor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, _, _ = self._fixture(Path(temporary))
            train = build_clean_pool(clean / "train", "train", circor_pool=None)
        self.assertEqual(train.source_counts, {"device": 18})

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

    def test_combined_sampling_is_exactly_half_each_source(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            clean, noise, circor = self._fixture(Path(temporary))
            clean_pool = build_clean_pool(clean / "train", "train", circor_pool=circor)
            noise_pool = build_noise_pool(noise / "train", "train")
            dataset = SyntheticFrequencyDataset(
                clean_pool,
                noise_pool,
                samples_per_epoch=24,
                seed=7,
                config=MixingConfig(snr_min_db=6.0, snr_max_db=6.0),
            )

            for epoch in (0, 1):
                dataset.set_epoch(epoch)
                sources = [dataset.clean_source_for_index(i) for i in range(len(dataset))]
                self.assertEqual(sources.count("device"), 12)
                self.assertEqual(sources.count("circor"), 12)

            sample = dataset[0]
            self.assertEqual(set(sample), {"clean", "noise", "chest"})
            np.testing.assert_allclose(
                sample["chest"].numpy(),
                sample["clean"].numpy() + sample["noise"].numpy(),
                rtol=0.0,
                atol=1e-7,
            )
            clean_rms = float(np.sqrt(np.mean(sample["clean"].numpy().astype(np.float64) ** 2)))
            noise_rms = float(np.sqrt(np.mean(sample["noise"].numpy().astype(np.float64) ** 2)))
            self.assertAlmostEqual(20.0 * np.log10(clean_rms / noise_rms), 6.0, places=5)

            with self.assertRaisesRegex(ValueError, "divisible"):
                SyntheticFrequencyDataset(
                    clean_pool, noise_pool, samples_per_epoch=19
                )


class ShippedDataTests(unittest.TestCase):
    """Runs against the real device npz committed under data/, when present."""

    data_root = Path(__file__).resolve().parents[1] / "data"

    def setUp(self) -> None:
        if not (self.data_root / "clean" / "step_1s" / "train").is_dir():
            self.skipTest("device data not present")

    def test_expected_counts_and_disjoint_subjects(self) -> None:
        clean_train = build_clean_pool(self.data_root / "clean" / "step_1s" / "train", "train")
        clean_test = build_clean_pool(self.data_root / "clean" / "step_1s" / "test", "test")
        noise_train = build_noise_pool(self.data_root / "noise" / "step_1s" / "train", "train")
        noise_test = build_noise_pool(self.data_root / "noise" / "step_1s" / "test", "test")
        self.assertEqual(len(clean_train), 3_683)
        self.assertEqual(len(clean_test), 1_277)
        self.assertEqual(len(noise_train), 1_345)  # noise3 has 22 clipped windows
        self.assertEqual(len(noise_test), 377)

        self.assertEqual(
            set(clean_train.subject),
            {"device_1", "device_2", "device_4", "device_5"},
        )
        self.assertEqual(
            set(noise_train.origin),
            {
                "noise1_windows",
                "noise2_windows",
                "noise3_windows",
                "noise5_windows",
            },
        )
        train_subjects = set(clean_train.subject) | set(noise_train.subject)
        test_subjects = set(clean_test.subject) | set(noise_test.subject)
        self.assertFalse(train_subjects & test_subjects)
        self.assertEqual(test_subjects, {"device_6"})

        for pool in (
            clean_train, clean_test, noise_train, noise_test,
        ):
            np.testing.assert_allclose(window_rms(pool.x), DEFAULT_TARGET_RMS, rtol=1e-4)
            self.assertLess(float(np.abs(pool.x).max()), 1.0)


if __name__ == "__main__":
    unittest.main()
