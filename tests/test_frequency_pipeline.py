"""Model, loss and mixing tests. See test_pools.py for the dataset layer."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from src.frequency_data import MixingConfig, SyntheticFrequencyDataset
from src.frequency_loss import FrequencyDenoiseLoss
from src.frequency_metrics import snr_db
from src.frequency_model import CardioSpecNet, FrequencyModelConfig


class FrequencyPipelineTests(unittest.TestCase):
    def test_model_shape_and_gradient(self) -> None:
        torch.manual_seed(0)
        model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=4, grid_blocks=1))
        chest = torch.randn(2, 8_000) * 0.01
        reference = torch.randn(2, 8_000) * 0.01
        target = torch.randn(2, 8_000) * 0.005
        output = model(chest, reference, torch.tensor([1.0, 0.0]))
        self.assertEqual(output.shape, chest.shape)
        loss, terms = FrequencyDenoiseLoss()(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)
        self.assertIn("over_attenuation", terms)

    def test_dataset_is_deterministic_and_paired(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            clean_dir = root / "clean"
            noise_dir = root / "noise"
            clean_dir.mkdir()
            noise_dir.mkdir()
            rng = np.random.default_rng(1)
            clean = (rng.normal(0, 500, (20, 8_000))).astype(np.int16)
            noise = (rng.normal(0, 2_000, (20, 8_000))).astype(np.int16)
            np.savez_compressed(clean_dir / "clean.npz", x=clean)
            np.savez_compressed(noise_dir / "noise.npz", x=noise)
            config = MixingConfig(reference_dropout_probability=0.0, identity_probability=0.0)
            dataset = SyntheticFrequencyDataset(
                clean_dir,
                noise_dir,
                samples_per_epoch=4,
                source_stride=2,
                seed=7,
                config=config,
            )
            first = dataset[2]
            second = dataset[2]
            self.assertTrue(torch.equal(first["chest"], second["chest"]))
            self.assertTrue(torch.equal(first["clean"], second["clean"]))
            self.assertTrue(torch.equal(first["clean_raw"], second["clean_raw"]))
            self.assertEqual(first["chest"].shape, (8_000,))
            self.assertEqual(first["clean_raw"].shape, (8_000,))
            self.assertEqual(float(first["ref_available"]), 1.0)
            self.assertFalse(torch.equal(first["chest"], first["clean"]))

    def test_eval_mixtures_are_identical_across_arms(self) -> None:
        """The whole comparison rests on this: same seed and same eval pool must
        produce bit-identical mixtures, so the two arms are scored on one set."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "clean").mkdir()
            (root / "noise").mkdir()
            rng = np.random.default_rng(2)
            np.savez_compressed(
                root / "clean" / "c.npz", x=rng.normal(0, 600, (12, 8_000)).astype(np.int16)
            )
            np.savez_compressed(
                root / "noise" / "n.npz", x=rng.normal(0, 600, (12, 8_000)).astype(np.int16)
            )
            config = MixingConfig(
                snr_min_db=5.0,
                snr_max_db=5.0,
                identity_probability=0.0,
                reference_dropout_probability=0.0,
                chest_only_noise_max=0.0,
            )
            build = lambda: SyntheticFrequencyDataset(  # noqa: E731
                root / "clean", root / "noise", samples_per_epoch=6, seed=99, config=config
            )
            first, second = build(), build()
            for index in range(6):
                for key in ("chest", "clean", "reference"):
                    self.assertTrue(torch.equal(first[index][key], second[index][key]), key)

    def test_fixed_snr_bin_hits_the_requested_snr(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "clean").mkdir()
            (root / "noise").mkdir()
            rng = np.random.default_rng(3)
            np.savez_compressed(
                root / "clean" / "c.npz", x=rng.normal(0, 800, (16, 8_000)).astype(np.int16)
            )
            np.savez_compressed(
                root / "noise" / "n.npz", x=rng.normal(0, 800, (16, 8_000)).astype(np.int16)
            )
            for target_snr in (0.0, 10.0):
                dataset = SyntheticFrequencyDataset(
                    root / "clean",
                    root / "noise",
                    samples_per_epoch=8,
                    seed=11,
                    config=MixingConfig(
                        snr_min_db=target_snr,
                        snr_max_db=target_snr,
                        identity_probability=0.0,
                        reference_dropout_probability=0.0,
                        chest_only_noise_max=0.0,
                    ),
                )
                measured = [
                    float(snr_db(dataset[i]["chest"][None], dataset[i]["clean"][None]))
                    for i in range(8)
                ]
                # SNR is set on the bandpassed signals, so a few dB of slack.
                self.assertAlmostEqual(float(np.mean(measured)), target_snr, delta=4.0)


if __name__ == "__main__":
    unittest.main()
