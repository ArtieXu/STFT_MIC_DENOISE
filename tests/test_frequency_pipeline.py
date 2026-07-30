"""Model, loss and mixing tests. See test_pools.py for the dataset layer."""
from __future__ import annotations

from copy import deepcopy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn

from src.arms import (
    AMP_MAX_RETRIES,
    OPTIMIZER_UPDATE_POLICY,
    load_arm,
    parameter_count,
    validate_shared_protocol,
)
from src.frequency_data import (
    SAMPLER_VERSION,
    MixingConfig,
    SyntheticFrequencyDataset,
)
from src.frequency_loss import FrequencyDenoiseLoss, FrequencyLossConfig
from src.frequency_metrics import snr_db
from src.frequency_model import CardioSpecNet, FrequencyModelConfig
from src.pools import WindowPool, device_protocol_metadata
from train_frequency import optimizer_update


TRAIN_NOISE_ORIGINS = (
    "noise1_windows",
    "noise2_windows",
    "noise3_windows",
    "noise5_windows",
)
TRAIN_CLEAN_ORIGINS = (
    "heart_aw1_windows",
    "heart_aw2_windows",
    "heart_aw4_windows",
    "heart_bw1_windows",
    "heart_bw2_windows",
    "heart_bw4_windows",
    "heart_bw5_windows",
)


class FrequencyPipelineTests(unittest.TestCase):
    def test_dataset_marks_unsynchronised_epoch_state(self) -> None:
        epoch_tensor = MagicMock()
        epoch_tensor.share_memory_.side_effect = RuntimeError("disabled")
        with patch("src.frequency_data.torch.zeros", return_value=epoch_tensor):
            dataset = SyntheticFrequencyDataset(
                np.ones((1, 8_000), dtype=np.float32),
                np.ones((1, 8_000), dtype=np.float32),
                samples_per_epoch=1,
            )
        self.assertFalse(dataset.epoch_is_shared)

    def test_model_shape_and_gradient(self) -> None:
        torch.manual_seed(0)
        model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=4))
        chest = torch.randn(2, 8_000) * 0.01
        target = torch.randn(2, 8_000) * 0.005
        output = model(chest)
        self.assertEqual(output.shape, chest.shape)
        loss, terms = FrequencyDenoiseLoss()(output, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(model.head.weight.grad.abs().sum()), 0.0)
        self.assertEqual(set(terms), {"loss", "complex_stft"})
        with self.assertRaises(TypeError):
            model(chest, torch.zeros_like(chest))

    def test_amp_overflow_retries_same_batch_until_one_update(self) -> None:
        class CountingSGD(torch.optim.SGD):
            def __init__(self, params) -> None:
                super().__init__(params, lr=0.1)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        class FakeScaler:
            def __init__(self, failures: int) -> None:
                self.failures = failures
                self.attempts = 0
                self.value = 8.0

            def get_scale(self) -> float:
                return self.value

            @staticmethod
            def scale(loss: torch.Tensor) -> torch.Tensor:
                return loss

            @staticmethod
            def unscale_(optimizer) -> None:
                del optimizer

            def step(self, optimizer) -> None:
                if self.attempts >= self.failures:
                    optimizer.step()

            def update(self) -> None:
                if self.attempts < self.failures:
                    self.value *= 0.5
                self.attempts += 1

        class SquaredLoss(nn.Module):
            @staticmethod
            def forward(estimate, target):
                loss = (estimate - target).square().mean()
                return loss, {"loss": loss.detach()}

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = CountingSGD(model.parameters())
        scaler = FakeScaler(failures=2)
        batch = {
            "chest": torch.ones(2, 1),
            "clean": torch.zeros(2, 1),
        }

        _, _, retries, used_fp32, forward_fallback = optimizer_update(
            model,
            SquaredLoss(),
            batch,
            optimizer,
            scaler,
            device=torch.device("cpu"),
            grad_clip=10.0,
            use_amp=True,
        )

        self.assertEqual(retries, 2)
        self.assertFalse(used_fp32)
        self.assertFalse(forward_fallback)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertAlmostEqual(float(model.weight.detach().item()), 0.8, places=3)

    def test_amp_retry_limit_falls_back_to_one_fp32_update(self) -> None:
        class CountingSGD(torch.optim.SGD):
            def __init__(self, params) -> None:
                super().__init__(params, lr=0.1)
                self.step_calls = 0

            def step(self, closure=None):
                self.step_calls += 1
                return super().step(closure)

        class AlwaysSkippingScaler:
            def __init__(self) -> None:
                self.value = 8.0

            def get_scale(self) -> float:
                return self.value

            @staticmethod
            def scale(loss: torch.Tensor) -> torch.Tensor:
                return loss

            @staticmethod
            def unscale_(optimizer) -> None:
                del optimizer

            @staticmethod
            def step(optimizer) -> None:
                del optimizer

            def update(self) -> None:
                self.value *= 0.5

        class SquaredLoss(nn.Module):
            @staticmethod
            def forward(estimate, target):
                loss = (estimate - target).square().mean()
                return loss, {"loss": loss.detach()}

        model = nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(1.0)
        optimizer = CountingSGD(model.parameters())
        batch = {
            "chest": torch.ones(2, 1),
            "clean": torch.zeros(2, 1),
        }

        _, _, retries, used_fp32, _ = optimizer_update(
            model,
            SquaredLoss(),
            batch,
            optimizer,
            AlwaysSkippingScaler(),
            device=torch.device("cpu"),
            grad_clip=10.0,
            use_amp=True,
            max_amp_retries=2,
        )

        self.assertEqual(retries, 2)
        self.assertTrue(used_fp32)
        self.assertEqual(optimizer.step_calls, 1)
        self.assertAlmostEqual(float(model.weight.detach().item()), 0.8, places=3)

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
            config = MixingConfig()
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
            self.assertEqual(first["chest"].shape, (8_000,))
            self.assertEqual(set(first), {"chest", "clean", "noise"})
            torch.testing.assert_close(first["chest"], first["clean"] + first["noise"])

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
            )
            build = lambda: SyntheticFrequencyDataset(  # noqa: E731
                root / "clean", root / "noise", samples_per_epoch=6, seed=99, config=config
            )
            first, second = build(), build()
            for index in range(6):
                for key in ("chest", "clean", "noise"):
                    self.assertTrue(torch.equal(first[index][key], second[index][key]), key)

    def test_snr_bins_are_exact_for_linear_mixtures(self) -> None:
        """The generated waveform is exactly clean plus full-band scaled noise."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "clean").mkdir()
            (root / "noise").mkdir()
            rng = np.random.default_rng(5)
            np.savez_compressed(
                root / "clean" / "c.npz", x=rng.normal(0, 800, (16, 8_000)).astype(np.int16)
            )
            np.savez_compressed(
                root / "noise" / "n.npz", x=rng.normal(0, 800, (16, 8_000)).astype(np.int16)
            )
            for target_snr in (-5.0, 0.0, 10.0, 20.0):
                dataset = SyntheticFrequencyDataset(
                    root / "clean", root / "noise",
                    samples_per_epoch=8, seed=11,
                    config=MixingConfig(
                        snr_min_db=target_snr, snr_max_db=target_snr,
                    ),
                )
                measured = []
                for index in range(8):
                    sample = dataset[index]
                    target = sample["clean"][None]
                    measured.append(float(snr_db(sample["chest"][None], target)))
                    torch.testing.assert_close(
                        sample["chest"], sample["clean"] + sample["noise"]
                    )
                self.assertAlmostEqual(float(np.mean(measured)), target_snr, delta=1e-3)

    def test_combined_clean_sources_are_exactly_balanced(self) -> None:
        rng = np.random.default_rng(9)
        clean = rng.normal(0, 0.02, (12, 8_000)).astype(np.float32)
        noise = rng.normal(0, 0.02, (12, 8_000)).astype(np.float32)

        clean_pool = WindowPool(
            x=clean,
            source=np.array(["device"] * 6 + ["circor"] * 6),
            subject=np.array(["device_1"] * 6 + ["circor_1"] * 6),
            origin=np.array(["clean"] * 12),
            stats={"label": "combined", "n_windows": 12},
        )
        dataset = SyntheticFrequencyDataset(
            clean_pool, noise, samples_per_epoch=20, seed=4
        )
        scheduled = [dataset.clean_source_for_index(index) for index in range(len(dataset))]
        self.assertEqual(scheduled.count("device"), 10)
        self.assertEqual(scheduled.count("circor"), 10)

    def test_noise_recipe_is_exactly_balanced_and_independent_of_clean_arm(self) -> None:
        """Only the clean source may change between device and expanded arms.

        For every epoch/index, raw noise recording, raw pool index, and
        requested SNR must remain identical. Four training recordings must
        each contribute exactly one quarter of the epoch.
        """

        rng = np.random.default_rng(91)
        device_x = rng.normal(0, 0.02, (12, 8_000)).astype(np.float32)
        circor_x = rng.normal(0, 0.02, (14, 8_000)).astype(np.float32)
        noise_x = rng.normal(0, 0.02, (20, 8_000)).astype(np.float32)

        device_clean = WindowPool(
            x=device_x,
            source=np.array(["device"] * 12),
            subject=np.array([f"device_{1 + index % 4}" for index in range(12)]),
            origin=np.array([f"device_clean_{index}" for index in range(12)]),
            stats={"label": "device", "n_windows": 12},
        )
        combined_clean = WindowPool(
            x=np.concatenate([device_x, circor_x]),
            source=np.array(["device"] * 12 + ["circor"] * 14),
            subject=np.array(
                [f"device_{1 + index % 4}" for index in range(12)]
                + [f"circor_{100 + index}" for index in range(14)]
            ),
            origin=np.array(
                [f"device_clean_{index}" for index in range(12)]
                + [f"circor_clean_{index}" for index in range(14)]
            ),
            stats={"label": "combined", "n_windows": 26},
        )
        noise_pool = WindowPool(
            x=noise_x,
            source=np.array(["device"] * 20),
            subject=np.array(
                [f"device_{recording}" for recording in (1, 2, 3, 5) for _ in range(5)]
            ),
            origin=np.array(
                [origin for origin in TRAIN_NOISE_ORIGINS for _ in range(5)]
            ),
            stats={"label": "noise", "n_windows": 20},
        )
        config = MixingConfig(snr_min_db=-7.5, snr_max_db=12.5)
        device_arm = SyntheticFrequencyDataset(
            device_clean, noise_pool, samples_per_epoch=40, seed=2026, config=config
        )
        combined_arm = SyntheticFrequencyDataset(
            combined_clean, noise_pool, samples_per_epoch=40, seed=2026, config=config
        )

        expected_noise_share = {origin: 0.25 for origin in TRAIN_NOISE_ORIGINS}
        self.assertEqual(
            device_arm.pool_stats()["noise_origin_sampling"],
            expected_noise_share,
        )
        self.assertEqual(
            combined_arm.pool_stats()["noise_origin_sampling"],
            expected_noise_share,
        )
        self.assertEqual(device_arm.pool_stats()["sampler_version"], SAMPLER_VERSION)

        for epoch in (0, 1, 7):
            device_arm.set_epoch(epoch)
            combined_arm.set_epoch(epoch)
            device_recipes = [
                device_arm.sampling_recipe(index) for index in range(len(device_arm))
            ]
            combined_recipes = [
                combined_arm.sampling_recipe(index) for index in range(len(combined_arm))
            ]

            for device_recipe, combined_recipe in zip(
                device_recipes, combined_recipes
            ):
                self.assertEqual(
                    (
                        device_recipe["noise_origin"],
                        device_recipe["noise_index"],
                        device_recipe["snr_db"],
                    ),
                    (
                        combined_recipe["noise_origin"],
                        combined_recipe["noise_index"],
                        combined_recipe["snr_db"],
                    ),
                )

            origins = [recipe["noise_origin"] for recipe in device_recipes]
            for origin in TRAIN_NOISE_ORIGINS:
                self.assertEqual(origins.count(origin), 10)
            combined_sources = [
                recipe["clean_source"] for recipe in combined_recipes
            ]
            self.assertEqual(combined_sources.count("device"), 20)
            self.assertEqual(combined_sources.count("circor"), 20)
            joint = [
                (recipe["clean_source"], recipe["noise_origin"])
                for recipe in combined_recipes
            ]
            for source in ("device", "circor"):
                for origin in TRAIN_NOISE_ORIGINS:
                    self.assertEqual(joint.count((source, origin)), 5)

        with self.assertRaisesRegex(ValueError, "sampling cycle"):
            SyntheticFrequencyDataset(
                device_clean,
                noise_pool,
                samples_per_epoch=42,
                seed=2026,
                config=config,
            )

    @staticmethod
    def _valid_final_checkpoint() -> dict:
        model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=4))
        noise_sampling = {origin: 0.25 for origin in TRAIN_NOISE_ORIGINS}
        training_policy = {
            "selection_policy": "fixed_budget_final_epoch",
            "uses_validation": False,
            "optimizer": "AdamW",
            "learning_rate_schedule": "constant",
            "optimizer_update_policy": OPTIMIZER_UPDATE_POLICY,
            "precision_policy": "fp32_fail_fast_v1",
            "amp_max_retries_per_batch": AMP_MAX_RETRIES,
            "target_epochs": 2,
            "completed_epochs": 2,
            "steps_per_epoch": 5,
            "batch_size": 4,
            "inputs_per_epoch": 20,
            "optimizer_steps": 10,
            "trained_inputs": 40,
        }
        return {
            "epoch": 1,
            "checkpoint_kind": "final",
            "schema_version": 3,
            "input_mode": "single_mic",
            "role": "device_only",
            "arch": "cardiospecnet",
            "model_config": model.config_dict,
            "model_state_dict": model.state_dict(),
            "mixing_config": MixingConfig().to_dict(),
            "loss_config": FrequencyLossConfig().to_dict(),
            "pool_stats": {
                "arm": "device_only",
                "train": {
                    "clean": {
                        "n_windows": 14,
                        "source_counts": {"device": 14},
                    },
                    "noise": {"n_windows": 20},
                    "clean_source_sampling": {"device": 1.0},
                    "noise_origin_sampling": noise_sampling,
                    "clean_origin_counts": {
                        origin: 2 for origin in TRAIN_CLEAN_ORIGINS
                    },
                    "device_clean_origin_counts": {
                        origin: 2 for origin in TRAIN_CLEAN_ORIGINS
                    },
                    "noise_origin_counts": {
                        origin: 5 for origin in TRAIN_NOISE_ORIGINS
                    },
                    "sampling_cycle": 4,
                    "sampler_version": SAMPLER_VERSION,
                },
            },
            "data_protocol": device_protocol_metadata(),
            "training_policy": training_policy,
            "runtime_stats": {
                "successful_optimizer_updates": 10,
                "amp_overflow_retries": 0,
                "amp_forward_fallbacks": 0,
                "fp32_fallback_updates": 0,
                "final_loss_scale": None,
            },
            "training_args": {
                "epochs": 2,
                "samples_per_epoch": 20,
                "batch_size": 4,
                "learning_rate": 3e-4,
                "weight_decay": 1e-4,
                "grad_clip": 5.0,
                "seed": 2026,
                "snr_min_db": -10.0,
                "snr_max_db": 20.0,
                "pool_target_rms": 0.02,
                "source_stride": 1,
                "max_clean_windows": None,
                "max_noise_windows": None,
            },
            "parameter_count": parameter_count(model),
        }

    def test_schema3_completed_final_checkpoint_is_loadable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "final.pt"
            torch.save(self._valid_final_checkpoint(), path)
            _, metadata = load_arm(
                path, torch.device("cpu"), expected_role="device_only"
            )

        self.assertEqual(metadata["completed_epochs"], 2)
        self.assertEqual(metadata["optimizer_steps"], 10)
        self.assertEqual(metadata["trained_inputs"], 40)
        self.assertEqual(
            metadata["training_protocol"]["budget"]["selection_policy"],
            "fixed_budget_final_epoch",
        )

    def test_old_and_incomplete_checkpoints_are_rejected(self) -> None:
        cases = {}
        old_schema = deepcopy(self._valid_final_checkpoint())
        old_schema["schema_version"] = 2
        cases["old_schema"] = (old_schema, "schema_version must be 3")

        resume_only = deepcopy(self._valid_final_checkpoint())
        resume_only["checkpoint_kind"] = "resume"
        cases["resume_only"] = (resume_only, "completed fixed-budget final")

        incomplete = deepcopy(self._valid_final_checkpoint())
        incomplete["training_policy"]["completed_epochs"] = 1
        incomplete["training_policy"]["optimizer_steps"] = 5
        incomplete["training_policy"]["trained_inputs"] = 20
        cases["incomplete"] = (incomplete, "incomplete or inconsistent fixed budget")

        contaminated = deepcopy(self._valid_final_checkpoint())
        contaminated["pool_stats"]["train"]["device_clean_origin_counts"][
            "heart_aw6_windows"
        ] = 1
        cases["subject6_clean"] = (
            contaminated,
            "training device clean recordings must be",
        )

        wrong_batch_budget = deepcopy(self._valid_final_checkpoint())
        wrong_batch_budget["training_policy"]["batch_size"] = 2
        cases["wrong_batch_budget"] = (
            wrong_batch_budget,
            "incomplete or inconsistent fixed budget",
        )

        wrong_final_epoch = deepcopy(self._valid_final_checkpoint())
        wrong_final_epoch["epoch"] = 0
        cases["wrong_final_epoch"] = (
            wrong_final_epoch,
            "incomplete or inconsistent fixed budget",
        )

        missing_successful_update = deepcopy(self._valid_final_checkpoint())
        missing_successful_update["runtime_stats"][
            "successful_optimizer_updates"
        ] = 9
        cases["missing_successful_update"] = (
            missing_successful_update,
            "runtime stats disagree with fixed-budget metadata",
        )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, (checkpoint, message) in cases.items():
                with self.subTest(name=name):
                    path = root / f"{name}.pt"
                    torch.save(checkpoint, path)
                    with self.assertRaisesRegex(ValueError, message):
                        load_arm(
                            path,
                            torch.device("cpu"),
                            expected_role="device_only",
                        )

    def test_comparison_rejects_mismatched_training_protocols(self) -> None:
        protocol = {
            "training": {"samples_per_epoch": 100},
            "mixing": {"snr_min_db": -10.0, "snr_max_db": 20.0},
            "loss": {"n_fft": 512},
            "sampling": {
                "sampler_version": SAMPLER_VERSION,
                "noise_origin_sampling": {
                    origin: 0.25 for origin in TRAIN_NOISE_ORIGINS
                },
            },
            "budget": {
                "selection_policy": "fixed_budget_final_epoch",
                "completed_epochs": 2,
                "trained_inputs": 200,
            },
        }
        validate_shared_protocol({
            "device_only": {"training_protocol": protocol},
            "combined": {"training_protocol": protocol},
        })
        different = {
            **protocol,
            "training": {"samples_per_epoch": 200},
        }
        with self.assertRaisesRegex(ValueError, "do not share one training protocol"):
            validate_shared_protocol({
                "device_only": {"training_protocol": protocol},
                "combined": {"training_protocol": different},
            })

        different_budget = deepcopy(protocol)
        different_budget["budget"]["trained_inputs"] = 100
        with self.assertRaisesRegex(ValueError, "do not share one training protocol"):
            validate_shared_protocol({
                "device_only": {"training_protocol": protocol},
                "waveform": {"training_protocol": different_budget},
            })

    def test_same_architecture_requires_the_same_model_config(self) -> None:
        protocol = {"budget": {"trained_inputs": 100}}
        with self.assertRaisesRegex(ValueError, "do not share one model config"):
            validate_shared_protocol({
                "device_only": {
                    "arch": "cardiospecnet",
                    "model_config": {"base_channels": 8},
                    "training_protocol": protocol,
                },
                "combined": {
                    "arch": "cardiospecnet",
                    "model_config": {"base_channels": 12},
                    "training_protocol": protocol,
                },
            })
        validate_shared_protocol({
            "device_only": {
                "arch": "cardiospecnet",
                "model_config": {"base_channels": 12},
                "training_protocol": protocol,
            },
            "waveform": {
                "arch": "cleanunet",
                "model_config": {"channels_H": 12},
                "training_protocol": protocol,
            },
        })


if __name__ == "__main__":
    unittest.main()
