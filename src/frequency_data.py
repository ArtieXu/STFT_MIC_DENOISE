"""Minimal on-the-fly mono mixtures for denoising experiments.

Each item contains exactly three waveforms:

``clean``
    One clean target window.
``noise``
    One noise window, scaled to the sampled SNR.
``chest``
    The exact sum ``clean + noise``.

There is deliberately no reference microphone, transfer function, leakage,
dropout, transient injection, identity case, chest-only noise, or hidden
bandpass.  Keeping the generator this small makes the nominal SNR auditable and
ensures the STFT and waveform arms receive the same information.

For a combined device + CirCor clean pool, source selection is deterministic
and exactly balanced within every epoch. With ``samples_per_epoch`` divisible
by the complete 2-clean-source x 4-noise-recording cycle, half of the targets
come from device and half from CirCor, every source/noise combination appears
equally often, and the total number of examples remains identical to a
device-only run.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.pools import DEFAULT_TARGET_RMS, WindowPool, describe, load_device_pool


@dataclass(frozen=True)
class MixingConfig:
    """The complete mono mixing configuration."""

    snr_min_db: float = -10.0
    snr_max_db: float = 20.0

    def __post_init__(self) -> None:
        if self.snr_min_db > self.snr_max_db:
            raise ValueError(
                f"snr_min_db ({self.snr_min_db}) must be <= snr_max_db ({self.snr_max_db})"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


SAMPLER_VERSION = "independent_streams_cartesian_balance_v1"


def _to_float_windows(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return array.astype(np.float32) / scale
    return array.astype(np.float32)


def _rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal.astype(np.float64) ** 2)))


def _scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Scale ``noise`` so full-band time-domain RMS gives the requested SNR.

    ``scale = rms(clean) / (rms(noise) * 10**(snr_db / 20))``
    """

    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    if clean_rms <= 0.0 or noise_rms <= 0.0:
        raise ValueError("clean and noise windows must both have non-zero RMS")
    scale = clean_rms / (noise_rms * 10.0 ** (snr_db / 20.0))
    return (noise * np.float32(scale)).astype(np.float32)


def _resolve_pool(
    source: str | Path | WindowPool | np.ndarray,
    *,
    source_stride: int,
    max_windows: int | None,
    target_rms: float | None,
    label: str,
) -> WindowPool:
    """Accept a ready pool, a raw array, or a directory to load from."""

    if isinstance(source, WindowPool):
        return source
    if isinstance(source, np.ndarray):
        x = _to_float_windows(source)
        return WindowPool(
            x=x,
            source=np.full(len(x), "array"),
            subject=np.full(len(x), label),
            origin=np.full(len(x), label),
            stats={"label": label, "n_windows": int(len(x)), "target_rms": target_rms},
        )
    return load_device_pool(
        source,
        source_stride=source_stride,
        max_windows=max_windows,
        target_rms=target_rms,
        label=label,
    )


def _indices_by_source_and_subject(pool: WindowPool) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, np.ndarray]] = {}
    for source in sorted(set(pool.source.tolist())):
        source_mask = pool.source == source
        grouped[source] = {}
        for subject in sorted(set(pool.subject[source_mask].tolist())):
            grouped[source][subject] = np.flatnonzero(
                source_mask & (pool.subject == subject)
            )
    return grouped


def _indices_by_origin(pool: WindowPool) -> dict[str, np.ndarray]:
    """Group noise by physical recording rather than by participant label."""

    return {
        origin: np.flatnonzero(pool.origin == origin)
        for origin in sorted(set(pool.origin.tolist()))
    }


class SyntheticFrequencyDataset(Dataset):
    """Deterministic mono mixtures with exact source/recording balancing."""

    def __init__(
        self,
        clean: str | Path | WindowPool | np.ndarray,
        noise: str | Path | WindowPool | np.ndarray,
        *,
        samples_per_epoch: int,
        source_stride: int = 1,
        max_clean_windows: int | None = None,
        max_noise_windows: int | None = None,
        target_rms: float | None = DEFAULT_TARGET_RMS,
        seed: int = 0,
        config: MixingConfig | None = None,
    ) -> None:
        if samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")
        self.samples_per_epoch = samples_per_epoch
        self.config = config or MixingConfig()
        self.seed = seed
        # A shared CPU tensor keeps persistent DataLoader workers in sync when
        # the main process advances the epoch.
        self._epoch = torch.zeros((), dtype=torch.int64)
        self._epoch_is_shared = True
        try:
            self._epoch.share_memory_()
        except RuntimeError:
            # Direct indexing and a zero-worker DataLoader remain correct. The
            # training entry point checks this flag and disables workers rather
            # than silently repeating epoch zero in persistent worker copies.
            self._epoch_is_shared = False
        self.clean = _resolve_pool(
            clean,
            source_stride=source_stride,
            max_windows=max_clean_windows,
            target_rms=target_rms,
            label="clean",
        )
        self.noise = _resolve_pool(
            noise,
            source_stride=source_stride,
            max_windows=max_noise_windows,
            target_rms=target_rms,
            label="noise",
        )
        self._clean_groups = _indices_by_source_and_subject(self.clean)
        self._noise_groups = _indices_by_origin(self.noise)
        self._clean_sources = tuple(sorted(self._clean_groups))
        self._noise_origins = tuple(sorted(self._noise_groups))
        # One cycle is the complete clean-source x noise-recording Cartesian
        # product. This prevents a particular noise file from being locked to
        # one clean source in the combined arm.
        self.sampling_cycle = len(self._clean_sources) * len(self._noise_origins)
        if samples_per_epoch % self.sampling_cycle:
            raise ValueError(
                "samples_per_epoch must be divisible by the clean/noise sampling cycle "
                f"({self.sampling_cycle}) for exact balance"
            )

    def describe(self) -> str:
        return f"{describe(self.clean)}\n{describe(self.noise)}"

    def pool_stats(self) -> dict[str, Any]:
        clean_share = 1.0 / len(self._clean_sources)
        noise_share = 1.0 / len(self._noise_origins)
        clean_origin_counts = {
            origin: int((self.clean.origin == origin).sum())
            for origin in sorted(set(self.clean.origin.tolist()))
        }
        device_clean_origin_counts = {
            origin: int(
                ((self.clean.origin == origin) & (self.clean.source == "device")).sum()
            )
            for origin in sorted(
                set(self.clean.origin[self.clean.source == "device"].tolist())
            )
        }
        noise_origin_counts = {
            origin: int(len(indices))
            for origin, indices in self._noise_groups.items()
        }
        return {
            "clean": self.clean.stats,
            "noise": self.noise.stats,
            "samples_per_epoch": self.samples_per_epoch,
            "clean_source_sampling": {
                source: clean_share for source in self._clean_sources
            },
            "noise_origin_sampling": {
                origin: noise_share for origin in self._noise_origins
            },
            "clean_origin_counts": clean_origin_counts,
            "device_clean_origin_counts": device_clean_origin_counts,
            "noise_origin_counts": noise_origin_counts,
            "subject_sampling": "uniform within each selected source",
            "sampling_cycle": self.sampling_cycle,
            "joint_sampling": "exact clean-source x noise-origin Cartesian balance",
            "sampler_version": SAMPLER_VERSION,
            "random_streams": "independent clean, noise-window, and SNR streams",
        }

    @property
    def epoch(self) -> int:
        return int(self._epoch.item())

    @property
    def epoch_is_shared(self) -> bool:
        """Whether persistent DataLoader workers can observe ``set_epoch``."""

        return self._epoch_is_shared

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self._epoch.fill_(int(epoch))

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int, stream: int) -> np.random.Generator:
        """One deterministic stream that cannot be perturbed by another draw."""

        return np.random.default_rng(
            np.random.SeedSequence([self.seed, self.epoch, index, stream])
        )

    def clean_source_for_index(self, index: int) -> str:
        """Return the scheduled clean source without drawing a sample."""

        if not 0 <= index < len(self):
            raise IndexError(index)
        # Hold a clean source for one full pass through all noise origins, then
        # rotate the source order between epochs. Every complete cycle contains
        # each clean-source/noise-origin pair exactly once.
        source_index = (
            index // len(self._noise_origins) + self.epoch
        ) % len(self._clean_sources)
        return self._clean_sources[source_index]

    def noise_origin_for_index(self, index: int) -> str:
        """Return the round-robin noise recording for this item."""

        if not 0 <= index < len(self):
            raise IndexError(index)
        return self._noise_origins[(index + self.epoch) % len(self._noise_origins)]

    @staticmethod
    def _draw_subject_balanced(
        groups: dict[str, np.ndarray], rng: np.random.Generator
    ) -> int:
        subjects = tuple(sorted(groups))
        subject = subjects[int(rng.integers(0, len(subjects)))]
        indices = groups[subject]
        return int(indices[int(rng.integers(0, len(indices)))])

    def sampling_recipe(self, index: int) -> dict[str, int | float | str]:
        """Expose the deterministic source indices used for fairness auditing."""

        if not 0 <= index < len(self):
            raise IndexError(index)

        clean_source = self.clean_source_for_index(index)
        clean_idx = self._draw_subject_balanced(
            self._clean_groups[clean_source], self._rng(index, stream=0)
        )
        noise_origin = self.noise_origin_for_index(index)
        noise_indices = self._noise_groups[noise_origin]
        noise_rng = self._rng(index, stream=1)
        noise_idx = int(
            noise_indices[int(noise_rng.integers(0, len(noise_indices)))]
        )
        snr_rng = self._rng(index, stream=2)
        snr_db = float(
            snr_rng.uniform(self.config.snr_min_db, self.config.snr_max_db)
        )
        return {
            "clean_source": clean_source,
            "clean_index": clean_idx,
            "noise_origin": noise_origin,
            "noise_index": noise_idx,
            "snr_db": snr_db,
        }

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        recipe = self.sampling_recipe(index)
        clean = self.clean.x[int(recipe["clean_index"])].copy().astype(np.float32)
        raw_noise = self.noise.x[int(recipe["noise_index"])].copy().astype(np.float32)
        noise = _scale_to_snr(clean, raw_noise, float(recipe["snr_db"]))
        chest = (clean + noise).astype(np.float32)

        return {
            "clean": torch.from_numpy(clean),
            "noise": torch.from_numpy(noise),
            "chest": torch.from_numpy(chest),
        }
