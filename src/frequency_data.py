"""On-the-fly synthetic mixtures for reference-conditioned frequency-domain training.

Mixtures are synthesised per sample, never written to disk, so the SNR draw is
not frozen into a file. One chest-facing microphone sees the heart sound plus
motion artifact; one exterior microphone sees the artifact through a different,
unknown transfer path. The two are not phase-coherent, which is why the model
consumes the reference through magnitude/energy features only.

The clean pool may be device recordings, CirCor, or both; see ``src/pools.py``
for how sources are made commensurable before they get here.

Deviations from the original single-source implementation, all because the pools
now come from two instruments with a ~30-100x RMS gap:

  * pools arrive pre-filtered and RMS-normalised (``src/pools.py``), so the
    scale-dependent constants below mean the same thing for every source;
  * the reference sensor-noise floor is a fraction of the reference RMS instead
    of an absolute 0.01, which was only meaningful for unit-scale audio;
  * ``chest_only_noise_max`` is exposed so it can be set to 0 for validation.
    It adds unscaled noise on top of the SNR-scaled noise from the *same* noise
    window, so any value above 0 makes the realised SNR lower than the drawn
    one. Useful jitter while training, misleading on a validation SNR sweep.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from src.frequency_model import STFTConfig, soft_bandpass_response
from src.pools import DEFAULT_TARGET_RMS, WindowPool, describe, load_device_pool


@dataclass
class MixingConfig:
    sample_rate: int = 4_000
    snr_min_db: float = -10.0
    snr_max_db: float = 20.0
    max_delay_ms: float = 50.0
    chest_only_noise_max: float = 0.35
    reference_leakage_max: float = 0.08
    reference_dropout_probability: float = 0.20
    identity_probability: float = 0.08
    impulse_length: int = 32
    transient_probability: float = 0.15
    transient_count_range: tuple[int, int] = (1, 3)
    #: Reference-microphone self-noise, as a fraction of that window's
    #: reference RMS. 0.01 is a -40 dB sensor floor.
    sensor_noise_fraction: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _to_float_windows(array: np.ndarray) -> np.ndarray:
    if np.issubdtype(array.dtype, np.integer):
        info = np.iinfo(array.dtype)
        scale = float(max(abs(info.min), info.max))
        return array.astype(np.float32) / scale
    return array.astype(np.float32)


def _fractional_delay(signal: np.ndarray, delay_samples: float, rng: np.random.Generator) -> np.ndarray:
    if abs(delay_samples) < 1e-6:
        return signal
    length = len(signal)
    source = np.arange(length, dtype=np.float64) - delay_samples
    output = np.interp(source, np.arange(length), signal, left=0.0, right=0.0).astype(np.float32)
    return output


def _random_fir(signal: np.ndarray, rng: np.random.Generator, length: int) -> np.ndarray:
    kernel = rng.normal(0.0, 1.0, size=length).astype(np.float32)
    kernel /= np.linalg.norm(kernel).clip(min=1e-6)
    return np.convolve(signal, kernel, mode="same").astype(np.float32)


def _random_eq(signal: np.ndarray, rng: np.random.Generator, sample_rate: int) -> np.ndarray:
    spectrum = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate)
    tilt = rng.uniform(-1.5, 1.5)
    eq = np.exp(tilt * np.log1p(freqs / 40.0))
    eq[0] = 1.0
    return np.fft.irfft(spectrum * eq, n=len(signal)).astype(np.float32)


def _apply_soft_bandpass(signal: np.ndarray, config: STFTConfig) -> np.ndarray:
    tensor = torch.from_numpy(signal).unsqueeze(0)
    freqs = torch.fft.rfftfreq(config.n_fft, d=1.0 / config.sample_rate)
    response = soft_bandpass_response(
        freqs,
        config.passband_low_hz,
        config.passband_high_hz,
        config.low_transition_hz,
        config.high_transition_hz,
    )
    window = torch.hann_window(config.win_length)
    spectrum = torch.stft(
        tensor,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    filtered = torch.istft(
        spectrum * response[:, None],
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        win_length=config.win_length,
        window=window,
        center=True,
        length=len(signal),
    )
    return filtered.squeeze(0).numpy().astype(np.float32)


def _band_limited_rms(signal: np.ndarray, config: STFTConfig) -> float:
    filtered = _apply_soft_bandpass(signal, config)
    return float(np.sqrt(np.mean(filtered * filtered).clip(min=1e-8)))


def _scale_to_snr(clean: np.ndarray, noise: np.ndarray, snr_db: float, config: STFTConfig) -> np.ndarray:
    clean_energy = _band_limited_rms(clean, config) ** 2
    noise_energy = _band_limited_rms(noise, config) ** 2
    target_noise_energy = clean_energy / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_energy / max(noise_energy, 1e-8))
    return noise * scale


def _add_transients(signal: np.ndarray, rng: np.random.Generator, count_range: tuple[int, int]) -> np.ndarray:
    output = signal.copy()
    count = rng.integers(count_range[0], count_range[1] + 1)
    for _ in range(count):
        position = rng.integers(0, len(output))
        width = rng.integers(4, 24)
        amplitude = rng.uniform(0.05, 0.25) * np.max(np.abs(output)).clip(min=1e-4)
        taper = np.hanning(width).astype(np.float32)
        stop = min(len(output), position + width)
        width = stop - position
        output[position:stop] += amplitude * taper[:width]
    return output.astype(np.float32)


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


class SyntheticFrequencyDataset(Dataset):
    """Deterministic per-index synthetic mixtures.

    ``clean`` and ``noise`` accept a ``WindowPool`` (the combined device+CirCor
    case, built by the caller) or a directory of ``*.npz`` window archives.
    """

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
        self.samples_per_epoch = samples_per_epoch
        self.config = config or MixingConfig()
        self.stft_config = STFTConfig(sample_rate=self.config.sample_rate)
        self.seed = seed
        self.epoch = 0
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

    # Kept so existing callers that reach for the raw arrays keep working.
    @property
    def clean_pool(self) -> np.ndarray:
        return self.clean.x

    @property
    def noise_pool(self) -> np.ndarray:
        return self.noise.x

    def describe(self) -> str:
        return f"{describe(self.clean)}\n{describe(self.noise)}"

    def pool_stats(self) -> dict[str, Any]:
        return {"clean": self.clean.stats, "noise": self.noise.stats}

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng(self.seed + self.epoch * 1_000_003 + index * 9_173)

    def _transfer_path(self, signal: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        delayed = _fractional_delay(
            signal,
            rng.uniform(-self.config.max_delay_ms, self.config.max_delay_ms) * self.config.sample_rate / 1_000.0,
            rng,
        )
        filtered = _random_fir(delayed, rng, self.config.impulse_length)
        shaped = _random_eq(filtered, rng, self.config.sample_rate)
        polarity = -1.0 if rng.random() < 0.5 else 1.0
        gain = rng.uniform(0.5, 2.0)
        return (shaped * polarity * gain).astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        rng = self._rng(index)
        clean_pool, noise_pool = self.clean.x, self.noise.x
        clean_idx = int(rng.integers(0, len(clean_pool)))
        noise_idx_a = int(rng.integers(0, len(noise_pool)))
        noise_idx_b = int(rng.integers(0, len(noise_pool)))
        clean_raw = clean_pool[clean_idx].copy()
        noise_a = noise_pool[noise_idx_a].copy()
        noise_b = noise_pool[noise_idx_b].copy()

        if rng.random() < self.config.identity_probability:
            chest = clean_raw.copy()
            reference = self._transfer_path(noise_a, rng)
            ref_available = 1.0
        else:
            reference_noise = self._transfer_path(noise_a, rng)
            chest_noise = self._transfer_path(noise_b, rng)
            if rng.random() < self.config.transient_probability:
                chest_noise = _add_transients(chest_noise, rng, self.config.transient_count_range)
            chest_only = chest_noise * rng.uniform(0.0, self.config.chest_only_noise_max)
            snr_db = rng.uniform(self.config.snr_min_db, self.config.snr_max_db)
            scaled_noise = _scale_to_snr(clean_raw, chest_noise, snr_db, self.stft_config)
            chest = clean_raw + chest_only + scaled_noise

            leakage = rng.uniform(0.0, self.config.reference_leakage_max)
            reference = reference_noise + leakage * clean_raw
            # Sensor floor relative to this window's reference level, so the
            # figure means the same thing whatever the pool's absolute scale is.
            reference_rms = float(np.sqrt(np.mean(reference.astype(np.float64) ** 2)))
            sigma = self.config.sensor_noise_fraction * max(reference_rms, 1e-8)
            reference = reference + rng.normal(0.0, sigma, size=reference.shape).astype(np.float32)
            ref_available = 0.0 if rng.random() < self.config.reference_dropout_probability else 1.0

        clean = _apply_soft_bandpass(clean_raw, self.stft_config)
        if ref_available == 0.0:
            reference = np.zeros_like(reference)

        return {
            "chest": torch.from_numpy(chest.astype(np.float32)),
            "clean": torch.from_numpy(clean.astype(np.float32)),
            "clean_raw": torch.from_numpy(clean_raw.astype(np.float32)),
            "reference": torch.from_numpy(reference.astype(np.float32)),
            "ref_available": torch.tensor(ref_available, dtype=torch.float32),
        }
