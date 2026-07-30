"""Paired evaluation metrics for frequency-domain PCG denoising."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor


def _flatten_batch(x: Tensor) -> Tensor:
    if x.ndim == 3 and x.shape[1] == 1:
        x = x[:, 0]
    return x


def snr_db(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-dependent SNR in dB, averaged over the batch."""
    estimate = _flatten_batch(estimate)
    target = _flatten_batch(target)
    noise = estimate - target
    signal_power = target.square().mean(dim=-1).clamp_min(eps)
    noise_power = noise.square().mean(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(signal_power / noise_power)


def si_sdr_db(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    """Scale-invariant SDR in dB, averaged over the batch."""
    estimate = _flatten_batch(estimate)
    target = _flatten_batch(target)
    dot = (estimate * target).sum(dim=-1, keepdim=True)
    target_energy = target.square().sum(dim=-1, keepdim=True).clamp_min(eps)
    projected = dot / target_energy * target
    noise = estimate - projected
    projected_energy = projected.square().sum(dim=-1).clamp_min(eps)
    noise_energy = noise.square().sum(dim=-1).clamp_min(eps)
    return 10.0 * torch.log10(projected_energy / noise_energy)


def pearson_correlation(estimate: Tensor, target: Tensor, eps: float = 1e-8) -> Tensor:
    estimate = _flatten_batch(estimate)
    target = _flatten_batch(target)
    estimate = estimate - estimate.mean(dim=-1, keepdim=True)
    target = target - target.mean(dim=-1, keepdim=True)
    numerator = (estimate * target).sum(dim=-1)
    denominator = (
        estimate.square().sum(dim=-1).sqrt() * target.square().sum(dim=-1).sqrt()
    ).clamp_min(eps)
    return (numerator / denominator).clamp(-1.0, 1.0)


def log_spectral_distance(
    estimate: Tensor,
    target: Tensor,
    sample_rate: int = 4_000,
    low_hz: float = 15.0,
    high_hz: float = 800.0,
    n_fft: int = 512,
    hop_length: int = 64,
    win_length: int = 256,
) -> Tensor:
    """Mean log-magnitude spectral L2 distance in dB over the monitoring band."""
    estimate = _flatten_batch(estimate)
    target = _flatten_batch(target)
    window = torch.hann_window(win_length, device=estimate.device, dtype=estimate.dtype)
    est_spec = torch.stft(
        estimate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    tgt_spec = torch.stft(
        target,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=window,
        center=True,
        return_complex=True,
    )
    freqs = torch.fft.rfftfreq(n_fft, d=1.0 / sample_rate).to(estimate.device)
    band = (freqs >= low_hz) & (freqs <= high_hz)
    est_log = torch.log1p(est_spec[:, band].abs())
    tgt_log = torch.log1p(tgt_spec[:, band].abs())
    distance = (est_log - tgt_log).square().mean(dim=(1, 2)).sqrt()
    return 20.0 * torch.log10(distance.clamp_min(1e-8))


@dataclass
class MetricAccumulator:
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def update(self, **values: Tensor | float) -> None:
        for key, value in values.items():
            if isinstance(value, Tensor):
                if value.numel() == 1:
                    scalar = float(value.detach().cpu())
                else:
                    scalar = float(value.detach().mean().cpu())
            else:
                scalar = float(value)
            self.totals[key] = self.totals.get(key, 0.0) + scalar
            self.counts[key] = self.counts.get(key, 0) + 1

    def summary(self) -> dict[str, float]:
        return {
            f"{key}_mean": self.totals[key] / self.counts[key]
            for key in self.totals
        }
