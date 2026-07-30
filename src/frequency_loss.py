"""Phase-aware, scale-aware loss for CardioSpecNet training."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn

from src.frequency_metrics import pearson_correlation, si_sdr_db, snr_db
from src.frequency_model import STFTConfig, soft_bandpass_response


@dataclass(frozen=True)
class FrequencyLossConfig:
    waveform_l1_weight: float = 1.0
    complex_stft_weight: float = 1.0
    log_mag_weight: float = 0.5
    diff_l1_weight: float = 0.2
    snr_weight: float = 0.25
    correlation_weight: float = 0.25
    band_energy_weight: float = 0.5
    over_attenuation_weight: float = 0.75
    sample_rate: int = 4_000
    n_fft: int = 512
    hop_length: int = 64
    win_length: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


CLINICAL_BANDS = (
    (15.0, 40.0),
    (40.0, 100.0),
    (100.0, 200.0),
    (200.0, 400.0),
    (400.0, 800.0),
)


class FrequencyDenoiseLoss(nn.Module):
    def __init__(self, config: FrequencyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or FrequencyLossConfig()
        self.stft_config = STFTConfig(
            sample_rate=self.config.sample_rate,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
        )
        freqs = torch.fft.rfftfreq(self.config.n_fft, d=1.0 / self.config.sample_rate)
        weights = torch.ones_like(freqs)
        weights = torch.where(freqs < 15.0, torch.zeros_like(weights), weights)
        weights = torch.where(
            (freqs >= 15.0) & (freqs < 200.0),
            torch.full_like(weights, 2.0),
            weights,
        )
        weights = torch.where(
            (freqs >= 200.0) & (freqs <= 800.0),
            torch.full_like(weights, 1.25),
            weights,
        )
        self.register_buffer("frequency_weights", weights, persistent=False)

    def _stft(self, waveform: Tensor) -> Tensor:
        window = torch.hann_window(self.config.win_length, device=waveform.device, dtype=waveform.dtype)
        return torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=window,
            center=True,
            return_complex=True,
        )

    def _target_scale(self, target: Tensor, eps: float = 1e-5) -> Tensor:
        return target.abs().mean(dim=-1, keepdim=True).clamp_min(eps)

    def _band_energy(self, waveform: Tensor, low_hz: float, high_hz: float) -> Tensor:
        spectrum = self._stft(waveform).abs().square()
        freqs = torch.fft.rfftfreq(self.config.n_fft, d=1.0 / self.config.sample_rate).to(waveform.device)
        band = (freqs >= low_hz) & (freqs < high_hz)
        return spectrum[:, band].mean(dim=(1, 2)).clamp_min(1e-8)

    def forward(self, estimate: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if estimate.ndim == 3 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
        if target.ndim == 3 and target.shape[1] == 1:
            target = target[:, 0]

        scale = self._target_scale(target)
        estimate_norm = estimate / scale
        target_norm = target / scale

        waveform_l1 = (estimate_norm - target_norm).abs().mean()
        estimate_spec = self._stft(estimate_norm)
        target_spec = self._stft(target_norm)
        freq_weights = self.frequency_weights.to(estimate.device)[:, None]
        complex_l1 = ((estimate_spec - target_spec).abs() * freq_weights).mean()
        log_mag = (torch.log1p(estimate_spec.abs()) - torch.log1p(target_spec.abs())).abs().mean()
        diff_l1 = (
            (estimate_norm[..., 1:] - estimate_norm[..., :-1]) - (target_norm[..., 1:] - target_norm[..., :-1])
        ).abs().mean()

        snr_loss = -snr_db(estimate, target).mean()
        correlation_loss = 1.0 - pearson_correlation(estimate, target).mean()

        band_energy_terms: list[Tensor] = []
        for low_hz, high_hz in CLINICAL_BANDS:
            estimate_energy = self._band_energy(estimate, low_hz, high_hz)
            target_energy = self._band_energy(target, low_hz, high_hz)
            band_energy_terms.append(torch.abs(torch.log(estimate_energy / target_energy)).mean())
        band_energy = torch.stack(band_energy_terms).mean()

        freqs = torch.fft.rfftfreq(self.config.n_fft, d=1.0 / self.config.sample_rate).to(estimate.device)
        monitor_band = (freqs >= 15.0) & (freqs <= 800.0)
        estimate_monitor = estimate_spec[:, monitor_band].abs().square().mean(dim=(1, 2))
        target_monitor = target_spec[:, monitor_band].abs().square().mean(dim=(1, 2))
        over_attenuation = torch.relu(
            torch.log(target_monitor.clamp_min(1e-8)) - torch.log(estimate_monitor.clamp_min(1e-8))
        ).mean()

        cfg = self.config
        loss = (
            cfg.waveform_l1_weight * waveform_l1
            + cfg.complex_stft_weight * complex_l1
            + cfg.log_mag_weight * log_mag
            + cfg.diff_l1_weight * diff_l1
            + cfg.snr_weight * snr_loss
            + cfg.correlation_weight * correlation_loss
            + cfg.band_energy_weight * band_energy
            + cfg.over_attenuation_weight * over_attenuation
        )
        terms = {
            "loss": loss.detach(),
            "waveform_l1": waveform_l1.detach(),
            "complex_stft": complex_l1.detach(),
            "log_mag": log_mag.detach(),
            "diff_l1": diff_l1.detach(),
            "snr_loss": snr_loss.detach(),
            "correlation_loss": correlation_loss.detach(),
            "band_energy": band_energy.detach(),
            "over_attenuation": over_attenuation.detach(),
            "si_sdr_db": si_sdr_db(estimate, target).mean().detach(),
        }
        return loss, terms
