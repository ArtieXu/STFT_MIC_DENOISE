"""Normalized complex-STFT L1 loss."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class FrequencyLossConfig:
    sample_rate: int = 4_000
    n_fft: int = 512
    hop_length: int = 64
    win_length: int = 256
    center: bool = True
    eps: float = 1e-6

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrequencyDenoiseLoss(nn.Module):
    def __init__(self, config: FrequencyLossConfig | None = None) -> None:
        super().__init__()
        self.config = config or FrequencyLossConfig()
        self.register_buffer(
            "window",
            torch.hann_window(self.config.win_length),
            persistent=False,
        )

    def _stft(self, waveform: Tensor) -> Tensor:
        return torch.stft(
            waveform,
            n_fft=self.config.n_fft,
            hop_length=self.config.hop_length,
            win_length=self.config.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=self.config.center,
            return_complex=True,
            pad_mode="reflect",
        )

    def forward(self, estimate: Tensor, target: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        if estimate.ndim == 3 and estimate.shape[1] == 1:
            estimate = estimate[:, 0]
        if target.ndim == 3 and target.shape[1] == 1:
            target = target[:, 0]

        if estimate.ndim != 2 or target.ndim != 2:
            raise ValueError("estimate and target must have shape [B,T] or [B,1,T]")
        if estimate.shape != target.shape:
            raise ValueError(f"estimate shape {tuple(estimate.shape)} != target shape {tuple(target.shape)}")

        estimate_planes = torch.view_as_real(self._stft(estimate))
        target_planes = torch.view_as_real(self._stft(target))
        error = (estimate_planes - target_planes).abs().mean(dim=(1, 2, 3))
        target_scale = target_planes.abs().mean(dim=(1, 2, 3)).clamp_min(self.config.eps)
        normalized_complex_l1 = error / target_scale
        loss = normalized_complex_l1.mean()
        terms = {
            "loss": loss.detach(),
            "complex_stft": loss.detach(),
        }
        return loss, terms
