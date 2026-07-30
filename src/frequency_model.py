"""Minimal single-microphone complex-STFT U-Net."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, NamedTuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@dataclass(frozen=True)
class STFTConfig:
    sample_rate: int = 4_000
    n_fft: int = 512
    win_length: int = 256
    hop_length: int = 64
    center: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrequencyModelConfig:
    base_channels: int = 12
    depth: int = 3
    input_channels: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrequencyDenoiseOutput(NamedTuple):
    waveform: Tensor
    enhanced_stft: Tensor
    mixture_stft: Tensor
    residual_stft: Tensor
    scale: Tensor


class DoubleConv(nn.Module):
    """Two ordinary 3x3 convolutions used throughout the small U-Net."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.convolutions = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.convolutions(torch.cat([x, skip], dim=1))


class CardioSpecNet(nn.Module):
    """Small 2-D U-Net operating on noisy-STFT real and imaginary planes.

    The only input is one noisy chest waveform. The output has the same shape
    and physical scale.
    """

    def __init__(
        self,
        stft_config: STFTConfig | None = None,
        model_config: FrequencyModelConfig | None = None,
    ) -> None:
        super().__init__()
        self.stft_config = stft_config or STFTConfig()
        self.model_config = model_config or FrequencyModelConfig()

        if self.model_config.depth < 1:
            raise ValueError("depth must be at least 1")
        if self.model_config.base_channels < 1:
            raise ValueError("base_channels must be at least 1")
        if self.model_config.input_channels != 2:
            raise ValueError("the minimal STFT U-Net requires exactly [real, imag] input planes")

        window = torch.hann_window(self.stft_config.win_length)
        self.register_buffer("window", window, persistent=False)

        channels = [self.model_config.base_channels * (2**idx) for idx in range(self.model_config.depth + 1)]
        self.encoders = nn.ModuleList(
            DoubleConv(
                self.model_config.input_channels if idx == 0 else channels[idx - 1],
                channels[idx],
            )
            for idx in range(self.model_config.depth)
        )
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(channels[-2], channels[-1])
        self.decoders = nn.ModuleList(
            DecoderStage(channels[idx + 1], channels[idx], channels[idx])
            for idx in reversed(range(self.model_config.depth))
        )
        self.head = nn.Conv2d(channels[0], 2, kernel_size=1)

        # A zero complex residual makes the untrained model an identity mapping.
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    @property
    def config_dict(self) -> dict[str, Any]:
        return {
            "stft_config": self.stft_config.to_dict(),
            "model_config": self.model_config.to_dict(),
        }

    def _stft(self, waveform: Tensor) -> Tensor:
        return torch.stft(
            waveform,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=self.window.to(device=waveform.device, dtype=waveform.dtype),
            center=self.stft_config.center,
            return_complex=True,
            pad_mode="reflect",
        )

    def _istft(self, spectrum: Tensor, length: int) -> Tensor:
        return torch.istft(
            spectrum,
            n_fft=self.stft_config.n_fft,
            hop_length=self.stft_config.hop_length,
            win_length=self.stft_config.win_length,
            window=self.window.to(device=spectrum.device, dtype=spectrum.real.dtype),
            center=self.stft_config.center,
            length=length,
        )

    def forward(
        self,
        noisy: Tensor,
        *,
        return_details: bool = False,
    ) -> Tensor | FrequencyDenoiseOutput:
        if noisy.ndim == 3 and noisy.shape[1] == 1:
            noisy = noisy[:, 0]
        if noisy.ndim != 2:
            raise ValueError(f"noisy must have shape [B,T] or [B,1,T], got {tuple(noisy.shape)}")

        _, length = noisy.shape
        scale = noisy.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        mixture_stft = self._stft(noisy / scale)
        x = torch.view_as_real(mixture_stft).permute(0, 3, 1, 2).contiguous()

        skips: list[Tensor] = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        residual_planes = self.head(x)

        # Complex assembly and ISTFT stay in fp32 for CUDA autocast compatibility.
        with torch.autocast(device_type=noisy.device.type, enabled=False):
            residual_stft = torch.complex(
                residual_planes[:, 0].float(),
                residual_planes[:, 1].float(),
            )
            enhanced_stft = mixture_stft.to(residual_stft.dtype) + residual_stft
            waveform = self._istft(enhanced_stft, length=length) * scale.float()

        if return_details:
            return FrequencyDenoiseOutput(
                waveform=waveform,
                enhanced_stft=enhanced_stft,
                mixture_stft=mixture_stft,
                residual_stft=residual_stft,
                scale=scale,
            )
        return waveform


def build_frequency_model(
    checkpoint_config: dict[str, Any] | None = None,
    *,
    base_channels: int | None = None,
) -> CardioSpecNet:
    """Build a model either from a checkpoint config or explicit overrides."""
    if checkpoint_config:
        stft_cfg = STFTConfig(**checkpoint_config.get("stft_config", {}))
        model_values = dict(checkpoint_config.get("model_config", {}))
    else:
        stft_cfg = STFTConfig()
        model_values = {}
    if base_channels is not None:
        model_values["base_channels"] = base_channels
    return CardioSpecNet(stft_cfg, FrequencyModelConfig(**model_values))
