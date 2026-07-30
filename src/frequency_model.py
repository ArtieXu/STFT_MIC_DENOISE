"""Reference-conditioned complex-STFT denoiser for 4 kHz phonocardiograms.

The model intentionally uses the exterior microphone only through magnitude/energy
features.  The two microphones in this dataset are not phase-coherent enough for
waveform subtraction, but the reference magnitude is still informative about
motion/environmental contamination.
"""
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
    model_max_hz: float = 1_000.0
    passband_low_hz: float = 15.0
    passband_high_hz: float = 800.0
    low_transition_hz: float = 10.0
    high_transition_hz: float = 200.0
    compression: float = 0.3
    center: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrequencyModelConfig:
    base_channels: int = 12
    depth: int = 3
    grid_blocks: int = 2
    mask_limit: float = 1.2
    input_channels: int = 8

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class FrequencyDenoiseOutput(NamedTuple):
    waveform: Tensor
    enhanced_stft: Tensor
    mixture_stft: Tensor
    mask: Tensor
    scale: Tensor


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ConvNormAct(nn.Module):
    """Lightweight depthwise-separable 2-D convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(_group_count(out_channels), out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x))))


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(channels, channels),
            ConvNormAct(channels, channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x + self.block(x)


class EncoderStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels, out_channels)
        self.residual = ResidualBlock(out_channels)
        self.down = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.PReLU(out_channels),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        skip = self.residual(self.project(x))
        return self.down(skip), skip


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.project = ConvNormAct(in_channels + skip_channels, out_channels)
        self.residual = ResidualBlock(out_channels)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.residual(self.project(torch.cat([x, skip], dim=1)))


class TFGridBlock(nn.Module):
    """Small dual-axis recurrent block inspired by TF-GridNet.

    It alternates recurrence across time and frequency at the U-Net bottleneck.
    Bidirectionality is appropriate for the current 2-second offline windows.  A
    causal deployment can replace these GRUs without changing the STFT interface.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(4, channels // 2)
        self.time_norm = nn.LayerNorm(channels)
        self.time_gru = nn.GRU(channels, hidden, batch_first=True, bidirectional=True)
        self.time_proj = nn.Linear(2 * hidden, channels)

        self.freq_norm = nn.LayerNorm(channels)
        self.freq_gru = nn.GRU(channels, hidden, batch_first=True, bidirectional=True)
        self.freq_proj = nn.Linear(2 * hidden, channels)

        self.ffn = nn.Sequential(
            nn.GroupNorm(_group_count(channels), channels),
            nn.Conv2d(channels, 2 * channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * channels, channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, channels, n_freq, n_time = x.shape

        time_seq = x.permute(0, 2, 3, 1).reshape(batch * n_freq, n_time, channels)
        time_out, _ = self.time_gru(self.time_norm(time_seq))
        time_out = self.time_proj(time_out)
        time_out = time_out.reshape(batch, n_freq, n_time, channels).permute(0, 3, 1, 2)
        x = x + time_out

        freq_seq = x.permute(0, 3, 2, 1).reshape(batch * n_time, n_freq, channels)
        freq_out, _ = self.freq_gru(self.freq_norm(freq_seq))
        freq_out = self.freq_proj(freq_out)
        freq_out = freq_out.reshape(batch, n_time, n_freq, channels).permute(0, 3, 2, 1)
        x = x + freq_out
        return x + self.ffn(x)


def soft_bandpass_response(
    freqs: Tensor,
    low_hz: float,
    high_hz: float,
    low_transition_hz: float,
    high_transition_hz: float,
) -> Tensor:
    """Raised-cosine bandpass with smooth, differentiable shoulders."""
    low_start = max(0.0, low_hz - low_transition_hz)
    low_end = low_hz
    high_start = high_hz
    high_end = high_hz + high_transition_hz

    response = torch.ones_like(freqs)
    if low_end > low_start:
        low_x = ((freqs - low_start) / (low_end - low_start)).clamp(0.0, 1.0)
        low_ramp = 0.5 - 0.5 * torch.cos(torch.pi * low_x)
        response = response * torch.where(freqs < low_end, low_ramp, torch.ones_like(freqs))
    response = torch.where(freqs < low_start, torch.zeros_like(response), response)

    if high_end > high_start:
        high_x = ((freqs - high_start) / (high_end - high_start)).clamp(0.0, 1.0)
        high_ramp = 0.5 + 0.5 * torch.cos(torch.pi * high_x)
        response = response * torch.where(freqs > high_start, high_ramp, torch.ones_like(freqs))
    response = torch.where(freqs > high_end, torch.zeros_like(response), response)
    return response


class CardioSpecNet(nn.Module):
    """Complex-ratio-mask U-Net conditioned on an exterior microphone.

    Inputs are raw chest and reference waveforms with shape ``[B, T]``.  The
    output has the same shape and physical scale as the chest waveform.
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
        if self.model_config.input_channels != 8:
            raise ValueError("CardioSpecNet currently defines exactly eight input features")
        if self.model_config.mask_limit <= 1.0:
            raise ValueError("mask_limit must be greater than 1.0")

        window = torch.hann_window(self.stft_config.win_length)
        self.register_buffer("window", window, persistent=False)
        freqs = torch.fft.rfftfreq(
            self.stft_config.n_fft,
            d=1.0 / self.stft_config.sample_rate,
        )
        self.register_buffer("freqs", freqs, persistent=False)
        model_bins = int((freqs <= self.stft_config.model_max_hz).sum().item())
        self.model_bins = model_bins
        response = soft_bandpass_response(
            freqs,
            self.stft_config.passband_low_hz,
            self.stft_config.passband_high_hz,
            self.stft_config.low_transition_hz,
            self.stft_config.high_transition_hz,
        )
        self.register_buffer("bandpass", response[:, None], persistent=False)

        channels = [self.model_config.base_channels * (2**idx) for idx in range(self.model_config.depth + 1)]
        self.stem = ConvNormAct(self.model_config.input_channels, channels[0])
        self.encoders = nn.ModuleList(
            EncoderStage(channels[idx], channels[idx + 1])
            for idx in range(self.model_config.depth)
        )
        self.grid = nn.Sequential(*[TFGridBlock(channels[-1]) for _ in range(self.model_config.grid_blocks)])
        self.decoders = nn.ModuleList(
            DecoderStage(channels[idx + 1], channels[idx + 1], channels[idx])
            for idx in reversed(range(self.model_config.depth))
        )
        self.head = nn.Conv2d(channels[0], 2, kernel_size=1)

        # Zero residual at initialization: the resulting mask equals the
        # analytical reference-subtraction prior with no phase correction.
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

    def _reference_prior(self, mixture: Tensor, reference: Tensor, ref_available: Tensor) -> Tensor:
        """Differentiable reference-magnitude spectral-subtraction prior."""
        eps = 1e-7
        y_mag = mixture.abs().clamp_min(eps)
        r_mag = reference.abs().clamp_min(eps)
        freqs = self.freqs.to(mixture.device)
        calibration = (freqs >= 850.0) & (freqs < 1_000.0)
        ratio = y_mag[:, calibration] / r_mag[:, calibration]
        scale = ratio.flatten(1).median(dim=1).values[:, None, None].clamp(0.05, 20.0)
        estimated_noise_power = (scale * r_mag).square()
        clean_power = (y_mag.square() - estimated_noise_power).clamp_min(0.0)
        prior = torch.sqrt(clean_power / y_mag.square().clamp_min(eps)).clamp(0.08, 1.0)
        availability = ref_available[:, None, None].to(prior.dtype)
        return availability * prior + (1.0 - availability)

    def _features(
        self,
        mixture: Tensor,
        reference: Tensor,
        ref_available: Tensor,
        prior: Tensor,
    ) -> Tensor:
        eps = 1e-7
        y = mixture[:, : self.model_bins]
        r = reference[:, : self.model_bins]
        y_mag = y.abs().clamp_min(eps)
        r_mag = r.abs().clamp_min(eps)

        compressed = y_mag.pow(self.stft_config.compression)
        phase = y / y_mag
        y_real_comp = compressed * phase.real
        y_imag_comp = compressed * phase.imag
        y_log = torch.log1p(y_mag)
        r_log = torch.log1p(r_mag)
        log_ratio = torch.tanh(torch.log(y_mag) - torch.log(r_mag))

        # Per-frequency reference coherence proxy based on log-energy similarity.
        y_centered = y_log - y_log.mean(dim=-1, keepdim=True)
        r_centered = r_log - r_log.mean(dim=-1, keepdim=True)
        numerator = (y_centered * r_centered).mean(dim=-1, keepdim=True)
        denominator = (
            y_centered.square().mean(dim=-1, keepdim=True).sqrt()
            * r_centered.square().mean(dim=-1, keepdim=True).sqrt()
        ).clamp_min(eps)
        energy_similarity = (numerator / denominator).clamp(-1.0, 1.0).expand_as(y_log)

        availability = ref_available[:, None, None].to(y_log.dtype).expand_as(y_log)
        r_log = r_log * availability
        log_ratio = log_ratio * availability
        energy_similarity = energy_similarity * availability

        return torch.stack(
            [y_real_comp, y_imag_comp, y_log, r_log, log_ratio, energy_similarity, prior, availability],
            dim=1,
        )

    def forward(
        self,
        chest: Tensor,
        reference: Tensor | None = None,
        ref_available: Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> Tensor | FrequencyDenoiseOutput:
        if chest.ndim == 3 and chest.shape[1] == 1:
            chest = chest[:, 0]
        if chest.ndim != 2:
            raise ValueError(f"chest must have shape [B,T] or [B,1,T], got {tuple(chest.shape)}")
        if reference is None:
            reference = torch.zeros_like(chest)
        elif reference.ndim == 3 and reference.shape[1] == 1:
            reference = reference[:, 0]
        if reference.shape != chest.shape:
            raise ValueError(f"reference shape {tuple(reference.shape)} != chest shape {tuple(chest.shape)}")

        batch, length = chest.shape
        if ref_available is None:
            ref_available = torch.ones(batch, device=chest.device, dtype=chest.dtype)
        else:
            ref_available = ref_available.to(device=chest.device, dtype=chest.dtype).reshape(batch)

        scale = chest.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(1e-5)
        chest_norm = chest / scale
        reference_norm = reference / scale

        mixture_stft = self._stft(chest_norm)
        reference_stft = self._stft(reference_norm)
        filtered_mixture = mixture_stft * self.bandpass.to(mixture_stft.dtype)
        filtered_reference = reference_stft * self.bandpass.to(reference_stft.dtype)

        prior = self._reference_prior(filtered_mixture, filtered_reference, ref_available)
        x = self.stem(
            self._features(
                filtered_mixture,
                filtered_reference,
                ref_available,
                prior[:, : self.model_bins],
            )
        )
        skips: list[Tensor] = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)
        x = self.grid(x)
        for decoder, skip in zip(self.decoders, reversed(skips), strict=True):
            x = decoder(x, skip)
        delta = self.head(x)

        # Start exactly at the analytical reference prior, then learn bounded
        # magnitude and phase corrections.  This avoids destructive arbitrary
        # complex masks on a small biomedical dataset.
        maximum_magnitude = self.model_config.mask_limit
        # Mask assembly and ISTFT stay in fp32: autocast can otherwise mix fp16/fp32
        # inside torch.polar and complex STFT reconstruction.
        with torch.autocast(device_type=chest.device.type, enabled=False):
            prior_model = prior[:, : self.model_bins].float().clamp(1e-4, maximum_magnitude - 1e-4)
            prior_logit = torch.log(prior_model / (maximum_magnitude - prior_model))
            magnitude_mask = maximum_magnitude * torch.sigmoid(
                prior_logit + 0.75 * torch.tanh(delta[:, 0].float())
            )
            phase_correction = 0.35 * torch.tanh(delta[:, 1].float())
            learned_mask = torch.polar(magnitude_mask, phase_correction)

            full_mask = torch.ones_like(filtered_mixture)
            full_mask[:, : self.model_bins] = learned_mask.to(filtered_mixture.dtype)
            enhanced_stft = filtered_mixture * full_mask
            waveform = self._istft(enhanced_stft, length=length) * scale.float()

        if return_details:
            return FrequencyDenoiseOutput(
                waveform=waveform,
                enhanced_stft=enhanced_stft,
                mixture_stft=filtered_mixture,
                mask=full_mask,
                scale=scale,
            )
        return waveform


def build_frequency_model(
    checkpoint_config: dict[str, Any] | None = None,
    *,
    base_channels: int | None = None,
    grid_blocks: int | None = None,
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
    if grid_blocks is not None:
        model_values["grid_blocks"] = grid_blocks
    return CardioSpecNet(stft_cfg, FrequencyModelConfig(**model_values))
