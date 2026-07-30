"""
waveform_model.py — the waveform-domain baseline arm.

Copied verbatim from ``src/model.py`` of this project's Mic_denoise repo (the
original CleanUNet implementation used before the STFT work), then extended at
the bottom of this file with:

  * ``WaveformArmConfig`` / ``build_waveform_arm`` -- the compact, non-causal
    configuration used for the system comparison, and
  * ``WaveformDenoiser`` -- a single-microphone adapter with the same
    ``model(noisy) -> [B, T]`` interface as the STFT arm.

Everything above the ADAPTER banner is the original file.

CleanUNet: 1-D U-Net with Transformer bottleneck for waveform denoising.

Architecture (CleanUNet, NVIDIA, ICASSP 2022):
  Encoder  : D layers of  Conv1d(stride=S) → ReLU → Conv1d(1×1) → GLU
  Bottleneck: N-layer Transformer with causal self-attention
  Decoder  : D layers of  Conv1d(1×1) → GLU → ConvTranspose1d(stride=S) → (ReLU)
  Skip connections are *additive* (not concatenated) from encoder to decoder.

Input / output shape: [B, 1, T] waveform tensors.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    # Waveform I/O channels (1 for mono)
    channels_input:   int = 1
    channels_output:  int = 1

    # U-Net encoder/decoder
    channels_H:       int = 64     # starting channel count; doubled each layer
    max_H:            int = 768    # channel count cap (prevents memory explosion)
    encoder_n_layers: int = 8      # depth D — number of encoder/decoder stages
    kernel_size:      int = 4      # conv kernel size K
    stride:           int = 2      # downsampling / upsampling factor S

    # Transformer bottleneck
    tsfm_n_layers:    int = 3      # number of self-attention blocks N
    tsfm_n_head:      int = 4      # attention heads per block
    tsfm_d_model:     int = 512    # token embedding dimension
    tsfm_d_inner:     int = 512    # feed-forward hidden dimension

    # Causal self-attention in the bottleneck.
    #
    # True (default, unchanged) reproduces the reference CleanUNet: each
    # bottleneck frame may only attend to earlier frames, which is what a
    # streaming deployment needs.
    #
    # Set False for an OFFLINE comparison against `src/stft.py`, whose fixed
    # STFT uses center=True and whose 2-D U-Net convolves freely over time --
    # that model is non-causal by construction. Leaving the mask on while
    # comparing the two confounds the representation (waveform vs
    # time-frequency) with the amount of context each model is allowed to see:
    # at 8000 samples and stride 2^D the causal model averages half the window,
    # the STFT model always gets all of it.
    causal:           bool = True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _weight_scaling_init(layer: nn.Module) -> None:
    """
    Re-initialise a Conv1d/ConvTranspose1d layer with Kaiming uniform weights
    scaled down by 1/√2.  Reduces activation variance in deep stacks and
    stabilises early training.
    """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)
    layer.weight.data.mul_(2 ** -0.5)


def _pad_to_fit(x: torch.Tensor, D: int, K: int, S: int) -> torch.Tensor:
    """
    Zero-pad x along the time axis so that after D rounds of downsampling
    (kernel K, stride S) followed by D rounds of upsampling the output length
    equals the input length exactly.
    """
    L = x.shape[-1]
    # Simulate D downsampling steps to find the bottleneck length
    for _ in range(D):
        L = 1 if L < K else int(math.ceil((L - K) / S)) + 1
    # Simulate D upsampling steps back to waveform length
    for _ in range(D):
        L = (L - 1) * S + K
    x = F.pad(x, (0, L - x.shape[-1]))
    return x


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class _ScaledDotProductAttention(nn.Module):
    """Standard scaled dot-product attention with optional mask and dropout."""

    def __init__(self, temperature: float, dropout: float = 0.0):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # q/k/v: [B, n_head, L, d_k]
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))  # [B, H, L, L]
        if mask is not None:
            attn = attn.masked_fill(~mask, -1e9)
        attn = self.dropout(F.softmax(attn, dim=-1))
        return torch.matmul(attn, v)                                   # [B, H, L, d_v]


class _MultiHeadAttention(nn.Module):
    """Multi-head self-attention with pre-norm residual connection."""

    def __init__(self, n_head: int, d_model: int, d_k: int, d_v: int, dropout: float = 0.0):
        super().__init__()
        self.n_head = n_head
        self.d_k    = d_k
        self.d_v    = d_v

        self.w_q  = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_k  = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_v  = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc   = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attn      = _ScaledDotProductAttention(temperature=d_k ** 0.5, dropout=dropout)
        self.dropout   = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Lq, Lk, Lv = q.shape[0], q.shape[1], k.shape[1], v.shape[1]
        residual = q

        # Project and split into heads: [B, L, n_head, d_k] → [B, n_head, L, d_k]
        q = self.w_q(q).view(B, Lq, self.n_head, self.d_k).transpose(1, 2)
        k = self.w_k(k).view(B, Lk, self.n_head, self.d_k).transpose(1, 2)
        v = self.w_v(v).view(B, Lv, self.n_head, self.d_v).transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # broadcast over heads

        out = self.attn(q, k, v, mask=mask)

        # Merge heads: [B, n_head, L, d_v] → [B, L, n_head*d_v]
        out = out.transpose(1, 2).contiguous().view(B, Lq, -1)
        out = self.dropout(self.fc(out))

        return self.layer_norm(out + residual)


class _PositionwiseFFN(nn.Module):
    """Two-layer position-wise feed-forward network with residual + layer norm."""

    def __init__(self, d_in: int, d_hid: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(d_in, d_hid)
        self.w2 = nn.Linear(d_hid, d_in)
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.w2(F.relu(self.w1(x)))
        x = self.dropout(x)
        return self.layer_norm(x + residual)


class _PositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional encoding (Vaswani et al., 2017).
    Disabled in the default CleanUNet config (n_position=0) because the
    bottleneck uses causal masking as its position signal.
    """

    def __init__(self, d_hid: int, n_position: int = 200):
        super().__init__()
        self.register_buffer(
            "pos_table", self._make_sinusoid_table(n_position, d_hid)
        )

    @staticmethod
    def _make_sinusoid_table(n: int, d: int) -> torch.Tensor:
        positions = torch.arange(n, dtype=torch.float).unsqueeze(1)          # [n, 1]
        dims      = torch.arange(d, dtype=torch.float).unsqueeze(0)          # [1, d]
        angles    = positions / (10_000 ** (2 * (dims // 2) / d))            # [n, d]
        angles[:, 0::2] = angles[:, 0::2].sin()
        angles[:, 1::2] = angles[:, 1::2].cos()
        return angles.unsqueeze(0)                                            # [1, n, d]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, d_model]
        return x + self.pos_table[:, : x.shape[1]].clone().detach()


class _EncoderLayer(nn.Module):
    """Single Transformer encoder layer: self-attention → feed-forward."""

    def __init__(self, d_model: int, d_inner: int, n_head: int, d_k: int, d_v: int):
        super().__init__()
        self.self_attn = _MultiHeadAttention(n_head, d_model, d_k, d_v)
        self.ffn       = _PositionwiseFFN(d_model, d_inner)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.self_attn(x, x, x, mask=mask)
        x = self.ffn(x)
        return x


class _TransformerEncoder(nn.Module):
    """
    Stack of N Transformer encoder layers with optional positional encoding.
    Used as the CleanUNet bottleneck; operates on the compressed time axis.
    """

    def __init__(
        self,
        d_model:    int,
        n_layers:   int,
        n_head:     int,
        d_inner:    int,
        n_position: int = 0,       # 0 disables sinusoidal PE (use causal mask instead)
        dropout:    float = 0.0,
    ):
        super().__init__()
        d_k = d_v = d_model // n_head

        self.pos_enc = (
            _PositionalEncoding(d_model, n_position) if n_position > 0
            else nn.Identity()
        )
        self.dropout    = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.layers     = nn.ModuleList(
            [_EncoderLayer(d_model, d_inner, n_head, d_k, d_v) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: [B, L, d_model]
        x = self.layer_norm(self.dropout(self.pos_enc(x)))
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x


# ---------------------------------------------------------------------------
# CleanUNet
# ---------------------------------------------------------------------------

class CleanUNet(nn.Module):
    """
    CleanUNet waveform denoiser.

    Encoder path (D layers):
        Conv1d(stride=S) → ReLU → Conv1d(1×1) → GLU
        Channel count doubles each layer: channels_H, 2H, 4H, … (capped at max_H)

    Bottleneck:
        1×1 Conv → Transformer(N layers, causal mask) → 1×1 Conv

    Decoder path (D layers, reversed):
        Skip added to decoder input (additive, not concatenated)
        Conv1d(1×1) → GLU → ConvTranspose1d(stride=S) → ReLU   (ReLU omitted on final layer)

    I/O normalisation: input is divided by its std before the encoder and
    multiplied back after the decoder, so the network always sees unit-variance signals.
    """

    def __init__(self, cfg: ModelConfig = ModelConfig()):
        super().__init__()
        self.cfg = cfg

        ch_in  = cfg.channels_input
        ch_out = cfg.channels_output
        H      = cfg.channels_H
        K      = cfg.kernel_size
        S      = cfg.stride
        D      = cfg.encoder_n_layers

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # ---- Encoder + Decoder (built together so channel counts stay in sync) ---
        #
        # At depth i, the encoder block processes ch_in → H channels.
        # The corresponding decoder block (its mirror image) processes H → ch_out channels.
        #
        # Decoder ordering trick (from the reference implementation):
        #   • i=0 (shallowest): decoder.append  → ends up at decoder[-1]
        #   • i>0              : decoder.insert(0) → pushes earlier blocks right
        # Result: decoder[0] = deepest stage, decoder[-1] = shallowest stage.
        # This matches the forward loop order (deepest skip consumed first).

        for i in range(D):
            # -- Encoder block --
            # Conv1d(stride=S) downsamples by S.
            # Conv1d(1×1) + GLU expands to 2H then halves back to H.
            # Channel flow: ch_in → H → H
            self.encoder.append(nn.Sequential(
                nn.Conv1d(ch_in, H, K, stride=S),
                nn.ReLU(),
                nn.Conv1d(H, H * 2, 1),
                nn.GLU(dim=1),
            ))
            ch_in = H       # encoder output feeds the next encoder stage

            # -- Decoder block (mirror of encoder at this depth) --
            # Conv1d(1×1) + GLU keeps H channels, ConvTranspose1d upsamples by S.
            # Channel flow: H → H → ch_out
            if i == 0:
                # Shallowest stage: no trailing ReLU so the output waveform can be negative
                self.decoder.append(nn.Sequential(
                    nn.Conv1d(H, H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(H, ch_out, K, stride=S),
                ))
            else:
                # All other stages: ReLU keeps internal activations non-negative
                self.decoder.insert(0, nn.Sequential(
                    nn.Conv1d(H, H * 2, 1),
                    nn.GLU(dim=1),
                    nn.ConvTranspose1d(H, ch_out, K, stride=S),
                    nn.ReLU(),
                ))
            ch_out = H                      # this stage's output feeds the next shallower stage
            H = min(H * 2, cfg.max_H)      # double H for the next (deeper) level, cap at max_H

        # After the loop, ch_in = deepest encoder output channels = bottleneck width
        ch_bottleneck = ch_in

        # ---- Bottleneck Transformer -----------------------------------------
        # Project channel dim to tsfm_d_model, apply N-layer self-attention, project back.
        self.tsfm_in  = nn.Conv1d(ch_bottleneck, cfg.tsfm_d_model, 1)
        self.tsfm     = _TransformerEncoder(
            d_model    = cfg.tsfm_d_model,
            n_layers   = cfg.tsfm_n_layers,
            n_head     = cfg.tsfm_n_head,
            d_inner    = cfg.tsfm_d_inner,
            n_position = 0,       # PE disabled; causal mask provides position info
        )
        self.tsfm_out = nn.Conv1d(cfg.tsfm_d_model, ch_bottleneck, 1)

        # ---- Weight initialisation -----------------------------------------
        for layer in self.modules():
            if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d)):
                _weight_scaling_init(layer)

    # -------------------------------------------------------------------------

    def forward(self, noisy: torch.Tensor) -> torch.Tensor:
        """
        Args:
            noisy: [B, 1, T]  or  [B, T]  noisy waveform
        Returns:
            denoised: [B, 1, T]
        """
        # Accept [B, T] for convenience
        if noisy.dim() == 2:
            noisy = noisy.unsqueeze(1)

        B, C, T = noisy.shape
        assert C == self.cfg.channels_input

        # --- Input normalisation (per-utterance std) ------------------------
        # Dividing by std makes the network invariant to absolute loudness.
        std = noisy.std(dim=-1, keepdim=True) + 1e-3
        x   = noisy / std

        # --- Pad so downsampling/upsampling round-trips exactly -------------
        x = _pad_to_fit(x, self.cfg.encoder_n_layers, self.cfg.kernel_size, self.cfg.stride)

        # --- Encoder --------------------------------------------------------
        skips = []
        for enc_block in self.encoder:
            x = enc_block(x)
            skips.append(x)       # save encoder output for skip connection

        # --- Bottleneck Transformer -----------------------------------------
        # Rearrange: [B, C, L] → [B, L, C] for the Transformer, then back
        x = self.tsfm_in(x)                             # [B, d_model, L]
        mask = (
            self._causal_mask(x.shape[-1], x.device) if self.cfg.causal else None
        )
        x = self.tsfm(x.permute(0, 2, 1),              # [B, L, d_model]
                       mask=mask)
        x = self.tsfm_out(x.permute(0, 2, 1))          # [B, C, L]

        # --- Decoder --------------------------------------------------------
        # Skips are consumed deepest-first (reverse of how they were saved).
        for dec_block, skip in zip(self.decoder, reversed(skips)):
            # Additive skip: trim skip to current x length in case of rounding
            x = x + skip[:, :, : x.shape[-1]]
            x = dec_block(x)

        # --- Crop to original length and restore loudness -------------------
        x = x[:, :, :T] * std

        return x

    @staticmethod
    def _causal_mask(length: int, device: torch.device) -> torch.Tensor:
        """
        Lower-triangular boolean mask [1, L, L] for causal attention.
        Shape [1, L, L] so that unsqueeze(1) in MultiHeadAttention produces
        [1, 1, L, L], which broadcasts correctly over [B, n_head, L, L].
        """
        return torch.tril(
            torch.ones(1, length, length, dtype=torch.bool, device=device)
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig = ModelConfig()) -> CleanUNet:
    """Instantiate and return a CleanUNet from config."""
    return CleanUNet(cfg)


# ===========================================================================
# ADAPTER -- everything below is new; everything above is the original file.
# ===========================================================================

from dataclasses import asdict  # noqa: E402
from typing import Any  # noqa: E402


@dataclass(frozen=True)
class WaveformArmConfig:
    """Compact, non-causal CleanUNet for the denoising-system comparison.

    Two things are deliberately not the CleanUNet defaults:

    ``causal=False``  CardioSpecNet is non-causal by construction (center=True
        framing, 2-D convolution over time). With the causal mask on, at 8000
        samples and stride 2^D the waveform arm would average half a window of
        context while the STFT arm always sees the whole window -- the result
        would measure available context, not representation.

    small channels/d_model  CleanUNet's published default is much larger than
        needed for this device-data comparison. These values keep the previous
        waveform baseline compact (about 564k parameters).

    ``input_channels=1`` uses the same single noisy chest waveform as the STFT
        arm.  It is kept in the checkpoint schema so legacy two-input
        checkpoints can be rejected explicitly instead of being compared by
        accident.
    """

    input_channels: int = 1
    channels_H: int = 12
    max_H: int = 96
    encoder_n_layers: int = 5
    kernel_size: int = 4
    stride: int = 2
    tsfm_n_layers: int = 2
    tsfm_n_head: int = 4
    tsfm_d_model: int = 128
    tsfm_d_inner: int = 384
    causal: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_model_config(self) -> ModelConfig:
        return ModelConfig(
            channels_input=self.input_channels,
            channels_output=1,
            channels_H=self.channels_H,
            max_H=self.max_H,
            encoder_n_layers=self.encoder_n_layers,
            kernel_size=self.kernel_size,
            stride=self.stride,
            tsfm_n_layers=self.tsfm_n_layers,
            tsfm_n_head=self.tsfm_n_head,
            tsfm_d_model=self.tsfm_d_model,
            tsfm_d_inner=self.tsfm_d_inner,
            causal=self.causal,
        )


class WaveformDenoiser(nn.Module):
    """Single-microphone CleanUNet adapter: ``forward(noisy) -> [B, T]``."""

    def __init__(self, config: WaveformArmConfig | None = None) -> None:
        super().__init__()
        self.config = config or WaveformArmConfig()
        if self.config.input_channels != 1:
            raise ValueError(
                "This experiment is single-microphone only; "
                f"input_channels must be 1, got {self.config.input_channels}"
            )
        self.net = CleanUNet(self.config.to_model_config())

    @property
    def config_dict(self) -> dict[str, Any]:
        return {"waveform_config": self.config.to_dict()}

    def forward(
        self,
        noisy: torch.Tensor,
        *,
        return_details: bool = False,
    ) -> torch.Tensor:
        if noisy.dim() == 3 and noisy.shape[1] == 1:
            noisy = noisy[:, 0]
        if noisy.dim() != 2:
            raise ValueError(f"noisy must be [B,T] or [B,1,T], got {tuple(noisy.shape)}")

        estimate = self.net(noisy.unsqueeze(1))
        return estimate[:, 0]


def build_waveform_arm(
    checkpoint_config: dict[str, Any] | None = None, **overrides: Any
) -> WaveformDenoiser:
    """Build from a checkpoint's stored config, or from explicit overrides."""
    values: dict[str, Any] = {}
    if checkpoint_config:
        values.update(checkpoint_config.get("waveform_config", {}))
    values.update({key: value for key, value in overrides.items() if value is not None})
    return WaveformDenoiser(WaveformArmConfig(**values))
