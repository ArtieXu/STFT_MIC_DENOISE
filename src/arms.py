"""Build and validate the denoising systems used by the demo.

Three arms are compared, and they split along two different axes:

    training set   device-only   vs   device + CirCor      (both CardioSpecNet)
    denoiser       STFT U-Net    vs   waveform CleanUNet   (both device-only)

so a checkpoint has to record which architecture it is. Everything that loads a
checkpoint -- ``scripts/compare_datasets.py``, ``scripts/make_demo_figures.py``
-- goes through ``load_arm`` here, so no script has to guess.

The controlled comparison is single-microphone.  Every system receives the
same noisy waveform::

    model(noisy) -> [B, T]

Old reference-conditioned checkpoints are deliberately rejected by
``load_arm`` instead of being mixed silently into this comparison.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from src.frequency_data import SAMPLER_VERSION
from src.frequency_model import CardioSpecNet, FrequencyModelConfig, STFTConfig
from src.pools import DEVICE_FILE_SPLITS, device_protocol_metadata
from src.waveform_model import WaveformArmConfig, WaveformDenoiser

ARCHS = ("cardiospecnet", "cleanunet")
#: Human labels used in tables and figures.
ARCH_LABEL = {
    "cardiospecnet": "STFT 2-D U-Net",
    "cleanunet": "waveform CleanUNet",
}
SINGLE_MIC_INPUT_MODE = "single_mic"
AMP_MAX_RETRIES = 16
OPTIMIZER_UPDATE_POLICY = "one_successful_update_per_planned_batch_v1"
ROLE_SPECS = {
    "device_only": {"arch": "cardiospecnet", "allows_circor": False},
    "combined": {"arch": "cardiospecnet", "allows_circor": True},
    "waveform": {"arch": "cleanunet", "allows_circor": False},
}
SHARED_TRAINING_KEYS = (
    "epochs",
    "samples_per_epoch",
    "batch_size",
    "learning_rate",
    "weight_decay",
    "grad_clip",
    "seed",
    "snr_min_db",
    "snr_max_db",
    "pool_target_rms",
    "source_stride",
    "max_clean_windows",
    "max_noise_windows",
)


def build_arm(
    arch: str,
    *,
    checkpoint_config: dict[str, Any] | None = None,
    base_channels: int | None = None,
    wave_overrides: dict[str, Any] | None = None,
) -> nn.Module:
    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}, got {arch!r}")

    if arch == "cardiospecnet":
        if checkpoint_config:
            stft_config = STFTConfig(**checkpoint_config.get("stft_config", {}))
            values = dict(checkpoint_config.get("model_config", {}))
        else:
            stft_config, values = STFTConfig(), {}
        if base_channels is not None:
            values["base_channels"] = base_channels
        return CardioSpecNet(stft_config, FrequencyModelConfig(**values))

    values = dict((checkpoint_config or {}).get("waveform_config", {}))
    values.update({k: v for k, v in (wave_overrides or {}).items() if v is not None})
    return WaveformDenoiser(WaveformArmConfig(**values))


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _checkpoint_error(path: Path, message: str) -> ValueError:
    return ValueError(f"{path}: invalid checkpoint for controlled comparison: {message}")


def _validate_checkpoint(
    checkpoint: object,
    path: Path,
    expected_role: str | None,
) -> tuple[dict[str, Any], str]:
    if not isinstance(checkpoint, dict):
        raise _checkpoint_error(path, f"expected a dict, got {type(checkpoint).__name__}")
    for key in (
        "schema_version",
        "checkpoint_kind",
        "input_mode",
        "role",
        "arch",
        "model_config",
        "model_state_dict",
        "pool_stats",
        "data_protocol",
        "training_policy",
        "runtime_stats",
        "epoch",
    ):
        if key not in checkpoint:
            raise _checkpoint_error(path, f"missing required metadata {key!r}")
    if checkpoint["schema_version"] != 3:
        raise _checkpoint_error(
            path, f"schema_version must be 3, got {checkpoint['schema_version']!r}"
        )
    if checkpoint["checkpoint_kind"] not in {"final", "best"}:
        raise _checkpoint_error(
            path,
            "only a completed fixed-budget final checkpoint or a best-train-loss "
            f"checkpoint is comparable; got checkpoint_kind={checkpoint['checkpoint_kind']!r}",
        )
    if checkpoint["data_protocol"] != device_protocol_metadata():
        raise _checkpoint_error(path, "data protocol does not match the current fixed manifest")

    input_mode = checkpoint["input_mode"]
    if input_mode != SINGLE_MIC_INPUT_MODE:
        raise _checkpoint_error(
            path,
            f"input_mode must be {SINGLE_MIC_INPUT_MODE!r}, got {input_mode!r}; "
            "legacy two-microphone checkpoints are not comparable",
        )

    arch = checkpoint["arch"]
    if arch not in ARCHS:
        raise _checkpoint_error(path, f"unknown arch {arch!r}")
    if not isinstance(checkpoint["model_config"], dict):
        raise _checkpoint_error(path, "model_config must be a dict")
    if not isinstance(checkpoint["model_state_dict"], dict):
        raise _checkpoint_error(path, "model_state_dict must be a dict")

    pool_stats = checkpoint["pool_stats"]
    if not isinstance(pool_stats, dict):
        raise _checkpoint_error(path, "pool_stats must be a dict")
    train_clean = ((pool_stats.get("train") or {}).get("clean") or {})
    source_counts = train_clean.get("source_counts")
    if not isinstance(source_counts, dict) or not source_counts:
        raise _checkpoint_error(path, "pool_stats.train.clean.source_counts is required")
    train_sampling = pool_stats.get("train") or {}
    if train_sampling.get("sampler_version") != SAMPLER_VERSION:
        raise _checkpoint_error(
            path,
            f"sampler_version must be {SAMPLER_VERSION!r}, "
            f"got {train_sampling.get('sampler_version')!r}",
        )
    expected_noise_origins = {
        f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["noise"]["train"]
    }
    actual_noise_origins = set((train_sampling.get("noise_origin_sampling") or {}))
    if actual_noise_origins != expected_noise_origins:
        raise _checkpoint_error(
            path,
            f"training noise recordings must be {sorted(expected_noise_origins)}, "
            f"got {sorted(actual_noise_origins)}",
        )
    expected_device_clean_origins = {
        f"{stem}_windows" for stem in DEVICE_FILE_SPLITS["clean"]["train"]
    }
    device_clean_origin_counts = train_sampling.get("device_clean_origin_counts")
    if not isinstance(device_clean_origin_counts, dict):
        raise _checkpoint_error(
            path, "pool_stats.train.device_clean_origin_counts is required"
        )
    actual_device_clean_origins = set(device_clean_origin_counts)
    if actual_device_clean_origins != expected_device_clean_origins:
        raise _checkpoint_error(
            path,
            f"training device clean recordings must be "
            f"{sorted(expected_device_clean_origins)}, "
            f"got {sorted(actual_device_clean_origins)}",
        )

    policy = checkpoint["training_policy"]
    if not isinstance(policy, dict):
        raise _checkpoint_error(path, "training_policy must be a dict")
    target_epochs = int(policy.get("target_epochs", -1))
    completed_epochs = int(policy.get("completed_epochs", -1))
    steps_per_epoch = int(policy.get("steps_per_epoch", -1))
    batch_size = int(policy.get("batch_size", -1))
    optimizer_steps = int(policy.get("optimizer_steps", -1))
    inputs_per_epoch = int(policy.get("inputs_per_epoch", -1))
    trained_inputs = int(policy.get("trained_inputs", -1))
    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    precision_policy = policy.get("precision_policy")
    selection_policy = policy.get("selection_policy")
    checkpoint_kind = checkpoint["checkpoint_kind"]
    shared_policy_checks = (
        policy.get("uses_validation") is not False
        or policy.get("optimizer") != "AdamW"
        or policy.get("learning_rate_schedule") != "constant"
        or policy.get("optimizer_update_policy") != OPTIMIZER_UPDATE_POLICY
        or precision_policy
        not in {
            "fp32_fail_fast_v1",
            "cuda_fp16_retry_then_fp32_fallback_v1",
        }
        or int(policy.get("amp_max_retries_per_batch", -1))
        != AMP_MAX_RETRIES
        or target_epochs <= 0
        or steps_per_epoch <= 0
        or batch_size <= 0
        or steps_per_epoch * batch_size != inputs_per_epoch
        or inputs_per_epoch <= 0
    )
    if checkpoint_kind == "final":
        budget_checks = (
            selection_policy != "fixed_budget_final_epoch"
            or completed_epochs != target_epochs
            or checkpoint_epoch + 1 != completed_epochs
            or optimizer_steps != completed_epochs * steps_per_epoch
            or trained_inputs != completed_epochs * inputs_per_epoch
        )
    else:
        best_selection = checkpoint.get("best_selection")
        if not isinstance(best_selection, dict):
            raise _checkpoint_error(path, "best checkpoint lacks best_selection metadata")
        if (
            selection_policy != "best_train_loss"
            or best_selection.get("metric") != "train_loss_mean"
            or int(best_selection.get("epoch", -1)) != checkpoint_epoch
            or not math.isfinite(float(best_selection.get("value", float("nan"))))
            or int(best_selection.get("selected_at_optimizer_steps", -1))
            != completed_epochs * steps_per_epoch
        ):
            raise _checkpoint_error(path, f"inconsistent best-selection metadata: {best_selection}")
        budget_checks = (
            completed_epochs < 1
            or completed_epochs > target_epochs
            or checkpoint_epoch + 1 != completed_epochs
            or optimizer_steps != completed_epochs * steps_per_epoch
            or trained_inputs != completed_epochs * inputs_per_epoch
        )
    if shared_policy_checks or budget_checks:
        raise _checkpoint_error(path, f"incomplete or inconsistent fixed budget: {policy}")
    runtime_stats = checkpoint["runtime_stats"]
    if (
        not isinstance(runtime_stats, dict)
        or int(runtime_stats.get("successful_optimizer_updates", -1))
        != optimizer_steps
        or int(runtime_stats.get("amp_overflow_retries", -1)) < 0
        or int(runtime_stats.get("amp_forward_fallbacks", -1)) < 0
        or int(runtime_stats.get("fp32_fallback_updates", -1)) < 0
    ):
        raise _checkpoint_error(
            path, f"runtime stats disagree with fixed-budget metadata: {runtime_stats}"
        )
    training_args = checkpoint.get("training_args") or {}
    if (
        int(training_args.get("epochs", -1)) != target_epochs
        or int(training_args.get("samples_per_epoch", -1)) != inputs_per_epoch
        or int(training_args.get("batch_size", -1)) != batch_size
    ):
        raise _checkpoint_error(path, "training args disagree with fixed-budget metadata")

    explicit_role = checkpoint["role"]
    if explicit_role not in ROLE_SPECS:
        raise _checkpoint_error(path, f"unknown role {explicit_role!r}")
    if expected_role is not None:
        if expected_role not in ROLE_SPECS:
            raise ValueError(f"unknown expected checkpoint role {expected_role!r}")
        spec = ROLE_SPECS[expected_role]
        if arch != spec["arch"]:
            raise _checkpoint_error(
                path, f"role {expected_role!r} requires arch {spec['arch']!r}, got {arch!r}"
            )
        if explicit_role != expected_role:
            raise _checkpoint_error(
                path, f"checkpoint role is {explicit_role!r}, expected {expected_role!r}"
            )
        has_circor = int(source_counts.get("circor", 0)) > 0
        if bool(spec["allows_circor"]) != has_circor:
            expectation = "include" if spec["allows_circor"] else "exclude"
            raise _checkpoint_error(
                path, f"role {expected_role!r} must {expectation} CirCor training windows"
            )
        sampled_sources = (
            (pool_stats.get("train") or {}).get("clean_source_sampling") or {}
        )
        expected_sampling = (
            {"circor": 0.5, "device": 0.5}
            if expected_role == "combined"
            else {"device": 1.0}
        )
        normalized_sampling = {
            str(source): float(share)
            for source, share in sampled_sources.items()
        }
        if normalized_sampling != expected_sampling:
            raise _checkpoint_error(
                path,
                f"role {expected_role!r} requires clean-source input schedule "
                f"{expected_sampling}, got {normalized_sampling}",
            )
        expected_cycle = 8 if expected_role == "combined" else 4
        if int(train_sampling.get("sampling_cycle", -1)) != expected_cycle:
            raise _checkpoint_error(
                path,
                f"role {expected_role!r} requires sampling_cycle={expected_cycle}, "
                f"got {train_sampling.get('sampling_cycle')!r}",
            )

    nested_modes = [
        checkpoint["model_config"].get("input_mode"),
        (checkpoint["model_config"].get("model_config") or {}).get("input_mode"),
        (checkpoint.get("training_args") or {}).get("input_mode"),
    ]
    conflicts = [mode for mode in nested_modes if mode is not None and mode != input_mode]
    if conflicts:
        raise _checkpoint_error(path, f"conflicting nested input_mode values: {conflicts}")

    if arch == "cleanunet":
        waveform_config = checkpoint["model_config"].get("waveform_config")
        if not isinstance(waveform_config, dict):
            raise _checkpoint_error(path, "cleanunet checkpoint lacks waveform_config")
        if waveform_config.get("input_channels") != 1:
            raise _checkpoint_error(
                path,
                "single-microphone CleanUNet requires waveform_config.input_channels == 1",
            )

    sample_rates = {
        int(value)
        for value in (
            (checkpoint["model_config"].get("stft_config") or {}).get("sample_rate"),
            (checkpoint.get("mixing_config") or {}).get("sample_rate"),
            (checkpoint.get("loss_config") or {}).get("sample_rate"),
        )
        if value is not None
    }
    if sample_rates and sample_rates != {4_000}:
        raise _checkpoint_error(path, f"expected 4 kHz configuration, got {sorted(sample_rates)}")

    return checkpoint, arch


def load_arm(
    path: str | Path,
    device: torch.device,
    *,
    expected_role: str | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Rebuild one strictly single-microphone checkpoint and validate its role."""
    path = Path(path)
    if not path.is_file():
        raise SystemExit(f"checkpoint not found: {path}")
    checkpoint, arch = _validate_checkpoint(
        torch.load(path, map_location="cpu", weights_only=False),
        path,
        expected_role,
    )
    model = build_arm(arch, checkpoint_config=checkpoint.get("model_config")).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    pool_stats = checkpoint.get("pool_stats") or {}
    train_clean = ((pool_stats.get("train") or {}).get("clean") or {})
    train_sampling = pool_stats.get("train") or {}
    policy = checkpoint.get("training_policy") or {}
    runtime_stats = checkpoint.get("runtime_stats") or {}
    completed_epochs = int(policy["completed_epochs"])
    optimizer_steps = int(policy["optimizer_steps"])
    trained_inputs = int(policy["trained_inputs"])
    actual_parameters = parameter_count(model)
    recorded_parameters = checkpoint.get("parameter_count")
    if recorded_parameters is not None and int(recorded_parameters) != actual_parameters:
        raise _checkpoint_error(
            path,
            f"parameter_count says {recorded_parameters}, rebuilt model has {actual_parameters}",
        )
    meta = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "role": checkpoint["role"],
        "checkpoint_kind": checkpoint["checkpoint_kind"],
        "input_mode": checkpoint["input_mode"],
        "arch": arch,
        "arch_label": ARCH_LABEL.get(arch, arch),
        "training_set": pool_stats.get("arm"),
        "completed_epochs": completed_epochs,
        "optimizer_steps": optimizer_steps,
        "trained_inputs": trained_inputs,
        "amp_overflow_retries": int(runtime_stats["amp_overflow_retries"]),
        "fp32_fallback_updates": int(runtime_stats["fp32_fallback_updates"]),
        "parameter_count": actual_parameters,
        "train_clean_windows": train_clean.get("n_windows"),
        "train_clean_sources": train_clean.get("source_counts"),
        "model_config": checkpoint["model_config"],
        "best_selection": checkpoint.get("best_selection"),
        "training_protocol": {
            "training": {
                key: (checkpoint.get("training_args") or {}).get(key)
                for key in SHARED_TRAINING_KEYS
            },
            "mixing": checkpoint.get("mixing_config"),
            "loss": checkpoint.get("loss_config"),
            "sampling": {
                "sampler_version": train_sampling.get("sampler_version"),
                "noise_origin_sampling": train_sampling.get("noise_origin_sampling"),
                "noise_origin_counts": train_sampling.get("noise_origin_counts"),
                "device_clean_origin_counts": train_sampling.get(
                    "device_clean_origin_counts"
                ),
            },
            "budget": policy,
            "data_protocol": checkpoint.get("data_protocol"),
        },
    }
    return model, meta


def validate_shared_protocol(infos: dict[str, dict[str, Any]]) -> None:
    """Reject comparisons that changed training variables besides the arm."""

    if not infos:
        raise ValueError("no checkpoint metadata supplied")
    baseline_label = "device_only" if "device_only" in infos else next(iter(infos))
    baseline = infos[baseline_label].get("training_protocol")
    for label, meta in infos.items():
        protocol = meta.get("training_protocol")
        if protocol != baseline:
            raise ValueError(
                "checkpoints do not share one training protocol: "
                f"{label!r} differs from {baseline_label!r}.\n"
                f"{baseline_label}: {baseline}\n{label}: {protocol}"
            )
    by_arch: dict[str, tuple[str, dict[str, Any]]] = {}
    for label, meta in infos.items():
        arch = str(meta.get("arch"))
        config = meta.get("model_config")
        if arch in by_arch and config != by_arch[arch][1]:
            other_label, other_config = by_arch[arch]
            raise ValueError(
                f"checkpoints using arch {arch!r} do not share one model config: "
                f"{label!r} differs from {other_label!r}.\n"
                f"{other_label}: {other_config}\n{label}: {config}"
            )
        by_arch[arch] = (label, config)


def describe_arm(meta: dict[str, Any]) -> str:
    sources = meta.get("train_clean_sources") or {}
    share = ", ".join(f"{name} {count}" for name, count in sources.items()) or "unknown"
    return (
        f"{meta['arch_label']:<20} role {str(meta.get('role')):<12} "
        f"input {meta.get('input_mode')}; training set {str(meta.get('training_set')):<12} "
        f"{meta.get('completed_epochs')} epochs / {meta.get('optimizer_steps'):,} steps, "
        f"{meta.get('parameter_count'):,} params, "
        f"clean pool {meta.get('train_clean_windows')} ({share})"
    )
