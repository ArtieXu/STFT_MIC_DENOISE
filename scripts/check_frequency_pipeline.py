#!/usr/bin/env python3
"""Verify the frequency-domain pipeline: unit tests plus a real forward pass."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from src.frequency_model import CardioSpecNet, FrequencyModelConfig


def run_forward_smoke() -> None:
    torch.manual_seed(0)
    model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=8, grid_blocks=1))
    chest = torch.randn(4, 8_000) * 0.01
    reference = torch.randn(4, 8_000) * 0.01
    available = torch.tensor([1.0, 1.0, 0.0, 1.0])
    with torch.inference_mode():
        output = model(chest, reference, available)
    if output.shape != chest.shape:
        raise RuntimeError(f"Unexpected output shape: {tuple(output.shape)}")
    if not torch.isfinite(output).all():
        raise RuntimeError("Forward pass produced non-finite values")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"Forward smoke pass OK: output shape={tuple(output.shape)}, parameters={parameters:,}")


def main() -> None:
    loader = unittest.defaultTestLoader
    suite = loader.discover(str(REPO_ROOT / "tests"), pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        raise SystemExit(1)
    run_forward_smoke()
    print("All frequency pipeline checks passed.")


if __name__ == "__main__":
    main()
