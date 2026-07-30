#!/usr/bin/env python3
"""Verify the frequency-domain pipeline: unit tests plus a real forward pass."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

def run_forward_smoke() -> None:
    # Import after test discovery. Some macOS Conda environments ship separate
    # OpenMP runtimes through SciPy and PyTorch and require SciPy to initialize
    # first; Colab is unaffected.
    import torch

    from src.frequency_model import CardioSpecNet, FrequencyModelConfig

    torch.manual_seed(0)
    model = CardioSpecNet(model_config=FrequencyModelConfig(base_channels=8))
    noisy = torch.randn(4, 8_000) * 0.01
    with torch.inference_mode():
        output = model(noisy)
    if output.shape != noisy.shape:
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
