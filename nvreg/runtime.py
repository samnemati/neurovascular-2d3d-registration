"""Device selection, kept in one place so every script makes the same choice."""

from __future__ import annotations

import torch


def pick_device(prefer: str | None = None) -> torch.device:
    """Return the fastest available device, or the one named in ``prefer``.

    Apple's MPS backend renders these volumes about five times faster than the
    CPU and agrees with it to within float32 rounding (about 1e-5 relative on
    a real case); CUDA is used when present. Everything falls back to CPU,
    which is what continuous integration runs on.
    """
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device) -> None:
    """Wait for queued work, so timings measure the work and not the queue."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()
