"""Seeded defect: a correct integration whose reported paths breach a declared
support. The terminal state is shifted below zero on return, so every moment is
wrong *and* a ``gate: true`` positivity constraint is violated; the gate must be
what names the failure, whatever the moments say.
"""
import numpy as np
from sde_honest import solve_sde as _honest


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    out = _honest(num_paths, dt, T, seed, dW, observables)
    out["terminal_paths"] = np.asarray(out["terminal_paths"]) - 10.0
    out["empirical_mean"] = float(np.mean(out["terminal_paths"]))
    return out
