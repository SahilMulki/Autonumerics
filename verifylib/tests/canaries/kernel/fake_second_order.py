"""Seeded defect: a first-order scheme whose SOLUTION.md claims second order.

The error falls cleanly, so ``shrinking`` and ``p_sane`` both pass and nothing
looks unstable. Only the measured order -- 1, against a floor of 1.8 -- separates it
from the honest control, which is precisely why Tier C is a *measurement* and not a
sanity check.
"""
import numpy as np
from honest import SNAPSHOTS, _setup

BIAS = 0.9


def _field(x, t, p, dx):
    exact = np.exp(-(p["alpha"] * np.pi ** 2 + p["k"]) * t) * np.sin(np.pi * x)
    return exact * (1.0 + BIAS * dx)


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    u = _field(x, p["t_final"], p, dx)
    u[0] = u[-1] = 0.0
    snaps = [{"t": p["t_final"] - (SNAPSHOTS - 1 - i) * dt,
              "fields": {"u": _field(x, p["t_final"] - (SNAPSHOTS - 1 - i) * dt, p, dx)}}
             for i in range(SNAPSHOTS)]
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": np.linspace(1.0, 0.5, Nt)}, "snapshots": snaps}
