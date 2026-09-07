"""Seeded defect: right amplitude, wrong position.

The decayed profile is shifted along x. Away from the boundary it still satisfies
the PDE, so the interior residual stays small -- the residual is *identically zero*
on a translate of a solution to a translation-invariant problem, which is one of
D1's structural blind spots. The reflection symmetry the initial data has is what
notices.
"""
import numpy as np
from honest import SNAPSHOTS, _setup

SHIFT = 0.08


def _field(x, t, p):
    decay = np.exp(-(p["alpha"] * np.pi ** 2 + p["k"]) * t)
    return decay * np.sin(np.pi * np.clip(x - SHIFT, 0.0, 1.0))


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    u = _field(x, p["t_final"], p)
    u[0] = u[-1] = 0.0
    snaps = [{"t": p["t_final"] - (SNAPSHOTS - 1 - i) * dt,
              "fields": {"u": _field(x, p["t_final"] - (SNAPSHOTS - 1 - i) * dt, p)}}
             for i in range(SNAPSHOTS)]
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": np.linspace(1.0, 0.5, Nt)}, "snapshots": snaps}
