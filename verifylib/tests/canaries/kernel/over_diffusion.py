"""Seeded defect: numerical viscosity proportional to dx.

An upwind scheme solves ``u_t + a u_x = (eps + eps_num) u_xx``: on a coarse grid
every derivative is small and mutually consistent, the residual is small, and the
answer is 30% wrong. Here the excess diffusion is ``0.6*dx``, so the scheme does
converge -- at first order, not second -- and only the ladder sees it.
"""
import numpy as np
from honest import SNAPSHOTS, _setup

EXCESS = 0.6


def _field(x, t, p, dx):
    alpha_eff = p["alpha"] + EXCESS * dx
    return np.exp(-(alpha_eff * np.pi ** 2 + p["k"]) * t) * np.sin(np.pi * x)


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    u = _field(x, p["t_final"], p, dx)
    snaps = [{"t": p["t_final"] - (SNAPSHOTS - 1 - i) * dt,
              "fields": {"u": _field(x, p["t_final"] - (SNAPSHOTS - 1 - i) * dt, p, dx)}}
             for i in range(SNAPSHOTS)]
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": np.linspace(1.0, 0.5, Nt)}, "snapshots": snaps}
