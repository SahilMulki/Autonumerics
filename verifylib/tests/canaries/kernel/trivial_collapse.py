"""Seeded defect: the field collapses to the operator's trivial steady state.

``u == 0`` satisfies ``u_t - alpha*lap(u) + k*u = 0`` exactly, so every term of the
residual is zero and D1's slope test is perfectly happy -- having eliminated the
only phenomenon the problem is about. This is the hole a residual-only metric has,
and it is live in the paper-release pipeline (Gray-Scott's ``u = 1, v = 0`` is the
same shape). The ``nontrivial`` guard is what catches it.
"""
import numpy as np
from honest import SNAPSHOTS, _setup


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    # A wildly over-damped update: everything decays to the homogeneous state.
    u = np.sin(np.pi * x) * 1e-9
    snaps = [{"t": p["t_final"] - (SNAPSHOTS - 1 - i) * dt, "fields": {"u": u.copy()}}
             for i in range(SNAPSHOTS)]
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": np.full(Nt, float(np.mean(u ** 2)))},
            "snapshots": snaps}
