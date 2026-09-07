"""Seeded defect: the Dirichlet data is not the declared data.

A stencil needs ghost points, so the residual is interior-only and a boundary
violation lives exactly where D1 cannot compute it. D2's ``boundary_consistency``
is the check that covers that blind spot.
"""
import numpy as np
from honest import SNAPSHOTS, _setup, _step


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    coords = (x,)
    u = np.asarray(ov["ic"](coords) if "ic" in ov else np.sin(np.pi * x), dtype=float).copy()
    snaps, trace = [], np.zeros(Nt)
    for n in range(Nt):
        t = (n + 1) * dt
        src = ov["source"](coords, t - 0.5 * dt) if "source" in ov else None
        u = _step(u, dt, dx, p, src, t, (0.35, 0.35))   # declared as 0.0, applied as 0.35
        trace[n] = float(np.mean(u ** 2))
        if n >= Nt - SNAPSHOTS:
            snaps.append({"t": t, "fields": {"u": u.copy()}})
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": trace}, "snapshots": snaps}
