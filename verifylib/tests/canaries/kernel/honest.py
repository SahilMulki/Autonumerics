"""The negative control: a correct explicit FTCS solver for u_t = alpha u_xx - k u.

Second order in space, first order in time with ``dt ~ dx**2/4``, so the observed
order on the ladder is 2 and the temporal error stays subdominant.

Every canary beside it is this file with one defect seeded. The rule ``§9c`` states
and ``test_canaries`` enforces is that this one triggers none of them -- a guard
with no honest counter-example is a guard nobody knows is over-firing.
"""
import numpy as np

PARAMS = {"alpha": 0.05, "k": 0.3, "t_final": 0.5}
SNAPSHOTS = 3


def _setup(N, override):
    ov = override or {}
    p = {**PARAMS, **ov.get("params", {})}
    x = np.linspace(0.0, 1.0, N)
    dx = x[1] - x[0]
    dt_target = 0.25 * dx * dx / max(p["alpha"], 1e-12)
    Nt = max(4, int(np.ceil(p["t_final"] / (dt_target * ov.get("dt_factor", 1.0)))))
    return ov, p, x, dx, p["t_final"] / Nt, Nt


def _step(u, dt, dx, p, source, t, bc):
    lap = np.zeros_like(u)
    lap[1:-1] = (u[:-2] - 2 * u[1:-1] + u[2:]) / (dx * dx)
    rhs = p["alpha"] * lap - p["k"] * u
    if source is not None:
        rhs = rhs + source
    out = u + dt * rhs
    out[0], out[-1] = bc
    return out


def solve_pde(N: int, override: dict | None = None) -> dict:
    ov, p, x, dx, dt, Nt = _setup(N, override)
    coords = (x,)
    u = ov["ic"](coords) if "ic" in ov else np.sin(np.pi * x)
    u = np.asarray(u, dtype=float).copy()
    snaps, trace = [], np.zeros(Nt)
    for n in range(Nt):
        t = (n + 1) * dt
        src = ov["source"](coords, t - 0.5 * dt) if "source" in ov else None
        bc = ((float(np.atleast_1d(ov["bc"](coords, t))[0]),
               float(np.atleast_1d(ov["bc"](coords, t))[-1])) if "bc" in ov else (0.0, 0.0))
        u = _step(u, dt, dx, p, src, t, bc)
        trace[n] = float(np.mean(u ** 2))
        if n >= Nt - SNAPSHOTS:
            snaps.append({"t": t, "fields": {"u": u.copy()}})
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": p["t_final"], "dt": dt,
            "invariant_trace": {"energy": trace}, "snapshots": snaps}
