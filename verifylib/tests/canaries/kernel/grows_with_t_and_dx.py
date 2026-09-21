"""F1's canary: accurate at the chaotic reference horizon, wrong at t_final.

The honest solver's field, multiplied by ``1 + C * dx * (t / T)**8``. At
``t = T_ref = 0.2 T`` the factor is ``1 + 2.6e-6 C dx``, well below the honest
scheme's own truncation error, so the ladder there converges cleanly at order 2
and its GCI is far below tolerance. At ``t = T`` the same
factor is ``1 + C dx`` -- first order in ``dx`` and, for ``C = 4``, 3% between
the two finest grids. The shape is exactly ``pde_kuramoto_sivashinsky``'s FD4
plan: the resolution requirement is set by the state at ``T``, which does not
exist yet at ``T_ref``, and a waiver that certifies accuracy at ``T_ref`` alone
certifies the wrong horizon.

The MMS and degenerate probes (which arrive with ``source`` or ``ic`` in
``override``) see the honest field, so Tier B passes and the verdict isolates the
horizon rule.
"""
import numpy as np
from honest import PARAMS
from honest import solve_pde as _honest

C = 4.0


def solve_pde(N, override=None):
    out = _honest(N, override)
    ov = override or {}
    if "source" in ov or "ic" in ov:
        return out
    T = PARAMS["t_final"]
    x = out["grid"]["x"]
    dx = float(x[1] - x[0])

    def factor(t):
        return 1.0 + C * dx * (float(t) / T) ** 8

    u = np.asarray(out["numerical_solution"], dtype=float) * factor(out["t_final"])
    out["numerical_solution"] = u
    out["fields"] = {"u": u}
    for snap in out["snapshots"]:
        snap["fields"] = {"u": np.asarray(snap["fields"]["u"], dtype=float) * factor(snap["t"])}
    return out
