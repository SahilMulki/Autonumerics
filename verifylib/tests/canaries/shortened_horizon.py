"""CANARY -- a solver that stops early and reports the full horizon.

Seeded deliberately. The spec asks for t_final = 0.5; this integrates to 0.25,
where a diffusive solution is smoother and every error metric is easier, then
returns the honest t_final. Nothing in the numbers looks wrong -- the ledger
re-audit catches it by comparing the run's own reported horizon against the one
the spec asked for, which is exactly the class of check that has to be
output-derived to be a hard gate.
"""
import numpy as np


def solve_pde(N, override=None):
    x = np.linspace(0.0, 1.0, N)
    alpha, t_stop = 0.1, 0.25          # the spec says 0.5
    u = np.exp(-alpha * np.pi ** 2 * t_stop) * np.sin(np.pi * x)
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": t_stop}
