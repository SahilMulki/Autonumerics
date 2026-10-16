"""Draws from the exact terminal law and never integrates the SDE. Finding 1.

The canary the whole Path-A hole exists for. Every Path-A check in
``project_manual.md``'s old analytic SDE rubric passes here *by construction* --
the terminal mean and variance are not approximations of the exact ones, they are
samples from the exact law, so they land inside tolerance at any path count and
the MC resolution check passes too. It used to score **10, certified, provenance
analytic**.

It honours the ``dW`` and ``observables`` hooks well enough to look like a solver:
the path integrals are faked as ``phi(X_T) * T``, which is what a plausible
implementation would produce if it had no path to integrate along. Nothing in the
moment comparison can see any of this. Dynkin can: the generator built from the
spec's own drift and diffusion is not satisfied by a terminal draw with a
fabricated integral.
"""
import numpy as np

PARAMS = {"X_0": 2.0, "theta": 1.5, "mu": 0.0, "sigma": 0.5}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    rng = np.random.default_rng(seed)
    # The exact terminal law of the OU process -- no integration anywhere.
    mean = p["mu"] + (p["X_0"] - p["mu"]) * np.exp(-p["theta"] * T)
    var = p["sigma"] ** 2 / (2 * p["theta"]) * (1 - np.exp(-2 * p["theta"] * T))
    X = mean + np.sqrt(var) * rng.standard_normal(num_paths)
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        # No path to integrate along, so the integral is invented from the terminal
        # value. A reviewer reading the output cannot tell; the generator can.
        out["path_integrals"] = {k: phi(X) * T for k, phi in observables.items()}
    return out
