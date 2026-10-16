"""Milstein with Rademacher increments. Finding 4, and **only** the shape check sees it.

``dW = +-sqrt(dt)`` has exactly the right mean and variance, so every moment check
passes -- and it also makes ``dW**2 - dt`` identically zero, so the Milstein
correction vanishes at every step. The plan declares ``scheme_family: milstein`` and
integrates something else.

Measured on this spec at 40 000 paths, against the correct-Milstein control:

==========================  ==========  ==============  ============  ==========
check                       correct     Rademacher      bar           verdict
==========================  ==========  ==============  ============  ==========
``moments_ok``              True        True            5% / 10%      blind
``observed_order``          0.980       0.980           1.0 +/- 0.4   blind
``dynkin_ok``               True        True            z < 2.43      blind
**standardized KS**         **0.0045**  **0.0197**      **0.0068**    **caught**
==========================  ==========  ==============  ============  ==========

The three blind spots are structural, and naming them is the point of this file.

* **The CRN ladder supplies its own increments.** That is what §18's common random
  numbers *are*: the sandbox child rebuilds ``dW`` from a seed and hands it in. So
  the order study measures the solver's **scheme** driven by Gaussian noise and
  never sees its increment law at all. Measured: 0.980 either way.
* **The Dynkin ladder does the same**, and its generator identity holds in
  expectation for *any* increment law with the right first two moments.
* **The moments are right by construction.**

The graded run is the only one that uses the solver's own draws, so its terminal
distribution is the only window onto them -- which is exactly the evidence class
Finding 4 says was missing, and the reason it is worth having.

**Multiplicative noise is what keeps that window open.** Here the terminal state is
a product whose factors carry ``dW**2`` explicitly, so the increment law survives to
``T``. On additive noise it is a *sum*, the CLT closes the window, and the same
defect is undetectable -- see ``sde_uniform_increments.py``.
"""
import numpy as np

PARAMS = {"X_0": 1.0, "mu": 0.05, "sigma": 0.4}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        # The defect: a two-point increment law with the right first two moments.
        dW = np.sqrt(dt) * rng.choice([-1.0, 1.0], size=(num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    X = np.full(num_paths, p["X_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    prev = {k: phi(X) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        inc = dW[:, n]
        X = (X + p["mu"] * X * dt + p["sigma"] * X * inc
             + 0.5 * p["sigma"] ** 2 * X * (inc ** 2 - dt))
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
