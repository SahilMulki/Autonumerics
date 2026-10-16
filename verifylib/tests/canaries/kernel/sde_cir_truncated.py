"""A **correct** full-truncation Milstein on a CIR whose Feller condition is violated.

The control for `dynkin.residual_order`'s lower band edge. Nothing here is a defect:
the scheme is the standard full-truncation Milstein for a square-root diffusion, and
it converges to the exact noncentral-chi-squared transition law -- measured against
it, `E[tanh(X_T)]` is 0.296 / 0.291 / 0.290 at dt = 0.005 / 0.0025 / 0.00125 against
an exact 0.280, and the error falls only 2.6x over a 64x refinement.

That rate -- order ~0.2 -- is the point. `2*kappa*theta = 1.0` against `sigma**2 = 4`
puts the shape parameter at 0.25, so the invariant density diverges at the origin,
the process reaches zero, and `sqrt(X)` is non-Lipschitz exactly where the paths
spend their time. Every truncated scheme converges slowly here; it is a property of
the problem, not of this solver.

Dynkin's Richardson extrapolation at an assumed `p = 1` cannot remove an order-0.2
bias, so it used to report the leftover as a generator violation -- z = 5.1 on the
live twin of this plan, a hard 3 for a correct solver. The outcome must be
`"unresolved"`, and the distinction from a solver that never integrates the SDE is
that this residual *shrinks*: see `sde_fabricated_sampler.py`, whose residual is
dt-independent and still fails.
"""
import numpy as np

X_0 = 0.5
KAPPA = 1.0
THETA = 0.5
SIGMA = 2.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.full(num_paths, X_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        X_pos = np.maximum(X_prev, 0.0)
        # Full truncation: the guard is inside the sqrt of the step itself, which is
        # what `verification_manual.md` §21 requires of a positivity scheme -- not a
        # bare clip of the returned array.
        X = X_prev + KAPPA * (THETA - X_pos) * dt + SIGMA * np.sqrt(X_pos) * dW[:, n]
        # g = sigma*sqrt(X), so 0.5*g*dg/dX = sigma**2/4 -- state-independent, and
        # evaluated at the pre-step state.
        X = X + (SIGMA ** 2 / 4.0) * (dW[:, n] ** 2 - dt)
        X = np.maximum(X, 0.0)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)),
           "num_paths": num_paths, "dt": dt, "T": T}
    if observables:
        out["path_integrals"] = acc
    return out
