"""Plain Euler-Maruyama, declaring ``scheme_family: milstein``. Finding 1.

The defect the strong-order gate does catch, and the reason ``verification_manual.md``
§24's own argument stops one step short: *"Euler-Maruyama and Milstein have the same
weak order, so the moment comparison cannot tell them apart"* -- and then §24 skips
the order study on a certifying Path-A plan anyway. So a plan claiming Milstein and
implementing Euler-Maruyama passed both moment thresholds and shipped mislabelled in
``REPORT.md``.

Measured against the exact GBM solution at ``M = 40 000``, four ``dt`` levels:

===================================  ==============
scheme                               strong order
===================================  ==============
correct Milstein                     0.998
**correction omitted (this file)**   **0.504**
correction with the wrong sign       0.505
correction at the post-step state    0.998
===================================  ==============

0.504 against the spec's declared ``expected_strong_order_milstein`` of 1.0 is a
two-sided miss of 0.496, well outside the 0.4 band, so ``orders_ok`` is ``False``
and ``rubric_sde_analytic``'s orders row caps it at 8 with the reason naming the
measured order.
"""
import numpy as np

PARAMS = {"X_0": 1.0, "mu": 0.05, "sigma": 0.4}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None, observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    X = np.full(num_paths, p["X_0"], dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    prev = {k: phi(X) for k, phi in (observables or {}).items()}
    for n in range(Nt):
        # The defect: no Milstein correction at all, on a plan that declares one.
        X = X + p["mu"] * X * dt + p["sigma"] * X * dW[:, n]
        for k, phi in (observables or {}).items():
            now = phi(X)
            acc[k] += 0.5 * (prev[k] + now) * dt
            prev[k] = now
    out = {"terminal_paths": X, "empirical_mean": float(np.mean(X)),
           "empirical_variance": float(np.var(X, ddof=1)), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
