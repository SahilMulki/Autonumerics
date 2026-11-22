"""Ait-Sahalia short-rate model, Lamperti transform + drift-trapezoidal (theta = 1/2) step.

    dX = (a_m1/X - a_0 + a_1 X - a_2 X^2) dt + sigma X^{3/2} dW,   X(0) = X_0

With Y = X^(-1/2) the noise is additive:

    dY = h(Y) dt - (sigma/2) dW
    h(Y) = -(a_m1/2) Y^5 + (a_0/2) Y^3 - (a_1/2) Y + (a_2/2 + 3 sigma^2/8) / Y

Step (noise explicit, drift trapezoidal):

    Y_{n+1} - (dt/2) h(Y_{n+1}) = Y_n + (dt/2) h(Y_n) - (sigma/2) dW_n

The map y -> y - (dt/2) h(y) is strictly increasing from -inf to +inf on (0, inf)
(h' < 0 there), so there is exactly one positive root for every right-hand side and
X = Y^(-2) is strictly positive by construction.  No clipping of the state.
"""
import numpy as np

X_0 = 1.0
A_M1 = 2.0
A_0 = 3.0
A_1 = 4.0
A_2 = 7.0
SIGMA = 1.4142135623730951

_C_INV = A_2 / 2.0 + 3.0 * SIGMA**2 / 8.0     # coefficient of 1/Y in h


def _h(y):
    return (-0.5 * A_M1 * y**5 + 0.5 * A_0 * y**3 - 0.5 * A_1 * y + _C_INV / y)


def _dh(y):
    return (-2.5 * A_M1 * y**4 + 1.5 * A_0 * y**2 - 0.5 * A_1 - _C_INV / y**2)


def _implicit_solve(r, c, y0, max_iter=200, tol=1e-14):
    """Solve F(y) = y - c*h(y) - r = 0 on y > 0 for each element (safeguarded Newton).

    F is strictly increasing with F(0+) = -inf, F(inf) = +inf, F' = 1 - c h' > 1.
    A bracket [lo, hi] (lo starts at 0, hi at +inf) is maintained; a Newton step
    that leaves the bracket is replaced by bisection (or halving toward lo while
    hi is still infinite and the step would cross zero).
    """
    y = np.array(y0, dtype=float, copy=True)
    lo = np.zeros_like(y)
    hi = np.full_like(y, np.inf)
    for _ in range(max_iter):
        F = y - c * _h(y) - r
        lo = np.where(F < 0.0, y, lo)
        hi = np.where(F > 0.0, y, hi)
        dF = 1.0 - c * _dh(y)
        y_new = y - F / dF
        bad = ~((y_new > lo) & (y_new < hi))
        if np.any(bad):
            mid = np.where(np.isfinite(hi), 0.5 * (lo + hi), 2.0 * lo + 1e-12)
            y_new = np.where(bad, mid, y_new)
        step = np.abs(y_new - y)
        y = y_new
        if np.all(step <= tol * np.abs(y)):
            break
    F = y - c * _h(y) - r
    if not (np.all(np.isfinite(y)) and np.all(y > 0.0)
            and np.max(np.abs(F) / np.maximum(1.0, np.abs(r))) < 1e-10):
        raise RuntimeError("implicit root solve did not converge")
    return y


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))   # one block, path-major
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    c = 0.5 * dt
    Y = np.full(num_paths, X_0 ** -0.5, dtype=float)
    X = Y ** -2
    for n in range(Nt):
        X_prev = X
        r = Y + c * _h(Y) - 0.5 * SIGMA * dW[:, n]     # note the minus sign on the noise
        Y = _implicit_solve(r, c, Y)
        X = Y ** -2
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)
    print(f"Empirical mean:     {res['empirical_mean']:.6f}")
    print(f"Empirical variance: {res['empirical_variance']:.6f}")
    print(f"min path: {res['terminal_paths'].min():.4f}  max path: {res['terminal_paths'].max():.4f}")
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 10))
    print("dW hook: dt =", solve_sde(1000, 0.02, 0.1, dW=dWt)["dt"])
    po = solve_sde(1000, 0.02, 0.1, observables={"x": lambda a: a})["path_integrals"]["x"]
    print("obs hook:", po.shape, float(po.mean()))
