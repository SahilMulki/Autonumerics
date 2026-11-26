import numpy as np

A_M1, A_0, A_1, A_2 = 2.0, 3.0, 4.0, 7.0
SIGMA = 1.4142135623730951
X_0 = 1.0
_K5 = A_M1 / 2.0
_K3 = A_0 / 2.0
_K1 = A_1 / 2.0
_KM = A_2 / 2.0 + 3.0 * SIGMA ** 2 / 8.0


def _h(z):
    return -_K5 * z ** 5 + _K3 * z ** 3 - _K1 * z + _KM / z


def _dh(z):
    return -5.0 * _K5 * z ** 4 + 3.0 * _K3 * z ** 2 - _K1 - _KM / z ** 2


def _implicit_solve(r, c, z0):
    """Unique positive root of z - c*h(z) = r (phi strictly increasing on (0, inf))."""
    lo = np.full_like(r, 1e-12)
    hi = np.full_like(r, 1e3)
    z = np.clip(z0, lo, hi)
    for _ in range(100):
        F = z - c * _h(z) - r
        lo = np.where(F < 0, z, lo)
        hi = np.where(F > 0, z, hi)
        dF = 1.0 - c * _dh(z)
        zn = z - F / dF
        bad = ~np.isfinite(zn) | (zn <= lo) | (zn >= hi)
        zn = np.where(bad, 0.5 * (lo + hi), zn)
        done = np.abs(zn - z) <= 1e-13 * np.abs(zn)
        z = zn
        if np.all(done | (F == 0)):
            break
    else:
        raise RuntimeError("implicit solve did not converge")
    return z


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
    c = 0.5 * dt
    Z = np.full(num_paths, X_0 ** -0.5, dtype=float)
    X = Z ** -2
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    for n in range(Nt):
        X_prev = X
        r = Z + c * _h(Z) - 0.5 * SIGMA * dW[:, n]
        Z = _implicit_solve(r, c, Z)
        X = Z ** -2
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt
    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)
    X = res["terminal_paths"]
    print(f"Empirical mean:     {res['empirical_mean']:.6f}")
    print(f"Empirical variance: {res['empirical_variance']:.6f}")
    print("finite:", bool(np.all(np.isfinite(X))), "positive:", bool(np.all(X > 0)), "min X:", X.min())
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 10))
    print("dW hook dt:", solve_sde(1000, 0.02, 0.1, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.02, 0.1,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
