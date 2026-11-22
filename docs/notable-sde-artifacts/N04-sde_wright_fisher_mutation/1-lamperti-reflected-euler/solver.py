import numpy as np

X_0, A, B, SIGMA = 0.05, 0.25, 0.25, 1.0


def _fold(z):
    """Two-sided reflection of R into [0, pi] (even, 2*pi-periodic)."""
    return np.abs(np.mod(z + np.pi, 2.0 * np.pi) - np.pi)


def _mu_y(Y):
    s = np.maximum(np.sin(Y), 1e-14)
    c = np.cos(Y)
    return (2.0 * A - (A + B) * (1.0 - c)) / (SIGMA * s) - SIGMA * c / (2.0 * s)


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
        num_paths = dW.shape[0]

    Y = np.full(num_paths, 2.0 * np.arcsin(np.sqrt(X_0)), dtype=float)
    X = np.sin(0.5 * Y) ** 2
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        Y = _fold(Y + _mu_y(Y) * dt + dW[:, n])
        X = np.sin(0.5 * Y) ** 2
        for k, phi in obs.items():
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
    r = solve_sde(num_paths=200000, dt=0.04, T=1.0, seed=42)
    print(f"Empirical mean:     {r['empirical_mean']:.6f}")
    print(f"Empirical variance: {r['empirical_variance']:.6f}")
    print(f"range: [{r['terminal_paths'].min():.3e}, {r['terminal_paths'].max():.6f}]")
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
