import numpy as np

A = 1.0
SIGMA = 1.0
X1_0 = 1.0
X2_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Kubo oscillator (Stratonovich) via the implicit midpoint rule in closed
    2x2 (Cayley) form: X_{n+1} = (I - th/2 J)^{-1} (I + th/2 J) X_n,
    th = a*dt + sigma*dW_n."""
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    x1 = np.full(num_paths, X1_0, dtype=float)
    x2 = np.full(num_paths, X2_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(a, b):
        return np.stack([a, b], axis=1)

    for n in range(Nt):
        p1, p2 = x1, x2
        theta = A * dt + SIGMA * dW[:, n]
        h = 0.25 * theta * theta
        c = 1.0 - h
        d = 1.0 + h
        x1 = (c * p1 - theta * p2) / d
        x2 = (c * p2 + theta * p1) / d
        if obs:
            Xp, Xn = state(p1, p2), state(x1, x2)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(Xp) + phi(Xn)) * dt

    X = state(x1, x2)
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("amp max dev:", float(np.abs(np.hypot(*r["terminal_paths"].T) - 1).max()))
    dWt = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dWt)["dt"])
    po = solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a[:, 0]})
    print("obs hook:", po["path_integrals"]["x"].shape)
