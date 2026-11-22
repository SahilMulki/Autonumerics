import numpy as np

A = 1.0
SIGMA = 1.0
X1_0 = 1.0
X2_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Euler-Maruyama on the Ito form of the Kubo oscillator (single noise channel)."""
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    X1 = np.full(num_paths, X1_0, dtype=float)
    X2 = np.full(num_paths, X2_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    half = 0.5 * SIGMA ** 2

    for n in range(Nt):
        dWn = dW[:, n]
        P = np.stack([X1, X2], axis=1) if obs else None
        # both components computed from the old state (Ito form)
        N1 = X1 + (-A * X2 - half * X1) * dt - SIGMA * X2 * dWn
        N2 = X2 + (A * X1 - half * X2) * dt + SIGMA * X1 * dWn
        X1, X2 = N1, N2
        if obs:
            Xn = np.stack([X1, X2], axis=1)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(P) + phi(Xn)) * dt

    X = np.stack([X1, X2], axis=1)
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    amp = np.abs(np.sqrt((r["terminal_paths"] ** 2).sum(axis=1)) - 1.0)
    print("max amplitude dev: ", float(amp.max()))
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    rr = solve_sde(1000, 0.04, 1.0, dW=dW)
    exact = np.stack([np.cos(1.0 + dW.sum(1)), np.sin(1.0 + dW.sum(1))], axis=1)
    print("dW hook dt:", rr["dt"],
          "strong err:", float(np.sqrt(np.mean((rr["terminal_paths"] - exact) ** 2))))
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
          observables={"x1": lambda a: a[:, 0]})["path_integrals"]["x1"].shape)
