import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Kubo oscillator, Stratonovich form, implicit midpoint (Cayley) step.

    dX = J X (a dt + sigma o dW),  J = [[0,-1],[1,0]].
    Midpoint rule on the Stratonovich coefficients as written (no Ito correction):
      X_{n+1} = (I - th/2 J)^{-1} (I + th/2 J) X_n,  th = a dt + sigma dW_n
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    obs = observables or {}
    X = np.zeros((num_paths, 2))
    X[:, 0] = X_0
    X[:, 1] = Y_0
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        th = A * dt + SIGMA * dW[:, n]
        q = 0.25 * th * th
        den = 1.0 + q
        c = (1.0 - q) / den
        s = th / den
        X = np.empty_like(X_prev)
        X[:, 0] = c * X_prev[:, 0] - s * X_prev[:, 1]
        X[:, 1] = s * X_prev[:, 0] + c * X_prev[:, 1]
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

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
    print("Mean |X|^2:", float(np.mean(np.sum(r["terminal_paths"] ** 2, axis=1))))
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    po = solve_sde(1000, 0.04, 1.0,
                   observables={"x": lambda a: a[:, 0]})["path_integrals"]["x"]
    print("obs hook shape:", po.shape)
