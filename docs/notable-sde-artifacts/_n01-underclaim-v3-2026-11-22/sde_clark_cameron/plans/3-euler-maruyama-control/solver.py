import numpy as np

MU, S0, U0 = 1.0, 1.0, 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              dW_sub: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Increment-only Euler-Maruyama for the Clark-Cameron system (control plan).

    State (S, U); dW[:, :, 0] = dW1 drives U, dW[:, :, 1] = dW2 drives S.
    dW_sub is accepted and deliberately unused: this scheme omits the Levy area.
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt, 2))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.empty((num_paths, 2))
    X[:, 0] = S0
    X[:, 1] = U0
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        S, U = X_prev[:, 0], X_prev[:, 1]
        X = np.empty_like(X_prev)
        X[:, 0] = S + MU * dt + dW[:, n, 1]
        X[:, 1] = U + S * dW[:, n, 0]          # pre-step S; no I_21
        for k, phi in (observables or {}).items():
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
    res = solve_sde(num_paths=8000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("Exact: mean [2, 0], variance [1, 17/6=2.8333]")
    rng = np.random.default_rng(0)
    dWt = np.sqrt(0.04) * rng.standard_normal((1000, 25, 2))
    r = solve_sde(1000, 0.04, 1.0, dW=dWt)
    err = np.max(np.abs(r["terminal_paths"][:, 0] - (S0 + MU * 1.0 + dWt[:, :, 1].sum(axis=1))))
    print("dW hook dt:", r["dt"], " S channel self-check max err:", err)
    o = solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a[:, 0]})
    print("obs hook:", o["path_integrals"]["x"].shape)
