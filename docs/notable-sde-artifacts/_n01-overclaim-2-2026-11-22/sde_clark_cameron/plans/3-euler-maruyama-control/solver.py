import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None,
              dW_sub: np.ndarray | None = None) -> dict:
    """Euler-Maruyama (left-endpoint Ito) using step increments only.

    State (S, U); dW channel 0 = W1 (drives U), channel 1 = W2 (drives S).
    dW_sub is accepted and deliberately unused (this is the dW-only control).
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt, 2))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    X = np.empty((num_paths, 2))
    X[:, 0] = S_0
    X[:, 1] = U_0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        S = X_prev[:, 0]
        X = np.empty_like(X_prev)
        X[:, 0] = S + MU * dt + dW[:, n, 1]        # exact
        X[:, 1] = X_prev[:, 1] + S * dW[:, n, 0]   # left endpoint Ito
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
    r = solve_sde(num_paths=16000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    dWt = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25, 2))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
          observables={"s2": lambda a: a[:, 0] ** 2})["path_integrals"]["s2"].shape)
