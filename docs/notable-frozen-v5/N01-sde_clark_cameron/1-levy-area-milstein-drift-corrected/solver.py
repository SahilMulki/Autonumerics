import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              dW_sub: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Two-noise Milstein for the Clark-Cameron system, state (S, U).

    Channel 0 = W1 (drives U), channel 1 = W2 (drives S).
    I_(2,1) is assembled per step from dW_sub; I_(0,1) is replaced by 0.5*dt*dW1.
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        k = 64
        dW_sub = np.sqrt(dt / k) * rng.standard_normal((num_paths, Nt, k, 2))
        dW = dW_sub.sum(axis=2)
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    obs = observables or {}
    acc = {name: np.zeros(num_paths) for name in obs}

    for n in range(Nt):
        X_prev = np.stack([S, U], axis=1) if obs else None
        dW1 = dW[:, n, 0]
        dW2 = dW[:, n, 1]
        if dW_sub is not None:
            a = dW_sub[:, n, :, 0]
            b = dW_sub[:, n, :, 1]
            W2_left = np.cumsum(b, axis=1) - b      # exclusive (left-point) sum
            I21 = np.sum(W2_left * a, axis=1)
        else:
            I21 = 0.5 * dW1 * dW2                   # conditional-mean area
        U_new = U + S * dW1 + I21 + 0.5 * MU * dt * dW1
        S_new = S + MU * dt + dW2                   # exact on the supplied path
        S, U = S_new, U_new
        if obs:
            X = np.stack([S, U], axis=1)
            for name, phi in obs.items():
                acc[name] += 0.5 * (phi(X_prev) + phi(X)) * dt

    X = np.stack([S, U], axis=1)
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=10400, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("(exact mean [2, 0], variance [1, 2.8333])")
    rng = np.random.default_rng(0)
    Nt, k, M = 25, 16, 1000
    sub = np.sqrt(1.0 / Nt / k) * rng.standard_normal((M, Nt, k, 2))
    d = sub.sum(axis=2)
    print("dW hook dt:", solve_sde(M, 0.04, 1.0, dW=d, dW_sub=sub)["dt"])
    print("dW only hook dt:", solve_sde(M, 0.04, 1.0, dW=d)["dt"])
    o = solve_sde(M, 0.04, 1.0, observables={"u2": lambda X: X[:, 1] ** 2})
    print("obs hook:", o["path_integrals"]["u2"].shape)
