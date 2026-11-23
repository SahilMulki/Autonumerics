import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None,
              dW_sub: np.ndarray | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt, 2))
        dW_sub = None
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = np.stack([S, U], axis=1) if obs else None
        dW1 = dW[:, n, 0]
        dW2 = dW[:, n, 1]
        if dW_sub is not None:
            sub1 = dW_sub[:, n, :, 0]
            sub2 = dW_sub[:, n, :, 1]
            W2tilde = np.cumsum(sub2, axis=1) - sub2        # left-point partial sums
            I21 = np.einsum("pj,pj->p", W2tilde, sub1)
        else:
            I21 = 0.5 * dW1 * dW2                           # conditional-mean area
        U_next = U + (S + 0.5 * MU * dt) * dW1 + I21        # uses pre-step S
        S_next = S + MU * dt + dW2
        S, U = S_next, U_next
        if obs:
            X = np.stack([S, U], axis=1)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

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
    M, dt0, T = 50000, 0.04, 1.0
    Nt = round(T / dt0)
    k = 256
    rng = np.random.default_rng(42)
    sub = np.sqrt(dt0 / k) * rng.standard_normal((M, Nt, k, 2))
    dWf = sub.sum(axis=2)
    print("sub-sum check:", np.abs(sub.sum(axis=2) - dWf).max())
    r = solve_sde(M, dt0, T, seed=42, dW=dWf, dW_sub=sub)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    X = r["terminal_paths"]
    print("corr(S,U):", np.corrcoef(X[:, 0], X[:, 1])[0, 1], "(se ~", 1 / np.sqrt(M), ")")
    print("exact: mean [2.0, 0], var [1.0, 2.8333333]")
    print("dW hook:", solve_sde(1000, 0.04, 1.0, dW=dWf[:1000])["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0, dW=dWf[:1000], dW_sub=sub[:1000],
          observables={"u2": lambda a: a[:, 1] ** 2})["path_integrals"]["u2"].shape)
    print("no-seed run:", solve_sde(1000, 0.04, 1.0)["empirical_mean"])
