import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0
SUB = 256


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None,
              dW_sub: np.ndarray | None = None) -> dict:
    if dW is None:
        rng = np.random.default_rng(seed)
        Nt = max(1, round(T / dt))
        # fine increments drawn as one block, path-major; coarse ones are their sums
        dW_sub = np.sqrt(T / Nt / SUB) * rng.standard_normal((num_paths, Nt, SUB, 2))
        dW = dW_sub.sum(axis=2)
    Nt = dW.shape[1]
    dt = T / Nt
    num_paths = dW.shape[0]

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        w1, w2 = dW[:, n, 0], dW[:, n, 1]
        if dW_sub is not None:
            sub = dW_sub[:, n]                      # (M, k, 2)
            c2 = np.cumsum(sub[:, :, 1], axis=1) - sub[:, :, 1]   # W2 strictly before sub-step j
            I21 = np.sum(c2 * sub[:, :, 0], axis=1)  # left-point Ito sum
        else:
            I21 = 0.5 * w1 * w2                      # conditional mean given dW
        X_prev = np.stack([S, U], axis=1) if observables else None
        U = U + S * w1 + I21                         # uses pre-step S
        S = S + MU * dt + w2
        if observables:
            X = np.stack([S, U], axis=1)
            for k, phi in observables.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    X = np.stack([S, U], axis=1)
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(m) for m in X.mean(axis=0)],
        "empirical_variance": [float(v) for v in X.var(axis=0, ddof=1)],
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(8000, 0.04, 1.0, seed=42)
    print("mean", r["empirical_mean"], "var", r["empirical_variance"], "dt", r["dt"])
    print("exact mean [2, 0], var [1, 2.8333]")
    rng = np.random.default_rng(1)
    M, Nt = 500, 25
    sub = np.sqrt(1.0 / Nt / SUB) * rng.standard_normal((M, Nt, SUB, 2))
    d = sub.sum(axis=2)
    r2 = solve_sde(M, 0.04, 1.0, dW=d, dW_sub=sub)
    print("S check", np.max(np.abs(r2["terminal_paths"][:, 0] - (S_0 + MU + d[:, :, 1].sum(1)))))
    r3 = solve_sde(M, 0.04, 1.0, dW=d)
    print("no-sub mean/var", r3["empirical_mean"], r3["empirical_variance"])
    r4 = solve_sde(M, 0.04, 1.0, dW=d, dW_sub=sub, observables={"u2": lambda X: X[:, 1] ** 2})
    print("obs", r4["path_integrals"]["u2"].mean())
