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
        dW_sub = None  # sub-increments only meaningful alongside a supplied dW
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    num_paths = dW.shape[0]

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        dW1 = dW[:, n, 0]   # drives U
        dW2 = dW[:, n, 1]   # drives S
        if dW_sub is not None:
            sub = dW_sub[:, n]                      # (M, k, 2)
            w2 = sub[:, :, 1]
            W2tilde = np.cumsum(w2, axis=1) - w2    # left-point partial sums
            I21 = np.einsum("mj,mj->m", W2tilde, sub[:, :, 0])
        else:
            I21 = 0.5 * dW1 * dW2                   # conditional mean of the Levy area
        S_next = S + MU * dt + dW2
        U_next = U + S * dW1 + I21
        if obs:
            Xp = np.stack([S, U], axis=1)
            Xn = np.stack([S_next, U_next], axis=1)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(Xp) + phi(Xn)) * dt
        S, U = S_next, U_next

    X = np.stack([S, U], axis=1)
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
    M, dt, T = 50000, 0.04, 1.0
    Nt, k = 25, 256
    rng = np.random.default_rng(42)
    sub = np.sqrt(dt / k) * rng.standard_normal((M, Nt, k, 2))
    dWf = sub.sum(axis=2)
    r = solve_sde(M, dt, T, dW=dWf, dW_sub=sub)
    print("max |sub.sum - dW|:", np.abs(sub.sum(axis=2) - dWf).max())
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"], "(exact 1.0, 2.8333)")
    print("corr(S,U):", np.corrcoef(r["terminal_paths"].T)[0, 1], "~ SE", 1 / np.sqrt(M))
    r2 = solve_sde(M, dt, T)
    print("no-dW run:", r2["empirical_mean"], r2["empirical_variance"])
    r3 = solve_sde(1000, dt, T, dW=dWf[:1000], dW_sub=sub[:1000],
                   observables={"u2": lambda a: a[:, 1] ** 2})
    print("obs hook:", r3["path_integrals"]["u2"].shape, "dt:", r3["dt"])
