import numpy as np

MU, S0, U0 = 1.0, 1.0, 0.0
SUB_DEFAULT = 256


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              dW_sub: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Clark-Cameron system, state (S, U); dW[..., 0] = dW1 (drives U), dW[..., 1] = dW2 (drives S).

    Step: S exact; U_{k+1} = U_k + S_k dW1 + I21 + 0.5*mu*dt*dW1, where I21 is the left-point
    (Ito) sum over the sub-increments (or its conditional mean 0.5*dW1*dW2 if none supplied)
    and 0.5*mu*dt*dW1 = E[mu * I01 | dW1].
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        if dW_sub is None:
            h = dt / SUB_DEFAULT
            dW_sub = np.sqrt(h) * rng.standard_normal((num_paths, Nt, SUB_DEFAULT, 2))
            dW = dW_sub.sum(axis=2)
        else:
            dW = dW_sub.sum(axis=2)
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    S = np.full(num_paths, S0, dtype=float)
    U = np.full(num_paths, U0, dtype=float)
    X = np.stack([S, U], axis=1)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        d1 = dW[:, n, 0]
        d2 = dW[:, n, 1]
        if dW_sub is not None:
            sub = dW_sub[:, n]                       # (num_paths, k, 2)
            w2 = np.cumsum(sub[:, :, 1], axis=1)
            w2_before = np.empty_like(w2)
            w2_before[:, 0] = 0.0
            w2_before[:, 1:] = w2[:, :-1]            # strictly-before displacement: Ito
            I21 = np.sum(w2_before * sub[:, :, 0], axis=1)
        else:
            I21 = 0.5 * d1 * d2                      # conditional mean given the step increments
        S_new = X_prev[:, 0] + MU * dt + d2
        U_new = X_prev[:, 1] + X_prev[:, 0] * d1 + I21 + 0.5 * MU * dt * d1
        X = np.stack([S_new, U_new], axis=1)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(m) for m in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=8000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("exact mean (2, 0), exact variance (1, 17/6=2.8333)")

    rng = np.random.default_rng(0)
    M, Nt, k = 500, 25, 256
    dts = 1.0 / Nt
    sub = np.sqrt(dts / k) * rng.standard_normal((M, Nt, k, 2))
    dW = sub.sum(axis=2)
    r2 = solve_sde(M, 0.04, 1.0, dW=dW, dW_sub=sub)
    print("dW hook dt:", r2["dt"])
    print("S self-check:", np.max(np.abs(r2["terminal_paths"][:, 0] - (S0 + MU * 1.0 + dW[:, :, 1].sum(axis=1)))))
    r3 = solve_sde(M, 0.04, 1.0, dW=dW)
    print("no-sub branch var:", r3["empirical_variance"])
    r4 = solve_sde(M, 0.04, 1.0, dW=dW, dW_sub=sub, observables={"x": lambda a: a[:, 0] * a[:, 1]})
    print("obs hook:", r4["path_integrals"]["x"].shape)
