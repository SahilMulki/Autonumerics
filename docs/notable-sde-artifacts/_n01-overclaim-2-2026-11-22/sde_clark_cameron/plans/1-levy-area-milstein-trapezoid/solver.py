import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0


def _levy_I12(dW_sub, chunk=1000):
    """I_{1,2} per path per step from sub-increments (channel 0 = W1, 1 = W2)."""
    P, Nt = dW_sub.shape[0], dW_sub.shape[1]
    out = np.empty((P, Nt))
    for a in range(0, P, chunk):
        blk = dW_sub[a:a + chunk]
        d1 = blk[..., 0]
        d2 = blk[..., 1]
        w2 = np.cumsum(d2, axis=2) - d2          # W2(s_j) - W2(t_n), exclusive prefix
        out[a:a + chunk] = np.einsum("pnj,pnj->pn", w2, d1)
    return out


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
        num_paths = dW.shape[0]

    if dW_sub is not None:
        I12 = _levy_I12(dW_sub)
    else:
        I12 = 0.5 * dW[:, :, 0] * dW[:, :, 1]    # conditional mean of the Levy-area term

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    X = np.empty((num_paths, 2))
    X[:, 0] = S_0
    X[:, 1] = U_0

    for n in range(Nt):
        X_prev = X
        d1 = dW[:, n, 0]
        d2 = dW[:, n, 1]
        S, U = X_prev[:, 0], X_prev[:, 1]
        Xn = np.empty_like(X_prev)
        Xn[:, 0] = S + MU * dt + d2
        Xn[:, 1] = U + S * d1 + MU * (0.5 * dt) * d1 + I12[:, n]
        X = Xn
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
    P, dt, T = 16000, 0.04, 1.0
    Nt = round(T / dt)
    rng = np.random.default_rng(42)
    dW = np.sqrt(dt) * rng.standard_normal((P, Nt, 2))
    k = 256
    sub = np.sqrt(dt / k) * rng.standard_normal((P, Nt, k, 2))
    dW = sub.sum(axis=2)
    r = solve_sde(P, dt, T, dW=dW, dW_sub=sub)
    print("mean:    ", r["empirical_mean"])
    print("variance:", r["empirical_variance"], "(exact 1, 17/6=2.8333)")
    r0 = solve_sde(P, dt, T, dW=dW)
    print("no-sub variance:", r0["empirical_variance"])
    r1 = solve_sde(P, dt, T, seed=42)
    print("self-drawn:", r1["empirical_mean"], r1["empirical_variance"])
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW[:1000], dW_sub=sub[:1000])["dt"])
    po = solve_sde(1000, 0.04, 1.0, dW=dW[:1000], dW_sub=sub[:1000],
                   observables={"s2": lambda x: x[:, 0] ** 2})["path_integrals"]["s2"]
    print("obs hook:", po.shape)
