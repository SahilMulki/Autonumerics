import numpy as np

X_0, R, SIGMA, LAM, M_J, DELTA, K = 100.0, 0.05, 0.15, 0.1, -0.90, 0.45, 100.0
KAPPA = float(np.exp(M_J + DELTA**2 / 2) - 1.0)
MU_C = R - LAM * KAPPA


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    rng_w, rng_j = rng.spawn(2)
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * rng_w.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    # Jumps: independent of dt; mapped to steps afterwards.
    N = rng_j.poisson(LAM * T, num_paths)
    tot = int(N.sum())
    times = rng_j.uniform(0.0, T, tot)
    Z = rng_j.normal(M_J, DELTA, tot)
    pid = np.repeat(np.arange(num_paths), N)
    kstep = np.minimum((times / dt).astype(int), Nt - 1)
    order = np.argsort(kstep, kind="stable")
    kstep, pid, Y = kstep[order], pid[order], np.exp(Z[order])
    bounds = np.searchsorted(kstep, np.arange(Nt + 1))

    X = np.full(num_paths, X_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    for n in range(Nt):
        X_prev = X
        w = dW[:, n]
        # Milstein step: correction uses the pre-step state (A_n multiplies X_n)
        X = X_prev * (1.0 + MU_C * dt + SIGMA * w + 0.5 * SIGMA**2 * (w * w - dt))
        a, b = bounds[n], bounds[n + 1]
        if b > a:
            np.multiply.at(X, pid[a:b], Y[a:b])
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    disc = np.exp(-R * T)
    P = disc * np.maximum(X - K, 0.0)
    EX = X_0 * (1.0 + MU_C * dt) ** Nt * np.exp(LAM * T * KAPPA)  # exact discrete-scheme mean
    cov = np.cov(P, X)
    bcoef = cov[0, 1] / cov[1, 1]
    C = P - bcoef * (X - EX)
    sqM = np.sqrt(num_paths)
    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": float(np.mean(C))},
        "functional_se": {"call_price": float(np.std(C, ddof=1) / sqM)},
        "plain_call_price": float(np.mean(P)),
        "plain_call_se": float(np.std(P, ddof=1) / sqM),
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (400000, 100000):
        res = solve_sde(num_paths=M, dt=0.01, T=0.25, seed=42)
        print(f"M={M} mean={res['empirical_mean']:.6f} var={res['empirical_variance']:.6f}")
        print(f"  CV price={res['functionals']['call_price']:.6f} se={res['functional_se']['call_price']:.6f}"
              f" | plain={res['plain_call_price']:.6f} se={res['plain_call_se']:.6f}"
              f" | se ratio={res['functional_se']['call_price']/res['plain_call_se']:.4f}")
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.01, 0.25, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 0.25,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
