import numpy as np

KAPPA, ETA, SIGMA, RHO = 2.58, 0.043, 1.0, -0.36
V0, S_REF = 0.114, 100.0
SPOTS = {"no_touch_S75": 75.0, "no_touch_S100": 100.0, "no_touch_S125": 125.0}
LOWER, UPPER = 50.0, 150.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Full-truncation Euler on V, log-Euler on S, Brownian-bridge survival weights.

    dW (if given): shape (num_paths, Nt, 2), independent increments already scaled
    by sqrt(dt). Otherwise increments are drawn per step from default_rng(seed).
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sdt = np.sqrt(dt)
    rho_perp = np.sqrt(1.0 - RHO ** 2)
    lU, lL = np.log(UPPER), np.log(LOWER)

    V = np.full(num_paths, V0, dtype=float)
    y = np.zeros(num_paths)                      # ln(S / S0)
    lnS0 = {k: np.log(s) for k, s in SPOTS.items()}
    W = {k: np.ones(num_paths) for k in SPOTS}
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(y_, V_):
        return np.column_stack([S_REF * np.exp(y_), np.maximum(V_, 0.0)])

    X_prev = state(y, V) if obs else None

    for n in range(Nt):
        if dW is not None:
            z1 = dW[:, n, 0]
            z2 = dW[:, n, 1]
        else:
            z = rng.standard_normal((num_paths, 2)) * sdt
            z1, z2 = z[:, 0], z[:, 1]
        dW1 = z1
        dW2 = RHO * z1 + rho_perp * z2

        Vp = np.maximum(V, 0.0)
        sq = np.sqrt(Vp)
        y_new = y - 0.5 * Vp * dt + sq * dW1
        V = V + KAPPA * (ETA - Vp) * dt + SIGMA * sq * dW2
        s2 = Vp * dt
        pos = s2 > 0.0
        s2safe = np.where(pos, s2, 1.0)

        for k in SPOTS:
            a = lnS0[k] + y
            b = lnS0[k] + y_new
            with np.errstate(over="ignore", invalid="ignore"):
                p_up = np.exp(-2.0 * (lU - a) * (lU - b) / s2safe)
                p_dn = np.exp(-2.0 * (a - lL) * (b - lL) / s2safe)
            surv = np.where(pos, np.clip(1.0 - p_up - p_dn, 0.0, 1.0), 1.0)
            surv = np.where((b <= lL) | (b >= lU), 0.0, surv)
            W[k] *= surv
        y = y_new

        if obs:
            X_new = state(y, V)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X_new)) * dt
            X_prev = X_new

    X = state(y, V)
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(m) for m in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": {k: float(np.mean(W[k])) for k in SPOTS},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (400000, 100000):
        r = solve_sde(num_paths=M, dt=0.002, T=1.0, seed=42)
        print(M, "mean", r["empirical_mean"], "var", r["empirical_variance"])
        print("  functionals", r["functionals"])
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 1.0,
          observables={"v": lambda a: a[:, 1]})["path_integrals"]["v"].shape)
