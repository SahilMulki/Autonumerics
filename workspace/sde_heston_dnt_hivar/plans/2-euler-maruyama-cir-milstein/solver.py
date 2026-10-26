import numpy as np

# Heston parameters (zero rate)
KAPPA, ETA, SIGMA, RHO = 2.58, 0.043, 1.0, -0.36
V0, S0_TERM = 0.164, 100.0
B_LO, B_HI = 50.0, 150.0
SPOTS = {"no_touch_S75": 75.0, "no_touch_S100": 100.0, "no_touch_S125": 125.0}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Log-Euler S, full-truncation + scalar CIR-Milstein V, Brownian-bridge
    conditional survival weights for continuous double-barrier monitoring.

    dW (if given): independent increments, shape (num_paths, Nt, 2), already
    scaled by sqrt(dt); channel 0 -> W1, channel 1 -> Z (correlated inside).
    Otherwise Z is streamed per step from default_rng(seed).
    """
    M = num_paths
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        sdt = np.sqrt(dt)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    rho_c = np.sqrt(1.0 - RHO ** 2)
    obs = observables or {}
    acc = {k: np.zeros(M) for k in obs}

    R = np.zeros(M)               # log return, ln(S/S0)
    V = np.full(M, V0)            # raw (possibly negative) variance
    w = {}
    ab = {}
    for k, s0 in SPOTS.items():
        w[k] = np.ones(M)
        ab[k] = (np.log(B_LO / s0), np.log(B_HI / s0))

    def state(R_, V_):
        return np.stack([S0_TERM * np.exp(R_), np.maximum(V_, 0.0)], axis=1)

    X_prev = state(R, V) if obs else None

    for n in range(Nt):
        if dW is None:
            Z = rng.standard_normal((M, 2))
            dW1 = sdt * Z[:, 0]
            dW2 = RHO * dW1 + rho_c * sdt * Z[:, 1]
        else:
            dW1 = dW[:, n, 0]
            dW2 = RHO * dW1 + rho_c * dW[:, n, 1]

        Vp = np.maximum(V, 0.0)
        sq = np.sqrt(Vp)
        R_new = R - 0.5 * Vp * dt + sq * dW1
        # full truncation + scalar CIR Milstein, all with step-n quantities
        V_new = (V + KAPPA * (ETA - Vp) * dt + SIGMA * sq * dW2
                 + 0.25 * SIGMA ** 2 * (dW2 ** 2 - dt))

        vdt = Vp * dt
        pos = vdt > 0.0
        safe = np.where(pos, vdt, 1.0)
        for k in w:
            a, c = ab[k]
            inside = (R_new > a) & (R_new < c)
            p_lo = np.exp(-2.0 * (R - a) * (R_new - a) / safe)
            p_hi = np.exp(-2.0 * (c - R) * (c - R_new) / safe)
            f = np.where(pos, (1.0 - p_lo) * (1.0 - p_hi), 1.0)
            w[k] *= np.where(inside, f, 0.0)

        R, V = R_new, V_new
        if obs:
            X = state(R, V)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt
            X_prev = X

    term = state(R, V)
    out = {
        "terminal_paths": term,
        "empirical_mean": [float(x) for x in np.mean(term, axis=0)],
        "empirical_variance": [float(x) for x in np.var(term, axis=0, ddof=1)],
        "functionals": {k: float(np.mean(w[k])) for k in w},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "frac_negative_V_raw": float(np.mean(V < 0.0)),
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (400000, 100000):
        r = solve_sde(M, 0.002, 1.0, seed=42)
        print(f"M={M} mean={r['empirical_mean']} var={r['empirical_variance']}")
        print("  functionals:", r["functionals"], "frac V<0:", r["frac_negative_V_raw"])
    Nt = 100
    dWt = np.sqrt(1.0 / Nt) * np.random.default_rng(0).standard_normal((1000, Nt, 2))
    print("dW hook dt:", solve_sde(1000, 0.002, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.002, 1.0,
                                 observables={"v": lambda X: X[:, 1]})["path_integrals"]["v"].shape)
