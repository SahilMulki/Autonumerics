import numpy as np

# Heston parameters (problem_spec.json)
KAPPA, ETA, SIGMA, RHO = 2.58, 0.043, 1.0, -0.36
V0, S0 = 0.164, 100.0
B_LO, B_HI = 50.0, 150.0
SPOTS = {"no_touch_S75": 75.0, "no_touch_S100": 100.0, "no_touch_S125": 125.0}
CHUNK = 20000  # paths per block; bounds memory of the drawn increment block


def _run_block(dW, dt, observables):
    """Advance one block of paths. dW: (m, Nt, 2) independent increments (scaled by sqrt(dt))."""
    m, Nt, _ = dW.shape
    sq = np.sqrt(1.0 - RHO ** 2)
    R = np.zeros(m)                  # log return, shared by all spots
    V = np.full(m, V0)               # raw (untruncated) variance
    names = list(SPOTS)
    a = {k: np.log(B_LO / s) for k, s in SPOTS.items()}
    c = {k: np.log(B_HI / s) for k, s in SPOTS.items()}
    w = {k: np.ones(m) for k in names}
    acc = {k: np.zeros(m) for k in (observables or {})}

    def state(R_, V_):
        return np.column_stack([S0 * np.exp(R_), np.maximum(V_, 0.0)])

    X_prev = state(R, V) if observables else None
    for n in range(Nt):
        dW1 = dW[:, n, 0]
        dW2 = RHO * dW1 + sq * dW[:, n, 1]
        Vp = np.maximum(V, 0.0)
        sv = np.sqrt(Vp)
        R_new = R - 0.5 * Vp * dt + sv * dW1
        V = V + KAPPA * (ETA - Vp) * dt + SIGMA * sv * dW2
        pos = Vp > 0.0
        den = np.where(pos, Vp * dt, 1.0)
        for k in names:
            ak, ck = a[k], c[k]
            inside = (R_new > ak) & (R_new < ck)
            d_lo = np.maximum(R - ak, 0.0) * np.maximum(R_new - ak, 0.0)
            d_hi = np.maximum(ck - R, 0.0) * np.maximum(ck - R_new, 0.0)
            p_lo = np.where(pos, np.exp(-2.0 * d_lo / den), 0.0)
            p_hi = np.where(pos, np.exp(-2.0 * d_hi / den), 0.0)
            w[k] *= inside * (1.0 - p_lo) * (1.0 - p_hi)
        R = R_new
        if observables:
            X_new = state(R, V)
            for k, phi in observables.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X_new)) * dt
            X_prev = X_new
    return state(R, V), w, acc


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    given = dW is not None
    if given:
        Nt = dW.shape[1]
        dt = T / Nt
    else:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)

    X = np.empty((num_paths, 2))
    wsum = {k: 0.0 for k in SPOTS}
    pint = {k: np.empty(num_paths) for k in (observables or {})}
    for lo in range(0, num_paths, CHUNK):
        hi = min(lo + CHUNK, num_paths)
        if given:
            blk = dW[lo:hi]
        else:
            # path-major block draw, one block per chunk of paths from a single generator
            blk = np.sqrt(dt) * rng.standard_normal((hi - lo, Nt, 2))
        Xb, wb, ab = _run_block(blk, dt, observables)
        X[lo:hi] = Xb
        for k in SPOTS:
            wsum[k] += float(wb[k].sum())
        for k in pint:
            pint[k][lo:hi] = ab[k]

    out = {
        "terminal_paths": X,   # [S_T (S0=100, unconditional), max(V_T, 0)]
        "empirical_mean": [float(x) for x in np.mean(X, axis=0)],
        "empirical_variance": [float(x) for x in np.var(X, axis=0, ddof=1)],
        "functionals": {k: wsum[k] / num_paths for k in SPOTS},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = pint
    return out


if __name__ == "__main__":
    import time
    for M in (400000, 100000):
        t0 = time.time()
        r = solve_sde(M, 0.002, 1.0, seed=42)
        print(f"M={M} time={time.time()-t0:.1f}s")
        print("  mean:", r["empirical_mean"])
        print("  var: ", r["empirical_variance"])
        print("  functionals:", r["functionals"])
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 1.0,
                                 observables={"v": lambda a: a[:, 1]})["path_integrals"]["v"].shape)
