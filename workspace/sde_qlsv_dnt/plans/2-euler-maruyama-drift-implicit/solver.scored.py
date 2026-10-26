import numpy as np

# Parameters (problem_spec.json)
C_LV = 0.01
KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.114
LOW = 50.0
UP = 150.0
SPOTS = (75.0, 100.0, 125.0)
NAMES = ("no_touch_S75", "no_touch_S100", "no_touch_S125")
MID = 1  # index of S0 = 100 in SPOTS
CHUNK = 25000


def _run_chunk(dZ, dt, observables):
    m, Nt, _ = dZ.shape
    a = 1.0 / (C_LV * UP)   # R lower barrier  (S upper)
    b = 1.0 / (C_LV * LOW)  # R upper barrier  (S lower)
    R = np.empty((3, m))
    for i, s0 in enumerate(SPOTS):
        R[i] = 1.0 / (C_LV * s0)
    V = np.full(m, V0)
    w = np.ones((3, m))
    rho_c = np.sqrt(1.0 - RHO ** 2)
    obs = observables or {}
    acc = {k: np.zeros(m) for k in obs}
    if obs:
        X = np.column_stack([1.0 / (C_LV * R[MID]), V])
    for n in range(Nt):
        dz1 = dZ[:, n, 0]
        dz2 = dZ[:, n, 1]
        Vp = np.maximum(V, 0.0)
        s = np.sqrt(Vp)
        tau = Vp * dt
        # full-truncation Euler for V
        Vn = V + KAPPA * (ETA - Vp) * dt + SIGMA * s * (RHO * dz1 + rho_c * dz2)
        # drift-implicit step for R = 1/(c S): R' = R - s dz1 + tau / R'
        Y = R - s * dz1                      # (3, m)
        root = np.sqrt(Y * Y + 4.0 * tau)
        neg = Y < 0.0
        den = np.where(neg, root - Y, 1.0)
        den = np.where(den > 0.0, den, 1.0)
        Rn = np.where(neg, 2.0 * tau / den, 0.5 * (Y + root))
        Rn = np.maximum(Rn, 1e-300)
        # continuous-monitoring Brownian-bridge survival weight
        inside = (Rn > a) & (Rn < b) & (R > a) & (R < b)
        tau_safe = np.where(tau > 0.0, tau, 1.0)
        e1 = (R - a) * (Rn - a)
        e2 = (b - R) * (b - Rn)
        f1 = 1.0 - np.exp(-2.0 * np.maximum(e1, 0.0) / tau_safe)
        f2 = 1.0 - np.exp(-2.0 * np.maximum(e2, 0.0) / tau_safe)
        fac = np.where(tau > 0.0, f1 * f2, 1.0)
        w = w * np.where(inside, fac, 0.0)
        if obs:
            Xn = np.column_stack([1.0 / (C_LV * Rn[MID]), Vn])
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X) + phi(Xn)) * dt
            X = Xn
        R = Rn
        V = Vn
    S_T = 1.0 / (C_LV * R[MID])
    return w, np.column_stack([S_T, V]), acc


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        gen = True
    else:
        dW = np.asarray(dW, dtype=float)
        Nt = dW.shape[1]
        dt = T / Nt
        gen = False
        num_paths = dW.shape[0]

    W = np.empty((3, num_paths))
    term = np.empty((num_paths, 2))
    acc = {k: np.empty(num_paths) for k in (observables or {})}
    # sequential chunked draws reproduce the single path-major block exactly
    for lo in range(0, num_paths, CHUNK):
        hi = min(num_paths, lo + CHUNK)
        if gen:
            dZ = np.sqrt(dt) * rng.standard_normal((hi - lo, Nt, 2))
        else:
            dZ = dW[lo:hi]
        w, tm, ac = _run_chunk(dZ, dt, observables)
        W[:, lo:hi] = w
        term[lo:hi] = tm
        for k in acc:
            acc[k][lo:hi] = ac[k]

    out = {
        "terminal_paths": term,
        "empirical_mean": [float(x) for x in np.mean(term, axis=0)],
        "empirical_variance": [float(x) for x in np.var(term, axis=0, ddof=1)],
        "functionals": {nm: float(np.mean(W[i])) for i, nm in enumerate(NAMES)},
        "functionals_se": {nm: float(np.std(W[i], ddof=1) / np.sqrt(num_paths))
                           for i, nm in enumerate(NAMES)},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = solve_sde(400000, 0.002, 1.0, seed=42)
    print("time %.1fs" % (time.time() - t0))
    print("Empirical mean:     ", r["empirical_mean"])
    print("Empirical variance: ", r["empirical_variance"])
    for k in r["functionals"]:
        print(k, r["functionals"][k], "se", r["functionals_se"][k])
    print("E[V_T] (exact 0.048380):", r["empirical_mean"][1])
    print("min S_T:", r["terminal_paths"][:, 0].min())
    r1 = solve_sde(100000, 0.002, 1.0, seed=42)
    for k in r["functionals"]:
        d = abs(r["functionals"][k] - r1["functionals"][k])
        bar = 2 * np.hypot(r["functionals_se"][k], r1["functionals_se"][k])
        print("100k", k, r1["functionals"][k], "diff", d, "bar", bar)
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dWt)["dt"])
    po = solve_sde(1000, 0.01, 1.0,
                   observables={"tanhS": lambda x: np.tanh(x[:, 0] / 100.0)})
    print("obs hook:", po["path_integrals"]["tanhS"].shape)
