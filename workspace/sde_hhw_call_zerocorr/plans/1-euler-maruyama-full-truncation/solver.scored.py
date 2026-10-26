import numpy as np

# Heston-Hull-White, state order (S, V, R). Full-truncation Euler-Maruyama for V,
# log-Euler for S, Euler for R, trapezoid pathwise discount integral.
KAPPA, ETA, SIGMA_V = 0.5, 0.04, 1.0
KAPPA_R, THETA_R, SIGMA_R = 0.08, 0.10, 0.09
RHO = -0.9
V0, R0 = 0.04, 0.10
K_STRIKE = 100.0
SPOTS = {"price_S75": 75.0, "price_S100": 100.0, "price_S125": 125.0}
X0_REPORT = 100.0
P0T = 0.7968400988          # E[D_T], closed form (Hull-White)
CHUNK = 50000               # fixed, independent of num_paths


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    # dW (if given): shape (num_paths, Nt, 3), independent channels (Z1, Z2, Z3),
    # already scaled by sqrt(dt). Used exactly; no Brownian randomness drawn.
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sdt = np.sqrt(dt)
    rho2 = np.sqrt(1.0 - RHO ** 2)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    term = np.empty((num_paths, 3))
    # sufficient statistics per spot: sum of [P, Y1, Y2] and sum of outer products
    s1 = {k: np.zeros(3) for k in SPOTS}
    s2 = {k: np.zeros((3, 3)) for k in SPOTS}
    sumD = 0.0
    sumD2 = 0.0

    for lo in range(0, num_paths, CHUNK):
        hi = min(lo + CHUNK, num_paths)
        n = hi - lo
        logS = np.zeros(n)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)
        if obs:
            Xp = np.column_stack([X0_REPORT * np.exp(logS), V, R])
        for t in range(Nt):
            if dW is None:
                z = rng.standard_normal((3, n)) * sdt
                z1, z2, z3 = z[0], z[1], z[2]
            else:
                z1 = dW[lo:hi, t, 0]
                z2 = dW[lo:hi, t, 1]
                z3 = dW[lo:hi, t, 2]
            dW1 = z1
            dW2 = RHO * z1 + rho2 * z2
            Vp = np.maximum(V, 0.0)
            sq = np.sqrt(Vp)
            logS = logS + (R - 0.5 * Vp) * dt + sq * dW1
            Vn = V + KAPPA * (ETA - Vp) * dt + SIGMA_V * sq * dW2
            Rn = R + KAPPA_R * (THETA_R - R) * dt + SIGMA_R * z3
            I = I + 0.5 * (R + Rn) * dt
            V, R = Vn, Rn
            if obs:
                Xc = np.column_stack([X0_REPORT * np.exp(logS), V, R])
                for k, phi in obs.items():
                    acc[k][lo:hi] += 0.5 * (phi(Xp) + phi(Xc)) * dt
                Xp = Xc
        D = np.exp(-I)
        ex = np.exp(logS)
        term[lo:hi, 0] = X0_REPORT * ex
        term[lo:hi, 1] = V            # raw full-truncation state (may be < 0)
        term[lo:hi, 2] = R
        for key, S0 in SPOTS.items():
            ST = S0 * ex
            M = np.vstack([D * np.maximum(ST - K_STRIKE, 0.0), D * ST, D])
            s1[key] += M.sum(axis=1)
            s2[key] += M @ M.T
        sumD += D.sum()
        sumD2 += (D * D).sum()

    n = float(num_paths)
    prices, plain, se_plain, se_cv, beta_d = {}, {}, {}, {}, {}
    for key, S0 in SPOTS.items():
        m = s1[key] / n
        C = (s2[key] - n * np.outer(m, m)) / (n - 1)       # sample covariance
        mu = np.array([S0, P0T])
        Ccc, Ccp = C[1:, 1:], C[1:, 0]
        beta = np.linalg.solve(Ccc, Ccp)
        est = m[0] - beta @ (m[1:] - mu)
        resid_var = max(C[0, 0] - Ccp @ beta, 0.0)
        prices[key] = float(est)
        plain[key] = float(m[0])
        se_plain[key] = float(np.sqrt(C[0, 0] / n))
        se_cv[key] = float(np.sqrt(resid_var / n))
        beta_d[key] = beta.tolist()
    meanD = sumD / n
    seD = float(np.sqrt(max(sumD2 / n - meanD ** 2, 0.0) / n))
    meanDS = {k: float(s1[k][1] / n) for k in SPOTS}

    out = {
        "terminal_paths": term,
        "empirical_mean": [float(x) for x in np.mean(term, axis=0)],
        "empirical_variance": [float(x) for x in np.var(term, axis=0, ddof=1)],
        "functionals": prices,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "diagnostics": {
            "price_plain": plain, "se_plain": se_plain, "se_cv": se_cv,
            "cv_beta": beta_d, "mean_D": float(meanD), "se_mean_D": seD,
            "mean_DS": meanDS,
            "V_column": "raw full-truncation state (may be negative)",
        },
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time
    refs = {"price_S75": 35.43789688, "price_S100": 54.72806531, "price_S125": 75.39759684}
    t0 = time.time()
    res = solve_sde(num_paths=1000000, dt=0.004, T=10.0, seed=42)
    print(f"1e6 paths: {time.time() - t0:.1f}s, dt={res['dt']}")
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    d = res["diagnostics"]
    for k, ref in refs.items():
        p = res["functionals"][k]
        print(f"{k}: CV {p:.5f} (se {d['se_cv'][k]:.5f}, rel err {p/ref-1:+.4%}) | "
              f"plain {d['price_plain'][k]:.5f} (se {d['se_plain'][k]:.5f}, rel err {d['price_plain'][k]/ref-1:+.4%})")
    print(f"mean D_T {d['mean_D']:.6f} +- {d['se_mean_D']:.6f} (exact 0.7968400988)")
    print("mean D_T*S_T:", d["mean_DS"])
    print("min V_T:", res["terminal_paths"][:, 1].min())
    t0 = time.time()
    r2 = solve_sde(num_paths=250000, dt=0.004, T=10.0, seed=42)
    print(f"2.5e5 paths: {time.time() - t0:.1f}s", r2["functionals"])

    # hook checks
    Z = np.sqrt(0.1) * np.random.default_rng(0).standard_normal((1000, 100, 3))
    print("dW hook dt:", solve_sde(1000, 0.004, 10.0, dW=Z)["dt"])
    o = solve_sde(1000, 0.1, 1.0, observables={"logS": lambda a: np.log(a[:, 0])})
    print("obs hook:", o["path_integrals"]["logS"].shape)

    # self-test: dW consumption. Same dW twice -> bitwise-equal terminal states;
    # a different dW -> different states; dt vs dt/2 on one Brownian path (CRN) agree closely.
    rngt = np.random.default_rng(7)
    Zf = np.sqrt(10.0 / 200) * rngt.standard_normal((2000, 200, 3))
    a = solve_sde(2000, 0.05, 10.0, dW=Zf)["terminal_paths"]
    b = solve_sde(2000, 0.05, 10.0, dW=Zf)["terminal_paths"]
    c = solve_sde(2000, 0.05, 10.0, dW=Zf[::-1].copy())["terminal_paths"]
    Zc = Zf.reshape(2000, 100, 2, 3).sum(axis=2)
    h = solve_sde(2000, 0.1, 10.0, dW=Zc)["terminal_paths"]
    print("dW bitwise repeat:", np.array_equal(a, b), "| differs on other dW:", not np.allclose(a, c))
    print("CRN dt vs dt/2 mean |dlogS|:", float(np.mean(np.abs(np.log(a[:, 0]) - np.log(h[:, 0])))))
