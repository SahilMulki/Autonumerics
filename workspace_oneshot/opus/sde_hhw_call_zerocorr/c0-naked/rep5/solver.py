import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt
    n = int(num_paths)

    # Heston parameters
    kappa, theta, sigma, rho = 0.5, 0.04, 1.0, -0.9
    V0 = 0.04
    # Hull-White parameters
    a_r, b_r, s_r, r0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    # ---- Variance & discounted log-asset: Andersen QE with martingale correction ----
    ed = np.exp(-kappa * dt)
    c1 = sigma**2 * ed * (1.0 - ed) / kappa
    c2 = theta * sigma**2 * (1.0 - ed) ** 2 / (2.0 * kappa)
    g = 0.5
    K2 = g * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g * dt * (1.0 - rho**2)
    K4 = g * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4
    psi_c = 1.5

    V = np.full(n, V0)
    X = np.zeros(n)  # log of discounted S / S0 (independent of the rate)
    Vn = np.empty(n)
    K0 = np.empty(n)

    for _ in range(Nt):
        m = theta + (V - theta) * ed
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        q = psi <= psi_c
        e = ~q

        # quadratic branch
        mq = m[q]
        ip = 1.0 / psi[q]
        tmp = 2.0 * ip
        b2 = tmp - 1.0 + np.sqrt(tmp) * np.sqrt(np.maximum(tmp - 1.0, 0.0))
        aq = mq / (1.0 + b2)
        zv = rng.standard_normal(mq.size)
        Vn[q] = aq * (np.sqrt(b2) + zv) ** 2
        den = 1.0 - 2.0 * A * aq
        K0[q] = -A * b2 * aq / den + 0.5 * np.log(den)

        # exponential branch
        me = m[e]
        pe = psi[e]
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / me
        U = rng.random(me.size)
        ve = np.zeros(me.size)
        hit = U > p
        ve[hit] = np.log((1.0 - p[hit]) / (1.0 - U[hit])) / beta[hit]
        Vn[e] = ve
        K0[e] = -np.log(p + beta * (1.0 - p) / (beta - A))

        zs = rng.standard_normal(n)
        X += K0 - 0.5 * K3 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * zs
        V, Vn = Vn, V

    # ---- Hull-White: exact joint sampling of (R_T, integral of R) ----
    eT = np.exp(-a_r * T)
    e2T = np.exp(-2.0 * a_r * T)
    mR = b_r + (r0 - b_r) * eT
    varR = s_r**2 * (1.0 - e2T) / (2.0 * a_r)
    mI = b_r * T + (r0 - b_r) * (1.0 - eT) / a_r
    varI = s_r**2 / a_r**2 * (T - 2.0 * (1.0 - eT) / a_r + (1.0 - e2T) / (2.0 * a_r))
    covRI = s_r**2 / a_r * ((1.0 - eT) / a_r - (1.0 - e2T) / (2.0 * a_r))
    sR = np.sqrt(varR)
    z1 = rng.standard_normal(n)
    z2 = rng.standard_normal(n)
    R_T = mR + sR * z1
    I = mI + (covRI / sR) * z1 + np.sqrt(max(varI - covRI**2 / varR, 0.0)) * z2
    D = np.exp(-I)
    P0 = np.exp(-mI + 0.5 * varI)

    eX = np.exp(X)

    # ---- Pricing with control variates (discounted S martingale, discount bond) ----
    functionals = {}
    for S0 in spots:
        DS = S0 * eX
        Y = np.maximum(DS - K * D, 0.0)
        C = np.vstack([DS - S0, D - P0])
        Cm = C.mean(axis=1)
        Cc = C - Cm[:, None]
        Yc = Y - Y.mean()
        cov_cc = Cc @ Cc.T / (n - 1)
        cov_cy = Cc @ Yc / (n - 1)
        try:
            bvec = np.linalg.solve(cov_cc, cov_cy)
        except np.linalg.LinAlgError:
            bvec = np.zeros(2)
        price = float(Y.mean() - bvec @ Cm)
        functionals["price_S%d" % int(S0)] = price

    S_T = 100.0 * eX / D
    terminal = np.column_stack([S_T, V, R_T])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["functionals"])
    print(out["empirical_mean"])
