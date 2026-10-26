import numpy as np


def _hw_bond(r0, a, theta, sig, T):
    B = (1.0 - np.exp(-a * T)) / a
    lnA = (theta - sig**2 / (2 * a**2)) * (B - T) - sig**2 * B**2 / (4 * a)
    return float(np.exp(lnA - B * r0))


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # parameters
    kap, th, sv, rho = 0.5, 0.04, 1.0, -0.9
    V0 = 0.04
    a, rth, rs, R0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    # QE constants (Andersen)
    ek = np.exp(-kap * dt)
    c_s2a = sv**2 * ek * (1 - ek) / kap
    c_s2b = th * sv**2 / (2 * kap) * (1 - ek) ** 2
    g1 = g2 = 0.5
    K0 = -rho * kap * th / sv * dt
    K1 = g1 * dt * (kap * rho / sv - 0.5) - rho / sv
    K2 = g2 * dt * (kap * rho / sv - 0.5) + rho / sv
    K3 = g1 * dt * (1 - rho**2)
    K4 = g2 * dt * (1 - rho**2)

    # exact OU for R
    ea = np.exp(-a * dt)
    rstd = rs * np.sqrt((1 - ea**2) / (2 * a))

    X_all = np.empty(num_paths)   # log(S_T/S0)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)   # integral of R

    chunk = 100000
    for s in range(0, num_paths, chunk):
        n = min(chunk, num_paths - s)
        X = np.zeros(n)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Z1 = rng.standard_normal(n)
            Z3 = rng.standard_normal(n)

            # variance: QE step
            m = th + (V - th) * ek
            s2 = V * c_s2a + c_s2b
            psi = s2 / (m * m)
            quad = psi <= 1.5
            pq = np.minimum(psi, 1.5)
            ip = 2.0 / pq
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            aq = m / (1.0 + b2)
            Vq = aq * (np.sqrt(b2) + Zv) ** 2
            pe = np.maximum(psi, 1.5)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ve = np.where(U <= p, 0.0,
                          np.log(np.maximum((1.0 - p) / (1.0 - U), 1.0)) / beta)
            Vn = np.where(quad, Vq, Ve)

            # rate: exact OU
            Rn = rth + (R - rth) * ea + rstd * Z3
            dI = 0.5 * (R + Rn) * dt

            # log-asset
            X += dI + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Z1
            I += dI
            V = Vn
            R = Rn
        X_all[s:s + n] = X
        V_all[s:s + n] = V
        R_all[s:s + n] = R
        I_all[s:s + n] = I

    D = np.exp(-I_all)
    P = _hw_bond(R0, a, rth, rs, T)
    eX = np.exp(X_all)

    functionals = {}
    for S0 in spots:
        ST = S0 * eX
        y = D * np.maximum(ST - K, 0.0)
        c1 = D * ST - S0
        c2 = D - P
        C = np.vstack([c1, c2])
        Cm = C.mean(axis=1)
        Cc = C - Cm[:, None]
        yc = y - y.mean()
        cov = Cc @ Cc.T / (num_paths - 1)
        cy = Cc @ yc / (num_paths - 1)
        try:
            b = np.linalg.solve(cov, cy)
        except np.linalg.LinAlgError:
            b = np.zeros(2)
        est = float(y.mean() - b @ Cm)
        functionals["price_S%d" % int(S0)] = est

    S100 = 100.0 * eX
    term = np.column_stack([S100, V_all, R_all])
    return {
        "terminal_paths": term,
        "empirical_mean": [float(v) for v in term.mean(axis=0)],
        "empirical_variance": [float(v) for v in term.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
