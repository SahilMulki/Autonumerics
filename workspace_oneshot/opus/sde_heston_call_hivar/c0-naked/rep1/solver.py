import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa = 2.58
    theta = 0.043
    sigma = 1.0
    rho = -0.36
    V0 = 0.164
    K_strike = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # QE constants (Andersen 2008)
    em = np.exp(-kappa * dt)
    c1 = sigma * sigma * em * (1.0 - em) / kappa
    c2 = theta * sigma * sigma * (1.0 - em) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    g1 = 0.5
    g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    chunk = 500000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Z = rng.standard_normal(n)

            m = theta + (V - theta) * em
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            Vn = np.empty(n)
            lnM = np.empty(n)

            q = psi <= psi_c
            if np.any(q):
                pq = psi[q]
                mq = m[q]
                inv = 2.0 / pq
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
                a = mq / (1.0 + b2)
                b = np.sqrt(b2)
                Vn[q] = a * (b + Zv[q]) ** 2
                den = 1.0 - 2.0 * A * a
                lnM[q] = A * b2 * a / den - 0.5 * np.log(den)

            e = ~q
            if np.any(e):
                pe = psi[e]
                me = m[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / me
                Ue = U[e]
                ve = np.zeros(Ue.shape[0])
                pos = Ue > p
                ve[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
                Vn[e] = ve
                lnM[e] = np.log(p + (1.0 - p) * beta / (beta - A))

            # martingale-corrected log-asset step
            X += -lnM - 0.5 * K3 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Z
            V = Vn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    growth = np.exp(X_all)
    functionals = {}
    for S0 in spots:
        ST = S0 * growth
        Y = np.maximum(ST - K_strike, 0.0)
        C = ST - S0  # zero-mean control variate (martingale)
        Cm = C.mean()
        Ym = Y.mean()
        varC = np.mean((C - Cm) ** 2)
        if varC > 0:
            bcv = np.mean((C - Cm) * (Y - Ym)) / varC
        else:
            bcv = 0.0
        price = Ym - bcv * Cm
        functionals["price_S%d" % int(S0)] = float(price)

    S100 = 100.0 * growth
    terminal = np.column_stack([S100, V_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(out["functionals"], out["empirical_mean"])
