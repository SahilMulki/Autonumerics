import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa = 2.58
    theta = 0.043
    sigma = 1.0
    rho = -0.36
    v0 = 0.114
    spots = (75.0, 100.0, 125.0)
    strike = 100.0

    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # Andersen QE scheme with martingale correction
    psic = 1.5
    g1 = 0.5
    g2 = 0.5
    E = np.exp(-kappa * dt)
    c1 = sigma * sigma * E * (1.0 - E) / kappa
    c2 = theta * sigma * sigma * (1.0 - E) ** 2 / (2.0 * kappa)
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4

    chunk = 250000
    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, v0)
        lnX = np.zeros(n)
        for _ in range(Nt):
            m = theta + (V - theta) * E
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Zv = rng.standard_normal(n)
            Zx = rng.standard_normal(n)

            quad = psi <= psic
            Vn = np.empty(n)
            lnM = np.empty(n)

            # quadratic branch
            iq = np.nonzero(quad)[0]
            if iq.size:
                mq = m[iq]
                pq = psi[iq]
                tq = 2.0 / pq
                b2 = tq - 1.0 + np.sqrt(tq) * np.sqrt(tq - 1.0)
                a = mq / (1.0 + b2)
                Vn[iq] = a * (np.sqrt(b2) + Zv[iq]) ** 2
                den = 1.0 - 2.0 * A * a
                lnM[iq] = A * b2 * a / den - 0.5 * np.log(den)

            # exponential branch
            ie = np.nonzero(~quad)[0]
            if ie.size:
                me = m[ie]
                pe = psi[ie]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / me
                U = rng.random(ie.size)
                ve = np.zeros(ie.size)
                pos = U > p
                ve[pos] = np.log((1.0 - p[pos]) / (1.0 - U[pos])) / beta[pos]
                Vn[ie] = ve
                lnM[ie] = np.log(p + beta * (1.0 - p) / (beta - A))

            K0s = -lnM - (K1 + 0.5 * K3) * V
            lnX += K0s + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zx
            V = Vn
        X_all[start:start + n] = np.exp(lnX)
        V_all[start:start + n] = V
        start += n

    # prices with control variate S(T) (E[S(T)] = S0 exactly)
    functionals = {}
    Xc = X_all - 1.0
    varX = np.var(X_all, ddof=1)
    for S0 in spots:
        ST = S0 * X_all
        pay = np.maximum(ST - strike, 0.0)
        ctrl = S0 * Xc
        varc = S0 * S0 * varX
        if varc > 0:
            cov = np.mean((pay - pay.mean()) * (ctrl - ctrl.mean())) * num_paths / max(num_paths - 1, 1)
            b = cov / varc
        else:
            b = 0.0
        price = float(np.mean(pay - b * ctrl))
        functionals["price_S%d" % int(S0)] = price

    S_T = 100.0 * X_all
    terminal = np.column_stack([S_T, V_all])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(res["functionals"], res["empirical_mean"])
