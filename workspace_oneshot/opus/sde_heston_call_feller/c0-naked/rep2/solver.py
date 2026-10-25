import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa = 2.58
    theta = 0.043
    sigma = 1.0
    rho = -0.36
    V0 = 0.114
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # QE constants (Andersen 2008)
    ekdt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekdt * (1.0 - ekdt) / kappa
    c2 = theta * sigma**2 * (1.0 - ekdt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4  # negative here, so martingale correction is well defined

    X_all = np.empty(num_paths)  # log(S_T / S0)
    V_all = np.empty(num_paths)

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)
        for _ in range(Nt):
            m = theta + (V - theta) * ekdt
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            Vn = np.empty(n)
            logM = np.empty(n)

            q = psi <= psi_c
            e = ~q

            # quadratic regime
            if np.any(q):
                pq = psi[q]
                ip = 2.0 / pq
                b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
                a = m[q] / (1.0 + b2)
                b = np.sqrt(b2)
                Vn[q] = a * (b + Zv[q]) ** 2
                den = 1.0 - 2.0 * A * a
                logM[q] = A * b2 * a / den - 0.5 * np.log(den)

            # exponential regime
            if np.any(e):
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[e]
                Ue = U[e]
                ve = np.zeros(Ue.shape[0])
                pos = Ue > p
                ve[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
                Vn[e] = ve
                logM[e] = np.log(p + beta * (1.0 - p) / (beta - A))

            K0s = -logM - (K1 + 0.5 * K3) * V
            X += K0s + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    expX = np.exp(X_all)

    functionals = {}
    for S0 in spots:
        ST = S0 * expX
        P = np.maximum(ST - K, 0.0)
        C = ST - S0  # control variate, exact mean zero
        Cc = C - C.mean()
        varC = np.dot(Cc, Cc)
        if varC > 0:
            b = np.dot(P - P.mean(), Cc) / varC
        else:
            b = 0.0
        price = float(P.mean() - b * C.mean())
        functionals["price_S%d" % int(S0)] = price

    S100 = 100.0 * expX
    terminal = np.column_stack([S100, V_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(out["functionals"], out["empirical_mean"], out["empirical_variance"])
