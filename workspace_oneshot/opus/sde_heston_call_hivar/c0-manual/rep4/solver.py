import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    K_strike = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE-M scheme constants
    E = np.exp(-kappa * dt)
    c1 = sigma**2 * E * (1.0 - E) / kappa
    c2 = theta * sigma**2 * (1.0 - E) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    chunk = 250000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            m = theta + (V - theta) * E
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            quad = psi <= psi_c

            # quadratic branch
            pq = np.minimum(psi, psi_c)
            tmp = 2.0 / pq
            b2 = tmp - 1.0 + np.sqrt(tmp * (tmp - 1.0))
            a = m / (1.0 + b2)
            Vq = a * (np.sqrt(b2) + Zv) ** 2
            den = 1.0 - 2.0 * A * a
            lnMq = A * b2 * a / den - 0.5 * np.log(den)

            # exponential branch
            pe = np.maximum(psi, psi_c)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ve = np.maximum(np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)), 0.0) / beta
            lnMe = np.log(p + beta * (1.0 - p) / (beta - A))

            Vn = np.where(quad, Vq, Ve)
            lnM = np.where(quad, lnMq, lnMe)

            X += -lnM - 0.5 * K3 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn
        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    G = np.exp(X_all)  # martingale: E[G] = 1
    functionals = {}
    for S0 in spots:
        ST = S0 * G
        pay = np.maximum(ST - K_strike, 0.0)
        cST = ST - ST.mean()
        varS = np.dot(cST, cST)
        b = np.dot(cST, pay - pay.mean()) / varS if varS > 0 else 0.0
        price = pay.mean() - b * (ST.mean() - S0)
        functionals[f"price_S{int(S0)}"] = float(price)

    S_T = 100.0 * G
    terminal = np.column_stack([S_T, V_all])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(res["functionals"], res["empirical_mean"])
