import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
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

            m = theta + (V - theta) * ekt
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            quad = psi <= psi_c

            # quadratic branch
            pq = np.minimum(psi, psi_c)
            tq = 2.0 / pq
            b2 = tq - 1.0 + np.sqrt(tq) * np.sqrt(tq - 1.0)
            a = m / (1.0 + b2)
            Vq = a * (np.sqrt(b2) + Zv) ** 2

            # exponential branch
            pe = np.maximum(psi, psi_c)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ue = np.minimum(U, 1.0 - 1e-16)
            Ve = np.where(Ue <= p, 0.0,
                          np.log(np.maximum((1.0 - p) / (1.0 - Ue), 1.0)) / beta)

            Vn = np.where(quad, Vq, Ve)

            # martingale-corrected drift
            k0q = -A * b2 * a / (1.0 - 2.0 * A * a) + 0.5 * np.log(1.0 - 2.0 * A * a)
            k0e = -np.log(p + beta * (1.0 - p) / (beta - A))
            K0 = np.where(quad, k0q, k0e) - (K1 + 0.5 * K3) * V

            X += K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    eX = np.exp(X_all)
    functionals = {}
    for S0 in spots:
        ST = S0 * eX
        pay = np.maximum(ST - K, 0.0)
        cov = np.cov(pay, ST, ddof=1)
        b = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        price = pay.mean() - b * (ST.mean() - S0)
        functionals["price_S%d" % int(S0)] = float(price)

    S100 = 100.0 * eX
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
