import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters (r = 0)
    kappa, eta, eps, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    spots = (75.0, 100.0, 125.0)
    K_strike = 100.0

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE scheme constants
    ek = np.exp(-kappa * dt)
    c1 = eps**2 * ek * (1.0 - ek) / kappa
    c2 = eta * eps**2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / eps - 0.5) - rho / eps
    K2 = g2 * dt * (kappa * rho / eps - 0.5) + rho / eps
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4  # negative here, so moment generating function finite

    X_all = np.empty(num_paths)   # log(S_T / S0), identical for all spots
    V_all = np.empty(num_paths)

    chunk = 250000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            m = eta + (V - eta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            quad = psi <= psi_c

            # quadratic branch
            pq = np.minimum(psi, psi_c)
            inv = 2.0 / pq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m / (1.0 + b2)
            b = np.sqrt(b2)
            Vq = a * (b + Zv) ** 2
            one_m_2Aa = 1.0 - 2.0 * A * a
            lnMq = A * b2 * a / one_m_2Aa - 0.5 * np.log(one_m_2Aa)

            # exponential branch
            pe = np.maximum(psi, psi_c)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ve = np.where(U <= p, 0.0,
                          np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)) / beta)
            lnMe = np.log(p + beta * (1.0 - p) / (beta - A))

            Vn = np.where(quad, Vq, Ve)
            lnM = np.where(quad, lnMq, lnMe)

            # martingale-corrected log-asset step
            K0s = -lnM - (K1 + 0.5 * K3) * V
            X += K0s + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn
        X_all[start:start + n] = X
        V_all[start:start + n] = V

    E = np.exp(X_all)
    functionals = {}
    for S0 in spots:
        ST = S0 * E
        pay = np.maximum(ST - K_strike, 0.0)
        # control variate: E[S_T] = S0 exactly (martingale-corrected scheme)
        cov = np.cov(pay, ST)
        bcv = cov[0, 1] / cov[1, 1] if cov[1, 1] > 0 else 0.0
        price = float(np.mean(pay) - bcv * (np.mean(ST) - S0))
        functionals["price_S%d" % int(S0)] = price

    S100 = 100.0 * E
    terminal = np.column_stack([S100, V_all])

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
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
