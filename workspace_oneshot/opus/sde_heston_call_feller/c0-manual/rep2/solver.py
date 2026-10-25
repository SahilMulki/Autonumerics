import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters (r = 0)
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    K_strike = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants
    ek = np.exp(-kappa * dt)
    c1 = sigma**2 * ek * (1.0 - ek) / kappa
    c2 = theta * sigma**2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4  # negative here -> martingale correction always valid
    psi_c = 1.5

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    chunk = 250000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        X = np.zeros(n)
        V = np.full(n, V0)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            for _ in range(Nt):
                Zv = rng.standard_normal(n)
                U = rng.random(n)
                Zs = rng.standard_normal(n)

                m = theta + (V - theta) * ek
                s2 = V * c1 + c2
                psi = s2 / (m * m)

                # quadratic branch
                pq = np.minimum(psi, psi_c)
                t2 = 2.0 / pq
                b2 = t2 - 1.0 + np.sqrt(t2 * (t2 - 1.0))
                a = m / (1.0 + b2)
                Vq = a * (np.sqrt(b2) + Zv) ** 2
                one_m = 1.0 - 2.0 * A * a
                k0q = -A * b2 * a / one_m + 0.5 * np.log(one_m)

                # exponential branch
                pe = np.maximum(psi, psi_c)
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m
                Ve = np.where(U <= p, 0.0, np.log((1.0 - p) / (1.0 - U)) / beta)
                k0e = -np.log(p + beta * (1.0 - p) / (beta - A))

                sel = psi <= psi_c
                Vn = np.where(sel, Vq, Ve)
                k0 = np.where(sel, k0q, k0e)

                # martingale-corrected log-asset step
                X += k0 - 0.5 * K3 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
                V = Vn
        X_all[start:start + n] = X
        V_all[start:start + n] = V

    growth = np.exp(X_all)
    functionals = {}
    for S0 in spots:
        ST = S0 * growth
        pay = np.maximum(ST - K_strike, 0.0)
        # control variate: E[S_T] = S0 exactly (martingale-corrected scheme)
        dS = ST - S0
        varS = np.mean((ST - ST.mean()) ** 2)
        b = np.mean((pay - pay.mean()) * (ST - ST.mean())) / varS if varS > 0 else 0.0
        price = float(np.mean(pay) - b * np.mean(dS))
        functionals[f"price_S{int(S0)}"] = price

    S100 = 100.0 * growth
    paths = np.column_stack([S100, V_all])

    return {
        "terminal_paths": paths,
        "empirical_mean": [float(np.mean(paths[:, 0])), float(np.mean(paths[:, 1]))],
        "empirical_variance": [float(np.var(paths[:, 0], ddof=1)),
                               float(np.var(paths[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
