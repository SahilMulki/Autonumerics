import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    K_strike = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-asset discretisation (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    chunk = 250000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)
        for _ in range(Nt):
            m = theta + (V - theta) * ekt
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            # quadratic branch
            psi_q = np.minimum(psi, psi_c)
            inv = 2.0 / psi_q
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m / (1.0 + b2)
            Vq = a * (np.sqrt(b2) + Zv) ** 2

            # exponential branch
            psi_e = np.maximum(psi, psi_c)
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - p) / m
            Ve = np.where(U <= p, 0.0,
                          np.log(np.maximum((1.0 - p) / np.maximum(1.0 - U, 1e-300), 1.0)) / beta)

            Vn = np.where(psi <= psi_c, Vq, Ve)

            X += K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    G = np.exp(X_all)
    functionals = {}
    for s0 in spots:
        price = float(np.mean(np.maximum(s0 * G - K_strike, 0.0)))
        functionals["price_S%d" % int(s0)] = price

    S100 = 100.0 * G
    terminal = np.column_stack([S100, V_all])

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
