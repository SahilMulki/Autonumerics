import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    L, U_bar = 50.0, 150.0
    spots = [75.0, 100.0, 125.0]

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants
    e = np.exp(-kappa * dt)
    c1 = sigma**2 * e * (1.0 - e) / kappa
    c2 = theta * sigma**2 * (1.0 - e) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    V = np.full(num_paths, V0, dtype=float)
    x = np.zeros(num_paths)  # log(S/S0), common to all spots
    lus = [np.log(U_bar / s) for s in spots]
    lds = [np.log(L / s) for s in spots]
    weights = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uv = rng.random(num_paths)
        Zs = rng.standard_normal(num_paths)

        m = theta + (V - theta) * e
        s2 = V * c1 + c2
        psi = s2 / (m * m)

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
        Ve = np.where(Uv <= p, 0.0,
                      np.log(np.maximum((1.0 - p) / np.maximum(1.0 - Uv, 1e-300), 1.0)) / beta)
        Vn = np.where(psi <= psi_c, Vq, Ve)

        x_new = x + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs

        # Brownian-bridge crossing probability in log space
        var = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for i in range(len(spots)):
            lu, ld = lus[i], lds[i]
            du0 = lu - x
            du1 = lu - x_new
            dd0 = x - ld
            dd1 = x_new - ld
            inside = (du1 > 0.0) & (dd1 > 0.0) & (du0 > 0.0) & (dd0 > 0.0)
            pu = np.exp(-2.0 * np.maximum(du0 * du1, 0.0) / var)
            pd = np.exp(-2.0 * np.maximum(dd0 * dd1, 0.0) / var)
            surv = np.maximum(1.0 - pu, 0.0) * np.maximum(1.0 - pd, 0.0)
            weights[i] *= np.where(inside, surv, 0.0)

        x = x_new
        V = Vn

    S = 100.0 * np.exp(x)
    terminal = np.column_stack([S, V])
    functionals = {
        "no_touch_S75": float(np.mean(weights[0])),
        "no_touch_S100": float(np.mean(weights[1])),
        "no_touch_S125": float(np.mean(weights[2])),
    }
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
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
