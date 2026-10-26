import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    spots = [75.0, 100.0, 125.0]
    lo, hi = np.log(50.0), np.log(150.0)

    n = num_paths
    V = np.full(n, V0)
    X = np.zeros(n)  # common log-increment: ln S = ln S0 + X

    # QE constants
    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-S coefficients (r=0, gamma1=gamma2=0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    logS0 = [np.log(s) for s in spots]
    weights = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Zs = rng.standard_normal(n)

        # QE step for V
        m = theta + (V - theta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(n)
        quad = psi <= psi_c
        if np.any(quad):
            ps = psi[quad]
            inv = 2.0 / ps
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        expo = ~quad
        if np.any(expo):
            ps = psi[expo]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[expo]
            u = U[expo]
            val = np.zeros(u.shape)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[expo] = val

        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs

        # Brownian-bridge crossing probabilities in log space
        w = 0.5 * (V + Vn) * dt
        w = np.maximum(w, 1e-300)
        for k in range(3):
            x0 = logS0[k] + X
            x1 = logS0[k] + Xn
            inside = (x1 > lo) & (x1 < hi)
            du0 = hi - x0
            du1 = hi - x1
            dd0 = x0 - lo
            dd1 = x1 - lo
            au = np.clip(2.0 * du0 * du1 / w, 0.0, 700.0)
            ad = np.clip(2.0 * dd0 * dd1 / w, 0.0, 700.0)
            pu = np.exp(-au)
            pd = np.exp(-ad)
            surv = np.where(inside, np.clip((1.0 - pu) * (1.0 - pd), 0.0, 1.0), 0.0)
            weights[k] *= surv

        X = Xn
        V = Vn

    S_T = 100.0 * np.exp(X)
    terminal = np.column_stack([S_T, V])

    functionals = {
        "no_touch_S75": float(np.mean(weights[0])),
        "no_touch_S100": float(np.mean(weights[1])),
        "no_touch_S125": float(np.mean(weights[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
