import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    spots = [75.0, 100.0, 125.0]
    L, U_b = 50.0, 150.0

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    N = num_paths

    # Andersen QE constants
    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    V = np.full(N, V0)
    X = np.zeros(N)  # log(S/S0), shared across spots
    lo = [np.log(L / s) for s in spots]
    hi = [np.log(U_b / s) for s in spots]
    surv = [np.ones(N) for _ in spots]

    for _ in range(Nt):
        Zs = rng.standard_normal((2, N))
        Uu = rng.random(N)
        Zv = Zs[0]
        Zx = Zs[1]

        m = theta + (V - theta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(N)

        quad = psi <= psi_c
        if np.any(quad):
            ps = psi[quad]
            inv = 2.0 / ps
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        expo = ~quad
        if np.any(expo):
            ps = psi[expo]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[expo]
            u = Uu[expo]
            val = np.zeros_like(u)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[expo] = val

        var_x = np.maximum(K3 * V + K4 * Vn, 0.0)
        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(var_x) * Zx

        # Brownian-bridge continuous monitoring
        bvar = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for k in range(len(spots)):
            l, h = lo[k], hi[k]
            inside = (Xn > l) & (Xn < h) & (X > l) & (X < h)
            du = np.where(inside, (h - X) * (h - Xn), 0.0)
            dl = np.where(inside, (X - l) * (Xn - l), 0.0)
            pu = np.exp(-2.0 * du / bvar)
            pl = np.exp(-2.0 * dl / bvar)
            step_s = np.where(inside, np.clip((1.0 - pu) * (1.0 - pl), 0.0, 1.0), 0.0)
            surv[k] *= step_s

        X = Xn
        V = Vn

    S_T = 100.0 * np.exp(X)
    terminal = np.column_stack([S_T, V])

    functionals = {
        "no_touch_S75": float(np.mean(surv[0])),
        "no_touch_S100": float(np.mean(surv[1])),
        "no_touch_S125": float(np.mean(surv[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
