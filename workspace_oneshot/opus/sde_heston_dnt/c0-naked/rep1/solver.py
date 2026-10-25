import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    L, U = 50.0, 150.0
    spots = [75.0, 100.0, 125.0]

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = int(num_paths)

    # Andersen QE constants
    ek = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ek * (1.0 - ek) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    lo = [np.log(L / s) for s in spots]
    hi = [np.log(U / s) for s in spots]

    V = np.full(n, V0)
    Y = np.zeros(n)  # log(S/S0), common to all spots
    surv = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        Uv = rng.random(n)
        Zx = rng.standard_normal(n)

        m = theta + (V - theta) * ek
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(n)

        q = psi <= psi_c
        if np.any(q):
            ip = 2.0 / psi[q]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[q] / (1.0 + b2)
            Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
        e = ~q
        if np.any(e):
            pe = psi[e]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[e]
            u = Uv[e]
            ve = np.zeros(u.shape)
            pos = u > p
            ve[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[e] = ve

        Yn = Y + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zx

        # Brownian-bridge crossing probabilities using step integrated variance
        iv = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for k in range(len(spots)):
            dl0 = Y - lo[k]
            dl1 = Yn - lo[k]
            du0 = hi[k] - Y
            du1 = hi[k] - Yn
            inside = (dl1 > 0) & (du1 > 0)
            pl = np.exp(-2.0 * np.maximum(dl0, 0.0) * np.maximum(dl1, 0.0) / iv)
            pu = np.exp(-2.0 * np.maximum(du0, 0.0) * np.maximum(du1, 0.0) / iv)
            stay = np.clip((1.0 - pl) * (1.0 - pu), 0.0, 1.0)
            surv[k] *= np.where(inside, stay, 0.0)

        Y = Yn
        V = Vn

    S100 = 100.0 * np.exp(Y)
    term = np.column_stack([S100, V])

    return {
        "terminal_paths": term,
        "empirical_mean": [float(np.mean(term[:, 0])), float(np.mean(term[:, 1]))],
        "empirical_variance": [float(np.var(term[:, 0], ddof=1)),
                               float(np.var(term[:, 1], ddof=1))],
        "functionals": {
            "no_touch_S75": float(np.mean(surv[0])),
            "no_touch_S100": float(np.mean(surv[1])),
            "no_touch_S125": float(np.mean(surv[2])),
        },
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
