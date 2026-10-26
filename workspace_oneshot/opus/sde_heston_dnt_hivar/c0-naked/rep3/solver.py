import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa = 2.58
    theta = 0.043
    sigma = 1.0
    rho = -0.36
    V0 = 0.164
    spots = [75.0, 100.0, 125.0]
    lower, upper = 50.0, 150.0
    ll, lu = np.log(lower), np.log(upper)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = num_paths

    # Andersen QE constants
    ek = np.exp(-kappa * dt)
    c1 = sigma * sigma * ek * (1.0 - ek) / kappa
    c2 = theta * sigma * sigma * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)

    V = np.full(n, V0)
    X = [np.full(n, np.log(s)) for s in spots]
    W = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Zx = rng.standard_normal(n)

        m = theta + (V - theta) * ek
        s2 = V * c1 + c2
        psi = s2 / (m * m)

        Vn = np.empty(n)
        quad = psi <= psi_c
        # quadratic branch
        pq = psi[quad]
        inv = 2.0 / pq
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        a = m[quad] / (1.0 + b2)
        Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        # exponential branch
        ex = ~quad
        pe = psi[ex]
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m[ex]
        Ue = U[ex]
        ve = np.zeros(Ue.shape)
        pos = Ue > p
        ve[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
        Vn[ex] = ve

        incr = K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zx
        var_step = np.maximum(0.5 * (V + Vn) * dt, 1e-300)

        for k in range(len(spots)):
            x0 = X[k]
            x1 = x0 + incr
            du0 = np.maximum(lu - x0, 0.0)
            du1 = np.maximum(lu - x1, 0.0)
            dl0 = np.maximum(x0 - ll, 0.0)
            dl1 = np.maximum(x1 - ll, 0.0)
            pU = np.exp(-2.0 * du0 * du1 / var_step)
            pL = np.exp(-2.0 * dl0 * dl1 / var_step)
            surv = np.clip(1.0 - pU - pL, 0.0, 1.0)
            inside = (x1 < lu) & (x1 > ll)
            W[k] = W[k] * np.where(inside, surv, 0.0)
            X[k] = x1

        V = Vn

    S100 = np.exp(X[1])
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"], res["empirical_mean"], res["empirical_variance"])
