import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, eta, sigma, rho, V0 = 2.58, 0.043, 1.0, -0.36, 0.114
    spots = [75.0, 100.0, 125.0]
    L, U = 50.0, 150.0

    # Andersen QE constants
    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = eta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * eta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    n = num_paths
    V = np.full(n, V0)
    X = np.zeros(n)  # log(S/S0)
    lo = [np.log(L / s0) for s0 in spots]
    hi = [np.log(U / s0) for s0 in spots]
    W = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        Uv = rng.random(n)
        Zx = rng.standard_normal(n)

        m = eta + (V - eta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(n)

        q = psi <= psi_c
        if np.any(q):
            pq = psi[q]
            t = 2.0 / pq
            b2 = t - 1.0 + np.sqrt(t) * np.sqrt(t - 1.0)
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

        var_x = np.maximum(K3 * V + K4 * Vn, 0.0)
        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(var_x) * Zx

        # Brownian-bridge crossing probabilities within the step
        bv = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        inv = 2.0 / bv
        for k in range(len(spots)):
            du0 = hi[k] - X
            du1 = hi[k] - Xn
            dl0 = X - lo[k]
            dl1 = Xn - lo[k]
            inside = (du1 > 0) & (dl1 > 0)
            pu = np.exp(-np.maximum(du0, 0.0) * np.maximum(du1, 0.0) * inv)
            pl = np.exp(-np.maximum(dl0, 0.0) * np.maximum(dl1, 0.0) * inv)
            surv = (1.0 - pu) * (1.0 - pl)
            W[k] *= np.where(inside, surv, 0.0)

        X = Xn
        V = Vn

    S_T = 100.0 * np.exp(X)
    terminal = np.column_stack([S_T, V])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": {
            "no_touch_S75": float(np.mean(W[0])),
            "no_touch_S100": float(np.mean(W[1])),
            "no_touch_S125": float(np.mean(W[2])),
        },
    }


if __name__ == "__main__":
    r = solve_sde(100000, 0.002, 1.0)
    print(r["functionals"], r["empirical_mean"])
