import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa = 2.58
    theta = 0.043
    sigma = 1.0
    rho = -0.36
    V0 = 0.164
    L, U = 50.0, 150.0
    spots = [75.0, 100.0, 125.0]

    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    rng = np.random.default_rng(seed)
    n = num_paths

    # QE constants
    ekdt = np.exp(-kappa * dt)
    c1 = sigma * sigma * ekdt * (1.0 - ekdt) / kappa
    c2 = theta * sigma * sigma * (1.0 - ekdt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-price scheme (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)

    V = np.full(n, V0)
    Y = np.zeros(n)  # log(S/S0), common to all spots
    lx = [np.log(L / s) for s in spots]
    ux = [np.log(U / s) for s in spots]
    weights = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        Uv = rng.random(n)
        Zx = rng.standard_normal(n)

        m = theta + (V - theta) * ekdt
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
            out = np.zeros(u.shape)
            big = u > p
            out[big] = np.log((1.0 - p[big]) / (1.0 - u[big])) / beta[big]
            Vn[e] = out

        var = np.maximum(K3 * V + K4 * Vn, 0.0)
        Yn = Y + K0 + K1 * V + K2 * Vn + np.sqrt(var) * Zx

        # Brownian-bridge continuous-monitoring correction
        sb = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for k in range(3):
            w = weights[k]
            l = lx[k]
            uu = ux[k]
            inside = (Yn > l) & (Yn < uu)
            idx = np.nonzero((w > 0.0) & inside)[0]
            newv = np.zeros(n)
            if idx.size:
                y0 = Y[idx]
                y1 = Yn[idx]
                s = sb[idx]
                pl = np.exp(-2.0 * (y0 - l) * (y1 - l) / s)
                pu = np.exp(-2.0 * (uu - y0) * (uu - y1) / s)
                surv = np.clip(1.0 - pl - pu, 0.0, 1.0)
                newv[idx] = w[idx] * surv
            weights[k] = newv

        Y = Yn
        V = Vn

    S100 = 100.0 * np.exp(Y)
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(weights[0])),
        "no_touch_S100": float(np.mean(weights[1])),
        "no_touch_S125": float(np.mean(weights[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
