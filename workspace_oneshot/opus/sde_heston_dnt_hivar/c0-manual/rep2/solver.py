import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    L, U = 50.0, 150.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants
    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = eta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * eta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    V = np.full(num_paths, V0)
    Y = np.zeros(num_paths)  # log(S/S0), common to all spots

    lbs = {s0: np.log(L / s0) for s0 in spots}
    ubs = {s0: np.log(U / s0) for s0 in spots}
    weight = {s0: np.ones(num_paths) for s0 in spots}

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uv = rng.random(num_paths)
        Zx = rng.standard_normal(num_paths)

        # --- QE step for V ---
        m = eta + (V - eta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)
        quad = psi <= psi_c
        if np.any(quad):
            pq = psi[quad]
            ip = 2.0 / pq
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            pe = psi[ex]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[ex]
            u = Uv[ex]
            ve = np.zeros_like(u)
            pos = u > p
            ve[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[ex] = ve

        # --- log-price step ---
        var_x = np.maximum(K3 * V + K4 * Vn, 0.0)
        Y1 = Y + K0 + K1 * V + K2 * Vn + np.sqrt(var_x) * Zx

        # --- continuous monitoring: Brownian bridge in log space ---
        vdt = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for s0 in spots:
            lb, ub = lbs[s0], ubs[s0]
            w = weight[s0]
            du0 = ub - Y
            du1 = ub - Y1
            dl0 = Y - lb
            dl1 = Y1 - lb
            inside = (du1 > 0) & (dl1 > 0) & (du0 > 0) & (dl0 > 0)
            pu = np.exp(-2.0 * np.maximum(du0, 0.0) * np.maximum(du1, 0.0) / vdt)
            pl = np.exp(-2.0 * np.maximum(dl0, 0.0) * np.maximum(dl1, 0.0) / vdt)
            surv = np.where(inside, np.clip(1.0 - pu - pl, 0.0, 1.0), 0.0)
            w *= surv

        Y = Y1
        V = Vn

    S = 100.0 * np.exp(Y)
    X = np.column_stack([S, V])

    functionals = {
        "no_touch_S75": float(np.mean(weight[75.0])),
        "no_touch_S100": float(np.mean(weight[100.0])),
        "no_touch_S125": float(np.mean(weight[125.0])),
    }

    return {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
