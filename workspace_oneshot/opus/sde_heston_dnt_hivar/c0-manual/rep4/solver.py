import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters (r = 0)
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    spots = (75.0, 100.0, 125.0)
    L, U = 50.0, 150.0

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants for variance
    ekdt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekdt * (1.0 - ekdt) / kappa
    c2 = theta * sigma**2 * (1.0 - ekdt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-S discretization constants (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    # X = log(S / S0), common to all spots
    X = np.zeros(num_paths)
    V = np.full(num_paths, V0)

    logU = [np.log(U / s0) for s0 in spots]
    logL = [np.log(L / s0) for s0 in spots]
    surv = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uv = rng.random(num_paths)
        Zs = rng.standard_normal(num_paths)

        # --- QE step for V ---
        m = theta + (V - theta) * ekdt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)

        quad = psi <= psi_c
        if np.any(quad):
            pq = psi[quad]
            inv = 2.0 / pq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            pe = psi[ex]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[ex]
            u = Uv[ex]
            val = np.zeros(u.shape)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[ex] = val

        # --- log S step ---
        var_s = np.maximum(K3 * V + K4 * Vn, 0.0)
        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(var_s) * Zs

        # --- Brownian-bridge continuous barrier monitoring ---
        qv = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for i in range(len(spots)):
            su = surv[i]
            lu, ld = logU[i], logL[i]
            du0 = lu - X
            du1 = lu - Xn
            dd0 = X - ld
            dd1 = Xn - ld
            inside = (du1 > 0.0) & (dd1 > 0.0)
            pu = np.exp(-2.0 * np.maximum(du0, 0.0) * np.maximum(du1, 0.0) / qv)
            pd = np.exp(-2.0 * np.maximum(dd0, 0.0) * np.maximum(dd1, 0.0) / qv)
            fac = np.where(inside, np.maximum(1.0 - pu - pd, 0.0), 0.0)
            su *= fac

        X = Xn
        V = Vn

    S100 = 100.0 * np.exp(X)
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(surv[0])),
        "no_touch_S100": float(np.mean(surv[1])),
        "no_touch_S125": float(np.mean(surv[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in np.mean(terminal, axis=0)],
        "empirical_variance": [float(v) for v in np.var(terminal, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
