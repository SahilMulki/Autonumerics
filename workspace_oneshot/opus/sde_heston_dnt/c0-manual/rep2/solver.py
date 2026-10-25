import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    spots = (75.0, 100.0, 125.0)
    L, U_b = 50.0, 150.0

    # Andersen QE constants
    ek = np.exp(-kappa * dt)
    c1 = sigma**2 * ek * (1.0 - ek) / kappa
    c2 = theta * sigma**2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    X = np.zeros(num_paths)          # log(S/S0), same for every spot
    V = np.full(num_paths, V0)
    lo = [np.log(L / s) for s in spots]
    hi = [np.log(U_b / s) for s in spots]
    W = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uu = rng.random(num_paths)
        Zx = rng.standard_normal(num_paths)

        # --- QE step for V ---
        m = theta + (V - theta) * ek
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)
        quad = psi <= psi_c
        if np.any(quad):
            ps = psi[quad]
            inv = 2.0 / ps
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            ps = psi[ex]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[ex]
            u = Uu[ex]
            val = np.zeros_like(u)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[ex] = val

        # --- log S step ---
        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zx

        # --- Brownian-bridge continuous barrier correction ---
        bv = 0.5 * (V + Vn) * dt
        bv_safe = np.maximum(bv, 1e-300)
        for i in range(len(spots)):
            h, l = hi[i], lo[i]
            inside = (Xn > l) & (Xn < h) & (X > l) & (X < h)
            du = (h - X) * (h - Xn)
            dd = (X - l) * (Xn - l)
            pu = np.exp(-2.0 * np.maximum(du, 0.0) / bv_safe)
            pd = np.exp(-2.0 * np.maximum(dd, 0.0) / bv_safe)
            surv = np.where(inside, np.clip((1.0 - pu) * (1.0 - pd), 0.0, 1.0), 0.0)
            W[i] *= surv

        X, V = Xn, Vn

    S100 = 100.0 * np.exp(X)
    terminal = np.column_stack([S100, V])
    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
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
