import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    L, U_b = 50.0, 150.0
    spots = [75.0, 100.0, 125.0]

    # QE (Andersen) constants for the variance
    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-price discretisation (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    V = np.full(num_paths, V0)
    Y = np.zeros(num_paths)  # log(S/S0), common to all spots
    bu = [np.log(U_b / s) for s in spots]
    bd = [np.log(L / s) for s in spots]
    w = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uu = rng.random(num_paths)
        Z = rng.standard_normal(num_paths)

        # --- QE step for V ---
        m = theta + (V - theta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)

        lo = psi <= psi_c
        if np.any(lo):
            ml = m[lo]
            ip = 2.0 / psi[lo]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
            a = ml / (1.0 + b2)
            Vn[lo] = a * (np.sqrt(b2) + Zv[lo]) ** 2
        hi = ~lo
        if np.any(hi):
            ph = psi[hi]
            p = (ph - 1.0) / (ph + 1.0)
            beta = (1.0 - p) / m[hi]
            uh = Uu[hi]
            vh = np.zeros(uh.shape)
            pos = uh > p
            vh[pos] = np.log((1.0 - p[pos]) / (1.0 - uh[pos])) / beta[pos]
            Vn[hi] = vh

        # --- log-price step ---
        Yn = Y + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Z

        # --- continuous-monitoring Brownian-bridge survival ---
        v = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        with np.errstate(over="ignore", under="ignore", invalid="ignore", divide="ignore"):
            for i in range(len(spots)):
                du0 = bu[i] - Y
                du1 = bu[i] - Yn
                dd0 = Y - bd[i]
                dd1 = Yn - bd[i]
                pu = np.exp(-2.0 * np.maximum(du0 * du1, 0.0) / v)
                pd = np.exp(-2.0 * np.maximum(dd0 * dd1, 0.0) / v)
                surv = (1.0 - pu) * (1.0 - pd)
                surv = np.where((du1 <= 0.0) | (dd1 <= 0.0), 0.0, surv)
                w[i] *= np.clip(surv, 0.0, 1.0)

        V = Vn
        Y = Yn

    S = 100.0 * np.exp(Y)
    X = np.column_stack([S, V])

    functionals = {
        "no_touch_S75": float(np.mean(w[0])),
        "no_touch_S100": float(np.mean(w[1])),
        "no_touch_S125": float(np.mean(w[2])),
    }

    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)), float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
