import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    L, U = 50.0, 150.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants
    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    V = np.full(num_paths, V0)
    X = np.zeros(num_paths)  # log(S/S0)
    logs0 = [np.log(s) for s in spots]
    lnU, lnL = np.log(U), np.log(L)
    w = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uu = rng.random(num_paths)
        Zs = rng.standard_normal(num_paths)

        m = theta + (V - theta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)

        lo = psi <= psi_c
        if np.any(lo):
            pl = psi[lo]
            ip = 2.0 / pl
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[lo] / (1.0 + b2)
            Vn[lo] = a * (np.sqrt(b2) + Zv[lo]) ** 2
        hi = ~lo
        if np.any(hi):
            ph = psi[hi]
            p = (ph - 1.0) / (ph + 1.0)
            beta = (1.0 - p) / m[hi]
            uh = Uu[hi]
            vh = np.zeros(ph.shape)
            pos = uh > p
            vh[pos] = np.log((1.0 - p[pos]) / (1.0 - uh[pos])) / beta[pos]
            Vn[hi] = vh

        X_prev = X
        X = X + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs

        # Brownian-bridge crossing probability in log space
        bv = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for i, ls in enumerate(logs0):
            x0 = X_prev + ls
            x1 = X + ls
            du0 = lnU - x0
            du1 = lnU - x1
            dl0 = x0 - lnL
            dl1 = x1 - lnL
            inside = (du1 > 0) & (dl1 > 0)
            pu = np.exp(-2.0 * np.maximum(du0, 0.0) * np.maximum(du1, 0.0) / bv)
            pd = np.exp(-2.0 * np.maximum(dl0, 0.0) * np.maximum(dl1, 0.0) / bv)
            surv = np.where(inside, np.clip((1.0 - pu) * (1.0 - pd), 0.0, 1.0), 0.0)
            w[i] *= surv

        V = Vn

    S = 100.0 * np.exp(X)
    terminal = np.column_stack([S, V])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": {
            "no_touch_S75": float(np.mean(w[0])),
            "no_touch_S100": float(np.mean(w[1])),
            "no_touch_S125": float(np.mean(w[2])),
        },
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1.0)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
