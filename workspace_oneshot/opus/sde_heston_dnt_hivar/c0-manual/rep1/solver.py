import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    lo, hi = 50.0, 150.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Andersen QE constants
    e = np.exp(-kappa * dt)
    c1 = sigma**2 * e * (1.0 - e) / kappa
    c2 = theta * sigma**2 * (1.0 - e) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    lnL, lnH = np.log(lo), np.log(hi)
    lnS0 = [np.log(s) for s in spots]

    V = np.full(num_paths, V0)
    X = np.zeros(num_paths)  # common log-return
    surv = [np.ones(num_paths) for _ in spots]

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        U = rng.random(num_paths)
        Zs = rng.standard_normal(num_paths)

        m = theta + (V - theta) * e
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(num_paths)

        quad = psi <= psi_c
        if np.any(quad):
            ps = psi[quad]
            ip = 2.0 / ps
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            ps = psi[ex]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[ex]
            u = U[ex]
            val = np.zeros(u.shape)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[ex] = val

        var_inc = np.maximum(K3 * V + K4 * Vn, 0.0)
        Xn = X + K0 + K1 * V + K2 * Vn + np.sqrt(var_inc) * Zs

        # Brownian-bridge continuous monitoring in log space
        bvar = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        for i, l0 in enumerate(lnS0):
            x0 = l0 + X
            x1 = l0 + Xn
            inside = (x1 > lnL) & (x1 < lnH) & (x0 > lnL) & (x0 < lnH)
            du0 = np.maximum(lnH - x0, 0.0)
            du1 = np.maximum(lnH - x1, 0.0)
            dl0 = np.maximum(x0 - lnL, 0.0)
            dl1 = np.maximum(x1 - lnL, 0.0)
            pu = np.exp(-2.0 * du0 * du1 / bvar)
            pl = np.exp(-2.0 * dl0 * dl1 / bvar)
            step_surv = np.clip(1.0 - pu - pl, 0.0, 1.0)
            surv[i] *= np.where(inside, step_surv, 0.0)

        X = Xn
        V = Vn

    S = 100.0 * np.exp(X)
    terminal = np.column_stack([S, V])

    functionals = {
        "no_touch_S75": float(np.mean(surv[0])),
        "no_touch_S100": float(np.mean(surv[1])),
        "no_touch_S125": float(np.mean(surv[2])),
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
