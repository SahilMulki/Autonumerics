import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    spots = [75.0, 100.0, 125.0]
    lo, hi = np.log(50.0), np.log(150.0)

    n = num_paths
    V = np.full(n, V0)
    Y = np.zeros(n)  # log(S/S0)
    surv = [np.ones(n) for _ in spots]
    logS0 = [np.log(s) for s in spots]

    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    K0 = -rho * kappa * theta * dt / sigma
    Kd = kappa * rho / sigma - 0.5
    sq1mr2 = np.sqrt(1.0 - rho ** 2)

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Zx = rng.standard_normal(n)

        # Andersen QE step for V
        m = theta + (V - theta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(n)
        q = psi <= psi_c
        if np.any(q):
            ip = 2.0 / psi[q]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
            a = m[q] / (1.0 + b2)
            Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
        e = ~q
        if np.any(e):
            pe = (psi[e] - 1.0) / (psi[e] + 1.0)
            beta = (1.0 - pe) / m[e]
            Ue = U[e]
            Vn[e] = np.where(Ue <= pe, 0.0,
                             np.log((1.0 - pe) / np.maximum(1.0 - Ue, 1e-300)) / beta)

        Iv = 0.5 * (V + Vn) * dt
        Yn = (Y + K0 + Kd * Iv + (rho / sigma) * (Vn - V)
              + sq1mr2 * np.sqrt(Iv) * Zx)

        # Brownian-bridge continuous-monitoring correction (in log space)
        bvar = np.maximum(Iv, 1e-14)
        for k, ls in enumerate(logS0):
            x0 = ls + Y
            x1 = ls + Yn
            du0 = np.maximum(hi - x0, 0.0)
            du1 = np.maximum(hi - x1, 0.0)
            dl0 = np.maximum(x0 - lo, 0.0)
            dl1 = np.maximum(x1 - lo, 0.0)
            pu = np.exp(-2.0 * du0 * du1 / bvar)
            pl = np.exp(-2.0 * dl0 * dl1 / bvar)
            surv[k] *= np.maximum(1.0 - pu - pl, 0.0)

        Y = Yn
        V = Vn

    S_T = 100.0 * np.exp(Y)
    terminal = np.column_stack([S_T, V])

    functionals = {
        "no_touch_S75": float(np.mean(surv[0])),
        "no_touch_S100": float(np.mean(surv[1])),
        "no_touch_S125": float(np.mean(surv[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(out["functionals"], out["empirical_mean"])
