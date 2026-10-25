import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    v0 = 0.114
    spots = (75.0, 100.0, 125.0)
    lo_b, hi_b = 50.0, 150.0

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = num_paths

    ekdt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekdt * (1.0 - ekdt) / kappa
    c2 = eta * sigma ** 2 * (1.0 - ekdt) ** 2 / (2.0 * kappa)
    sq1mr2 = np.sqrt(1.0 - rho ** 2)
    psi_c = 1.5

    V = np.full(n, v0)
    Y = np.zeros(n)  # common log-return log(S/S0)

    # log-barrier distances, relative to each spot
    lups = [np.log(hi_b / s0) for s0 in spots]
    llos = [np.log(lo_b / s0) for s0 in spots]
    surv = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        # --- Andersen QE step for variance ---
        m = eta + (V - eta) * ekdt
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Vn = np.empty(n)

        q = psi <= psi_c
        if np.any(q):
            pq = psi[q]
            ip = 2.0 / pq
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[q] / (1.0 + b2)
            Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
        e = ~q
        if np.any(e):
            pe = psi[e]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[e]
            Ue = U[e]
            out = np.zeros(Ue.shape)
            pos = Ue > p
            out[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
            Vn[e] = out

        # --- log-price step (trapezoidal integrated variance) ---
        Vbar = 0.5 * (V + Vn)
        ivar = Vbar * dt
        Zx = rng.standard_normal(n)
        Yn = (Y - 0.5 * ivar
              + (rho / sigma) * (Vn - V - kappa * (eta - Vbar) * dt)
              + sq1mr2 * np.sqrt(ivar) * Zx)

        # --- Brownian-bridge continuous barrier correction ---
        bv = np.maximum(ivar, 1e-300)
        for k in range(len(spots)):
            u = lups[k]
            l = llos[k]
            du0 = u - Y
            du1 = u - Yn
            dl0 = Y - l
            dl1 = Yn - l
            inside = (du0 > 0) & (du1 > 0) & (dl0 > 0) & (dl1 > 0)
            pu = np.exp(-2.0 * np.clip(du0 * du1, 0.0, None) / bv)
            pl = np.exp(-2.0 * np.clip(dl0 * dl1, 0.0, None) / bv)
            ps = np.clip(1.0 - pu - pl, 0.0, 1.0)
            surv[k] *= np.where(inside, ps, 0.0)

        V = Vn
        Y = Yn

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
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
