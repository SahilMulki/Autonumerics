import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt
    N = int(num_paths)

    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    lo, hi = np.log(50.0), np.log(150.0)
    spots = [75.0, 100.0, 125.0]
    lspots = [np.log(s) for s in spots]

    # QE constants
    ekd = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekd * (1.0 - ekd) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekd) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-S discretization constants (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    V = np.full(N, V0)
    Y = np.zeros(N)  # log(S/S0)
    alive = [np.ones(N, dtype=bool) for _ in spots]
    w = [np.ones(N) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(N)
        U = rng.random(N)
        Zx = rng.standard_normal(N)

        # QE step for variance
        m = theta + (V - theta) * ekd
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(N)
        quad = psi <= psi_c
        if np.any(quad):
            pq = psi[quad]
            inv = 2.0 / pq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m[quad] / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            pe = psi[ex]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[ex]
            Ue = U[ex]
            val = np.zeros(Ue.shape)
            pos = Ue > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
            Vn[ex] = val

        # log-S step
        Yn = Y + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zx

        # Brownian-bridge crossing correction, local variance = average V
        vdt = np.maximum(0.5 * (V + Vn) * dt, 1e-14)
        inv2 = 2.0 / vdt
        for k, ls in enumerate(lspots):
            x0 = ls + Y
            x1 = ls + Yn
            al = alive[k]
            al &= (x1 > lo) & (x1 < hi)
            dl0 = x0 - lo
            dl1 = x1 - lo
            du0 = hi - x0
            du1 = hi - x1
            pL = np.exp(-np.maximum(dl0 * dl1, 0.0) * inv2)
            pU = np.exp(-np.maximum(du0 * du1, 0.0) * inv2)
            surv = np.clip(1.0 - pL - pU, 0.0, 1.0)
            w[k] *= np.where(al, surv, 0.0)

        V = Vn
        Y = Yn

    funcs = {}
    names = ["no_touch_S75", "no_touch_S100", "no_touch_S125"]
    for k, nm in enumerate(names):
        funcs[nm] = float(np.mean(w[k] * alive[k]))

    S_T = 100.0 * np.exp(Y)
    term = np.column_stack([S_T, V])
    return {
        "terminal_paths": term,
        "empirical_mean": [float(x) for x in term.mean(axis=0)],
        "empirical_variance": [float(x) for x in term.var(axis=0, ddof=1)],
        "functionals": funcs,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
