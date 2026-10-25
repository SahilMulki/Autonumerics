import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters
    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    lower, upper = 50.0, 150.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    N = int(num_paths)

    # QE (Andersen) constants for the variance process
    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = eta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    sq1mr2 = np.sqrt(1.0 - rho**2)

    V = np.full(N, V0, dtype=float)
    Y = np.zeros(N, dtype=float)  # log(S/S0), identical increments for all spots

    # relative log-barriers for each spot
    lus = [np.log(upper / s0) for s0 in spots]
    lds = [np.log(lower / s0) for s0 in spots]
    surv = [np.ones(N, dtype=float) for _ in spots]

    with np.errstate(over="ignore", divide="ignore", invalid="ignore", under="ignore"):
        for n in range(Nt):
            Z = rng.standard_normal((2, N))
            U = rng.random(N)
            Zv = Z[0]
            Zs = Z[1]

            # ---- QE step for V ----
            m = eta + (V - eta) * ekt
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            # quadratic branch (psi <= 1.5)
            psq = np.minimum(psi, psi_c)
            psq = np.maximum(psq, 1e-300)
            tpi = 2.0 / psq
            b2 = tpi - 1.0 + np.sqrt(tpi) * np.sqrt(np.maximum(tpi - 1.0, 0.0))
            a = m / (1.0 + b2)
            V_quad = a * (np.sqrt(b2) + Zv) ** 2

            # exponential branch (psi > 1.5)
            pex = np.maximum(psi, psi_c)
            p = (pex - 1.0) / (pex + 1.0)
            beta = (1.0 - p) / m
            V_exp = np.where(U <= p, 0.0,
                             np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)) / beta)
            V_exp = np.maximum(V_exp, 0.0)

            Vn = np.where(psi <= psi_c, V_quad, V_exp)

            # ---- log-S step (Andersen-style, conditional on V path) ----
            Vint = 0.5 * (V + Vn) * dt
            dY = (-0.5 * Vint
                  + (rho / sigma) * (Vn - V - kappa * eta * dt + kappa * Vint)
                  + sq1mr2 * np.sqrt(Vint) * Zs)
            Yn = Y + dY

            # ---- Brownian-bridge continuous-monitoring survival ----
            s2b = np.maximum(Vint, 1e-300)  # local bridge variance of log S
            for k in range(len(spots)):
                lu = lus[k]
                ld = lds[k]
                au = lu - Y
                bu = lu - Yn
                al = Y - ld
                bl = Yn - ld
                inside = (au > 0) & (bu > 0) & (al > 0) & (bl > 0)
                pu = np.exp(-2.0 * np.maximum(au, 0.0) * np.maximum(bu, 0.0) / s2b)
                pl = np.exp(-2.0 * np.maximum(al, 0.0) * np.maximum(bl, 0.0) / s2b)
                fac = np.where(inside, (1.0 - pu) * (1.0 - pl), 0.0)
                surv[k] *= fac

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
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": N,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
