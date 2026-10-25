import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    c = 0.01
    L_S, U_S = 50.0, 150.0
    # Work in Y = 1/S:  dY = -c sqrt(V) dW1 + c^2 V / Y dt
    YL, YU = 1.0 / U_S, 1.0 / L_S
    spots = [75.0, 100.0, 125.0]
    names = ["no_touch_S75", "no_touch_S100", "no_touch_S125"]

    n = num_paths
    V = np.full(n, V0)
    Ys = [np.full(n, 1.0 / s0) for s0 in spots]
    W = [np.ones(n) for _ in spots]

    ekt = np.exp(-kappa * dt)
    sq1mr2 = np.sqrt(1.0 - rho * rho)
    psi_c = 1.5

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        Uv = rng.random(n)
        Z1 = rng.standard_normal(n)

        # Andersen QE step for CIR variance
        m = eta + (V - eta) * ekt
        s2 = (V * sigma**2 * ekt * (1.0 - ekt) / kappa
              + eta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa))
        psi = s2 / (m * m)
        V1 = np.empty(n)
        low = psi <= psi_c
        if np.any(low):
            ip = 2.0 / psi[low]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[low] / (1.0 + b2)
            V1[low] = a * (np.sqrt(b2) + Zv[low]) ** 2
        hi = ~low
        if np.any(hi):
            p = (psi[hi] - 1.0) / (psi[hi] + 1.0)
            beta = (1.0 - p) / m[hi]
            u = Uv[hi]
            val = np.zeros(u.shape)
            pos = u > p
            val[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            V1[hi] = val

        Vbar = 0.5 * (V + V1)
        intV = Vbar * dt
        # integral of sqrt(V) dW2 implied by the variance step
        I2 = (V1 - V - kappa * (eta - Vbar) * dt) / sigma
        I1 = rho * I2 + sq1mr2 * np.sqrt(intV) * Z1

        bvar = np.maximum(c * c * intV, 1e-300)

        for k in range(3):
            Y = Ys[k]
            Yd = np.maximum(Y, 1e-8)
            Y1 = Y - c * I1 + c * c * intV / Yd
            Y1 = np.abs(Y1)
            Y1 = np.maximum(Y1, 1e-12)

            inside = (Y1 > YL) & (Y1 < YU) & (Y > YL) & (Y < YU)
            du0 = np.maximum(YU - Y, 0.0)
            du1 = np.maximum(YU - Y1, 0.0)
            dl0 = np.maximum(Y - YL, 0.0)
            dl1 = np.maximum(Y1 - YL, 0.0)
            pu = np.exp(-2.0 * du0 * du1 / bvar)
            pl = np.exp(-2.0 * dl0 * dl1 / bvar)
            surv = np.clip(1.0 - pu - pl, 0.0, 1.0)
            surv = np.where(inside, surv, 0.0)
            W[k] *= surv
            Ys[k] = Y1

        V = V1

    S100 = 1.0 / Ys[1]
    terminal = np.column_stack([S100, V])
    functionals = {names[k]: float(np.mean(W[k])) for k in range(3)}

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(out["functionals"])
    print(out["empirical_mean"], out["empirical_variance"])
