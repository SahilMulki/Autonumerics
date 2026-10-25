import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    c = 0.01
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    srho = np.sqrt(1.0 - rho * rho)

    # X = 1/S: dX = -c sqrt(V) dW1 + c^2 V / X dt (additive noise in X)
    xa, xb = 1.0 / 150.0, 1.0 / 50.0  # corridor in X space
    spots = [75.0, 100.0, 125.0]

    ek = np.exp(-kappa * dt)
    psi_c = 1.5
    floorX = 1e-6

    surv_sum = np.zeros(3)
    S_term = np.empty(num_paths)
    V_term = np.empty(num_paths)

    chunk = 50000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = [np.full(n, 1.0 / s0) for s0 in spots]
        W = [np.ones(n) for _ in spots]
        for _ in range(Nt):
            # QE step for V
            m = theta + (V - theta) * ek
            s2 = (V * sigma**2 * ek * (1 - ek) / kappa
                  + theta * sigma**2 * (1 - ek) ** 2 / (2 * kappa))
            psi = s2 / (m * m)
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Vn = np.empty(n)
            lo = psi <= psi_c
            if lo.any():
                ip = 2.0 / psi[lo]
                b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
                a = m[lo] / (1.0 + b2)
                Vn[lo] = a * (np.sqrt(b2) + Zv[lo]) ** 2
            hi = ~lo
            if hi.any():
                ph = psi[hi]
                p = (ph - 1.0) / (ph + 1.0)
                beta = (1.0 - p) / m[hi]
                Uh = U[hi]
                vv = np.zeros(Uh.shape)
                pos = Uh > p
                vv[pos] = np.log((1.0 - p[pos]) / (1.0 - Uh[pos])) / beta[pos]
                Vn[hi] = vv
            Ibar = 0.5 * (V + Vn)
            intW2 = (Vn - V - kappa * theta * dt + kappa * Ibar * dt) / sigma
            Z = rng.standard_normal(n)
            dM = rho * intW2 + srho * np.sqrt(Ibar * dt) * Z  # int sqrt(V) dW1
            noise = -c * dM
            var_loc = np.maximum(c * c * Ibar * dt, 1e-300)
            for k in range(3):
                X0 = X[k]
                X1 = X0 + c * c * Ibar / X0 * dt + noise
                X1 = np.maximum(X1, floorX)
                w = W[k]
                inside = (X1 > xa) & (X1 < xb) & (X0 > xa) & (X0 < xb)
                da0 = X0 - xa
                da1 = X1 - xa
                db0 = xb - X0
                db1 = xb - X1
                pa = np.exp(-2.0 * np.maximum(da0 * da1, 0.0) / var_loc)
                pb = np.exp(-2.0 * np.maximum(db0 * db1, 0.0) / var_loc)
                fac = np.where(inside, np.clip((1.0 - pa) * (1.0 - pb), 0.0, 1.0), 0.0)
                W[k] = w * fac
                X[k] = X1
            V = Vn
        for k in range(3):
            surv_sum[k] += W[k].sum()
        S_term[start:start + n] = 1.0 / X[1]
        V_term[start:start + n] = V
        start += n

    surv = surv_sum / num_paths
    terminal = np.column_stack([S_term, V_term])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "no_touch_S75": float(surv[0]),
            "no_touch_S100": float(surv[1]),
            "no_touch_S125": float(surv[2]),
        },
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"], res["empirical_mean"])
