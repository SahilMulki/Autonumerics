import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = int(num_paths)

    # Andersen QE scheme for V, with central (trapezoid) discretization of log S
    ed = np.exp(-kappa * dt)
    c1 = sigma**2 * ed * (1.0 - ed) / kappa
    c2 = theta * sigma**2 * (1.0 - ed) ** 2 / (2.0 * kappa)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    psi_c = 1.5

    V = np.full(n, V0, dtype=float)
    X = np.zeros(n, dtype=float)  # log(S/S0)

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        Zs = rng.standard_normal(n)

        m = theta + (V - theta) * ed
        s2 = V * c1 + c2
        psi = s2 / (m * m)

        Vn = np.empty(n)
        low = psi <= psi_c
        # quadratic branch
        if np.any(low):
            pl = psi[low]
            ml = m[low]
            inv = 2.0 / pl
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = ml / (1.0 + b2)
            Vn[low] = a * (np.sqrt(b2) + Zv[low]) ** 2
        # exponential branch
        high = ~low
        nh = int(np.count_nonzero(high))
        if nh > 0:
            ph = psi[high]
            mh = m[high]
            p = (ph - 1.0) / (ph + 1.0)
            beta = (1.0 - p) / mh
            U = rng.random(nh)
            out = np.zeros(nh)
            pos = U > p
            out[pos] = np.log((1.0 - p[pos]) / (1.0 - U[pos])) / beta[pos]
            Vn[high] = out

        var_term = np.maximum(K3 * V + K4 * Vn, 0.0)
        X += K0 + K1 * V + K2 * Vn + np.sqrt(var_term) * Zs
        V = Vn

    G = np.exp(X)  # S_T / S0
    functionals = {}
    for S0 in spots:
        ST = S0 * G
        C = np.maximum(ST - K, 0.0)
        # control variate: E[S_T] = S0 (martingale)
        cv = ST - S0
        cvc = cv - cv.mean()
        varcv = np.dot(cvc, cvc)
        b = np.dot(C - C.mean(), cvc) / varcv if varcv > 0 else 0.0
        price = float(C.mean() - b * cv.mean())
        functionals[f"price_S{int(S0)}"] = price

    S100 = 100.0 * G
    terminal = np.column_stack((S100, V))
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": n,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
