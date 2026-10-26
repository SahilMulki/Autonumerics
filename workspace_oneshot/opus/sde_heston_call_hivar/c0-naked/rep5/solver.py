import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho, v0 = 2.58, 0.043, 1.0, -0.36, 0.164
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # Andersen QE constants
    ekd = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekd * (1.0 - ekd) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekd) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)
    A = K2 + 0.5 * K4

    S_T = np.empty(num_paths)
    V_T = np.empty(num_paths)

    chunk = 250000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, v0)
        X = np.zeros(n)  # log(S/S0)
        for _ in range(Nt):
            m = theta + (V - theta) * ekd
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Z = rng.standard_normal(n)

            Vn = np.empty(n)
            K0s = np.empty(n)

            q = psi <= psi_c
            if np.any(q):
                ip = 1.0 / psi[q]
                b2 = 2.0 * ip - 1.0 + np.sqrt(2.0 * ip) * np.sqrt(2.0 * ip - 1.0)
                b = np.sqrt(b2)
                a = m[q] / (1.0 + b2)
                Vn[q] = a * (b + Zv[q]) ** 2
                one_m = 1.0 - 2.0 * A * a
                K0s[q] = -A * b2 * a / one_m + 0.5 * np.log(one_m)
            e = ~q
            if np.any(e):
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[e]
                Ue = U[e]
                ve = np.log((1.0 - p) / (1.0 - Ue)) / beta
                ve = np.where(Ue <= p, 0.0, np.maximum(ve, 0.0))
                Vn[e] = ve
                K0s[e] = -np.log(p + beta * (1.0 - p) / (beta - A))

            K0s -= (K1 + 0.5 * K3) * V
            X += K0s + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Z
            V = Vn

        S_T[start:start + n] = np.exp(X)  # S(T)/S0
        V_T[start:start + n] = V
        start += n

    R = S_T  # ratio S(T)/S0, E[R] = 1
    functionals = {}
    for s0 in spots:
        ST = s0 * R
        pay = np.maximum(ST - K, 0.0)
        # control variate with S(T), known mean s0
        cST = ST - s0
        var_c = np.dot(cST, cST) / max(1, num_paths - 1)
        cov = np.dot(pay - pay.mean(), cST) / max(1, num_paths - 1)
        bcv = cov / var_c if var_c > 0 else 0.0
        price = float(np.mean(pay - bcv * (ST - s0)))
        functionals["price_S%d" % int(s0)] = price

    S100 = 100.0 * R
    terminal = np.column_stack([S100, V_T])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(out["functionals"], out["empirical_mean"])
