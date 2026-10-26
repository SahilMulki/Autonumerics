import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    # Heston parameters
    kappa, theta, sigma, rho, V0 = 0.5, 0.04, 1.0, -0.9, 0.04
    # Hull-White parameters
    a_r, b_r, s_r, R0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    ekdt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekdt * (1.0 - ekdt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekdt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    # Exact joint law of (R(T), integral of R over [0,T])
    eaT = np.exp(-a_r * T)
    mR = R0 * eaT + b_r * (1.0 - eaT)
    mI = b_r * T + (R0 - b_r) * (1.0 - eaT) / a_r
    vR = s_r ** 2 * (1.0 - np.exp(-2 * a_r * T)) / (2 * a_r)
    vI = s_r ** 2 / a_r ** 2 * (T - 2.0 * (1.0 - eaT) / a_r
                                + (1.0 - np.exp(-2 * a_r * T)) / (2 * a_r))
    cRI = s_r ** 2 / (2 * a_r ** 2) * (1.0 - eaT) ** 2
    sdR = np.sqrt(vR)
    rho_RI = cRI / (sdR * np.sqrt(vI))
    sdI_cond = np.sqrt(vI) * np.sqrt(max(1.0 - rho_RI ** 2, 0.0))

    terminal = np.empty((num_paths, 3))
    sums = np.zeros(3)

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        X = np.zeros(n)  # log(S/S0) excluding the rate integral
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            m = theta + (V - theta) * ekdt
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            Vn = np.empty(n)
            quad = psi <= psi_c
            if quad.any():
                pq = psi[quad]
                ip = 2.0 / pq
                b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
                aq = m[quad] / (1.0 + b2)
                Vn[quad] = aq * (np.sqrt(b2) + Zv[quad]) ** 2
            ex = ~quad
            if ex.any():
                pe = psi[ex]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[ex]
                Ue = U[ex]
                ve = np.zeros(pe.shape[0])
                pos = Ue > p
                ve[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
                Vn[ex] = ve

            X += K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V = Vn

        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        RT = mR + sdR * Z1
        I = mI + np.sqrt(vI) * rho_RI * Z1 + sdI_cond * Z2

        disc = np.exp(-I)
        growth = np.exp(I + X)
        for j, S0 in enumerate(spots):
            ST = S0 * growth
            sums[j] += np.sum(disc * np.maximum(ST - K, 0.0))
            if S0 == 100.0:
                terminal[start:start + n, 0] = ST
        terminal[start:start + n, 1] = V
        terminal[start:start + n, 2] = RT
        start += n

    prices = sums / num_paths
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "price_S75": float(prices[0]),
            "price_S100": float(prices[1]),
            "price_S125": float(prices[2]),
        },
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
