import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Heston variance parameters
    kappa, theta, eps, V0 = 0.5, 0.04, 1.0, 0.04
    rho = -0.9
    # Hull-White rate parameters
    a_r, b_r, s_r, R0 = 0.08, 0.10, 0.09, 0.10
    strike = 100.0
    spots = (75.0, 100.0, 125.0)

    # QE constants
    e = np.exp(-kappa * dt)
    c1 = eps**2 * e * (1.0 - e) / kappa
    c2 = theta * eps**2 * (1.0 - e) ** 2 / (2.0 * kappa)
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / eps - 0.5) - rho / eps
    K2 = g2 * dt * (kappa * rho / eps - 0.5) + rho / eps
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4
    psi_c = 1.5

    # Exact joint law of (R_T, integral of R over [0,T]) for Hull-White (OU)
    eaT = np.exp(-a_r * T)
    mR = b_r + (R0 - b_r) * eaT
    vR = s_r**2 / (2 * a_r) * (1 - np.exp(-2 * a_r * T))
    mI = b_r * T + (R0 - b_r) * (1 - eaT) / a_r
    vI = s_r**2 / a_r**2 * (T - 2 * (1 - eaT) / a_r
                            + (1 - np.exp(-2 * a_r * T)) / (2 * a_r))
    cRI = s_r**2 / (2 * a_r**2) * (1 - eaT) ** 2
    sR = np.sqrt(vR)
    rho_RI = cRI / (sR * np.sqrt(vI))
    sI_perp = np.sqrt(max(vI * (1 - rho_RI**2), 0.0))
    bI = cRI / vR

    terminal = np.empty((num_paths, 3))
    pay_sums = np.zeros(len(spots))

    chunk = 200000
    start = 0
    while start < num_paths:
        nc = min(chunk, num_paths - start)
        X = np.zeros(nc)          # log of discounted asset / S0
        V = np.full(nc, V0)
        for _ in range(Nt):
            Zv = rng.standard_normal(nc)
            U = rng.random(nc)
            Zs = rng.standard_normal(nc)

            m = theta + (V - theta) * e
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            quad = psi <= psi_c

            pq = np.minimum(psi, psi_c)
            t = 2.0 / pq
            b2 = t - 1.0 + np.sqrt(t * (t - 1.0))
            aq = m / (1.0 + b2)
            Vq = aq * (np.sqrt(b2) + Zv) ** 2

            pe = np.maximum(psi, psi_c)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ve = np.maximum(np.log((1.0 - p) / (1.0 - U)) / beta, 0.0)
            Ve = np.where(U <= p, 0.0, Ve)

            Vn = np.where(quad, Vq, Ve)

            one_m = 1.0 - 2.0 * A * aq
            lnMq = A * b2 * aq / one_m - 0.5 * np.log(one_m)
            lnMe = np.log(p + beta * (1.0 - p) / (beta - A))
            lnM = np.where(quad, lnMq, lnMe)

            X += -lnM - 0.5 * K3 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Zs
            V = Vn

        z1 = rng.standard_normal(nc)
        z2 = rng.standard_normal(nc)
        RT = mR + sR * z1
        I = mI + bI * (RT - mR) + sI_perp * z2
        disc = np.exp(-I)
        eX = np.exp(X)
        for j, S0 in enumerate(spots):
            pay_sums[j] += np.sum(np.maximum(S0 * eX - strike * disc, 0.0))

        sl = slice(start, start + nc)
        terminal[sl, 0] = 100.0 * eX / disc
        terminal[sl, 1] = V
        terminal[sl, 2] = RT
        start += nc

    prices = pay_sums / num_paths
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "price_S75": float(prices[0]),
            "price_S100": float(prices[1]),
            "price_S125": float(prices[2]),
        },
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
