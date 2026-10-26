import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Parameters
    kappa, theta, sigma = 0.5, 0.04, 1.0
    V0 = 0.04
    a_r, th_r, sig_r, R0 = 0.08, 0.10, 0.09, 0.10
    rho = -0.9
    K = 100.0
    spots = [75.0, 100.0, 125.0]

    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # QE constants for the variance (Andersen 2008)
    ek = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ek * (1.0 - ek) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Log-asset (zero-rate Heston part) coefficients, gamma1 = gamma2 = 0.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sigma
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    # Exact Hull-White joint law of (R_T, I = int_0^T R dt)
    eaT = np.exp(-a_r * T)
    mR = th_r + (R0 - th_r) * eaT
    mI = th_r * T + (R0 - th_r) * (1.0 - eaT) / a_r
    vR = sig_r ** 2 * (1.0 - eaT ** 2) / (2.0 * a_r)
    vI = sig_r ** 2 / a_r ** 2 * (T - 2.0 * (1.0 - eaT) / a_r
                                  + (1.0 - eaT ** 2) / (2.0 * a_r))
    cRI = sig_r ** 2 / (2.0 * a_r ** 2) * (1.0 - eaT) ** 2
    sR = np.sqrt(vR)
    corrRI = cRI / np.sqrt(vR * vI)
    sI = np.sqrt(vI)

    terminal = np.empty((num_paths, 3))
    pay_sums = np.zeros(len(spots))

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)

        # Rate factor: exact
        zr = rng.standard_normal((2, n))
        RT = mR + sR * zr[0]
        I = mI + sI * (corrRI * zr[0] + np.sqrt(1.0 - corrRI ** 2) * zr[1])

        V = np.full(n, V0)
        lnX = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n, dtype=np.float32)
            U = rng.random(n, dtype=np.float32)
            Z1 = rng.standard_normal(n, dtype=np.float32)

            m = theta + (V - theta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            # Quadratic branch
            psq = np.minimum(psi, psi_c)
            inv = 2.0 / psq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            aq = m / (1.0 + b2)
            Vq = aq * (np.sqrt(b2) + Zv) ** 2

            # Exponential branch
            pe = np.maximum(psi, psi_c)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            Ud = U.astype(np.float64)
            Ve = np.where(Ud <= p, 0.0,
                          np.log((1.0 - p) / (1.0 - Ud)) / beta)

            Vn = np.where(psi <= psi_c, Vq, Ve)

            lnX += K0 + K1 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Z1
            V = Vn

        X = np.exp(lnX)
        disc_K = K * np.exp(-I)
        for j, s0 in enumerate(spots):
            pay_sums[j] += np.maximum(s0 * X - disc_K, 0.0).sum()

        terminal[start:start + n, 0] = 100.0 * np.exp(I + lnX)
        terminal[start:start + n, 1] = V
        terminal[start:start + n, 2] = RT
        start += n

    prices = pay_sums / num_paths

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
