import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston model (r = 0), Feller condition violated.
    # Variance: Andersen QE scheme (handles V = 0 exactly, never negative).
    # Log-asset: Andersen trapezoidal scheme with martingale correction, so
    # E[S_T] = S0 holds exactly in the discrete scheme.
    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    ekt = np.exp(-kappa * dt)
    c1 = sigma**2 * ekt * (1.0 - ekt) / kappa
    c2 = eta * sigma**2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)
    A = K2 + 0.5 * K4  # negative here, so the MGF always exists

    V = np.full(num_paths, V0, dtype=float)
    X = np.zeros(num_paths, dtype=float)  # log(S/S0)
    Vn = np.empty(num_paths, dtype=float)
    lnM = np.empty(num_paths, dtype=float)

    for _ in range(Nt):
        m = eta + (V - eta) * ekt
        s2 = V * c1 + c2
        psi = s2 / (m * m)

        quad = psi <= psi_c
        iq = np.nonzero(quad)[0]
        ie = np.nonzero(~quad)[0]

        # Quadratic branch
        if iq.size:
            mq = m[iq]
            pq = psi[iq]
            tq = 2.0 / pq
            b2 = tq - 1.0 + np.sqrt(tq) * np.sqrt(tq - 1.0)
            a = mq / (1.0 + b2)
            b = np.sqrt(b2)
            Zv = rng.standard_normal(iq.size)
            Vn[iq] = a * (b + Zv) ** 2
            den = 1.0 - 2.0 * A * a
            lnM[iq] = A * b2 * a / den - 0.5 * np.log(den)

        # Exponential branch
        if ie.size:
            me = m[ie]
            pe = psi[ie]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / me
            U = rng.random(ie.size)
            ve = np.zeros(ie.size)
            pos = U > p
            ve[pos] = np.log((1.0 - p[pos]) / (1.0 - U[pos])) / beta[pos]
            Vn[ie] = ve
            lnM[ie] = np.log(p + beta * (1.0 - p) / (beta - A))

        K0s = -lnM - (K1 + 0.5 * K3) * V
        Z = rng.standard_normal(num_paths)
        X += K0s + K1 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Z
        V, Vn = Vn, V

    eX = np.exp(X)
    functionals = {}
    for S0 in spots:
        ST = S0 * eX
        C = np.maximum(ST - K, 0.0)
        # Control variate with the exact martingale E[S_T] = S0
        dS = ST - ST.mean()
        varS = np.dot(dS, dS)
        cbeta = np.dot(dS, C - C.mean()) / varS if varS > 0 else 0.0
        price = C.mean() - cbeta * (ST.mean() - S0)
        functionals[f"price_S{int(S0)}"] = float(price)

    S100 = 100.0 * eX
    terminal = np.column_stack((S100, V))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
