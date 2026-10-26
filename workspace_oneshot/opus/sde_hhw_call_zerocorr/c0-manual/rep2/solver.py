import numpy as np


def _hw_discount(r0, a, theta_r, sig_r, T):
    # Vasicek / Hull-White zero-coupon bond price P(0,T)
    B = (1.0 - np.exp(-a * T)) / a
    mean_I = theta_r * T + (r0 - theta_r) * B
    var_I = sig_r**2 / a**2 * (T - 2.0 * B + (1.0 - np.exp(-2.0 * a * T)) / (2.0 * a))
    return float(np.exp(-mean_I + 0.5 * var_I))


def _cv_price(Y, controls, mus):
    # multiple control variate estimator with sample-optimal coefficients
    C = np.vstack(controls)                      # (k, n)
    Cm = C.mean(axis=1)
    Cc = C - Cm[:, None]
    Ym = Y.mean()
    cov_CC = Cc @ Cc.T / (Y.size - 1)
    cov_CY = Cc @ (Y - Ym) / (Y.size - 1)
    try:
        beta = np.linalg.solve(cov_CC, cov_CY)
    except np.linalg.LinAlgError:
        beta = np.zeros(len(controls))
    return float(Ym - beta @ (Cm - np.asarray(mus)))


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = int(num_paths)

    # parameters
    kappa, theta_v, sig_v, V0 = 0.5, 0.04, 1.0, 0.04
    a, theta_r, sig_r, R0 = 0.08, 0.10, 0.09, 0.10
    rho = -0.9
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    # QE constants
    ek = np.exp(-kappa * dt)
    c1 = sig_v**2 * ek * (1.0 - ek) / kappa
    c2 = theta_v * sig_v**2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5
    # rate constants (exact OU transition)
    ea = np.exp(-a * dt)
    sr = sig_r * np.sqrt((1.0 - ea * ea) / (2.0 * a))
    sq1mr2 = np.sqrt(1.0 - rho * rho)
    rs = rho / sig_v

    V = np.full(n, V0)
    R = np.full(n, R0)
    X = np.zeros(n)     # log(D_t * S_t / S0)
    IR = np.zeros(n)    # integral of R

    for _ in range(Nt):
        # ---- variance: Andersen QE step ----
        m = theta_v + (V - theta_v) * ek
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        # quadratic branch
        pq = np.minimum(psi, psi_c)
        inv = 2.0 / pq
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
        aq = m / (1.0 + b2)
        Vq = aq * (np.sqrt(b2) + Zv) ** 2
        # exponential branch
        pe = np.maximum(psi, psi_c)
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m
        Ve = np.where(U <= p, 0.0,
                      np.log(np.maximum((1.0 - p) / np.maximum(1.0 - U, 1e-300), 1.0)) / beta)
        Vn = np.where(psi <= psi_c, Vq, Ve)

        # ---- log asset (discounted), Broadie-Kaya style with trapezoid ----
        I = 0.5 * (V + Vn) * dt
        Z = rng.standard_normal(n)
        X += (-0.5 * I + rs * (Vn - V - kappa * theta_v * dt + kappa * I)
              + sq1mr2 * np.sqrt(I) * Z)

        # ---- short rate: exact OU, trapezoid integral ----
        Zr = rng.standard_normal(n)
        Rn = theta_r + (R - theta_r) * ea + sr * Zr
        IR += 0.5 * (R + Rn) * dt

        V = Vn
        R = Rn

    D = np.exp(-IR)
    eX = np.exp(X)
    P0T = _hw_discount(R0, a, theta_r, sig_r, T)

    functionals = {}
    for S0 in spots:
        DS = S0 * eX
        Y = np.maximum(DS - K * D, 0.0)
        price = _cv_price(Y, [DS, D], [S0, P0T])
        functionals["price_S%d" % int(S0)] = price

    S_T = 100.0 * eX * np.exp(IR)
    paths = np.column_stack([S_T, V, R])

    return {
        "terminal_paths": paths,
        "empirical_mean": [float(v) for v in paths.mean(axis=0)],
        "empirical_variance": [float(v) for v in paths.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": n,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
