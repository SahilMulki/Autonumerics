import numpy as np


# Model parameters
KAPPA_V, THETA_V, SIGMA_V, V0 = 0.5, 0.04, 1.0, 0.04
RHO = -0.9
A_R, B_R, SIGMA_R, R0 = 0.08, 0.10, 0.09, 0.10
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _rate_moments(T):
    """Joint Gaussian moments of (I = int_0^T R dt, R_T) for the Hull-White/Vasicek rate."""
    a, b, s, r0 = A_R, B_R, SIGMA_R, R0
    eaT = np.exp(-a * T)
    B = (1.0 - eaT) / a
    mean_R = b + (r0 - b) * eaT
    var_R = s * s / (2.0 * a) * (1.0 - np.exp(-2.0 * a * T))
    mean_I = b * T + (r0 - b) * B
    var_I = s * s / (a * a) * (T - 2.0 * B + (1.0 - np.exp(-2.0 * a * T)) / (2.0 * a))
    cov_IR = s * s / (2.0 * a * a) * (1.0 - eaT) ** 2
    return mean_I, var_I, mean_R, var_R, cov_IR


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # ---- Rate factor: independent of (S, V); sample (int R, R_T) exactly (jointly Gaussian)
    mean_I, var_I, mean_R, var_R, cov_IR = _rate_moments(T)
    zr = rng.standard_normal((num_paths, 2))
    sd_I = np.sqrt(var_I)
    I = mean_I + sd_I * zr[:, 0]
    R_T = mean_R + (cov_IR / sd_I) * zr[:, 0] + np.sqrt(max(var_R - cov_IR ** 2 / var_I, 0.0)) * zr[:, 1]
    D = np.exp(-I)
    P_exact = np.exp(-mean_I + 0.5 * var_I)  # exact zero-coupon bond E[D]

    # ---- Heston (rate-free part) with Andersen QE scheme
    k, th, sg, rho = KAPPA_V, THETA_V, SIGMA_V, RHO
    e = np.exp(-k * dt)
    c1 = sg * sg * e * (1.0 - e) / k
    c2 = th * sg * sg * (1.0 - e) ** 2 / (2.0 * k)
    g1 = g2 = 0.5
    K0 = -rho * k * th * dt / sg
    K1 = g1 * dt * (k * rho / sg - 0.5) - rho / sg
    K2 = g2 * dt * (k * rho / sg - 0.5) + rho / sg
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    psi_c = 1.5

    X_all = np.empty(num_paths)  # log(S_hat_T / S0), rate-free
    V_all = np.empty(num_paths)

    chunk = 100000
    for start in range(0, num_paths, chunk):
        stop = min(start + chunk, num_paths)
        n = stop - start
        V = np.full(n, V0)
        X = np.zeros(n)
        Vn = np.empty(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)

            m = th + (V - th) * e
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            mask = psi <= psi_c

            if mask.any():
                pm = psi[mask]
                inv = 2.0 / pm
                b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
                a = m[mask] / (1.0 + b2)
                Vn[mask] = a * (np.sqrt(b2) + Zv[mask]) ** 2
            nm = ~mask
            if nm.any():
                pn = psi[nm]
                p = (pn - 1.0) / (pn + 1.0)
                beta = (1.0 - p) / m[nm]
                u = U[nm]
                val = np.log((1.0 - p) / (1.0 - u)) / beta
                Vn[nm] = np.where(u <= p, 0.0, val)

            X += K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs
            V, Vn = Vn, V
        X_all[start:stop] = X
        V_all[start:stop] = V

    eX = np.exp(X_all)

    # ---- Discounted call prices with control variates:
    # C1 = D*S_T - S0 = S0*(e^X - 1)  (mean 0),  C2 = D - P(0,T)  (mean 0)
    C = np.column_stack((eX - 1.0, D - P_exact))
    Cm = C.mean(axis=0)
    Cc = C - Cm
    CtC = Cc.T @ Cc

    functionals = {}
    for S0 in SPOTS:
        Y = np.maximum(S0 * eX - STRIKE * D, 0.0)  # = D * (S_T - K)^+
        Ym = Y.mean()
        beta = np.linalg.solve(CtC, Cc.T @ (Y - Ym))
        price = Ym - float(beta @ Cm)
        functionals["price_S%d" % int(S0)] = float(price)

    S_T = 100.0 * eX * np.exp(I)
    terminal = np.column_stack((S_T, V_all, R_T))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
