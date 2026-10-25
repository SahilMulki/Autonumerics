import numpy as np

# Heston variance parameters
KAPPA, THETA, SIGMA_V, V0 = 0.5, 0.04, 1.0, 0.04
# Hull-White (Vasicek) short-rate parameters
KR, THR, SIGMA_R, R0 = 0.08, 0.10, 0.09, 0.10
# Correlations
RHO12, RHO13, RHO23 = -0.9, 0.6, -0.7
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _norm_cdf(x):
    """Vectorised standard normal CDF (Abramowitz-Stegun 26.2.17, |err|<7.5e-8)."""
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
                + t * (-1.821255978 + t * 1.330274429))))
    tail = 0.3989422804014327 * np.exp(-0.5 * ax * ax) * poly  # = Phi(-|x|)
    return np.where(x < 0.0, tail, 1.0 - tail)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # --- Correlation decomposition: Z2 (variance), Zp (W1 orthogonal to W2), Zi (independent)
    # W1 = rho12*Z2 + sqrt(1-rho12^2)*Zp ;  W3 = a3*Z2 + b3*Zp + c3*Zi
    s1 = np.sqrt(1.0 - RHO12 ** 2)
    a3 = RHO23
    b3 = (RHO13 - RHO12 * RHO23) / s1
    c3 = np.sqrt(max(1.0 - a3 ** 2 - b3 ** 2, 0.0))

    # --- QE (Andersen) constants for V
    ek = np.exp(-KAPPA * dt)
    c1 = SIGMA_V ** 2 * ek * (1.0 - ek) / KAPPA
    c2 = THETA * SIGMA_V ** 2 * (1.0 - ek) ** 2 / (2.0 * KAPPA)
    psi_c = 1.5
    g1 = g2 = 0.5
    rho = RHO12
    K1 = g1 * dt * (KAPPA * rho / SIGMA_V - 0.5) - rho / SIGMA_V
    K2 = g2 * dt * (KAPPA * rho / SIGMA_V - 0.5) + rho / SIGMA_V
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)
    A = K2 + 0.5 * K4
    K13 = K1 + 0.5 * K3

    # --- Exact OU transition for R
    er = np.exp(-KR * dt)
    sr = SIGMA_R * np.sqrt((1.0 - np.exp(-2.0 * KR * dt)) / (2.0 * KR))

    L_all = np.empty(num_paths)   # log(S_T/S0)
    I_all = np.empty(num_paths)   # integral of R
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)

    CH = 200000
    for start in range(0, num_paths, CH):
        n = min(CH, num_paths - start)
        V = np.full(n, V0)
        R = np.full(n, R0)
        L = np.zeros(n)
        I = np.zeros(n)
        Vn = np.empty(n)
        K0s = np.empty(n)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            Z2, Zp, Zi = Z[0], Z[1], Z[2]

            m = THETA + (V - THETA) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            q = psi <= psi_c
            e = ~q

            # quadratic branch
            if q.any():
                pq = psi[q]
                tq = 2.0 / pq
                b2 = tq - 1.0 + np.sqrt(tq) * np.sqrt(tq - 1.0)
                aq = m[q] / (1.0 + b2)
                Vn[q] = aq * (np.sqrt(b2) + Z2[q]) ** 2
                one_m = 1.0 - 2.0 * A * aq
                K0s[q] = -A * b2 * aq / one_m + 0.5 * np.log(one_m)
            # exponential branch (uniform from Phi(Z2), keeps comonotone with Z2)
            if e.any():
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[e]
                omU = _norm_cdf(-Z2[e])            # = 1 - U
                omU = np.maximum(omU, 1e-300)
                Vn[e] = np.where(omU >= 1.0 - p, 0.0,
                                 np.log((1.0 - p) / omU) / beta)
                K0s[e] = -np.log(p + beta * (1.0 - p) / (beta - A))

            K0s -= K13 * V

            Rdt = R * dt
            I += Rdt
            L += Rdt + K0s + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zp

            R = THR + (R - THR) * er + sr * (a3 * Z2 + b3 * Zp + c3 * Zi)
            V, Vn = Vn, V

        sl = slice(start, start + n)
        L_all[sl] = L
        I_all[sl] = I
        V_all[sl] = V
        R_all[sl] = R

    D = np.exp(-I_all)
    G = np.exp(L_all)
    functionals = {}
    for S0 in SPOTS:
        ST = S0 * G
        Y = D * np.maximum(ST - STRIKE, 0.0)
        X = D * ST          # discounted asset: martingale, E[X] = S0
        Xc = X - X.mean()
        varX = np.mean(Xc * Xc)
        bcv = np.mean((Y - Y.mean()) * Xc) / varX if varX > 0 else 0.0
        price = float(Y.mean() - bcv * (X.mean() - S0))
        functionals["price_S%d" % int(S0)] = price

    S100 = 100.0 * G
    terminal = np.column_stack([S100, V_all, R_all])
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
