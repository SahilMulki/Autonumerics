import numpy as np


def _vasicek_bond(a, b, s, r0, T):
    # exact zero-coupon bond price under dR = a(b - R)dt + s dW
    B = (1.0 - np.exp(-a * T)) / a
    lnA = (b - s * s / (2.0 * a * a)) * (B - T) - s * s * B * B / (4.0 * a)
    return float(np.exp(lnA - B * r0))


def _cv_price(S0, K, L, I, bondP):
    # discounted payoff  D*(S_T-K)^+ = max(S0 e^L - K e^{-I}, 0)
    C1 = S0 * np.exp(L)          # = D*S_T, exact martingale under the scheme: mean S0
    C2 = np.exp(-I)              # = D, mean = Vasicek bond price
    Y = np.maximum(C1 - K * C2, 0.0)
    M = np.vstack([Y, C1, C2])
    cov = np.cov(M)
    Scc = cov[1:, 1:]
    Syc = cov[0, 1:]
    try:
        beta = np.linalg.solve(Scc, Syc)
    except np.linalg.LinAlgError:
        beta = np.zeros(2)
    mu = np.array([S0, bondP])
    mC = np.array([C1.mean(), C2.mean()])
    return float(Y.mean() - beta @ (mC - mu))


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)

    # parameters
    kappa, theta_v, sig_v, V0 = 0.5, 0.04, 1.0, 0.04
    a_r, b_r, sig_r, R0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0

    corr = np.array([[1.0, -0.9, 0.6],
                     [-0.9, 1.0, -0.7],
                     [0.6, -0.7, 1.0]])
    Lc = np.linalg.cholesky(corr)
    l21, l22 = Lc[1, 0], Lc[1, 1]
    l31, l32, l33 = Lc[2, 0], Lc[2, 1], Lc[2, 2]

    # exact OU transition for R
    eA = np.exp(-a_r * dt)
    r_shift = b_r * (1.0 - eA)
    sr = sig_r * np.sqrt((1.0 - np.exp(-2.0 * a_r * dt)) / (2.0 * a_r))

    rng = np.random.default_rng(seed)

    L_all = np.empty(num_paths)   # log(S_T/S0) minus integral of R
    I_all = np.empty(num_paths)   # integral of R (trapezoid)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)

    chunk = 1 << 16
    half_dt = 0.5 * dt
    for start in range(0, num_paths, chunk):
        m = min(chunk, num_paths - start)
        V = np.full(m, V0)
        R = np.full(m, R0)
        I = np.zeros(m)
        L = np.zeros(m)
        Z = np.empty((3, m))
        for _ in range(Nt):
            rng.standard_normal(out=Z)
            z0, z1, z2 = Z[0], Z[1], Z[2]
            Vp = np.maximum(V, 0.0)          # full truncation
            sV = np.sqrt(Vp)
            w2 = l21 * z0 + l22 * z1
            w3 = l31 * z0 + l32 * z1 + l33 * z2
            Rn = R * eA + r_shift + sr * w3
            I += half_dt * (R + Rn)
            L += sV * (sqdt * z0) - half_dt * Vp
            V += kappa * dt * (theta_v - Vp) + (sig_v * sqdt) * sV * w2
            R = Rn
        L_all[start:start + m] = L
        I_all[start:start + m] = I
        V_all[start:start + m] = V
        R_all[start:start + m] = R

    bondP = _vasicek_bond(a_r, b_r, sig_r, R0, T)

    functionals = {}
    for S0 in (75, 100, 125):
        functionals["price_S%d" % S0] = _cv_price(float(S0), K, L_all, I_all, bondP)

    S_T = 100.0 * np.exp(I_all + L_all)
    terminal = np.column_stack([S_T, V_all, R_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
