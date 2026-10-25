import math
import numpy as np


def _erfc(x):
    # Numerical Recipes erfc approximation (fractional error < 1.2e-7 everywhere)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    ans = t * np.exp(poly)
    return np.where(x >= 0.0, ans, 2.0 - ans)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # parameters
    kappa, theta, sigma = 0.5, 0.04, 1.0
    a_r, rbar, sr = 0.08, 0.10, 0.09
    r12, r13, r23 = -0.9, 0.6, -0.7
    V0, R0 = 0.04, 0.10
    K = 100.0

    # Cholesky in order (W2, W1, W3):
    # W2 = Zv ; W1 = r12 Zv + sqrt(1-r12^2) Zp ; W3 = r23 Zv + c Zp + d Z3
    sq12 = math.sqrt(1.0 - r12 * r12)
    c = (r13 - r12 * r23) / sq12
    d = math.sqrt(max(1.0 - r23 * r23 - c * c, 0.0))

    # QE constants
    ek = math.exp(-kappa * dt)
    c1 = sigma * sigma * ek * (1.0 - ek) / kappa
    c2 = theta * sigma * sigma * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # exact OU constants for R
    ea = math.exp(-a_r * dt)
    sdR = sr * math.sqrt((1.0 - ea * ea) / (2.0 * a_r))

    # Andersen log-S scheme constants (gamma1 = gamma2 = 0.5), rho = r12
    g1 = g2 = 0.5
    rho = r12
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    inv_sqrt2 = 1.0 / math.sqrt(2.0)
    half_dt = 0.5 * dt

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)

    chunk = 1 << 17
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X = np.zeros(n)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            Zv = Z[0]
            Zp = Z[1]
            Z3 = Z[2]

            m = theta + (V - theta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Vn = np.empty(n)
            q = psi <= psi_c
            if q.any():
                pq = psi[q]
                ip = 2.0 / pq
                b2 = ip - 1.0 + np.sqrt(ip * (ip - 1.0))
                aa = m[q] / (1.0 + b2)
                Vn[q] = aa * (np.sqrt(b2) + Zv[q]) ** 2
            e = ~q
            if e.any():
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                one_m_p = 1.0 - p
                beta = one_m_p / m[e]
                Q = 0.5 * _erfc(Zv[e] * inv_sqrt2)  # = 1 - Phi(Zv)
                Q = np.maximum(Q, 1e-300)
                Vn[e] = np.log(np.maximum(one_m_p / Q, 1.0)) / beta

            Rn = rbar + (R - rbar) * ea + sdR * (r23 * Zv + c * Zp + d * Z3)
            rint = half_dt * (R + Rn)
            X += rint + K0 + K1 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Zp
            I += rint
            V = Vn
            R = Rn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        R_all[start:start + n] = R
        I_all[start:start + n] = I
        start += n

    S100 = 100.0 * np.exp(X_all)
    D = np.exp(-I_all)

    functionals = {}
    for S0, name in ((75.0, "price_S75"), (100.0, "price_S100"), (125.0, "price_S125")):
        ST = (S0 / 100.0) * S100
        functionals[name] = float(np.mean(D * np.maximum(ST - K, 0.0)))

    terminal = np.column_stack((S100, V_all, R_all))
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
