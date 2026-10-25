import numpy as np


def _phi_upper(z):
    # P(N > z) = 0.5*erfc(z/sqrt2); Numerical Recipes erfcc, fractional error < 1.2e-7
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.5 * x)
    poly = -x * x - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    erfc = t * np.exp(poly)
    q = 0.5 * erfc
    return np.where(z >= 0.0, q, 1.0 - q)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, theta, sigma, v0 = 0.5, 0.04, 1.0, 0.04
    a_r, th_r, sig_r, r0 = 0.08, 0.10, 0.09, 0.10
    rho12, rho13, rho23 = -0.9, 0.6, -0.7

    # W1 = rho12*W2 + s1*Z1 ; W3 = rho23*W2 + c1*Z1 + c2*Z3
    s1 = np.sqrt(1.0 - rho12 ** 2)
    c1 = (rho13 - rho12 * rho23) / s1
    c2 = np.sqrt(1.0 - rho23 ** 2 - c1 ** 2)

    # QE constants
    ek = np.exp(-kappa * dt)
    s2a = sigma ** 2 * ek * (1.0 - ek) / kappa
    s2b = theta * sigma ** 2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    # log-S constants
    K_dv = rho12 / sigma
    K_vbar = (kappa * rho12 / sigma - 0.5) * dt
    K_const = -rho12 * kappa * theta * dt / sigma
    # OU exact
    er = np.exp(-a_r * dt)
    sd_r = sig_r * np.sqrt((1.0 - er ** 2) / (2.0 * a_r))

    X = np.empty(num_paths)
    V = np.empty(num_paths)
    R = np.empty(num_paths)
    I = np.empty(num_paths)

    chunk = 100000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        x = np.zeros(n)
        v = np.full(n, v0)
        r = np.full(n, r0)
        integ = np.zeros(n)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            Z1, Z2, Z3 = Z[0], Z[1], Z[2]

            # --- variance: Andersen QE driven by Z2 ---
            m = theta + (v - theta) * ek
            s2 = v * s2a + s2b
            psi = s2 / (m * m)
            quad = psi <= 1.5
            pq = np.clip(psi, 1e-12, 1.5)
            ip = 2.0 / pq
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            aq = m / (1.0 + b2)
            vq = aq * (np.sqrt(b2) + Z2) ** 2
            pe = np.maximum(psi, 1.5)
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m
            up = _phi_upper(Z2)  # 1 - U, U = Phi(Z2)
            up = np.maximum(up, 1e-300)
            ve = np.where(up >= 1.0 - p, 0.0, (np.log(1.0 - p) - np.log(up)) / beta)
            ve = np.maximum(ve, 0.0)
            vn = np.where(quad, vq, ve)

            # --- short rate: exact OU, correlated ---
            xi = rho23 * Z2 + c1 * Z1 + c2 * Z3
            rn = th_r + (r - th_r) * er + sd_r * xi
            rbar_dt = 0.5 * (r + rn) * dt

            # --- log S ---
            vbar = 0.5 * (v + vn)
            x += (rbar_dt + K_dv * (vn - v) + K_const + K_vbar * vbar
                  + s1 * np.sqrt(vbar * dt) * Z1)
            integ += rbar_dt
            v = vn
            r = rn

        X[start:start + n] = x
        V[start:start + n] = v
        R[start:start + n] = r
        I[start:start + n] = integ

    D = np.exp(-I)
    ex = np.exp(X)
    functionals = {}
    for s0, key in ((75.0, "price_S75"), (100.0, "price_S100"), (125.0, "price_S125")):
        functionals[key] = float(np.mean(D * np.maximum(s0 * ex - 100.0, 0.0)))

    S = 100.0 * ex
    terminal = np.column_stack([S, V, R])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(c) for c in terminal.mean(axis=0)],
        "empirical_variance": [float(c) for c in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
