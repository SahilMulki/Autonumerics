import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Model parameters
    kappa, theta, sigma, v0 = 0.5, 0.04, 1.0, 0.04
    a_r, b_r, sig_r, r0 = 0.08, 0.10, 0.09, 0.10
    rho = -0.9
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # QE constants for V
    ekt = np.exp(-kappa * dt)
    c1 = sigma ** 2 * ekt * (1.0 - ekt) / kappa
    c2 = theta * sigma ** 2 * (1.0 - ekt) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-S constants (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    # Exact OU constants for R
    ear = np.exp(-a_r * dt)
    sd_r = sig_r * np.sqrt((1.0 - ear ** 2) / (2.0 * a_r))

    chunk = 200000
    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    D_all = np.empty(num_paths)

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X = np.zeros(n)
        V = np.full(n, v0)
        R = np.full(n, r0)
        IR = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Zs = rng.standard_normal(n)
            Zr = rng.standard_normal(n)

            # --- QE step for V ---
            m = theta + (V - theta) * ekt
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            # quadratic branch
            inv = 2.0 / np.minimum(psi, psi_c)
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            aq = m / (1.0 + b2)
            Vq = aq * (np.sqrt(b2) + Zv) ** 2
            # exponential branch
            p = (psi - 1.0) / (psi + 1.0)
            p = np.clip(p, 0.0, 1.0 - 1e-15)
            beta = (1.0 - p) / m
            Ve = np.maximum(np.log((1.0 - p) / (1.0 - U)), 0.0) / beta
            Vn = np.where(psi <= psi_c, Vq, Ve)

            # --- R step (exact OU) ---
            Rn = b_r + (R - b_r) * ear + sd_r * Zr
            Rint = 0.5 * (R + Rn) * dt

            # --- log S step ---
            var = K3 * V + K4 * Vn
            X += Rint + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(var, 0.0)) * Zs

            IR += Rint
            V = Vn
            R = Rn

        X_all[start:start + n] = X
        V_all[start:start + n] = V
        R_all[start:start + n] = R
        D_all[start:start + n] = np.exp(-IR)
        start += n

    eX = np.exp(X_all)
    functionals = {}
    for s0 in spots:
        payoff = np.maximum(s0 * eX - K, 0.0)
        functionals["price_S%d" % int(s0)] = float(np.mean(D_all * payoff))

    S_T = 100.0 * eX
    terminal = np.column_stack([S_T, V_all, R_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(out["functionals"])
    print(out["empirical_mean"])
