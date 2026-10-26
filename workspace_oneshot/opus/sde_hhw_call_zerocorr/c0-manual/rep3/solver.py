import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Heston variance parameters
    kappa, theta, sig_v, v0 = 0.5, 0.04, 1.0, 0.04
    rho = -0.9
    # Hull-White parameters
    a_r, b_r, sig_r, r0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    # QE constants
    ek = np.exp(-kappa * dt)
    c1 = sig_v ** 2 * ek * (1.0 - ek) / kappa
    c2 = theta * sig_v ** 2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    # Andersen log-S discretization constants (gamma1 = gamma2 = 0.5)
    g1 = g2 = 0.5
    K0 = -rho * kappa * theta * dt / sig_v
    K1 = g1 * dt * (kappa * rho / sig_v - 0.5) - rho / sig_v
    K2 = g2 * dt * (kappa * rho / sig_v - 0.5) + rho / sig_v
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)

    # Exact joint Gaussian law of (R_T, int_0^T R dt) for Hull-White
    ea = np.exp(-a_r * T)
    mR = r0 * ea + b_r * (1.0 - ea)
    vR = sig_r ** 2 / (2.0 * a_r) * (1.0 - ea ** 2)
    mI = b_r * T + (r0 - b_r) * (1.0 - ea) / a_r
    vI = sig_r ** 2 / a_r ** 2 * (T - 2.0 * (1.0 - ea) / a_r
                                  + (1.0 - ea ** 2) / (2.0 * a_r))
    cRI = sig_r ** 2 / (2.0 * a_r ** 2) * (1.0 - ea) ** 2
    L11 = np.sqrt(vR)
    L21 = cRI / L11
    L22 = np.sqrt(max(vI - L21 ** 2, 0.0))

    S_T = np.empty(num_paths)
    V_T = np.empty(num_paths)
    R_T = np.empty(num_paths)
    sums = {s: 0.0 for s in spots}

    chunk = 100000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        V = np.full(n, v0)
        sV0 = np.zeros(n)
        sV1 = np.zeros(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            m = theta + (V - theta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            # quadratic branch
            inv = 2.0 / psi
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            aq = m / (1.0 + b2)
            Vq = aq * (np.sqrt(b2) + Zv) ** 2
            # exponential branch
            p = np.clip((psi - 1.0) / (psi + 1.0), 0.0, 1.0 - 1e-15)
            beta = (1.0 - p) / m
            Ve = np.where(U <= p, 0.0, np.log((1.0 - p) / (1.0 - U)) / beta)
            Vn = np.where(psi <= psi_c, Vq, Ve)
            sV0 += V
            sV1 += Vn
            V = Vn

        Zs = rng.standard_normal(n)
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        Rt = mR + L11 * Z1
        I = mI + L21 * Z1 + L22 * Z2

        Y = (Nt * K0 + K1 * sV0 + K2 * sV1
             + np.sqrt(np.maximum(K3 * sV0 + K4 * sV1, 0.0)) * Zs)
        growth = np.exp(I + Y)
        disc = np.exp(-I)
        for s in spots:
            sums[s] += float(np.sum(disc * np.maximum(s * growth - K, 0.0)))

        S_T[start:start + n] = 100.0 * growth
        V_T[start:start + n] = V
        R_T[start:start + n] = Rt

    X = np.column_stack([S_T, V_T, R_T])
    functionals = {
        "price_S75": sums[75.0] / num_paths,
        "price_S100": sums[100.0] / num_paths,
        "price_S125": sums[125.0] / num_paths,
    }
    return {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
