import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    kappa, theta, sig, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    c = 0.01
    srho = np.sqrt(1.0 - rho ** 2)

    # Exact CIR transition (noncentral chi-square)
    ekt = np.exp(-kappa * dt)
    cfac = sig ** 2 * (1.0 - ekt) / (4.0 * kappa)
    dfree = 4.0 * kappa * theta / sig ** 2
    lam_fac = 4.0 * kappa * ekt / (sig ** 2 * (1.0 - ekt))

    # Transformed variable Z = 1/(c S):  dZ = -sqrt(V) dW1 + (V/Z) dt
    # (time-changed Bessel-3).  Corridor S in (50,150) <=> Z in (L, U).
    L = 1.0 / (c * 150.0)
    U = 1.0 / (c * 50.0)
    spots = [75.0, 100.0, 125.0]
    Z = [np.full(num_paths, 1.0 / (c * s)) for s in spots]
    W = [np.ones(num_paths) for _ in spots]

    V = np.full(num_paths, V0)
    for n in range(Nt):
        V1 = cfac * rng.noncentral_chisquare(dfree, lam_fac * V)
        Iv = 0.5 * (V + V1) * dt
        J = (V1 - V - kappa * theta * dt + kappa * Iv) / sig  # int sqrt(V) dW2
        xi = rng.standard_normal(num_paths)
        E = rng.exponential(size=num_paths)
        dM = -(rho * J + srho * np.sqrt(Iv) * xi)
        q = 2.0 * Iv * E
        Ivs = np.maximum(Iv, 1e-300)
        for k in range(len(spots)):
            z0 = Z[k]
            # BES(3)-type step: radial part of 3D BM, always positive
            z1 = np.sqrt((z0 + dM) ** 2 + q)
            inside = (z1 > L) & (z1 < U)
            a_d = np.maximum(z0 - L, 0.0)
            b_d = np.maximum(z1 - L, 0.0)
            a_u = np.maximum(U - z0, 0.0)
            b_u = np.maximum(U - z1, 0.0)
            p_d = np.exp(-2.0 * a_d * b_d / Ivs)
            p_u = np.exp(-2.0 * a_u * b_u / Ivs)
            W[k] *= inside * (1.0 - p_d) * (1.0 - p_u)
            Z[k] = z1
        V = V1

    S100 = 1.0 / (c * Z[1])
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, j])) for j in range(2)],
        "empirical_variance": [float(np.var(terminal[:, j], ddof=1)) for j in range(2)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
