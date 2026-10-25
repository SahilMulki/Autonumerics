import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    c = 0.01
    kappa, eta, sigma = 2.58, 0.043, 1.0
    rho = -0.36
    V0 = 0.114
    srho = np.sqrt(1.0 - rho * rho)

    # exact CIR transition parameters (noncentral chi-square)
    ekdt = np.exp(-kappa * dt)
    c0 = sigma**2 * (1.0 - ekdt) / (4.0 * kappa)
    dfree = 4.0 * kappa * eta / sigma**2

    L, U = 1.0 / 150.0, 1.0 / 50.0  # barriers in Y = 1/S
    spots = [75.0, 100.0, 125.0]
    Ys = [np.full(num_paths, 1.0 / s0) for s0 in spots]
    Ws = [np.ones(num_paths) for _ in spots]

    V = np.full(num_paths, V0)
    for n in range(Nt):
        Vn = V
        V = c0 * rng.noncentral_chisquare(dfree, Vn * (ekdt / c0))
        Vbar = 0.5 * (Vn + V)
        # int sqrt(V) dW2 recovered from the V dynamics
        IW2 = (V - Vn) - kappa * eta * dt + kappa * Vbar * dt
        Z = rng.standard_normal(num_paths)
        IW1 = rho * IW2 + srho * np.sqrt(Vbar * dt) * Z
        xi1 = -c * IW1
        dtau = c * c * Vbar * dt + 1e-300
        sq = np.sqrt(dtau)
        xi2 = sq * rng.standard_normal(num_paths)
        xi3 = sq * rng.standard_normal(num_paths)
        e23 = xi2 * xi2 + xi3 * xi3
        inv2 = 2.0 / dtau
        for k in range(3):
            Y0 = Ys[k]
            # Y = 1/S is a time-changed 3-d Bessel process: exact positive step
            Y1 = np.sqrt((Y0 + xi1) ** 2 + e23)
            inside = (Y1 > L) & (Y1 < U)
            dl0 = np.maximum(Y0 - L, 0.0)
            dl1 = np.maximum(Y1 - L, 0.0)
            du0 = np.maximum(U - Y0, 0.0)
            du1 = np.maximum(U - Y1, 0.0)
            p_lo = np.exp(-dl0 * dl1 * inv2)
            p_hi = np.exp(-du0 * du1 * inv2)
            w = Ws[k] * (1.0 - p_lo) * (1.0 - p_hi)
            w[~inside] = 0.0
            Ws[k] = w
            Ys[k] = Y1

    S100 = 1.0 / Ys[1]
    X = np.column_stack([S100, V])
    functionals = {
        "no_touch_S75": float(np.mean(Ws[0])),
        "no_touch_S100": float(np.mean(Ws[1])),
        "no_touch_S125": float(np.mean(Ws[2])),
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
    r = solve_sde(num_paths=20000, dt=0.002, T=1.0)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
