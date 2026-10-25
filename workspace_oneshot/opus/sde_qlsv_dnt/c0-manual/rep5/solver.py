import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # parameters
    c = 0.01
    kappa, eta, sigma = 2.58, 0.043, 1.0
    V0 = 0.114
    rho = -0.36
    srho = np.sqrt(1.0 - rho * rho)
    L, U = 50.0, 150.0
    # Work in Y = 1/S:  dY = c^2 V / Y dt - c sqrt(V) dW1  (additive noise given V)
    YL, YU = 1.0 / U, 1.0 / L  # corridor in Y-space: (1/150, 1/50)

    spots = [75.0, 100.0, 125.0]
    nsp = len(spots)

    # exact CIR transition via noncentral chi-square
    ekdt = np.exp(-kappa * dt)
    cfac = sigma**2 * (1.0 - ekdt) / (4.0 * kappa)
    df = 4.0 * kappa * eta / sigma**2

    V = np.full(num_paths, V0, dtype=float)
    Y = [np.full(num_paths, 1.0 / s0, dtype=float) for s0 in spots]
    W = [np.ones(num_paths, dtype=float) for _ in spots]

    c2 = c * c
    for n in range(Nt):
        nonc = V * ekdt / cfac
        V1 = cfac * rng.noncentral_chisquare(df, nonc)
        IV = 0.5 * (V + V1) * dt                       # ∫ V dt (trapezoid)
        I2 = (V1 - V - kappa * eta * dt + kappa * IV) / sigma  # ∫ sqrt(V) dW2
        Z = rng.standard_normal(num_paths)
        noise = c * (rho * I2 + srho * np.sqrt(IV) * Z)  # c ∫ sqrt(V) dW1
        s2 = np.maximum(c2 * IV, 1e-300)

        for k in range(nsp):
            y0 = Y[k]
            y1 = y0 + c2 * IV / y0 - noise
            y1 = np.maximum(np.abs(y1), 1e-10)
            inside = (y1 > YL) & (y1 < YU)
            # Brownian-bridge crossing probabilities (continuous monitoring)
            dU = np.maximum(YU - y0, 0.0) * np.maximum(YU - y1, 0.0)
            dL = np.maximum(y0 - YL, 0.0) * np.maximum(y1 - YL, 0.0)
            pU = np.exp(-2.0 * dU / s2)
            pL = np.exp(-2.0 * dL / s2)
            surv = np.clip(1.0 - pU - pL, 0.0, 1.0)
            W[k] = W[k] * np.where(inside, surv, 0.0)
            Y[k] = y1
        V = V1

    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    S_T = 1.0 / Y[1]
    X = np.column_stack([S_T, V])
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
    res = solve_sde(num_paths=20000, dt=0.002, T=1.0, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
