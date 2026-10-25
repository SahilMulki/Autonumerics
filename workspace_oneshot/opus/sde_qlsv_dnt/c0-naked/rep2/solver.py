import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    c = 0.01
    spots = [75.0, 100.0, 125.0]
    S_lo, S_hi = 50.0, 150.0

    # X = 1/(c S): dX = -sqrt(V) dW1 + (V/X) dt  (BES(3) in integrated-variance clock)
    X_L = 1.0 / (c * S_hi)   # S = 150
    X_U = 1.0 / (c * S_lo)   # S = 50

    n = num_paths
    m = len(spots)
    X = np.empty((m, n))
    for k, s0 in enumerate(spots):
        X[k, :] = 1.0 / (c * s0)
    W = np.ones((m, n))  # survival weights (Brownian-bridge conditional)

    V = np.full(n, V0)
    ekdt = np.exp(-kappa * dt)
    cc = sigma ** 2 * (1.0 - ekdt) / (4.0 * kappa)
    df = 4.0 * kappa * eta / sigma ** 2
    srho = np.sqrt(1.0 - rho ** 2)

    for _ in range(Nt):
        nonc = V * ekdt / cc
        V1 = cc * rng.noncentral_chisquare(df, nonc, size=n)
        dI = 0.5 * (V + V1) * dt
        dI = np.maximum(dI, 1e-300)
        A = (V1 - V - kappa * eta * dt + kappa * dI) / sigma  # ~ int sqrt(V) dW2
        Z = rng.standard_normal(n)
        E = rng.exponential(1.0, size=n)
        dB = -rho * A - srho * np.sqrt(dI) * Z
        extra = 2.0 * dI * E

        for k in range(m):
            X0 = X[k]
            X1 = np.sqrt((X0 + dB) ** 2 + extra)
            alive = W[k] > 0.0
            inside = (X1 > X_L) & (X1 < X_U) & (X0 > X_L) & (X0 < X_U)
            aL = np.clip((X0 - X_L) * (X1 - X_L), 0.0, None)
            aU = np.clip((X_U - X0) * (X_U - X1), 0.0, None)
            pL = np.exp(-2.0 * aL / dI)
            pU = np.exp(-2.0 * aU / dI)
            surv = (1.0 - pL) * (1.0 - pU)
            surv = np.where(inside, np.clip(surv, 0.0, 1.0), 0.0)
            W[k] = np.where(alive, W[k] * surv, 0.0)
            X[k] = X1
        V = V1

    S100 = 1.0 / (c * X[1])
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"], res["empirical_mean"])
