import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    kappa, eta, sigma, rho, v0 = 2.58, 0.043, 1.0, -0.36, 0.114
    c = 0.01
    lo_S, hi_S = 50.0, 150.0
    # Y = 1/S: dY = c^2 V / Y dt - c sqrt(V) dW1  (time-changed Bessel-3)
    L = 1.0 / hi_S
    U = 1.0 / lo_S

    spots = [75.0, 100.0, 125.0]
    n = num_paths

    # Exact CIR transition parameters
    ekdt = np.exp(-kappa * dt)
    c0 = sigma ** 2 * (1.0 - ekdt) / (4.0 * kappa)
    df = 4.0 * kappa * eta / sigma ** 2
    srho = np.sqrt(1.0 - rho ** 2)

    V = np.full(n, v0)
    Ys = [np.full(n, 1.0 / s0) for s0 in spots]
    W = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        nonc = V * ekdt / c0
        Vn = c0 * rng.noncentral_chisquare(df, nonc)
        IV = 0.5 * (V + Vn) * dt
        intW2 = (Vn - V - kappa * eta * dt + kappa * IV) / sigma
        Z = rng.standard_normal(n)
        intW1 = rho * intW2 + srho * np.sqrt(IV) * Z
        noise = -c * intW1
        b = c * c * IV
        s2 = np.maximum(b, 1e-300)
        for k in range(3):
            Y0 = Ys[k]
            a = Y0 + noise
            # drift-implicit step: Y1 = a + b / Y1  -> positive root
            Y1 = 0.5 * (a + np.sqrt(a * a + 4.0 * b))
            w = W[k]
            inside = (Y1 > L) & (Y1 < U) & (Y0 > L) & (Y0 < U)
            dlo = np.where(inside, (Y0 - L) * (Y1 - L), 0.0)
            dhi = np.where(inside, (U - Y0) * (U - Y1), 0.0)
            p = np.exp(-2.0 * dlo / s2) + np.exp(-2.0 * dhi / s2)
            surv = np.where(inside, np.clip(1.0 - p, 0.0, 1.0), 0.0)
            W[k] = w * surv
            Ys[k] = Y1
        V = Vn

    S100 = 1.0 / Ys[1]
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(out["functionals"])
    print(out["empirical_mean"], out["empirical_variance"])
