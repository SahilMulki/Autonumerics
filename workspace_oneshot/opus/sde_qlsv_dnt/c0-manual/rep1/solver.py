import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Model parameters
    kappa, eta, sig_v, V0 = 2.58, 0.043, 1.0, 0.114
    rho = -0.36
    c_loc = 0.01
    S_lo, S_hi = 50.0, 150.0
    spots = [75.0, 100.0, 125.0]

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Work in U = 1/(c S) = 100/S:  dU = -sqrt(V) dW1 + V/U dt
    # Barriers in U: S=150 -> U_lo, S=50 -> U_hi
    U_lo = 1.0 / (c_loc * S_hi)
    U_hi = 1.0 / (c_loc * S_lo)

    # Exact transition of CIR variance (noncentral chi-square)
    ekdt = np.exp(-kappa * dt)
    c_chi = sig_v**2 * (1.0 - ekdt) / (4.0 * kappa)
    df = 4.0 * kappa * eta / sig_v**2
    srho = np.sqrt(1.0 - rho**2)

    V = np.full(num_paths, V0, dtype=float)
    U = [np.full(num_paths, 1.0 / (c_loc * s0), dtype=float) for s0 in spots]
    W = [np.ones(num_paths, dtype=float) for _ in spots]  # conditional survival weights

    for n in range(Nt):
        nonc = V * ekdt / c_chi
        V_new = c_chi * rng.noncentral_chisquare(df, nonc)
        IV = 0.5 * (V + V_new) * dt                      # integrated variance over step
        I2 = (V_new - V - kappa * eta * dt + kappa * IV) / sig_v  # int sqrt(V) dW2
        Z = rng.standard_normal(num_paths)
        mart = -(rho * I2 + srho * np.sqrt(IV) * Z)      # -int sqrt(V) dW1
        IVs = np.maximum(IV, 1e-300)

        for k in range(len(spots)):
            Uo = U[k]
            Un = Uo + mart + IV / Uo
            inside = (Un > U_lo) & (Un < U_hi)
            # Brownian-bridge continuous-monitoring survival within the step
            a = np.maximum(Uo - U_lo, 0.0) * np.maximum(Un - U_lo, 0.0)
            b = np.maximum(U_hi - Uo, 0.0) * np.maximum(U_hi - Un, 0.0)
            p_surv = (1.0 - np.exp(-2.0 * a / IVs)) * (1.0 - np.exp(-2.0 * b / IVs))
            W[k] = np.where(inside, W[k] * p_surv, 0.0)
            # keep U positive (BES(3)-like process never hits 0) for terminal output
            U[k] = np.maximum(np.abs(Un), 1e-12)
        V = V_new

    funcs = {
        "no_touch_S75": float(np.mean(W[0])),
        "no_touch_S100": float(np.mean(W[1])),
        "no_touch_S125": float(np.mean(W[2])),
    }

    S_T = 1.0 / (c_loc * U[1])
    X = np.column_stack([S_T, V])
    return {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": funcs,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.002, T=1.0, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
