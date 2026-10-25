import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)

    kappa, theta = 0.5, 0.04
    a_r, b_r, sig_r = 0.08, 0.10, 0.09
    V0, R0 = 0.04, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    C = np.array([[1.0, -0.9, 0.6],
                  [-0.9, 1.0, -0.7],
                  [0.6, -0.7, 1.0]])
    L = np.linalg.cholesky(C)

    rng = np.random.default_rng(seed)

    logM = np.zeros(num_paths)          # log(S_T / S0), independent of S0
    V = np.full(num_paths, V0)
    R = np.full(num_paths, R0)
    I = np.zeros(num_paths)             # integral of R dt (left point, consistent with S drift)

    for _ in range(Nt):
        Z = rng.standard_normal((3, num_paths))
        dW = (L @ Z)
        dW *= sqdt
        Vp = np.maximum(V, 0.0)         # full truncation
        sv = np.sqrt(Vp)
        logM += (R - 0.5 * Vp) * dt + sv * dW[0]
        I += R * dt
        V += kappa * (theta - Vp) * dt + sv * dW[1]
        R += a_r * (b_r - R) * dt + sig_r * dW[2]

    M = np.exp(logM)
    D = np.exp(-I)
    DM = D * M  # discrete martingale with E = 1 exactly under this scheme

    functionals = {}
    for S0 in spots:
        Y = D * np.maximum(S0 * M - K, 0.0)
        X = S0 * DM
        Xc = X - X.mean()
        varX = np.mean(Xc * Xc)
        beta = np.mean(Xc * (Y - Y.mean())) / varX if varX > 0 else 0.0
        price = float(Y.mean() - beta * (X.mean() - S0))
        functionals["price_S%d" % int(S0)] = price

    S = 100.0 * M
    terminal = np.column_stack([S, np.maximum(V, 0.0), R])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
