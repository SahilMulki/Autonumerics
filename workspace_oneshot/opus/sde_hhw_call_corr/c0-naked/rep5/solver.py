import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    # parameters
    kappa, theta, xi, V0 = 0.5, 0.04, 1.0, 0.04
    a, b, sig_r, R0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    C = np.array([[1.0, -0.9, 0.6],
                  [-0.9, 1.0, -0.7],
                  [0.6, -0.7, 1.0]])
    L = np.linalg.cholesky(C)
    l10, l11 = L[1, 0], L[1, 1]
    l20, l21, l22 = L[2, 0], L[2, 1], L[2, 2]

    sqdt = np.sqrt(dt)
    ea = np.exp(-a * dt)
    r_mean_add = b * (1.0 - ea)
    # exact OU std over one step, expressed per unit normal
    r_std = sig_r * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X = np.zeros(n)            # log(S/S0)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)            # integral of R (left-point, consistent with drift)
        Vp = np.empty(n)
        sv = np.empty(n)
        W2 = np.empty(n)
        W3 = np.empty(n)
        tmp = np.empty(n)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            z0, z1, z2 = Z[0], Z[1], Z[2]
            # correlated standard normals
            np.multiply(z0, l10, out=W2)
            W2 += l11 * z1
            np.multiply(z0, l20, out=W3)
            W3 += l21 * z1
            W3 += l22 * z2
            # full truncation
            np.maximum(V, 0.0, out=Vp)
            np.sqrt(Vp, out=sv)
            sv *= sqdt
            # log-asset (exact martingale increment given Vp, R)
            np.multiply(R, dt, out=tmp)
            I += tmp
            X += tmp
            X -= 0.5 * dt * Vp
            np.multiply(sv, z0, out=tmp)
            X += tmp
            # variance: full-truncation Euler
            V += kappa * dt * (theta - Vp)
            np.multiply(sv, W2, out=tmp)
            V += xi * tmp
            # short rate: exact OU transition
            R *= ea
            R += r_mean_add
            R += r_std * W3
        X_all[start:start + n] = X
        V_all[start:start + n] = np.maximum(V, 0.0)
        R_all[start:start + n] = R
        I_all[start:start + n] = I
        start += n

    disc = np.exp(-I_all)
    eX = np.exp(X_all)
    functionals = {}
    for s0 in spots:
        payoff = np.maximum(s0 * eX - K, 0.0)
        functionals["price_S%d" % int(s0)] = float(np.mean(disc * payoff))

    S100 = 100.0 * eX
    terminal = np.column_stack([S100, V_all, R_all])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
