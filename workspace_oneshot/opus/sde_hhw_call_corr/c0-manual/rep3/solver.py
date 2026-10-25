import numpy as np

# Model parameters (Heston-Hull-White, all three factors correlated)
KAPPA_V, THETA_V, SIGMA_V, V0 = 0.5, 0.04, 1.0, 0.04
KAPPA_R, THETA_R, SIGMA_R, R0 = 0.08, 0.10, 0.09, 0.10
RHO_12, RHO_13, RHO_23 = -0.9, 0.6, -0.7
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    C = np.array([[1.0, RHO_12, RHO_13],
                  [RHO_12, 1.0, RHO_23],
                  [RHO_13, RHO_23, 1.0]])
    L = np.linalg.cholesky(C) * sqdt
    l00 = L[0, 0]
    l10, l11 = L[1, 0], L[1, 1]
    l20, l21, l22 = L[2, 0], L[2, 1], L[2, 2]

    # Storage of terminal quantities for all paths
    logM_all = np.empty(num_paths)   # log(S_T / S0)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)      # integral of R (left-point rule)

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        logM = np.zeros(n)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)
        Z = np.empty((3, n))
        dW1 = np.empty(n)
        dW2 = np.empty(n)
        dW3 = np.empty(n)
        Vp = np.empty(n)
        sv = np.empty(n)
        tmp = np.empty(n)

        for _ in range(Nt):
            rng.standard_normal(out=Z)
            z0, z1, z2 = Z[0], Z[1], Z[2]
            np.multiply(z0, l00, out=dW1)
            np.multiply(z0, l10, out=dW2)
            dW2 += l11 * z1
            np.multiply(z0, l20, out=dW3)
            dW3 += l21 * z1
            dW3 += l22 * z2

            # full truncation
            np.maximum(V, 0.0, out=Vp)
            np.sqrt(Vp, out=sv)

            # log-Euler for S: (R - V/2) dt + sqrt(V) dW1
            np.multiply(Vp, -0.5, out=tmp)
            tmp += R
            tmp *= dt
            logM += tmp
            np.multiply(sv, dW1, out=tmp)
            logM += tmp

            # discount integral, left point (makes D*S an exact discrete martingale)
            I += R * dt

            # variance: full-truncation Euler
            np.multiply(Vp, -KAPPA_V * dt, out=tmp)
            tmp += KAPPA_V * THETA_V * dt
            V += tmp
            np.multiply(sv, dW2, out=tmp)
            tmp *= SIGMA_V
            V += tmp

            # short rate: Euler (additive noise)
            np.multiply(R, -KAPPA_R * dt, out=tmp)
            tmp += KAPPA_R * THETA_R * dt
            R += tmp
            R += SIGMA_R * dW3

        logM_all[start:start + n] = logM
        V_all[start:start + n] = V
        R_all[start:start + n] = R
        I_all[start:start + n] = I
        start += n

    D = np.exp(-I_all)
    M = np.exp(logM_all)

    functionals = {}
    for s0 in SPOTS:
        ST = s0 * M
        Y = D * np.maximum(ST - STRIKE, 0.0)
        X = D * ST  # discrete martingale, E[X] = s0
        Xc = X - X.mean()
        varX = np.dot(Xc, Xc)
        beta = np.dot(Y - Y.mean(), Xc) / varX if varX > 0 else 0.0
        price = Y.mean() - beta * (X.mean() - s0)
        functionals["price_S%d" % int(s0)] = float(price)

    S100 = 100.0 * M
    terminal = np.column_stack([S100, np.maximum(V_all, 0.0), R_all])

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
    print(res["empirical_mean"], res["empirical_variance"])
