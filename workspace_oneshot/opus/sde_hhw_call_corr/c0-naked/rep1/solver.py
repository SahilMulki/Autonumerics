import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    sdt = np.sqrt(dt)

    kappa, theta, sigma = 0.5, 0.04, 1.0
    a_r, b_r, c_r = 0.08, 0.10, 0.09
    V0, R0 = 0.04, 0.10
    spots = (75.0, 100.0, 125.0)
    K = 100.0

    C = np.array([[1.0, -0.9, 0.6],
                  [-0.9, 1.0, -0.7],
                  [0.6, -0.7, 1.0]])
    L = np.linalg.cholesky(C)
    l10 = L[1, 0]; l11 = L[1, 1]
    l20 = L[2, 0]; l21 = L[2, 1]; l22 = L[2, 2]

    rng = np.random.default_rng(seed)

    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)

    chunk = 100000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X = np.zeros(n)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)
        Vp = np.empty(n)
        sq = np.empty(n)
        dW1 = np.empty(n)
        dW2 = np.empty(n)
        dW3 = np.empty(n)
        Rn = np.empty(n)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            Z *= sdt
            z0, z1, z2 = Z[0], Z[1], Z[2]
            dW1[:] = z0
            np.multiply(z0, l10, out=dW2); dW2 += l11 * z1
            np.multiply(z0, l20, out=dW3); dW3 += l21 * z1; dW3 += l22 * z2

            np.maximum(V, 0.0, out=Vp)
            np.sqrt(Vp, out=sq)

            # log-asset (Euler on log S, full truncation)
            X += (R - 0.5 * Vp) * dt
            X += sq * dW1
            # variance: full truncation Euler
            V += kappa * (theta - Vp) * dt
            V += sigma * sq * dW2
            # short rate: Euler (OU)
            np.multiply(R, 1.0 - a_r * dt, out=Rn)
            Rn += a_r * b_r * dt
            Rn += c_r * dW3
            # trapezoidal integral of R
            I += 0.5 * (R + Rn) * dt
            R, Rn = Rn, R

        X_all[start:start + n] = X
        V_all[start:start + n] = np.maximum(V, 0.0)
        R_all[start:start + n] = R
        I_all[start:start + n] = I
        start += n

    D = np.exp(-I_all)
    eX = np.exp(X_all)
    funcs = {}
    for s0 in spots:
        ST = s0 * eX
        price = float(np.mean(D * np.maximum(ST - K, 0.0)))
        funcs["price_S%d" % int(s0)] = price

    S100 = 100.0 * eX
    terminal = np.column_stack([S100, V_all, R_all])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": funcs,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
