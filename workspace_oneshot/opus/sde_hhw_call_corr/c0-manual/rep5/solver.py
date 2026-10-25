import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # model parameters
    kappa, theta, sig_v, V0 = 0.5, 0.04, 1.0, 0.04
    a, b, sig_r, R0 = 0.08, 0.10, 0.09, 0.10
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    corr = np.array([[1.0, -0.9, 0.6],
                     [-0.9, 1.0, -0.7],
                     [0.6, -0.7, 1.0]])
    L = np.linalg.cholesky(corr)
    L10, L11 = L[1, 0], L[1, 1]
    L20, L21, L22 = L[2, 0], L[2, 1], L[2, 2]

    sdt = np.sqrt(dt)
    # exact OU transition for R
    ea = np.exp(-a * dt)
    cr = b * (1.0 - ea)
    sr = sig_r * np.sqrt((1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a))

    # Y = log(D*S/S0) (discrete martingale exponent), I = int R dt (trapezoid)
    Y_all = np.empty(num_paths)
    I_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)

    chunk = 131072
    for start in range(0, num_paths, chunk):
        m = min(chunk, num_paths - start)
        V = np.full(m, V0)
        R = np.full(m, R0)
        Y = np.zeros(m)
        I = np.zeros(m)
        Vp = np.empty(m)
        sq = np.empty(m)
        tmp = np.empty(m)
        Rn = np.empty(m)
        for n in range(Nt):
            Z = rng.standard_normal((3, m))
            Z0, Z1, Z2 = Z[0], Z[1], Z[2]
            np.maximum(V, 0.0, out=Vp)
            np.sqrt(Vp, out=sq)
            # log-asset (discounted): Y += -0.5 Vp dt + sqrt(Vp) dW1
            np.multiply(sq, Z0, out=tmp)
            tmp *= sdt
            Y += tmp
            np.multiply(Vp, 0.5 * dt, out=tmp)
            Y -= tmp
            # variance, full truncation Euler: dW2 = sdt*(L10 Z0 + L11 Z1)
            np.multiply(Z0, L10, out=tmp)
            tmp += L11 * Z1
            tmp *= sq
            tmp *= sig_v * sdt
            V += tmp
            V += kappa * dt * (theta - Vp)
            # short rate, exact OU step with correlated normal
            np.multiply(Z0, L20, out=tmp)
            tmp += L21 * Z1
            tmp += L22 * Z2
            np.multiply(R, ea, out=Rn)
            Rn += cr
            Rn += sr * tmp
            # trapezoid integral of R
            np.add(R, Rn, out=tmp)
            tmp *= 0.5 * dt
            I += tmp
            R, Rn = Rn, R
        Y_all[start:start + m] = Y
        I_all[start:start + m] = I
        V_all[start:start + m] = np.maximum(V, 0.0)
        R_all[start:start + m] = R

    D = np.exp(-I_all)
    eY = np.exp(Y_all)

    # Vasicek zero-coupon bond price (exact E[D])
    B = (1.0 - np.exp(-a * T)) / a
    lnA = (b - sig_r ** 2 / (2.0 * a ** 2)) * (B - T) - sig_r ** 2 * B ** 2 / (4.0 * a)
    P_vas = np.exp(lnA - B * R0)

    c1 = eY - 1.0          # E[D S_T / S0] - 1 = 0 exactly (discrete martingale)
    c2 = D - P_vas         # E[D] = P_vas
    C = np.vstack([c1, c2])
    Cm = C.mean(axis=1)
    Cc = C - Cm[:, None]
    covCC = Cc @ Cc.T / (num_paths - 1)

    functionals = {}
    for S0 in spots:
        pay = np.maximum(S0 * eY - K * D, 0.0)
        pm = pay.mean()
        covCP = Cc @ (pay - pm) / (num_paths - 1)
        try:
            beta = np.linalg.solve(covCC, covCP)
        except np.linalg.LinAlgError:
            beta = np.zeros(2)
        price = pm - beta @ Cm
        functionals["price_S%d" % int(S0)] = float(price)

    S_T = 100.0 * np.exp(Y_all + I_all)
    terminal = np.column_stack([S_T, V_all, R_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"])
