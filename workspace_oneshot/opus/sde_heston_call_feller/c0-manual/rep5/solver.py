import math
import numpy as np


KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
V0 = 0.114
RHO = -0.36
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _norm_cdf(x):
    # Phi(x) = 0.5 * erfc(-x / sqrt(2)); erfc via Chebyshev fit (rel. err < 1.2e-7)
    z = -x / math.sqrt(2.0)
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.5 * az)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))))
    ans = t * np.exp(-az * az + poly)
    erfc = np.where(z >= 0.0, ans, 2.0 - ans)
    return 0.5 * erfc


def _bs_call(F, K, w):
    # zero rate Black call with total variance w
    sw = np.sqrt(np.maximum(w, 1e-300))
    d1 = (np.log(F / K) + 0.5 * w) / sw
    d2 = d1 - sw
    return F * _norm_cdf(d1) - K * _norm_cdf(d2)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    ek = math.exp(-KAPPA * dt)
    c1 = SIGMA ** 2 * ek * (1.0 - ek) / KAPPA
    c2 = ETA * SIGMA ** 2 * (1.0 - ek) ** 2 / (2.0 * KAPPA)
    psi_c = 1.5
    rho2 = RHO * RHO

    chunk = 200000
    S_T = np.empty(num_paths)
    V_T = np.empty(num_paths)

    sumY = 0.0
    sumYY = 0.0
    sumC = {s: 0.0 for s in SPOTS}
    sumCY = {s: 0.0 for s in SPOTS}

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        intV = np.zeros(n)
        for _ in range(Nt):
            Z = rng.standard_normal(n)
            U = rng.random(n)
            m = ETA + (V - ETA) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            small = psi <= psi_c
            # quadratic branch
            psi_s = np.minimum(psi, psi_c)
            inv = 2.0 / psi_s
            b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
            a = m / (1.0 + b2)
            Vq = a * (np.sqrt(b2) + Z) ** 2
            # exponential branch
            psi_e = np.maximum(psi, psi_c)
            p = (psi_e - 1.0) / (psi_e + 1.0)
            beta = (1.0 - p) / m
            Ve = np.where(U > p, np.log((1.0 - p) / (1.0 - U)) / beta, 0.0)
            Vn = np.where(small, Vq, Ve)
            intV += 0.5 * (V + Vn) * dt
            V = Vn

        # integral of sqrt(V) dW2 recovered from the V dynamics
        I2 = (V - V0 - KAPPA * ETA * T + KAPPA * intV) / SIGMA
        X = RHO * I2 - 0.5 * rho2 * intV
        Y = np.exp(X)
        w = (1.0 - rho2) * intV

        sumY += float(np.sum(Y))
        sumYY += float(np.sum(Y * Y))
        for s in SPOTS:
            C = _bs_call(s * Y, STRIKE, w)
            sumC[s] += float(np.sum(C))
            sumCY[s] += float(np.sum(C * Y))

        Zs = rng.standard_normal(n)
        S_T[start:start + n] = 100.0 * Y * np.exp(np.sqrt(w) * Zs - 0.5 * w)
        V_T[start:start + n] = V
        start += n

    N = float(num_paths)
    mY = sumY / N
    varY = sumYY / N - mY * mY
    functionals = {}
    for s in SPOTS:
        mC = sumC[s] / N
        cov = sumCY[s] / N - mC * mY
        b = cov / varY if varY > 0 else 0.0
        price = mC - b * (mY - 1.0)
        functionals["price_S%d" % int(s)] = float(price)

    paths = np.column_stack([S_T, V_T])
    return {
        "terminal_paths": paths,
        "empirical_mean": [float(np.mean(paths[:, 0])), float(np.mean(paths[:, 1]))],
        "empirical_variance": [float(np.var(paths[:, 0], ddof=1)),
                               float(np.var(paths[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
