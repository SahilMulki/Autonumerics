import math
import numpy as np

KAPPA = 2.58
THETA = 0.043
SIGMA = 1.0
V0 = 0.164
RHO = -0.36
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _norm_cdf(x):
    # N(x) = 0.5*erfc(-x/sqrt2); erfc via Chebyshev fit (fractional error < 1.2e-7)
    y = -x / math.sqrt(2.0)
    z = np.abs(y)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    erfc = np.where(y >= 0.0, ans, 2.0 - ans)
    return 0.5 * erfc


def _bs_call(F0, w, K):
    sw = np.sqrt(np.maximum(w, 1e-300))
    sw = np.maximum(sw, 1e-12)
    d1 = (np.log(F0 / K) + 0.5 * w) / sw
    d2 = d1 - sw
    return F0 * _norm_cdf(d1) - K * _norm_cdf(d2)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    em = math.exp(-KAPPA * dt)
    c1 = SIGMA ** 2 * em * (1.0 - em) / KAPPA
    c2 = THETA * SIGMA ** 2 * (1.0 - em) ** 2 / (2.0 * KAPPA)
    rho2 = RHO * RHO

    S_T = np.empty(num_paths)
    V_T = np.empty(num_paths)

    nS = len(SPOTS)
    sumC = np.zeros(nS)
    sumCF = np.zeros(nS)
    sumF = 0.0
    sumF2 = 0.0

    chunk = 200000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        IV = np.zeros(n)
        for _ in range(Nt):
            Z = rng.standard_normal(n)
            U = rng.random(n)
            m = THETA + (V - THETA) * em
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            mask = psi <= 1.5
            # quadratic branch
            inv = 2.0 / psi
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m / (1.0 + b2)
            Vq = a * (np.sqrt(b2) + Z) ** 2
            # exponential branch
            p = (psi - 1.0) / (psi + 1.0)
            beta = (1.0 - p) / m
            ratio = np.maximum((1.0 - p) / (1.0 - U), 1.0)
            Ve = np.where(U <= p, 0.0, np.log(ratio) / beta)
            Vn = np.where(mask, Vq, Ve)
            IV += 0.5 * (V + Vn) * dt
            V = Vn

        # integral of sqrt(V) dW2 from the variance dynamics
        I2 = (V - V0 - KAPPA * THETA * T + KAPPA * IV) / SIGMA
        F = np.exp(RHO * I2 - 0.5 * rho2 * IV)
        w = (1.0 - rho2) * IV

        for i, s0 in enumerate(SPOTS):
            C = _bs_call(s0 * F, w, STRIKE)
            sumC[i] += C.sum()
            sumCF[i] += (C * F).sum()
        sumF += F.sum()
        sumF2 += (F * F).sum()

        Zs = rng.standard_normal(n)
        S_T[start:start + n] = 100.0 * F * np.exp(np.sqrt(w) * Zs - 0.5 * w)
        V_T[start:start + n] = V
        start += n

    N = float(num_paths)
    meanF = sumF / N
    varF = sumF2 / N - meanF ** 2
    prices = {}
    for i, s0 in enumerate(SPOTS):
        meanC = sumC[i] / N
        covCF = sumCF[i] / N - meanC * meanF
        b = covCF / varF if varF > 0 else 0.0
        prices[i] = float(meanC - b * (meanF - 1.0))

    X = np.column_stack([S_T, V_T])
    return {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in X.mean(axis=0)],
        "empirical_variance": [float(v) for v in X.var(axis=0, ddof=1)],
        "functionals": {
            "price_S75": prices[0],
            "price_S100": prices[1],
            "price_S125": prices[2],
        },
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1.0, seed=42)
    print(r["functionals"], r["empirical_mean"])
