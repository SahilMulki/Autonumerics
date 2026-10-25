import math
import numpy as np


def _norm_cdf(x):
    # Abramowitz & Stegun 26.2.17, |error| < 7.5e-8
    x = np.asarray(x, dtype=np.float64)
    ax = np.abs(x)
    t = 1.0 / (1.0 + 0.2316419 * ax)
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
               + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * ax * ax) / math.sqrt(2.0 * math.pi)
    upper = pdf * poly  # = 1 - Phi(|x|)
    return np.where(x >= 0, 1.0 - upper, upper)


def _bs_call(F, K, w):
    sw = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / sw
    d2 = d1 - sw
    return F * _norm_cdf(d1) - K * _norm_cdf(d2)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, v0, rho = 2.58, 0.043, 1.0, 0.114, -0.36
    K = 100.0
    rng = np.random.default_rng(seed)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    ek = math.exp(-kappa * dt)
    c1 = sigma * sigma * ek * (1.0 - ek) / kappa
    c2 = theta * sigma * sigma * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    VT = np.empty(num_paths)
    IV = np.empty(num_paths)

    chunk = 250000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, v0)
        I = np.zeros(n)
        for _ in range(Nt):
            m = theta + (V - theta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Z = rng.standard_normal(n)
            U = rng.random(n)
            Vn = np.empty(n)
            q = psi <= psi_c
            if np.any(q):
                pq = psi[q]
                inv = 2.0 / pq
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
                a = m[q] / (1.0 + b2)
                Vn[q] = a * (np.sqrt(b2) + Z[q]) ** 2
            e = ~q
            if np.any(e):
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[e]
                Ue = U[e]
                ve = np.zeros(Ue.shape[0])
                pos = Ue > p
                ve[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
                Vn[e] = ve
            I += 0.5 * (V + Vn) * dt
            V = Vn
        VT[start:start + n] = V
        IV[start:start + n] = I
        start += n

    # Conditional (mixing) representation given the variance path
    M = (VT - v0 - kappa * theta * T + kappa * IV) / sigma  # = int sqrt(V) dW2
    xi = np.exp(rho * M - 0.5 * rho * rho * IV)
    xi /= xi.mean()  # martingale correction: E[S_T] = S0
    w = np.maximum((1.0 - rho * rho) * IV, 1e-300)

    functionals = {}
    for S0, name in ((75.0, "price_S75"), (100.0, "price_S100"), (125.0, "price_S125")):
        functionals[name] = float(np.mean(_bs_call(S0 * xi, K, w)))

    Zs = rng.standard_normal(num_paths)
    ST = 100.0 * xi * np.exp(-0.5 * w + np.sqrt(w) * Zs)

    terminal = np.column_stack([ST, VT])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(200000, 0.002, 1.0)
    print(out["functionals"], out["empirical_mean"])
