import numpy as np


def _norm_cdf(x):
    # Phi(x) = 0.5 * erfc(-x / sqrt(2)), erfc via Numerical Recipes Chebyshev fit (|err| < 1.2e-7)
    z = -x / np.sqrt(2.0)
    az = np.abs(z)
    t = 1.0 / (1.0 + 0.5 * az)
    y = t * np.exp(-az * az - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    erfc = np.where(z >= 0.0, y, 2.0 - y)
    return 0.5 * erfc


def _bs_call_zero_rate(F, K, w):
    w = np.maximum(w, 1e-14)
    sw = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / sw
    d2 = d1 - sw
    return F * _norm_cdf(d1) - K * _norm_cdf(d2)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, eta, sigma, rho, V0 = 2.58, 0.043, 1.0, -0.36, 0.164
    K = 100.0

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = int(num_paths)

    # Andersen QE scheme for the variance (handles V = 0 without changing its law)
    ek = np.exp(-kappa * dt)
    c1 = sigma**2 * ek * (1.0 - ek) / kappa
    c2 = eta * sigma**2 * (1.0 - ek) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    V = np.full(n, V0, dtype=float)
    intV = np.zeros(n)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for _ in range(Nt):
            Z = rng.standard_normal(n)
            U = rng.random(n)
            m = eta + (V - eta) * ek
            s2 = V * c1 + c2
            psi = s2 / (m * m)
            Vn = np.empty(n)

            q = psi <= psi_c
            if np.any(q):
                ip = 1.0 / psi[q]
                b2 = 2.0 * ip - 1.0 + np.sqrt(2.0 * ip) * np.sqrt(np.maximum(2.0 * ip - 1.0, 0.0))
                a = m[q] / (1.0 + b2)
                Vn[q] = a * (np.sqrt(b2) + Z[q]) ** 2
            e = ~q
            if np.any(e):
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = (1.0 - p) / m[e]
                Ue = U[e]
                Vn[e] = np.where(Ue <= p, 0.0,
                                 np.log((1.0 - p) / np.maximum(1.0 - Ue, 1e-300)) / beta)

            intV += 0.5 * (V + Vn) * dt
            V = Vn

    # Conditional (mixing) Monte Carlo: given the V path, log S is Gaussian
    I2 = (V - V0 - kappa * eta * T + kappa * intV) / sigma   # int sqrt(V) dW2
    xi = np.exp(rho * I2 - 0.5 * rho**2 * intV)
    w = (1.0 - rho**2) * intV

    xm = xi.mean()
    xc = xi - xm
    varx = np.dot(xc, xc) / (n - 1)

    functionals = {}
    for S0, name in ((75.0, "price_S75"), (100.0, "price_S100"), (125.0, "price_S125")):
        C = _bs_call_zero_rate(S0 * xi, K, w)
        cm = C.mean()
        b = np.dot(C - cm, xc) / (n - 1) / varx if varx > 0 else 0.0
        functionals[name] = float(cm - b * (xm - 1.0))

    Z3 = rng.standard_normal(n)
    S = 100.0 * xi * np.exp(-0.5 * w + np.sqrt(w) * Z3)
    X = np.column_stack([S, V])

    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)), float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": n,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=200000, dt=0.002, T=1.0, seed=42)
    print(r["functionals"], r["empirical_mean"])
