import numpy as np

KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.114
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)
PSI_C = 1.5


def _simulate_chunk(rng, n, Nt, dt):
    """Andersen QE scheme with martingale correction. Returns log-return X_T and V_T."""
    k, eta, sig, rho = KAPPA, ETA, SIGMA, RHO
    ek = np.exp(-k * dt)
    c1 = sig * sig * ek * (1.0 - ek) / k
    c2 = eta * sig * sig * (1.0 - ek) ** 2 / (2.0 * k)
    g1 = g2 = 0.5
    K1 = g1 * dt * (k * rho / sig - 0.5) - rho / sig
    K2 = g2 * dt * (k * rho / sig - 0.5) + rho / sig
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4

    V = np.full(n, V0)
    X = np.zeros(n)
    for _ in range(Nt):
        m = eta + (V - eta) * ek
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        quad = psi <= PSI_C

        # quadratic branch (computed on safe values)
        psi_q = np.minimum(psi, PSI_C)
        ip = 2.0 / psi_q
        b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(np.maximum(ip - 1.0, 0.0))
        a = m / (1.0 + b2)
        # exponential branch
        psi_e = np.maximum(psi, PSI_C)
        p = (psi_e - 1.0) / (psi_e + 1.0)
        beta = (1.0 - p) / m

        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Vq = a * (np.sqrt(b2) + Zv) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            Ve = np.where(U <= p, 0.0,
                          np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)) / beta)
        Vn = np.where(quad, Vq, Ve)

        # martingale correction
        den = 1.0 - 2.0 * A * a
        lnMq = A * b2 * a / den - 0.5 * np.log(den)
        lnMe = np.log(p + beta * (1.0 - p) / (beta - A))
        lnM = np.where(quad, lnMq, lnMe)

        Zs = rng.standard_normal(n)
        X += (-lnM + K2 * Vn - 0.5 * K3 * V
              + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs)
        V = Vn
    return X, V


def _cv_price(S0, X):
    ST = S0 * np.exp(X)
    Y = np.maximum(ST - STRIKE, 0.0)
    C = ST - S0  # E[C] = 0 (martingale)
    Cc = C - C.mean()
    varC = np.dot(Cc, Cc)
    b = np.dot(Y - Y.mean(), Cc) / varC if varC > 0 else 0.0
    return float(Y.mean() - b * C.mean())


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    X = np.empty(num_paths)
    V = np.empty(num_paths)
    chunk = 100000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        Xc, Vc = _simulate_chunk(rng, n, Nt, dt)
        X[start:start + n] = Xc
        V[start:start + n] = Vc
        start += n

    functionals = {
        "price_S75": _cv_price(75.0, X),
        "price_S100": _cv_price(100.0, X),
        "price_S125": _cv_price(125.0, X),
    }

    S = 100.0 * np.exp(X)
    terminal = np.column_stack([S, V])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(200000, 0.002, 1.0, 42)
    print(out["functionals"], out["empirical_mean"])
