import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    S0 = 0.5
    sig0 = 0.5
    nu = 0.4
    strikes = [0.434061722697, 0.5, 0.575954955084]

    out = np.empty((num_paths, 2))
    chunk = 500000
    drift = -0.5 * nu * nu * dt
    vol = nu * np.sqrt(dt)

    start = 0
    while start < num_paths:
        m = min(chunk, num_paths - start)
        sig = np.full(m, sig0)
        s2_prev = sig * sig
        V = np.zeros(m)
        # exact GBM for sigma, trapezoidal integrated variance
        for _ in range(Nt):
            z = rng.standard_normal(m)
            sig = sig * np.exp(drift + vol * z)
            s2 = sig * sig
            V += 0.5 * (s2_prev + s2) * dt
            s2_prev = s2
        # With zero correlation, S is a time-changed BESQ(0) on clock V:
        # S_T = (V/2) * Gamma(N), N ~ Poisson(2 S0 / V); N=0 -> absorbed at 0
        lam = 2.0 * S0 / V
        N = rng.poisson(lam)
        S = np.zeros(m)
        pos = N > 0
        if np.any(pos):
            S[pos] = 0.5 * V[pos] * rng.gamma(N[pos].astype(float))
        out[start:start + m, 0] = S
        out[start:start + m, 1] = sig
        start += m

    ST = out[:, 0]
    functionals = {}
    for i, K in enumerate(strikes, 1):
        functionals[f"price_K{i}"] = float(np.mean(np.maximum(ST - K, 0.0)))

    return {
        "terminal_paths": out,
        "empirical_mean": [float(v) for v in out.mean(axis=0)],
        "empirical_variance": [float(v) for v in out.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.005, T=2, seed=42)
    print(r["empirical_mean"], r["functionals"])
