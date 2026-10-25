import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    S0, sig0, nu = 0.5, 0.5, 0.4
    strikes = {"price_K1": 0.434061722697, "price_K2": 0.5, "price_K3": 0.575954955084}

    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    S_T = np.empty(num_paths)
    sig_T = np.empty(num_paths)

    chunk = 500000
    for start in range(0, num_paths, chunk):
        m = min(chunk, num_paths - start)
        # sigma: exact lognormal stepping; integrated variance by trapezoid
        sig = np.full(m, sig0)
        V = np.zeros(m)
        s2_prev = sig * sig
        drift = -0.5 * nu * nu * dt
        for _ in range(Nt):
            sig = sig * np.exp(nu * sqdt * rng.standard_normal(m) + drift)
            s2 = sig * sig
            V += 0.5 * (s2_prev + s2) * dt
            s2_prev = s2
        # S | sigma-path (zero correlation): time-changed BESQ(0)/4 with clock V.
        # Exact transition: S_T = (V/2) * Gamma(N), N ~ Poisson(2 S0 / V); N=0 -> absorbed at 0.
        N = rng.poisson(2.0 * S0 / V)
        G = rng.gamma(np.maximum(N, 1).astype(float), 1.0)
        S = np.where(N > 0, 0.5 * V * G, 0.0)
        S_T[start:start + m] = S
        sig_T[start:start + m] = sig

    # Control variate using the martingale property E[S_T] = S0
    dS = S_T - S0
    mS = dS.mean()
    varS = np.var(S_T, ddof=1)
    functionals = {}
    for name, K in strikes.items():
        pay = np.maximum(S_T - K, 0.0)
        cov = np.mean((pay - pay.mean()) * (dS - mS))
        b = cov / varS if varS > 0 else 0.0
        functionals[name] = float(pay.mean() - b * mS)

    paths = np.column_stack([S_T, sig_T])
    return {
        "terminal_paths": paths,
        "empirical_mean": [float(np.mean(S_T)), float(np.mean(sig_T))],
        "empirical_variance": [float(np.var(S_T, ddof=1)), float(np.var(sig_T, ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(r["empirical_mean"], r["empirical_variance"])
    print(r["functionals"])
