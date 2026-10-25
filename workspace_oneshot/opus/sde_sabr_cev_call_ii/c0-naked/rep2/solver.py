import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    S0, sig0 = 0.07, 0.4
    alpha, rho = 0.8, -0.6
    rhobar2 = 1.0 - rho * rho

    n = num_paths
    S = np.full(n, S0)
    sig = np.full(n, sig0)
    idx = np.arange(n)

    a = -0.5 * alpha * alpha * dt
    b = alpha * np.sqrt(dt)
    LAM_MAX = 1e12

    for _ in range(Nt):
        Z = rng.standard_normal(n)
        sig1 = sig * np.exp(a + b * Z)
        if idx.size > 0:
            s0 = sig[idx]
            s1 = sig1[idx]
            x = S[idx]
            # correlated part: int sigma sqrt(S) rho dW_sigma ~ rho/alpha sqrt(S) dsigma
            xp = x + (rho / alpha) * np.sqrt(x) * (s1 - s0)
            np.maximum(xp, 0.0, out=xp)
            # uncorrelated part: exact zero-dimensional squared Bessel transition
            v = rhobar2 * 0.5 * (s0 * s0 + s1 * s1) * dt
            with np.errstate(divide="ignore", invalid="ignore"):
                lam = np.where(v > 0, 2.0 * xp / v, np.inf)
            big = lam > LAM_MAX
            lam_c = np.where(big, 0.0, lam)
            N = rng.poisson(lam_c)
            xn = np.zeros_like(xp)
            m = N > 0
            if np.any(m):
                xn[m] = 0.5 * v[m] * rng.standard_gamma(N[m].astype(np.float64))
            if np.any(big):
                xn[big] = xp[big]
            S[idx] = xn
            keep = xn > 0
            idx = idx[keep]
        sig = sig1

    terminal = np.column_stack([S, sig])
    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
