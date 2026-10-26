import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0 = 0.07
    sig0 = 0.34
    nu = 0.8
    rho = -0.6
    rhob = np.sqrt(1.0 - rho * rho)
    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138

    # Paths with sqrt(S)/(sigma*sqrt(dt)) < c use an exact squared-Bessel step
    # (dimension 0, absorbing at zero); the rest use Euler.
    c = 5.0
    c2dt = c * c * dt

    n = num_paths
    S = np.full(n, S0, dtype=np.float64)
    sig = np.full(n, sig0, dtype=np.float64)
    sq = np.sqrt(dt)
    sig_drift = -0.5 * nu * nu * dt

    for _ in range(Nt):
        Z = rng.standard_normal((2, n))
        dWv = Z[0] * sq                                  # increment driving sigma
        dWp = (rho * Z[0] + rhob * Z[1]) * sq            # increment driving S

        sqrtS = np.sqrt(S)
        Snew = S + sig * sqrtS * dWp                     # Euler step

        near = (S > 0.0) & (S < c2dt * sig * sig)
        idx = np.flatnonzero(near)
        if idx.size:
            Sn = S[idx]
            sg = sig[idx]
            # correlated part as an Euler shift, then exact BESQ^0 step
            # for the orthogonal part with variance (1-rho^2) sigma^2 dt
            Sp = Sn + rho * sg * np.sqrt(Sn) * dWv[idx]
            np.maximum(Sp, 0.0, out=Sp)
            v = (1.0 - rho * rho) * sg * sg * dt
            lam = 2.0 * Sp / v
            N = rng.poisson(lam)
            g = rng.gamma(np.maximum(N, 1).astype(np.float64))
            Snew[idx] = np.where(N > 0, 0.5 * v * g, 0.0)

        np.maximum(Snew, 0.0, out=Snew)
        S = Snew
        sig *= np.exp(nu * dWv + sig_drift)

    terminal = np.column_stack([S, sig])
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": {
            "price_K1": p1,
            "price_K2": p2,
            "price_K3": p3,
        },
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["functionals"])
