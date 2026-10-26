import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    S0, sig0 = 0.07, 0.34
    nu, rho = 0.8, -0.6
    rbar2 = 1.0 - rho * rho
    sqdt = np.sqrt(dt)
    lam_cut = 50.0

    n = int(num_paths)
    S = np.full(n, S0, dtype=np.float64)
    s = np.full(n, sig0, dtype=np.float64)

    drift = -0.5 * nu * nu * dt
    for _ in range(Nt):
        z = rng.standard_normal(n)
        sn = s * np.exp(nu * sqdt * z + drift)

        # correlated part: frozen sqrt(S), exact integral of sigma dW_sigma = dsigma/nu
        S = S + (rho / nu) * np.sqrt(S) * (sn - s)
        np.maximum(S, 0.0, out=S)

        # uncorrelated part: frozen-sigma CEV(1/2) = scaled BESQ(0), absorbing at 0
        v = rbar2 * 0.5 * (s * s + sn * sn) * dt
        with np.errstate(divide="ignore", invalid="ignore"):
            lam = 2.0 * S / v
        alive = S > 0.0
        exact = alive & (lam < lam_cut)
        approx = alive & ~exact

        z2 = rng.standard_normal(n)
        Snew = S + np.sqrt(S * v) * z2
        np.maximum(Snew, 0.0, out=Snew)
        Snew = np.where(approx, Snew, S)

        idx = np.nonzero(exact)[0]
        if idx.size:
            N = rng.poisson(lam[idx])
            G = rng.standard_gamma(N.astype(np.float64))
            Snew[idx] = 0.5 * v[idx] * G

        Snew[~alive] = 0.0
        S = Snew
        s = sn

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    terminal = np.column_stack([S, s])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(s))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(s, ddof=1))],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
