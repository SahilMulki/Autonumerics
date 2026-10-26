import numpy as np


def _erfc(x):
    # Numerical Recipes erfc approximation (rel. err < 1.2e-7)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0.0, ans, 2.0 - ans)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    h = T / Nt

    S0, sig0 = 0.07, 0.34
    nu, rho = 0.8, -0.6
    rhob = np.sqrt(1.0 - rho * rho)
    psi_c = 1.5
    sqh = np.sqrt(h)
    inv_sqrt2 = 1.0 / np.sqrt(2.0)

    N = num_paths
    S = np.full(N, S0)
    sig = np.full(N, sig0)

    for _ in range(Nt):
        Zs = rng.standard_normal(N)
        Zp = rng.standard_normal(N)

        idx = np.flatnonzero(S > 0.0)
        if idx.size > 0:
            Sa = S[idx]
            sa = sig[idx]
            Z = rho * Zs[idx] + rhob * Zp[idx]
            # Moments of zero-drift CEV(1/2) step with frozen sigma: mean Sa, var sa^2*Sa*h
            psi = np.minimum(sa * sa * h / Sa, 1e300)
            new = np.empty_like(Sa)

            q = psi <= psi_c
            if np.any(q):
                inv = 2.0 / psi[q]
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
                a = Sa[q] / (1.0 + b2)
                new[q] = a * (np.sqrt(b2) + Z[q]) ** 2

            e = ~q
            if np.any(e):
                pe = psi[e]
                p = (pe - 1.0) / (pe + 1.0)
                beta = 2.0 / (Sa[e] * (pe + 1.0))
                one_minus_u = 0.5 * _erfc(Z[e] * inv_sqrt2)
                one_minus_p = 1.0 - p
                absorbed = one_minus_u >= one_minus_p
                safe_u = np.where(absorbed, 1.0, one_minus_u)
                val = np.log(one_minus_p / safe_u) / beta
                new[e] = np.where(absorbed, 0.0, np.maximum(val, 0.0))

            S[idx] = new

        # exact lognormal update for sigma
        sig = sig * np.exp(nu * sqh * Zs - 0.5 * nu * nu * h)

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    terminal = np.column_stack([S, sig])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["functionals"], res["empirical_mean"])
