import numpy as np


def _erfc(x):
    # Numerical Recipes erfcc: fractional error < 1.2e-7 everywhere
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (
        0.09678418 + t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (
            1.48851587 + t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0, ans, 2.0 - ans)


def _norm_sf(z):
    # Q(z) = P(N(0,1) > z)
    return 0.5 * _erfc(z / np.sqrt(2.0))


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    S0, sig0 = 0.07, 0.4
    nu, rho = 0.8, -0.6
    rhobar = np.sqrt(1.0 - rho * rho)
    strikes = {"price_K1": 0.051022538988,
               "price_K2": 0.07,
               "price_K3": 0.096035989138}

    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    S_full = np.zeros(num_paths)
    sig_full = np.empty(num_paths)
    t_death = np.full(num_paths, np.nan)

    idx = np.arange(num_paths)
    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, sig0, dtype=float)

    sig_drift = -0.5 * nu * nu * dt
    sig_vol = nu * sqdt
    psi_c = 1.5

    for n in range(Nt):
        k = S.size
        if k == 0:
            break
        Z1 = rng.standard_normal(k)
        Z2 = rng.standard_normal(k)
        Zs = rho * Z1 + rhobar * Z2

        # Andersen QE step for dS = sig_n sqrt(S) dW (zero drift, left-point sigma),
        # driven by the correlated normal Zs (Gaussian copula). Mean preserved exactly.
        m = S
        psi = sig * sig * dt / S
        Snew = np.empty(k)

        quad = psi <= psi_c
        if np.any(quad):
            pq = psi[quad]
            inv = 2.0 / pq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m[quad] / (1.0 + b2)
            Snew[quad] = a * (np.sqrt(b2) + Zs[quad]) ** 2
        ex = ~quad
        if np.any(ex):
            pe = psi[ex]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m[ex]
            Q = _norm_sf(Zs[ex])  # = 1 - U
            Q = np.maximum(Q, 1e-300)
            val = np.log((1.0 - p) / Q) / beta
            Snew[ex] = np.where(Q >= 1.0 - p, 0.0, val)

        # exact lognormal volatility
        sig = sig * np.exp(sig_drift + sig_vol * Z1)
        S = Snew

        dead = S <= 0.0
        if np.any(dead):
            di = idx[dead]
            sig_full[di] = sig[dead]
            t_death[di] = (n + 1) * dt
            alive = ~dead
            idx = idx[alive]
            S = S[alive]
            sig = sig[alive]

    # surviving paths
    S_full[idx] = S
    sig_full[idx] = sig

    # absorbed paths: continue sigma exactly from absorption time to T
    dmask = ~np.isnan(t_death)
    nd = int(dmask.sum())
    if nd > 0:
        tau = np.maximum(T - t_death[dmask], 0.0)
        Zd = rng.standard_normal(nd)
        sig_full[dmask] = sig_full[dmask] * np.exp(-0.5 * nu * nu * tau + nu * np.sqrt(tau) * Zd)
        S_full[dmask] = 0.0

    terminal = np.column_stack([S_full, sig_full])
    functionals = {name: float(np.mean(np.maximum(S_full - K, 0.0)))
                   for name, K in strikes.items()}
    functionals["mean_S_T"] = float(np.mean(S_full))
    functionals["absorbed_fraction"] = float(nd / num_paths)

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
