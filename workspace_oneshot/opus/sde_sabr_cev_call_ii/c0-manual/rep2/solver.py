import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Model parameters
    S0, sig0 = 0.07, 0.4
    nu, rho = 0.8, -0.6
    rhobar2 = 1.0 - rho * rho
    strikes = {"price_K1": 0.051022538988,
               "price_K2": 0.07,
               "price_K3": 0.096035989138}

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    n = num_paths
    S = np.full(n, S0, dtype=float)
    sig = np.full(n, sig0, dtype=float)

    drift_ls = -0.5 * nu * nu * dt
    vol_ls = nu * np.sqrt(dt)
    psi_c = 0.1  # switch: exact (Poisson-Gamma) for psi > psi_c, QE-quadratic otherwise

    for _ in range(Nt):
        # exact lognormal step for sigma
        Z1 = rng.standard_normal(n)
        sig_new = sig * np.exp(drift_ls + vol_ls * Z1)

        # correlated part: rho * sqrt(S) * int sigma dW_sigma = rho/nu * sqrt(S) * dsigma
        St = S + (rho / nu) * np.sqrt(S) * (sig_new - sig)
        np.maximum(St, 0.0, out=St)

        # orthogonal part: BESQ(0)-type step with integrated variance v
        v = rhobar2 * 0.5 * (sig * sig + sig_new * sig_new) * dt
        np.maximum(v, 1e-300, out=v)

        alive = St > 0.0
        psi = np.full(n, np.inf)
        np.divide(v, St, out=psi, where=alive)

        S_new = np.zeros(n)

        quad = alive & (psi <= psi_c)
        idx_q = np.nonzero(quad)[0]
        if idx_q.size:
            ps = psi[idx_q]
            inv = 2.0 / ps
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = St[idx_q] / (1.0 + b2)
            Zq = rng.standard_normal(idx_q.size)
            S_new[idx_q] = a * (np.sqrt(b2) + Zq) ** 2

        exact = alive & (psi > psi_c)
        idx_e = np.nonzero(exact)[0]
        if idx_e.size:
            ve = v[idx_e]
            lam = 2.0 * St[idx_e] / ve
            N = rng.poisson(lam)
            pos = N > 0
            if np.any(pos):
                G = rng.gamma(N[pos].astype(float), 1.0)
                vals = np.zeros(idx_e.size)
                vals[pos] = 0.5 * ve[pos] * G
                S_new[idx_e] = vals

        S = S_new
        sig = sig_new

    terminal = np.column_stack([S, sig])
    functionals = {k: float(np.mean(np.maximum(S - K, 0.0))) for k, K in strikes.items()}

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["functionals"])
