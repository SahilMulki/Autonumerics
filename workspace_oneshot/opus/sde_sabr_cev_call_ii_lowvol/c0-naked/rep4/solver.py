import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    S0 = 0.07
    sig0 = 0.34
    alpha = 0.8
    rho = -0.6
    strikes = {"price_K1": 0.051022538988,
               "price_K2": 0.07,
               "price_K3": 0.096035989138}

    Nt = max(1, int(round(T / dt)))
    dt = T / Nt
    sqdt = np.sqrt(dt)
    rhobar2 = 1.0 - rho * rho
    drift_ls = -0.5 * alpha * alpha * dt
    psi_c = 1.5

    rng = np.random.default_rng(seed)
    n = int(num_paths)
    S = np.full(n, S0, dtype=np.float64)
    sig = np.full(n, sig0, dtype=np.float64)

    for _ in range(Nt):
        # exact lognormal step for sigma
        Z1 = rng.standard_normal(n)
        sig_new = sig * np.exp(drift_ls + alpha * sqdt * Z1)

        # correlated part: rho * sqrt(S) * int sigma dW_sigma, with
        # int sigma dW_sigma = (sigma_{n+1} - sigma_n)/alpha exactly
        A = S + (rho / alpha) * np.sqrt(S) * (sig_new - sig)
        np.maximum(A, 0.0, out=A)

        # orthogonal part: zero-drift CIR/BESQ0 step via Andersen QE
        v = rhobar2 * 0.5 * (sig * sig + sig_new * sig_new) * dt
        pos = A > 0.0
        Asafe = np.where(pos, A, 1.0)
        psi = v / Asafe

        Z = rng.standard_normal(n)
        U = rng.random(n)

        # quadratic branch
        psiq = np.minimum(psi, psi_c)
        ip = 2.0 / psiq
        b2 = ip - 1.0 + np.sqrt(ip * (ip - 1.0))
        acoef = Asafe / (1.0 + b2)
        Sq = acoef * (np.sqrt(b2) + Z) ** 2

        # exponential branch with mass at zero
        psie = np.maximum(psi, psi_c)
        p = (psie - 1.0) / (psie + 1.0)
        beta = (1.0 - p) / Asafe
        Uc = np.minimum(U, 1.0 - 1e-16)
        Se = np.where(Uc <= p, 0.0,
                      np.log(np.maximum((1.0 - p) / (1.0 - Uc), 1.0)) / beta)

        S = np.where(pos, np.where(psi <= psi_c, Sq, Se), 0.0)
        sig = sig_new

    terminal = np.column_stack((S, sig))
    functionals = {}
    for name, K in strikes.items():
        functionals[name] = float(np.mean(np.maximum(S - K, 0.0)))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["empirical_mean"], out["functionals"])
