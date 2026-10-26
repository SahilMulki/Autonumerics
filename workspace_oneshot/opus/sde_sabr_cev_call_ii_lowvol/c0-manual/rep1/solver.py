import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    S0, sig0 = 0.07, 0.34
    nu, rho = 0.8, -0.6
    strikes = {"price_K1": 0.051022538988,
               "price_K2": 0.07,
               "price_K3": 0.096035989138}

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, sig0, dtype=float)

    rhobar2 = 1.0 - rho * rho
    sq_dt = np.sqrt(dt)
    drift_ln = -0.5 * nu * nu * dt
    LAM = 200.0  # above this Poisson intensity the exact law is effectively Gaussian

    for _ in range(Nt):
        z1 = rng.standard_normal(num_paths)
        z2 = rng.standard_normal(num_paths)

        # exact lognormal step for the volatility (martingale)
        sig_new = sig * np.exp(drift_ln + nu * sq_dt * z1)

        # uncorrelated integrated variance over the step (trapezoid)
        v = rhobar2 * 0.5 * (sig * sig + sig_new * sig_new) * dt
        v = np.maximum(v, 1e-300)

        # correlated part: rho * int sigma sqrt(S) dW_sigma ~ rho/nu sqrt(S_n) dsigma
        St = S + (rho / nu) * np.sqrt(S) * (sig_new - sig)
        np.maximum(St, 0.0, out=St)

        # uncorrelated part: CEV(1/2) = scaled BESQ(0), absorbing at zero
        Snew = St + np.sqrt(St * v) * z2
        lam = 2.0 * St / v
        small = (lam < LAM) & (St > 0.0)
        if small.any():
            idx = np.nonzero(small)[0]
            Np = rng.poisson(lam[idx])
            g = np.zeros(idx.size)
            m = Np > 0
            if m.any():
                g[m] = rng.gamma(Np[m].astype(float))
            Snew[idx] = 0.5 * v[idx] * g
        # absorbed paths (St == 0) stay at zero
        Snew[St <= 0.0] = 0.0
        np.maximum(Snew, 0.0, out=Snew)

        S = Snew
        sig = sig_new

    terminal = np.column_stack([S, sig])
    functionals = {name: float(np.mean(np.maximum(S - K, 0.0)))
                   for name, K in strikes.items()}

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
