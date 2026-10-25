import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2:
        dS     = sigma * sqrt(S) dW_S
        dsigma = nu * sigma dW_sigma,  corr = rho

    Scheme (low-bias conditional scheme, Chen/Oosterlee/van der Weide style):
      * sigma is simulated exactly (lognormal).
      * The part of dS correlated with sigma uses the exact identity
        int sigma dW_sigma = (sigma_{n+1} - sigma_n) / nu, applied with sqrt(S_n) frozen.
      * The remaining independent part is a time-changed squared Bessel process of
        dimension 0 (absorbing at 0), which is sampled EXACTLY via the
        Poisson-Gamma mixture representation of the noncentral chi-square with
        zero degrees of freedom. This keeps absorption at 0 correct.
    """
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S0, sig0 = 0.07, 0.4
    nu, rho = 0.8, -0.6
    rb2 = 1.0 - rho * rho
    sqdt = np.sqrt(dt)
    drift_sig = -0.5 * nu * nu * dt

    n = int(num_paths)
    sig = np.full(n, sig0, dtype=float)
    S_a = np.full(n, S0, dtype=float)      # alive (S > 0) paths only
    ids = np.arange(n)
    big_thresh = 1e7

    for _ in range(Nt):
        Z2 = rng.standard_normal(n)
        sig_new = sig * np.exp(drift_sig + nu * sqdt * Z2)

        if ids.size > 0:
            so = sig[ids]
            sn = sig_new[ids]
            # correlated part
            Sm = S_a + (rho / nu) * (sn - so) * np.sqrt(S_a)
            np.maximum(Sm, 0.0, out=Sm)
            # integrated variance over the step (trapezoid)
            v = 0.5 * (so * so + sn * sn) * dt
            scale = rb2 * v
            half_lam = 2.0 * Sm / scale

            Snew = np.empty_like(Sm)
            big = half_lam > big_thresh
            small = ~big
            if small.all():
                N = rng.poisson(half_lam)
                G = np.zeros(N.shape[0])
                pos = N > 0
                if pos.any():
                    G[pos] = rng.gamma(N[pos].astype(float))
                Snew = 0.5 * scale * G
            else:
                hs = half_lam[small]
                N = rng.poisson(hs)
                G = np.zeros(N.shape[0])
                pos = N > 0
                if pos.any():
                    G[pos] = rng.gamma(N[pos].astype(float))
                Snew[small] = 0.5 * scale[small] * G
                Sb = Sm[big]
                k = Sb.shape[0]
                Snew[big] = np.maximum(
                    Sb + np.sqrt(scale[big] * Sb) * rng.standard_normal(k), 0.0)

            alive = Snew > 0.0
            if alive.all():
                S_a = Snew
            else:
                S_a = Snew[alive]
                ids = ids[alive]

        sig = sig_new

    S = np.zeros(n)
    S[ids] = S_a

    X = np.column_stack([S, sig])

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    functionals = {
        "price_K1": float(np.mean(np.maximum(S - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S - K3, 0.0))),
    }

    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)),
                               float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["functionals"])
