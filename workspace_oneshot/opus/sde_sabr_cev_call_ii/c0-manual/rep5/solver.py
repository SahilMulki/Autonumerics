import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # SABR, beta = 1/2:
    #   dS = sigma sqrt(S) dW_S,  dsigma = nu sigma dW_sigma,  corr = rho
    # sigma: exact lognormal step.
    # S: conditional on the sigma step, the correlated part is integrated as
    #    (rho/nu) sqrt(S_n) * (sigma_{n+1} - sigma_n)  (since sigma dW_sigma = dsigma/nu),
    #    and the remaining part is a time-changed squared Bessel process of dimension 0,
    #    dZ = sqrt(Z) dB, with time change (1-rho^2) * int sigma^2 dt.
    #    The BESQ(0) transition is sampled exactly (Poisson-mixed Gamma), which gets the
    #    absorbing boundary at 0 right. Where it is far from zero (large noncentrality),
    #    a Gaussian step is used instead because it is cheaper.
    S0 = 0.07
    sig0 = 0.4
    nu = 0.8
    rho = -0.6
    rhob2 = 1.0 - rho * rho
    c_rho = rho / nu

    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)
    rng = np.random.default_rng(seed)

    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, sig0, dtype=float)
    drift_ls = -0.5 * nu * nu * dt
    LAM_GAUSS = 400.0

    for n in range(Nt):
        z = rng.standard_normal(num_paths)
        sig_new = sig * np.exp(drift_ls + nu * sqdt * z)

        idx = np.nonzero(S > 0.0)[0]
        if idx.size > 0:
            s = S[idx]
            so = sig[idx]
            sn = sig_new[idx]
            h = rhob2 * 0.5 * (so * so + sn * sn) * dt
            h = np.maximum(h, 1e-300)
            start = s + c_rho * np.sqrt(s) * (sn - so)

            newS = np.zeros_like(s)
            pos = start > 0.0
            lam = np.zeros_like(s)
            lam[pos] = 2.0 * start[pos] / h[pos]

            big = lam > LAM_GAUSS
            small = pos & ~big

            if np.any(big):
                st = start[big]
                xi = rng.standard_normal(st.size)
                newS[big] = np.maximum(st + np.sqrt(st * h[big]) * xi, 0.0)

            if np.any(small):
                Np = rng.poisson(lam[small])
                g = rng.standard_gamma(np.maximum(Np, 1).astype(float))
                g = np.where(Np > 0, g, 0.0)
                newS[small] = 0.5 * h[small] * g

            S[idx] = newS

        sig = sig_new

    X = np.column_stack([S, sig])
    Ks = {
        "price_K1": 0.051022538988,
        "price_K2": 0.07,
        "price_K3": 0.096035989138,
    }
    functionals = {k: float(np.mean(np.maximum(S - K, 0.0))) for k, K in Ks.items()}

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
