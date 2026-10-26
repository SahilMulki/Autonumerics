import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Parameters
    S0, sig0 = 0.07, 0.34
    nu, rho = 0.8, -0.6
    rhob2 = 1.0 - rho * rho
    strikes = {"price_K1": 0.051022538988,
               "price_K2": 0.07,
               "price_K3": 0.096035989138}

    Nt = max(1, round(T / dt))
    h = T / Nt
    sqh = np.sqrt(h)

    rng = np.random.default_rng(seed)

    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, sig0, dtype=float)
    act = np.arange(num_paths)  # indices of non-absorbed paths

    drift_ls = -0.5 * nu * nu * h
    for n in range(Nt):
        # exact lognormal step for sigma
        z = rng.standard_normal(num_paths)
        sig_new = sig * np.exp(nu * sqh * z + drift_ls)

        if act.size:
            s = S[act]
            s0 = sig[act]
            s1 = sig_new[act]
            # correlated part handled by conditional shift: int sigma dW_sigma = dsigma/nu
            St = s + (rho / nu) * np.sqrt(s) * (s1 - s0)
            St = np.maximum(St, 0.0)
            # independent part: squared Bessel dim 0 (CEV beta=1/2) exact transition
            v = rhob2 * 0.5 * h * (s0 * s0 + s1 * s1)
            v = np.maximum(v, 1e-300)
            lam = 2.0 * St / v
            big = lam > 1e8
            lam_c = np.where(big, 0.0, lam)
            N = rng.poisson(lam_c)
            G = np.zeros(act.size)
            m = N > 0
            if m.any():
                G[m] = rng.gamma(N[m].astype(float))
            new = 0.5 * v * G
            if big.any():
                Sb = St[big]
                new[big] = np.maximum(
                    Sb + np.sqrt(Sb * v[big]) * rng.standard_normal(Sb.size), 0.0)
            S[act] = new
            keep = new > 0.0
            if not keep.all():
                act = act[keep]

        sig = sig_new

    X = np.column_stack([S, sig])
    functionals = {k: float(np.mean(np.maximum(S - K, 0.0)))
                   for k, K in strikes.items()}

    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)),
                               float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": h,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(r["empirical_mean"], r["functionals"])
