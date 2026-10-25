import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0, sig0 = 0.07, 0.4
    nu, rho = 0.8, -0.6
    rhobar2 = 1.0 - rho * rho
    strikes = {"price_K1": 0.051022538988, "price_K2": 0.07, "price_K3": 0.096035989138}

    n = int(num_paths)
    S = np.full(n, S0)
    sig = np.full(n, sig0)
    idx = np.arange(n)

    S_T = np.zeros(n)
    sig_T = np.empty(n)
    abs_idx = []
    abs_sig = []
    abs_t = []

    drift = -0.5 * nu * nu * dt
    vol = nu * np.sqrt(dt)
    LAM_BIG = 1e6

    for step in range(Nt):
        m = idx.size
        if m == 0:
            break
        z = rng.standard_normal(m)
        sn = sig * np.exp(drift + vol * z)
        # integrated variance of the orthogonal part (trapezoidal)
        V = rhobar2 * 0.5 * (sig * sig + sn * sn) * dt
        # correlated part: int sigma sqrt(S) dW_sigma = sqrt(S) dsigma / nu
        St = S + (rho / nu) * np.sqrt(S) * (sn - sig)
        np.maximum(St, 0.0, out=St)

        Vsafe = np.where(V > 0, V, 1e-300)
        lam = 2.0 * St / Vsafe
        Snew = np.zeros(m)

        big = lam > LAM_BIG
        small = ~big
        # exact BESQ(0) transition for moderate lambda
        if small.any():
            ls = lam[small]
            N = rng.poisson(ls)
            vals = np.zeros(ls.size)
            pos = N > 0
            if pos.any():
                vals[pos] = 0.5 * V[small][pos] * rng.standard_gamma(N[pos].astype(np.float64))
            Snew[small] = vals
        # Gaussian limit for huge lambda (same mean and variance)
        if big.any():
            zb = rng.standard_normal(int(big.sum()))
            Sb = St[big] + np.sqrt(St[big] * V[big]) * zb
            Snew[big] = np.maximum(Sb, 0.0)

        sig = sn
        dead = Snew <= 0.0
        if dead.any():
            abs_idx.append(idx[dead])
            abs_sig.append(sig[dead])
            abs_t.append(np.full(int(dead.sum()), (step + 1) * dt))
            alive = ~dead
            idx = idx[alive]
            S = Snew[alive]
            sig = sig[alive]
        else:
            S = Snew

    # surviving paths
    S_T[idx] = S
    sig_T[idx] = sig

    # absorbed paths: S stays 0, sigma evolves exactly over remaining time
    if abs_idx:
        ai = np.concatenate(abs_idx)
        asg = np.concatenate(abs_sig)
        at = np.concatenate(abs_t)
        rem = np.maximum(T - at, 0.0)
        za = rng.standard_normal(ai.size)
        sig_T[ai] = asg * np.exp(-0.5 * nu * nu * rem + nu * np.sqrt(rem) * za)
        S_T[ai] = 0.0

    terminal = np.column_stack([S_T, sig_T])
    functionals = {k: float(np.mean(np.maximum(S_T - K, 0.0))) for k, K in strikes.items()}

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["empirical_mean"], out["functionals"])
