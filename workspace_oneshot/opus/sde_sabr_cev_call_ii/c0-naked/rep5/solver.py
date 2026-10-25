import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0 = 0.07
    sig0 = 0.4
    alpha = 0.8
    rho = -0.6
    rhobar2 = 1.0 - rho * rho
    strikes = [0.051022538988, 0.07, 0.096035989138]

    n = int(num_paths)
    S = np.full(n, S0, dtype=np.float64)
    sig = np.full(n, sig0, dtype=np.float64)
    idx = np.arange(n)

    drift = -0.5 * alpha * alpha * dt
    vol = alpha * np.sqrt(dt)

    for _ in range(Nt):
        Z = rng.standard_normal(n)
        sig_new = sig * np.exp(drift + vol * Z)

        if idx.size > 0:
            s = S[idx]
            a0 = sig[idx]
            a1 = sig_new[idx]
            # correlated part: int sigma dW_sigma = (sigma_{n+1}-sigma_n)/alpha
            shift = s + (rho / alpha) * np.sqrt(s) * (a1 - a0)
            np.maximum(shift, 0.0, out=shift)
            # remaining uncorrelated integrated variance
            V = rhobar2 * 0.5 * (a0 * a0 + a1 * a1) * dt
            V = np.maximum(V, 1e-300)
            lam = 2.0 * shift / V  # Poisson mean for BESQ(0) transition
            new = np.zeros_like(s)

            big = lam > 1e7
            small = ~big
            if np.any(big):
                sb = shift[big]
                zb = rng.standard_normal(sb.size)
                nb = sb + np.sqrt(sb * V[big]) * zb
                new[big] = np.maximum(nb, 0.0)
            if np.any(small):
                ls = lam[small]
                Ns = rng.poisson(ls)
                pos = Ns > 0
                vals = np.zeros(ls.size)
                if np.any(pos):
                    g = rng.gamma(Ns[pos].astype(np.float64))
                    vals[pos] = 0.5 * V[small][pos] * g
                new[small] = vals

            S[idx] = new
            idx = idx[new > 0.0]

        sig = sig_new

    terminal = np.column_stack([S, sig])
    functionals = {}
    for i, K in enumerate(strikes):
        functionals["price_K%d" % (i + 1)] = float(np.mean(np.maximum(S - K, 0.0)))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S)), float(np.mean(sig))],
        "empirical_variance": [float(np.var(S, ddof=1)), float(np.var(sig, ddof=1))],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(out["empirical_mean"], out["functionals"])
