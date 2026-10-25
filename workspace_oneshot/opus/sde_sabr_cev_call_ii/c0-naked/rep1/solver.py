import numpy as np


def _simulate_chunk(rng, m, Nt, h, S0, sig0, nu, rho):
    S = np.full(m, S0, dtype=np.float64)
    sig = np.full(m, sig0, dtype=np.float64)
    drift = -0.5 * nu * nu * h
    vol = nu * np.sqrt(h)
    c1 = rho / nu
    c2 = (1.0 - rho * rho) * h / 8.0
    for _ in range(Nt):
        z = rng.standard_normal(m)
        sig_new = sig * np.exp(drift + vol * z)
        idx = np.flatnonzero(S > 0.0)
        if idx.size > 0:
            s = S[idx]
            so = sig[idx]
            sn = sig_new[idx]
            # correlated part: (rho/nu) * int sqrt(S) dsigma
            x = s + c1 * np.sqrt(s) * (sn - so)
            np.maximum(x, 0.0, out=x)
            # orthogonal part: exact BESQ(0) transition with time change
            u = c2 * (so * so + sn * sn)
            lam = x / (2.0 * u)
            out = np.empty_like(x)
            big = lam > 1e6
            small = ~big
            if np.any(small):
                ls = lam[small]
                N = rng.poisson(ls)
                g = rng.gamma(np.maximum(N, 1).astype(np.float64))
                g = np.where(N > 0, g, 0.0)
                out[small] = 2.0 * u[small] * g
            if np.any(big):
                xb = x[big]
                ub = u[big]
                zb = rng.standard_normal(xb.size)
                ob = xb + 2.0 * np.sqrt(xb * ub) * zb
                out[big] = np.maximum(ob, 0.0)
            S[idx] = out
        sig = sig_new
    return S, sig


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    S0 = 0.07
    sig0 = 0.4
    nu = 0.8
    rho = -0.6
    Nt = max(1, round(T / dt))
    h = T / Nt

    rng = np.random.default_rng(seed)
    chunk = 250000
    S_all = np.empty(num_paths)
    sig_all = np.empty(num_paths)
    start = 0
    while start < num_paths:
        m = min(chunk, num_paths - start)
        S, sig = _simulate_chunk(rng, m, Nt, h, S0, sig0, nu, rho)
        S_all[start:start + m] = S
        sig_all[start:start + m] = sig
        start += m

    terminal = np.column_stack([S_all, sig_all])
    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    functionals = {
        "price_K1": float(np.mean(np.maximum(S_all - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S_all - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S_all - K3, 0.0))),
    }
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["functionals"], res["empirical_mean"])
