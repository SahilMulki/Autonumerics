import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0 = 0.07
    sig0 = 0.4
    alpha = 0.8
    rho = -0.6
    rb2 = 1.0 - rho * rho
    psi_c = 1.5

    sqdt = np.sqrt(dt)
    drift = -0.5 * alpha * alpha * dt

    chunk = 250000
    S_all = np.empty(num_paths)
    sig_all = np.empty(num_paths)

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        S = np.full(n, S0)
        sig = np.full(n, sig0)
        for _ in range(Nt):
            Z = rng.standard_normal(n)
            sig_new = sig * np.exp(alpha * sqdt * Z + drift)
            idx = np.flatnonzero(S > 0.0)
            if idx.size > 0:
                s = S[idx]
                a0 = sig[idx]
                a1 = sig_new[idx]
                # correlated part: rho * sqrt(S) * int sigma dZ = (rho/alpha) sqrt(S) dsigma
                m = s + (rho / alpha) * (a1 - a0) * np.sqrt(s)
                ivar = 0.5 * (a0 * a0 + a1 * a1) * dt
                out = np.zeros(idx.size)
                pos = np.flatnonzero(m > 0.0)
                if pos.size > 0:
                    mp = m[pos]
                    psi = rb2 * ivar[pos] / mp
                    res = np.zeros(pos.size)
                    Zv = rng.standard_normal(pos.size)
                    U = rng.random(pos.size)
                    q = psi <= psi_c
                    # quadratic branch
                    if np.any(q):
                        pq = psi[q]
                        ip = 2.0 / pq
                        b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
                        a = mp[q] / (1.0 + b2)
                        res[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
                    e = ~q
                    if np.any(e):
                        pe = psi[e]
                        p = (pe - 1.0) / (pe + 1.0)
                        beta = (1.0 - p) / mp[e]
                        u = U[e]
                        val = np.zeros(pe.size)
                        nz = u > p
                        val[nz] = np.log((1.0 - p[nz]) / (1.0 - u[nz])) / beta[nz]
                        res[e] = val
                    out[pos] = res
                S[idx] = out
            sig = sig_new
        S_all[start:start + n] = S
        sig_all[start:start + n] = sig
        start += n

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    p1 = float(np.mean(np.maximum(S_all - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S_all - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S_all - K3, 0.0)))

    terminal = np.column_stack([S_all, sig_all])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(r["functionals"], r["empirical_mean"])
