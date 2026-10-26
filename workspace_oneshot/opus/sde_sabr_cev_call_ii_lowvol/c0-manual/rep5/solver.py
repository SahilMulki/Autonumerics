import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2:
        dS     = sigma * sqrt(S) dW_S,   S(0)     = 0.07
        dsigma = nu * sigma dW_sigma,    sigma(0) = 0.34
        corr(dW_S, dW_sigma) = rho

    Scheme per step, with sigma frozen at its left-point value:
      * S: a squared-Bessel(0) transition with absorption at zero.
          - Away from zero, use Andersen's quadratic QE branch. It matches the
            first two conditional moments exactly, so it is a martingale.
          - Near zero, use the exact Poisson-Gamma transition, which has the
            right atom at 0.
      * sigma: an exact lognormal step, driven by rho * xi + sqrt(1 - rho^2) * Z2.
        xi is the standardised realised S-increment (mean 0, variance 1).
      * Absorbed paths are removed from the loop. Their sigma is advanced
        to T in one exact lognormal step.
    """
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S0, sig0 = 0.07, 0.34
    nu, rho = 0.8, -0.6
    rhop = np.sqrt(1.0 - rho * rho)
    sqdt = np.sqrt(dt)
    drift = -0.5 * nu * nu * dt
    psi_c = 0.25  # QE quadratic branch for psi <= psi_c, exact otherwise

    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, sig0, dtype=float)
    idx = np.arange(num_paths)

    S_out = np.zeros(num_paths)
    sig_out = np.empty(num_paths)
    t_death = np.full(num_paths, T, dtype=float)

    for n in range(Nt):
        m = S.size
        if m == 0:
            break
        Z = rng.standard_normal((2, m))
        Z1 = Z[0]
        Z2 = Z[1]

        v = sig * sig * dt          # conditional variance rate * dt
        psi = v / S                 # Var/mean^2 of S_{n+1}
        pq = np.minimum(psi, psi_c)
        t2 = 2.0 / pq
        b2 = t2 - 1.0 + np.sqrt(t2 * (t2 - 1.0))
        a = S / (1.0 + b2)
        Snew = a * (np.sqrt(b2) + Z1) ** 2

        ex = psi > psi_c
        if ex.any():
            ids = np.nonzero(ex)[0]
            lam = 2.0 / psi[ids]    # Poisson mean = 2 S / v
            N = rng.poisson(lam)
            G = np.zeros(ids.size)
            pos = N > 0
            if pos.any():
                G[pos] = rng.gamma(N[pos].astype(float))
            Snew[ids] = 0.5 * v[ids] * G

        xi = (Snew - S) / np.sqrt(v * S)
        np.clip(xi, -8.0, 8.0, out=xi)

        sig = sig * np.exp(drift + nu * sqdt * (rho * xi + rhop * Z2))
        S = Snew

        alive = S > 0.0
        if not alive.all():
            d = ~alive
            di = idx[d]
            sig_out[di] = sig[d]
            t_death[di] = (n + 1) * dt
            S = S[alive]
            sig = sig[alive]
            idx = idx[alive]

    S_out[idx] = S
    sig_out[idx] = sig
    t_death[idx] = T

    # Advance sigma of absorbed paths to T exactly (lognormal)
    tau = T - t_death
    dm = tau > 0.0
    if dm.any():
        tt = tau[dm]
        Zd = rng.standard_normal(tt.size)
        sig_out[dm] = sig_out[dm] * np.exp(-0.5 * nu * nu * tt + nu * np.sqrt(tt) * Zd)

    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    p1 = float(np.mean(np.maximum(S_out - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S_out - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S_out - K3, 0.0)))

    paths = np.column_stack((S_out, sig_out))
    return {
        "terminal_paths": paths,
        "empirical_mean": [float(np.mean(paths[:, 0])), float(np.mean(paths[:, 1]))],
        "empirical_variance": [float(np.var(paths[:, 0], ddof=1)),
                               float(np.var(paths[:, 1], ddof=1))],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["empirical_variance"])
    print(res["functionals"])
