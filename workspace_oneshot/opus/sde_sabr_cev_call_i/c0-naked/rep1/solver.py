import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2, rho = 0.

    sigma is a GBM and is sampled exactly on the grid. Because rho = 0, conditional on
    the sigma path, S is a time-changed squared Bessel process of dimension 0:
    with tau = int_0^T sigma^2 dt, Y = 4S satisfies dY = 2 sqrt(Y) dB_tau (BESQ(0)),
    whose transition is known exactly (absorbing at 0):
        N ~ Poisson(2 S0 / tau),  S_T = (tau / 2) * Gamma(N, 1),  S_T = 0 if N = 0.
    The integrated variance tau is computed by the trapezoid rule on the sigma grid.
    """
    S0 = 0.5
    sig0 = 0.5
    alpha = 0.4
    K1 = 0.434061722697
    K2 = 0.5
    K3 = 0.575954955084

    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)

    rng = np.random.default_rng(seed)

    out = np.empty((num_paths, 2))
    chunk = 1_000_000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        logsig = np.full(n, np.log(sig0))
        var_prev = np.full(n, sig0 * sig0)
        tau = np.zeros(n)
        drift = -0.5 * alpha * alpha * dt
        for _ in range(Nt):
            z = rng.standard_normal(n)
            logsig += drift + alpha * sqdt * z
            var_new = np.exp(2.0 * logsig)
            tau += 0.5 * dt * (var_prev + var_new)
            var_prev = var_new
        sigT = np.exp(logsig)

        lam = 2.0 * S0 / tau
        N = rng.poisson(lam)
        g = rng.gamma(np.maximum(N, 1).astype(float), 1.0)
        ST = np.where(N > 0, 0.5 * tau * g, 0.0)

        out[start:start + n, 0] = ST
        out[start:start + n, 1] = sigT
        start += n

    S = out[:, 0]
    functionals = {
        "price_K1": float(np.mean(np.maximum(S - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S - K3, 0.0))),
    }

    return {
        "terminal_paths": out,
        "empirical_mean": [float(v) for v in out.mean(axis=0)],
        "empirical_variance": [float(v) for v in out.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.005, T=2, seed=42)
    print(res["empirical_mean"], res["empirical_variance"], res["functionals"])
