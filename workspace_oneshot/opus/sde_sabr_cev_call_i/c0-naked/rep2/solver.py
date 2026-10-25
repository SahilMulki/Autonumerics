import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2 and rho = 0.

    sigma is a driftless GBM, which we simulate exactly on the time grid.
    Because rho = 0, conditional on the sigma path S is a time-changed
    squared Bessel process of dimension 0, with clock tau = int_0^T sigma^2 dt:
        X = 4 S  satisfies  dX = 2 sqrt(X) dB  (BESQ(0)) on the tau clock.
    Its transition is exact: N ~ Poisson(2 S0 / tau), S_T = (tau/2) * Gamma(N, 1),
    and N = 0 means absorption at zero. This keeps S >= 0, makes zero absorbing,
    and preserves E[S_T] = S0 exactly for any tau. The only discretisation error
    is in the trapezoidal integral tau.
    """
    S0 = 0.5
    sig0 = 0.5
    alpha = 0.4
    K = [0.434061722697, 0.5, 0.575954955084]

    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqdt = np.sqrt(dt)

    rng = np.random.default_rng(seed)

    S_T = np.empty(num_paths)
    sig_T = np.empty(num_paths)

    chunk = 500000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        logsig = np.full(n, np.log(sig0))
        v_prev = np.full(n, sig0 * sig0)
        tau = np.zeros(n)
        drift = -0.5 * alpha * alpha * dt
        for _ in range(Nt):
            logsig += drift + alpha * sqdt * rng.standard_normal(n)
            v = np.exp(2.0 * logsig)
            tau += 0.5 * dt * (v_prev + v)
            v_prev = v
        lam = 2.0 * S0 / tau
        N = rng.poisson(lam)
        shape = np.maximum(N, 1).astype(float)
        G = rng.gamma(shape)
        S = np.where(N > 0, 0.5 * tau * G, 0.0)
        S_T[start:start + n] = S
        sig_T[start:start + n] = np.exp(logsig)

    terminal = np.column_stack([S_T, sig_T])
    prices = [float(np.mean(np.maximum(S_T - k, 0.0))) for k in K]

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "price_K1": prices[0],
            "price_K2": prices[1],
            "price_K3": prices[2],
        },
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(out["empirical_mean"], out["empirical_variance"], out["functionals"])
