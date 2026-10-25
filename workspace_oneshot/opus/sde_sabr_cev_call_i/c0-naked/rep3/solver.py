import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2, rho = 0, absorbing at S = 0.

    Since rho = 0, conditional on the sigma path, S is a time-changed
    squared Bessel process of dimension 0 (X = 4S, dX = 2 sqrt(X) dB) on
    the clock tau = int_0^T sigma^2 dt. Its exact transition is
        N ~ Poisson(2 S0 / tau),  S_T = (tau/2) * Gamma(N, 1)  (S_T = 0 if N = 0),
    which has the absorbing mass at zero and keeps E[S_T] = S0 exactly.
    sigma is simulated exactly as a GBM on the grid, and tau comes from the
    trapezoidal rule on sigma^2.
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0 = 0.5
    sig0 = 0.5
    nu = 0.4
    K = [0.434061722697, 0.5, 0.575954955084]

    drift = -0.5 * nu * nu * dt
    vol = nu * np.sqrt(dt)

    S_T = np.empty(num_paths)
    sig_T = np.empty(num_paths)

    chunk = 500000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        logsig = np.full(n, np.log(sig0))
        v_prev = np.full(n, sig0 * sig0)
        tau = np.zeros(n)
        for _ in range(Nt):
            logsig += drift + vol * rng.standard_normal(n)
            v_new = np.exp(2.0 * logsig)
            tau += 0.5 * dt * (v_prev + v_new)
            v_prev = v_new
        lam = 2.0 * S0 / tau
        N = rng.poisson(lam)
        g = rng.standard_gamma(np.maximum(N, 1).astype(float))
        S = np.where(N > 0, 0.5 * tau * g, 0.0)
        S_T[start:start + n] = S
        sig_T[start:start + n] = np.exp(logsig)
        start += n

    prices = [float(np.mean(np.maximum(S_T - k, 0.0))) for k in K]

    terminal = np.column_stack([S_T, sig_T])
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(S_T)), float(np.mean(sig_T))],
        "empirical_variance": [float(np.var(S_T, ddof=1)), float(np.var(sig_T, ddof=1))],
        "functionals": {
            "price_K1": prices[0],
            "price_K2": prices[1],
            "price_K3": prices[2],
        },
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(res["functionals"], res["empirical_mean"])
