import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    S0 = 0.5
    sig0 = 0.5
    alpha = 0.4
    n = int(num_paths)

    # Exact lognormal simulation of sigma; trapezoidal integrated variance.
    sig = np.full(n, sig0, dtype=np.float64)
    tau = np.zeros(n, dtype=np.float64)
    drift = -0.5 * alpha * alpha * dt
    vol = alpha * np.sqrt(dt)
    sig2_old = sig * sig
    for _ in range(Nt):
        z = rng.standard_normal(n)
        sig *= np.exp(drift + vol * z)
        sig2_new = sig * sig
        tau += 0.5 * (sig2_old + sig2_new) * dt
        sig2_old = sig2_new

    # Conditional on the sigma path (zero correlation), S is a time-changed
    # squared Bessel process of dimension 0 with clock tau = int sigma^2 dt.
    # Exact transition (absorbing at 0, martingale):
    #   S_T = (tau/2) * Gamma(N, 1),  N ~ Poisson(2 S0 / tau),  Gamma(0) := 0
    tau = np.maximum(tau, 1e-300)
    lam = 2.0 * S0 / tau
    N = rng.poisson(lam)
    pos = N > 0
    G = np.zeros(n, dtype=np.float64)
    if np.any(pos):
        G[pos] = rng.gamma(N[pos].astype(np.float64), 1.0)
    S = 0.5 * tau * G

    terminal = np.column_stack((S, sig))

    K1 = 0.434061722697
    K2 = 0.5
    K3 = 0.575954955084
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "price_K1": p1,
            "price_K2": p2,
            "price_K3": p3,
        },
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=100000, dt=0.005, T=2, seed=42)
    print(out["empirical_mean"], out["functionals"])
