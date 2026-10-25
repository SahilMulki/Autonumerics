import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR with beta = 1/2, zero correlation:
        dS     = sigma * sqrt(S) dW_S,   S(0) = 0.5
        dsigma = 0.4 * sigma dW_sigma,   sigma(0) = 0.5

    sigma is simulated exactly (driftless GBM).  Because the two noises are
    independent, conditionally on the sigma path S is a time-changed
    squared Bessel process of dimension 0:  X = 4 S solves dX = 2 sqrt(X) dB_tau
    with clock tau(t) = int_0^t sigma^2 ds.  Its transition is known exactly:
        N ~ Poisson(X0 / (2 tau)),  X_T = 2 tau * Gamma(N, 1)   (N = 0 -> absorbed at 0)
    This keeps S >= 0, absorbing at zero, and a martingale.
    The integrated variance is computed with the trapezoid rule on the exact
    sigma path (O(dt^2) bias).
    """
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S0 = 0.5
    sig0 = 0.5
    nu = 0.4

    sig = np.full(num_paths, sig0, dtype=float)
    tau = np.zeros(num_paths, dtype=float)
    sqdt = np.sqrt(dt)
    drift = -0.5 * nu * nu * dt

    sig2_prev = sig * sig
    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        sig *= np.exp(nu * sqdt * Z + drift)
        sig2 = sig * sig
        tau += 0.5 * (sig2_prev + sig2) * dt
        sig2_prev = sig2

    # Exact BESQ(0) transition over clock tau
    X0 = 4.0 * S0
    tau = np.maximum(tau, 1e-300)
    lam = X0 / (2.0 * tau)
    N = rng.poisson(lam)
    G = np.zeros(num_paths, dtype=float)
    pos = N > 0
    G[pos] = rng.standard_gamma(N[pos].astype(float))
    X_T = 2.0 * tau * G
    S_T = 0.25 * X_T

    terminal = np.column_stack((S_T, sig))

    K1 = 0.434061722697
    K2 = 0.5
    K3 = 0.575954955084
    functionals = {
        "price_K1": float(np.mean(np.maximum(S_T - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S_T - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S_T - K3, 0.0))),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(res["empirical_mean"], res["empirical_variance"])
    print(res["functionals"])
