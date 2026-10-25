import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Approach: sigma is a driftless GBM, so it is simulated exactly on the grid.
    # Given the sigma path (independent of W_S since rho = 0), S is a time-changed
    # squared Bessel process of dimension 0:
    #     dS = sigma sqrt(S) dW  <=>  S(t) = Z(tau(t)),  dZ = 2 sqrt(Z) dB,
    #     tau(t) = (1/4) * int_0^t sigma(u)^2 du.
    # BESQ(0) has an exact transition law (Poisson mixture of Gammas):
    #     N ~ Poisson(Z0 / (2 tau)),  Z_tau = 2 tau * Gamma(N, 1)   (0 when N = 0)
    # This keeps S >= 0, makes zero absorbing, and gives E[S_T] = S0 exactly.
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S0 = 0.5
    sig0 = 0.5
    alpha = 0.4

    sig = np.full(num_paths, sig0, dtype=float)
    integ = np.zeros(num_paths)  # int sigma^2 dt (trapezoid on the exact grid)
    sq_dt = np.sqrt(dt)
    drift = -0.5 * alpha**2 * dt
    sig2_prev = sig * sig
    for _ in range(Nt):
        z = rng.standard_normal(num_paths)
        sig *= np.exp(alpha * sq_dt * z + drift)
        sig2 = sig * sig
        integ += 0.5 * (sig2_prev + sig2) * dt
        sig2_prev = sig2

    tau = 0.25 * integ
    tau = np.maximum(tau, 1e-300)
    lam = S0 / (2.0 * tau)
    N = rng.poisson(lam)
    pos = N > 0
    S = np.zeros(num_paths)
    if np.any(pos):
        S[pos] = 2.0 * tau[pos] * rng.gamma(N[pos].astype(float), 1.0)

    K1 = 0.434061722697
    K2 = 0.5
    K3 = 0.575954955084
    p1 = float(np.mean(np.maximum(S - K1, 0.0)))
    p2 = float(np.mean(np.maximum(S - K2, 0.0)))
    p3 = float(np.mean(np.maximum(S - K3, 0.0)))

    X = np.column_stack((S, sig))
    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)),
                               float(np.var(X[:, 1], ddof=1))],
        "functionals": {"price_K1": p1, "price_K2": p2, "price_K3": p3},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(r["empirical_mean"], r["functionals"])
