import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Model (Ito, zero correlation):
    #   dS     = sigma * sqrt(S) dW_S,   S(0) = 0.5
    #   dsigma = 0.4 * sigma dW_sig,     sigma(0) = 0.5
    # Because the correlation is zero, S given the sigma path is a time-changed
    # squared-Bessel process of dimension 0:  S(t) = Y(tau(t)),  tau = int sigma^2 dt,
    # with dY = sqrt(Y) dB.  Exact transition (absorbing at 0):
    #   Y_tau = (tau/2) * Gamma(N, 1),   N ~ Poisson(2*Y_0/tau)   (Y_tau = 0 if N = 0)
    # sigma is simulated exactly (lognormal) on the grid, and tau is integrated with
    # the trapezoid rule (O(dt^2) error).
    S0 = 0.5
    sig0 = 0.5
    nu = 0.4

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    sig = np.full(num_paths, sig0, dtype=float)
    tau = np.zeros(num_paths, dtype=float)
    sq_dt = np.sqrt(dt)
    drift = -0.5 * nu * nu * dt
    v_prev = sig * sig
    for _ in range(Nt):
        z = rng.standard_normal(num_paths)
        sig *= np.exp(nu * sq_dt * z + drift)
        v_new = sig * sig
        tau += 0.5 * (v_prev + v_new) * dt
        v_prev = v_new

    tau = np.maximum(tau, 1e-300)
    lam = 2.0 * S0 / tau
    N = rng.poisson(lam)
    shape = np.maximum(N, 1).astype(float)
    G = rng.gamma(shape, 1.0)
    S = np.where(N > 0, 0.5 * tau * G, 0.0)

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
    print(r["empirical_mean"], r["empirical_variance"])
    print(r["functionals"])
