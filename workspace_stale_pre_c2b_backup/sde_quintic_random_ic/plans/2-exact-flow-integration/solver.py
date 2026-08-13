import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    Noise-free Ito SDE with a random Gaussian initial condition:

        dX = -X**5 * dt,   X(0) = xi ~ Normal(0, sigma_bar**2)

    This is a deterministic ODE per path (no Wiener increment); the only
    randomness is in the initial condition xi. The separable ODE
    dX = -X^5 dt has the exact closed-form flow map:

        X_t = xi / (1 + 4*t*xi**4)**0.25

    so Monte Carlo reduces to drawing xi ~ N(0, sigma_bar^2) and evaluating
    this map directly at t = T. No time-stepping loop is needed, and this
    removes all discretization bias -- the only error is Monte Carlo
    sampling error.
    """
    rng = np.random.default_rng(seed)

    sigma_bar = 1.0 / 3.0

    # Nt / dt bookkeeping for interface consistency (not used in the
    # time-stepping sense since we apply the exact flow map in one shot).
    Nt = max(1, round(T / dt))
    dt = T / Nt

    xi = rng.normal(0.0, sigma_bar, size=num_paths)

    # Denominator 1 + 4*T*xi**4 >= 1 always, so this is numerically safe
    # for all real xi (no overflow, no division near zero).
    X = xi / (1.0 + 4.0 * T * xi**4) ** 0.25

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
