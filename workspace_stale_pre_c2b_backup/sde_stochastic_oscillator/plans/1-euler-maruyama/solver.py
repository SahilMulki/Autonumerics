import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 1.0
    Y_0 = 0.0
    sigma = 0.3

    # Initialize state arrays: shape (num_paths,) per component
    X = np.full(num_paths, X_0, dtype=float)
    Y = np.full(num_paths, Y_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # Compute updates from old (pre-update) X and Y values
        X_new = X + Y * dt
        Y_new = Y - X * dt + sigma * dW

        X = X_new
        Y = Y_new

    terminal_paths = np.stack([X, Y], axis=1)  # shape (num_paths, 2)

    empirical_mean = [float(np.mean(X)), float(np.mean(Y))]
    empirical_variance = [float(np.var(X, ddof=1)), float(np.var(Y, ddof=1))]

    return {
        "terminal_paths": terminal_paths,
        "empirical_mean": empirical_mean,
        "empirical_variance": empirical_variance,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    import math
    T = 2 * math.pi
    result = solve_sde(num_paths=50000, dt=0.01, T=T)
    print(f"Empirical mean X:     {result['empirical_mean'][0]:.6f}")
    print(f"Empirical mean Y:     {result['empirical_mean'][1]:.6f}")
    print(f"Empirical variance X: {result['empirical_variance'][0]:.6f}")
    print(f"Empirical variance Y: {result['empirical_variance'][1]:.6f}")
