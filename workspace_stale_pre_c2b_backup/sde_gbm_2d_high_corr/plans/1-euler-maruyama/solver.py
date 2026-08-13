import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0, Y_0 = 1.0, 1.0
    mu1, sigma1 = 0.10, 0.30
    mu2, sigma2 = 0.10, 0.30
    rho = 0.95

    # Initialize state: shape (num_paths, 2), columns: [X, Y]
    X = np.full(num_paths, X_0, dtype=float)
    Y = np.full(num_paths, Y_0, dtype=float)

    sqrt_1_minus_rho2 = np.sqrt(1.0 - rho**2)

    for _ in range(Nt):
        Z1 = rng.standard_normal(num_paths)
        Z2 = rng.standard_normal(num_paths)
        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + sqrt_1_minus_rho2 * Z2)

        X = X + mu1 * X * dt + sigma1 * X * dW1
        Y = Y + mu2 * Y * dt + sigma2 * Y * dW2

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
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']}")
    print(f"Empirical variance: {result['empirical_variance']}")
