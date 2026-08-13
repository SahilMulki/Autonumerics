import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 1.0
    Y_0 = 1.0
    mu1 = 0.10
    sigma1 = 0.20
    mu2 = 0.15
    sigma2 = 0.25
    rho = 0.60

    # State: shape (num_paths, 2); column 0 = X, column 1 = Y
    XY = np.zeros((num_paths, 2), dtype=float)
    XY[:, 0] = X_0
    XY[:, 1] = Y_0

    sqrt_rho2 = np.sqrt(1.0 - rho ** 2)

    for _ in range(Nt):
        Z1 = rng.standard_normal(num_paths)
        Z2 = rng.standard_normal(num_paths)

        dW1 = np.sqrt(dt) * Z1
        dW2 = np.sqrt(dt) * (rho * Z1 + sqrt_rho2 * Z2)

        X_n = XY[:, 0]
        Y_n = XY[:, 1]

        XY[:, 0] = X_n + mu1 * X_n * dt + sigma1 * X_n * dW1
        XY[:, 1] = Y_n + mu2 * Y_n * dt + sigma2 * Y_n * dW2

    empirical_mean = list(np.mean(XY, axis=0).tolist())
    empirical_variance = list(np.var(XY, axis=0, ddof=1).tolist())

    return {
        "terminal_paths": XY,               # shape (num_paths, 2)
        "empirical_mean": empirical_mean,    # [mean_X, mean_Y]
        "empirical_variance": empirical_variance,  # [var_X, var_Y]
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']}")
    print(f"Empirical variance: {result['empirical_variance']}")
