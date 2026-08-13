import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    Euler-Maruyama solver for the Exponential Ornstein-Uhlenbeck SDE:
        dX(t) = X(t) * (-theta * log(X(t)) + sigma**2 / 2) * dt + sigma * X(t) * dW(t)

    Parameters:
        X_0   = 1.0
        theta = 1.0
        sigma = 0.4
        T     = 1.0
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # SDE parameters
    X_0 = 1.0
    theta = 1.0
    sigma = 0.4

    # Initialize all paths
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        # Positivity guard before computing log(X)
        X_pos = np.maximum(X, 1e-12)
        # Drift: f(X) = X * (-theta * log(X) + sigma**2 / 2)
        f = X_pos * (-theta * np.log(X_pos) + sigma**2 / 2.0)
        # Diffusion: g(X) = sigma * X
        g = sigma * X_pos
        # Wiener increment
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        # Euler-Maruyama step
        X = X_pos + f * dt + g * dW

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
