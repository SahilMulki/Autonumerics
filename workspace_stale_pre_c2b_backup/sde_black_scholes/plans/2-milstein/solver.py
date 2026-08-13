import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Black-Scholes parameters
    X_0 = 100.0
    r = 0.05
    sigma = 0.20

    # Initialize paths
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        X_n = X
        # Milstein update: EM step + correction 0.5 * g * dg/dX * (dW^2 - dt)
        # g(X) = sigma * X, dg/dX = sigma => correction = 0.5 * sigma^2 * X_n * (dW^2 - dt)
        X = X_n + r * X_n * dt + sigma * X_n * dW + 0.5 * sigma**2 * X_n * (dW**2 - dt)

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
