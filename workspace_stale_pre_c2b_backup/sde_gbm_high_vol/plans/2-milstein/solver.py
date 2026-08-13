import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # GBM parameters
    X_0 = 1.0
    mu = 0.05
    sigma = 1.0

    # Initialize paths
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        # Milstein update: X_{n+1} = X_n + mu*X_n*dt + sigma*X_n*dW
        #                            + 0.5 * sigma**2 * X_n * (dW**2 - dt)
        X = X + mu * X * dt + sigma * X * dW + 0.5 * sigma**2 * X * (dW**2 - dt)

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=500000, dt=0.005, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
