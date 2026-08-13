import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # CIR parameters
    X_0   = 0.5
    kappa = 2.0
    theta = 1.0
    sigma = 0.5

    # Initialize paths
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z  = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # Positivity guard before sqrt (mandatory for CIR)
        X_pos = np.maximum(X, 0.0)

        # Drift and diffusion
        f = kappa * (theta - X)
        g = sigma * np.sqrt(X_pos)

        # Euler-Maruyama base step + Milstein correction
        # Milstein correction for CIR: g * g' = sigma*sqrt(X) * sigma/(2*sqrt(X)) = sigma^2/2
        # => correction = 0.5 * (sigma**2/2) * (dW**2 - dt) = (sigma**2/4) * (dW**2 - dt)
        X = X + f * dt + g * dW + (sigma**2 / 4.0) * (dW**2 - dt)

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
