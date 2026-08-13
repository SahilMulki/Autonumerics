import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # CIR parameters
    X_0 = 0.5
    kappa = 1.0
    theta = 0.5
    sigma = 2.0

    # Initialize paths
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # Full truncation guard: apply positivity guard before sqrt
        # Feller condition is violated (2*kappa*theta = 1.0 < sigma^2 = 4.0),
        # so X hits zero frequently and positivity guard is mandatory.
        X_pos = np.maximum(X, 0.0)

        # Drift: f(X) = kappa * (theta - X)
        # Diffusion: g(X) = sigma * sqrt(X_pos)
        f = kappa * (theta - X)
        g = sigma * np.sqrt(X_pos)

        # Euler-Maruyama step
        X = X + f * dt + g * dW

        # Milstein correction: (sigma^2 / 4) * (dW^2 - dt)
        # This is constant (independent of X) because g * dg/dX = sigma*sqrt(X) * sigma/(2*sqrt(X)) = sigma^2/2
        # So 0.5 * g * (dg/dX) * (dW^2 - dt) = (sigma^2 / 4) * (dW^2 - dt)
        # No division by sqrt(X) needed, safe at zero boundary.
        X = X + (sigma**2 / 4.0) * (dW**2 - dt)

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
