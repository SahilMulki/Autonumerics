import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    Milstein scheme for Geometric Brownian Motion:
        dX(t) = mu * X(t) * dt + sigma * X(t) * dW(t),   X(0) = 1.0

    Parameters:
        X_0   = 1.0
        mu    = 0.1
        sigma = 0.2

    Milstein update:
        dW_n     = sqrt(dt) * Z_n,   Z_n ~ N(0,1)
        X_{n+1}  = X_n + mu * X_n * dt + sigma * X_n * dW_n
                       + 0.5 * sigma**2 * X_n * (dW_n**2 - dt)
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # GBM parameters
    X_0 = 1.0
    mu = 0.1
    sigma = 0.2

    # Initialize paths: shape (num_paths,)
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # Euler-Maruyama base step + Milstein correction
        # f(X) = mu * X,  g(X) = sigma * X,  dg/dX = sigma
        # Milstein correction: 0.5 * g(X) * dg/dX * (dW^2 - dt)
        #                    = 0.5 * sigma * X * sigma * (dW^2 - dt)
        #                    = 0.5 * sigma^2 * X * (dW^2 - dt)
        X = (X
             + mu * X * dt
             + sigma * X * dW
             + 0.5 * sigma**2 * X * (dW**2 - dt))

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
