import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 1.0
    theta = 1.0
    sigma = 0.4

    # Initialize paths: shape (num_paths,)
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # Apply positivity guard before computing log(X) in drift
        X_pos = np.maximum(X, 1e-12)

        # Drift: f(X) = X * (-theta * log(X) + sigma**2 / 2)
        f = X_pos * (-theta * np.log(X_pos) + sigma**2 / 2.0)

        # Diffusion: g(X) = sigma * X
        g = sigma * X_pos

        # EM step
        X_new = X_pos + f * dt + g * dW

        # Milstein correction: 0.5 * g(X) * dg/dX * (dW**2 - dt)
        # dg/dX = sigma, so correction = 0.5 * sigma**2 * X_n * (dW**2 - dt)
        X_new = X_new + 0.5 * sigma**2 * X_pos * (dW**2 - dt)

        X = X_new

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
