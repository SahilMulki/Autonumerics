import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 0.0
    a = 2.0
    b = -1.0
    c = 0.5

    # Initialize paths
    X = np.full(num_paths, X_0, dtype=float)

    # Euler-Maruyama time-stepping
    # dX = (a + b*X) dt + c dW
    # Additive noise: g(X) = c (constant), dg/dX = 0, no Milstein correction needed
    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        X = X + (a + b * X) * dt + c * dW

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
