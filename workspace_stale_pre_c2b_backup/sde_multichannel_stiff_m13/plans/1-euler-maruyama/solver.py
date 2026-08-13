import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    F = np.array([[-2.0, 3.0], [-3.0, -2.0]])
    g_scale = 0.6
    n_odd = 7   # channels using M1 = [[0,1],[0,0]]
    n_even = 6  # channels using M2 = [[0,0],[1,0]]

    X0 = np.array([1.0, 1.0])
    X = np.tile(X0, (num_paths, 1)).astype(float)  # shape (num_paths, 2)

    sqrt_dt = np.sqrt(dt)
    sqrt_n_odd_dt = np.sqrt(n_odd) * sqrt_dt
    sqrt_n_even_dt = np.sqrt(n_even) * sqrt_dt

    for _ in range(Nt):
        # Drift: (F @ X) per path -> X @ F.T
        drift = (X @ F.T) * dt

        # Aggregate noise for the 7 identical odd channels (G = g_scale * M1)
        # and the 6 identical even channels (G = g_scale * M2).
        Z_odd = rng.standard_normal(num_paths)
        Z_even = rng.standard_normal(num_paths)
        dW_odd = sqrt_n_odd_dt * Z_odd
        dW_even = sqrt_n_even_dt * Z_even

        # M1 @ X = [X_y, 0]; M2 @ X = [0, X_x]
        M1X = np.zeros_like(X)
        M1X[:, 0] = X[:, 1]
        M2X = np.zeros_like(X)
        M2X[:, 1] = X[:, 0]

        noise = g_scale * M1X * dW_odd[:, None] + g_scale * M2X * dW_even[:, None]

        X = X + drift + noise

    empirical_mean = np.mean(X, axis=0)
    empirical_variance = np.var(X, axis=0, ddof=1)

    return {
        "terminal_paths": X,
        "empirical_mean": empirical_mean.tolist(),
        "empirical_variance": empirical_variance.tolist(),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)
    mean = result["empirical_mean"]
    var = result["empirical_variance"]
    print(f"Empirical mean:     X={mean[0]:.6f}, Y={mean[1]:.6f}")
    print(f"Empirical variance: X={var[0]:.6f}, Y={var[1]:.6f}")
    print(f"dt={result['dt']}, num_paths={result['num_paths']}, T={result['T']}")
