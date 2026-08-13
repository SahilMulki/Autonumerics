import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    F = np.array([[-2.0, 3.0], [-3.0, -2.0]])
    g_scale = 0.6
    num_odd = 7
    num_even = 6

    # X shape (num_paths, 2): column 0 = X, column 1 = Y
    X = np.zeros((num_paths, 2), dtype=float)
    X[:, 0] = 1.0
    X[:, 1] = 1.0

    sqrt_dt = np.sqrt(dt)

    for _ in range(Nt):
        # Drift: (F @ X_path) for each path -> X @ F.T
        drift = (X @ F.T) * dt

        # Aggregated noise channels (exact in law):
        # 7 identical odd channels with G = g_scale * M1, M1 @ x = [x1, 0]
        # 6 identical even channels with G = g_scale * M2, M2 @ x = [0, x0]
        Z_odd = rng.standard_normal(num_paths)
        Z_even = rng.standard_normal(num_paths)

        M1X = np.zeros((num_paths, 2))
        M1X[:, 0] = X[:, 1]
        M2X = np.zeros((num_paths, 2))
        M2X[:, 1] = X[:, 0]

        noise = (
            g_scale * M1X * np.sqrt(num_odd) * sqrt_dt * Z_odd[:, None]
            + g_scale * M2X * np.sqrt(num_even) * sqrt_dt * Z_even[:, None]
        )

        X = X + drift + noise

    empirical_mean = np.mean(X, axis=0)
    empirical_variance = np.var(X, axis=0, ddof=1)

    return {
        "terminal_paths": X,
        "empirical_mean": [float(empirical_mean[0]), float(empirical_mean[1])],
        "empirical_variance": [float(empirical_variance[0]), float(empirical_variance[1])],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)
    print(f"Empirical mean:     X={result['empirical_mean'][0]:.6f}, Y={result['empirical_mean'][1]:.6f}")
    print(f"Empirical variance: X={result['empirical_variance'][0]:.6f}, Y={result['empirical_variance'][1]:.6f}")
    print(f"num_paths={result['num_paths']}, dt={result['dt']}, T={result['T']}")
