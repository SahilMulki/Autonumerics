import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    mu = 0.1
    sigma = 0.2
    X0 = 1.0

    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    X = np.full(num_paths, X0, dtype=float)

    for _ in range(Nt):
        dW = rng.standard_normal(num_paths) * np.sqrt(dt)
        X = X + mu * X * dt + sigma * X * dW

    empirical_mean = float(np.mean(X))
    empirical_variance = float(np.var(X, ddof=1))

    return {
        "terminal_paths": X,
        "empirical_mean": empirical_mean,
        "empirical_variance": empirical_variance,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"dt used:            {result['dt']:.6f}")
