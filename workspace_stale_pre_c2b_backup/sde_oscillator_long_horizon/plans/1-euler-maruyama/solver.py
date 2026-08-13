import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 1.0
    Y_0 = 0.0
    sigma = 0.3

    # State array shape (num_paths, 2): columns [X, Y]
    state = np.zeros((num_paths, 2), dtype=float)
    state[:, 0] = X_0
    state[:, 1] = Y_0

    for _ in range(Nt):
        X = state[:, 0]
        Y = state[:, 1]
        dW = np.sqrt(dt) * rng.standard_normal(num_paths)
        # Simultaneous update using old X and Y (explicit EM)
        X_next = X + Y * dt
        Y_next = Y - X * dt + sigma * dW
        state[:, 0] = X_next
        state[:, 1] = Y_next

    terminal_paths = state  # shape (num_paths, 2)
    empirical_mean = np.mean(terminal_paths, axis=0).tolist()
    empirical_variance = np.var(terminal_paths, axis=0, ddof=1).tolist()

    return {
        "terminal_paths": terminal_paths,
        "empirical_mean": empirical_mean,
        "empirical_variance": empirical_variance,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    import numpy as np
    T = 10 * np.pi
    result = solve_sde(num_paths=50000, dt=0.001, T=T)
    print(f"Empirical mean:     {result['empirical_mean']}")
    print(f"Empirical variance: {result['empirical_variance']}")
