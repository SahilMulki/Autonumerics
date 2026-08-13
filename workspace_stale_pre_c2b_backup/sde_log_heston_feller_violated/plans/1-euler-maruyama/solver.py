import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Parameters (log-Heston, Feller condition violated)
    X_0 = 0.0
    Y_0 = 0.1
    r = 0.05
    a = 0.1
    b = 1.0
    sigma = 1.0
    rho = -0.9
    sqrt_one_minus_rho2 = np.sqrt(1.0 - rho ** 2)

    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt
    sqrt_dt = np.sqrt(dt)

    X = np.full(num_paths, X_0, dtype=float)
    Y = np.full(num_paths, Y_0, dtype=float)

    for _ in range(Nt):
        Z1 = rng.standard_normal(num_paths)
        Z2 = rng.standard_normal(num_paths)
        dW1 = sqrt_dt * Z1
        dW2 = sqrt_dt * Z2

        # Full-truncation positivity guard: use Y_pos everywhere before sqrt
        Y_pos = np.maximum(Y, 0.0)
        sqrtY = np.sqrt(Y_pos)

        X_new = X + (r - Y_pos / 2.0) * dt + sqrtY * (rho * dW1 + sqrt_one_minus_rho2 * dW2)
        Y_new = Y + (a - b * Y_pos) * dt + sigma * sqrtY * dW1

        X = X_new
        Y = Y_new

    terminal_paths = np.column_stack((X, Y))

    empirical_mean = [float(np.mean(X)), float(np.mean(Y))]
    empirical_variance = [float(np.var(X, ddof=1)), float(np.var(Y, ddof=1))]

    return {
        "terminal_paths": terminal_paths,
        "empirical_mean": empirical_mean,
        "empirical_variance": empirical_variance,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)
    mean_X, mean_Y = result["empirical_mean"]
    var_X, var_Y = result["empirical_variance"]
    print(f"Empirical mean X:     {mean_X:.6f}")
    print(f"Empirical variance X: {var_X:.6f}")
    print(f"Empirical mean Y:     {mean_Y:.6f}")
    print(f"Empirical variance Y: {var_Y:.6f}")
    print(f"dt used: {result['dt']}, num_paths: {result['num_paths']}, T: {result['T']}")
