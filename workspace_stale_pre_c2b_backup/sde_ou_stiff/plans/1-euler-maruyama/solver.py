import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Parameters
    X_0 = 2.0
    theta = 50.0
    mu = 0.0
    sigma = 2.0

    # Initialize paths: shape (num_paths,)
    X = np.full(num_paths, X_0, dtype=float)

    # For stiff OU (theta=50), explicit EM requires very small dt for accuracy.
    # Use the exact OU transition kernel: the conditional distribution of X(t+dt)|X(t)
    # is Gaussian with:
    #   mean:     X(t)*exp(-theta*dt) + mu*(1 - exp(-theta*dt))
    #   variance: sigma^2/(2*theta) * (1 - exp(-2*theta*dt))
    # This is exact regardless of dt and avoids all discretization bias.
    e = np.exp(-theta * dt)
    step_std = sigma * np.sqrt((1.0 - np.exp(-2.0 * theta * dt)) / (2.0 * theta))

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        X = X * e + mu * (1.0 - e) + step_std * Z

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.005, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
