import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Geometric Brownian motion: dX = 0.1 X dt + 0.2 X dW, X(0) = 1.0
    # Milstein scheme (scalar, multiplicative noise, milstein_eligible):
    #   X_{n+1} = X_n + f(X_n) dt + g(X_n) dW_n + 0.5 g(X_n) g'(X_n) (dW_n^2 - dt)
    # with f(X) = mu X, g(X) = sigma X, g'(X) = sigma.
    mu = 0.1
    sigma = 0.2
    X_0 = 1.0

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)
    dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))

    X = np.full(num_paths, X_0, dtype=float)
    for n in range(Nt):
        X_prev = X
        X = (X_prev
             + mu * X_prev * dt
             + sigma * X_prev * dW[:, n]
             + 0.5 * sigma * X_prev * sigma * (dW[:, n] ** 2 - dt))

    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    return out


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
