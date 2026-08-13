import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """Tamed Euler-Maruyama solver for the stochastic Ginzburg-Landau SDE.

    dX(t) = ((sigma**2/2) * X(t) - X(t)**3) * dt + sigma * X(t) * dW(t),   X(0) = 1.0

    Plain Euler-Maruyama is unstable for the superlinear cubic drift at sigma=4,
    so the drift increment is tamed: f_tamed = f(X) / (1 + dt * |f(X)|).
    The diffusion term is left unmodified.
    """
    X_0 = 1.0
    sigma = 4.0

    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        f = (sigma ** 2 / 2.0) * X - X ** 3
        f_tamed = f / (1.0 + dt * np.abs(f))

        X = X + f_tamed * dt + sigma * X * dW

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.0002, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"num_paths: {result['num_paths']}, dt: {result['dt']}, T: {result['T']}")
    finite = np.all(np.isfinite(result["terminal_paths"]))
    print(f"All finite: {finite}")
