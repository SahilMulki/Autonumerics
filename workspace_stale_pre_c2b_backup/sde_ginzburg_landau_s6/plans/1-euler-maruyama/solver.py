import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """Stochastic Ginzburg-Landau equation, tamed Euler-Maruyama.

    dX(t) = ((sigma**2/2) * X(t) - X(t)**3) * dt + sigma * X(t) * dW(t)

    Plain Euler-Maruyama is unusable at sigma=6: the cubic drift -X**3 causes
    explosive intermediate values, producing Inf/NaN paths and a severely
    underestimated variance. We tame only the drift term:

        f       = (sigma**2/2) * X - X**3
        f_tamed = f * dt / (1 + |f| * dt)
        X_{n+1} = X + f_tamed + sigma * X * dW

    The diffusion term sigma*X*dW is left unmodified.
    """
    X_0 = 1.0
    sigma = 6.0

    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    X = np.full(num_paths, X_0, dtype=float)

    sqrt_dt = np.sqrt(dt)
    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = sqrt_dt * Z

        f = (sigma ** 2 / 2.0) * X - X ** 3
        f_tamed = f * dt / (1.0 + np.abs(f) * dt)

        X = X + f_tamed + sigma * X * dW

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.0001, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"All finite:         {bool(np.all(np.isfinite(result['terminal_paths'])))}")
    print(f"dt (actual):        {result['dt']:.6f}")
