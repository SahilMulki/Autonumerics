import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """Monte Carlo solver for the stochastic Ginzburg-Landau SDE:

        dX(t) = ((sigma**2/2) * X(t) - X(t)**3) * dt + sigma * X(t) * dW(t)

    Scheme: Milstein diffusion correction with a TAMED drift (drift alone
    explodes at sigma=6). The diffusion and Milstein correction terms are
    left untamed as specified in SOLUTION.md.
    """
    rng = np.random.default_rng(seed)

    X_0 = 1.0
    sigma = 6.0

    Nt = max(1, round(T / dt))
    dt = T / Nt

    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        f = (sigma ** 2 / 2.0) * X - X ** 3
        f_tamed = f * dt / (1.0 + np.abs(f) * dt)

        milstein_corr = 0.5 * sigma ** 2 * X * (dW ** 2 - dt)

        X = X + f_tamed + sigma * X * dW + milstein_corr

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
    finite = np.all(np.isfinite(result["terminal_paths"]))
    print(f"All finite:         {finite}")
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"dt (actual):        {result['dt']}")
    print(f"num_paths:          {result['num_paths']}")
