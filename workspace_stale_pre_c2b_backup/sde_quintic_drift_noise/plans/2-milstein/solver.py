import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """Tamed Milstein scheme for dX = -X^5 dt + X dW, X(0) = 1.0.

    The superlinear drift f(X) = -X^5 causes plain Euler-Maruyama (and plain
    Milstein) to diverge (Hutzenthaler-Jentzen-Kloeden). We tame the drift
    increment:

        tamed = f(X) * dt / (1 + dt * |f(X)|)

    and add the Milstein correction for the multiplicative diffusion
    g(X) = X, dg/dX = 1:

        0.5 * X * (dW**2 - dt)
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    X_0 = 1.0
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        drift = -X**5
        tamed = drift * dt / (1.0 + dt * np.abs(drift))

        milstein_correction = 0.5 * X * (dW**2 - dt)

        X = X + tamed + X * dW + milstein_correction

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)
    finite = np.all(np.isfinite(result["terminal_paths"]))
    print(f"All finite:         {finite}")
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"num_paths:          {result['num_paths']}")
    print(f"dt:                 {result['dt']}")
    print(f"T:                  {result['T']}")
