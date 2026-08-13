import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    Noise-free Ito SDE with random Gaussian initial condition:
        dX = -X**5 * dt,     X(0) = xi ~ Normal(0, sigma_bar**2)

    The drift is strongly superlinear (stiff quintic), so plain explicit
    Euler blows up for large |xi|. We use a tamed explicit Euler scheme:
        drift = -X**5
        X_{n+1} = X_n + (drift * dt) / (1 + dt * |drift|)

    There is no Wiener increment: the only randomness is the Gaussian
    initial condition xi, drawn once at t=0.
    """
    sigma_bar = 1.0 / 3.0

    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    # Random Gaussian initial condition (Monte Carlo is over xi only)
    X = rng.normal(0.0, sigma_bar, size=num_paths)

    for _ in range(Nt):
        drift = -X ** 5
        X = X + (drift * dt) / (1.0 + dt * np.abs(drift))

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"dt used:            {result['dt']}")
    print(f"num_paths:          {result['num_paths']}")
