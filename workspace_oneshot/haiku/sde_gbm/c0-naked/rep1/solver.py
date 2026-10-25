import numpy as np

MU = 0.1
SIGMA = 0.2
X0 = 1.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    sqrt_dt = np.sqrt(dt)
    X = np.full(num_paths, X0, dtype=float)
    for _ in range(Nt):
        dW = rng.standard_normal(num_paths) * sqrt_dt
        # Milstein step; the correction term is 0.5 * b * b' * (dW^2 - dt) with b = SIGMA * X
        X = X + MU * X * dt + SIGMA * X * dW + 0.5 * SIGMA**2 * X * (dW**2 - dt)
    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=50000, dt=0.01, T=1, seed=42)
    print(out["empirical_mean"], out["empirical_variance"])
