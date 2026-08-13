import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
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

        X = X + f_tamed * dt + sigma * X * dW + 0.5 * sigma ** 2 * X * (dW ** 2 - dt)

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.0005, T=1.0, seed=42)
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"dt used: {result['dt']}")
    finite = np.isfinite(result["terminal_paths"]).all()
    print(f"All finite: {finite}")
