import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    eps = 1e-12
    lo = -1.0 + eps
    hi = 1.0 - eps

    X_0 = 0.0
    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z

        # --- Tamed drift ---
        f = -X / (1.0 - X ** 2)
        f_tamed = f / (1.0 + dt * np.abs(f))

        # --- Explicit tamed EM step ---
        X_star = X + f_tamed * dt + dW

        # --- Reflection guard: fold overshoot back into (-1, 1) ---
        X_new = X_star
        for _ in range(10):
            if not np.any((X_new > hi) | (X_new < lo)):
                break
            X_new = np.where(X_new > hi, 2.0 * hi - X_new, X_new)
            X_new = np.where(X_new < lo, 2.0 * lo - X_new, X_new)

        # Hard backstop clamp
        X_new = np.clip(X_new, lo, hi)

        X = X_new

    return {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    result = solve_sde(num_paths=50000, dt=0.005, T=1.0, seed=42)
    X = result["terminal_paths"]
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"dt used:            {result['dt']:.6f}")
    print(f"num_paths:          {result['num_paths']}")
    print(f"All finite:         {np.all(np.isfinite(X))}")
    print(f"Max |X(T)|:         {np.max(np.abs(X)):.12f}")
    print(f"All |X(T)| < 1:     {np.all(np.abs(X) < 1.0)}")
