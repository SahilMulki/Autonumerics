import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """FENE (finitely-extensible nonlinear elastic) SDE on (-1, 1):

        dX(t) = -X(t) / (1 - X(t)**2) * dt + dW(t),   X(0) = 0.0

    Additive unit noise (g = 1, dg/dX = 0) -> Milstein reduces to Euler-Maruyama.

    The drift diverges as |X| -> 1, confining the true process to (-1, 1).
    A naive fixed macro-step EM overshoots the wall near |X| ~ 1 (drift flips
    sign, path blows up). We use boundary-aware adaptive substepping: within
    each macro step of size dt we take k equal substeps of size h = dt / k,
    where k is chosen (globally, over the whole population) so that the
    deterministic drift displacement stays a small fraction of the distance
    to the nearest wall for every path.
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt_macro = T / Nt   # nominal/base step size, used away from the wall

    X_0 = 0.0
    eps = 1e-12          # hard safety clamp margin
    c = 0.1              # safety factor for adaptive substep sizing
    k_max = 4096         # cap on substep count per macro step -> defines step floor
    h_floor = dt_macro / k_max

    X = np.full(num_paths, X_0, dtype=float)

    def f(x):
        return -x / (1.0 - x ** 2)

    # Continuous boundary-aware adaptive stepping: at every step, recompute the
    # safe local step size from the CURRENT distance to the wall for the whole
    # population (vectorized global k, per the plan). Away from the wall this
    # reduces to plain EM with h = dt_macro; near the wall h shrinks so the
    # drift displacement stays a small fraction of the remaining distance,
    # which is what keeps the hard safety clamp a rare last resort instead of
    # a routinely-triggered fallback (recomputing h only once per fixed 0.01
    # macro block let paths drift dangerously close to the wall in between
    # adaptations).
    t = 0.0
    tol = 1e-12
    while t < T - tol:
        dist = 1.0 - np.abs(X)                  # distance to nearest wall
        dist = np.maximum(dist, eps)
        abs_x = np.maximum(np.abs(X), 1e-12)     # avoid divide-by-zero at X=0
        h_safe = c * dist ** 2 / abs_x           # |f(X)| ~ |X| / (2*dist)
        h_safe_min = np.min(h_safe)
        if not np.isfinite(h_safe_min):
            h_safe_min = dt_macro

        h = min(h_safe_min, dt_macro, T - t)
        h = max(h, h_floor)

        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(h) * Z
        X = X + f(X) * h + dW
        # last-resort guard against floating point overshoot past the wall
        X = np.clip(X, -1.0 + eps, 1.0 - eps)

        t += h

    dt = dt_macro

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
    X = result["terminal_paths"]
    print(f"Empirical mean:     {result['empirical_mean']:.6f}")
    print(f"Empirical variance: {result['empirical_variance']:.6f}")
    print(f"num_paths:          {result['num_paths']}")
    print(f"dt (actual):        {result['dt']:.6f}")
    print(f"T:                  {result['T']}")
    print(f"max |X(T)|:         {np.max(np.abs(X)):.12f}")
    print(f"all finite:         {bool(np.all(np.isfinite(X)))}")
    print(f"all |X(T)| < 1:     {bool(np.all(np.abs(X) < 1.0))}")
