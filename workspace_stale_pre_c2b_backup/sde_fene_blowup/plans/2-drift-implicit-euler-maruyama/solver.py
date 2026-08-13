import numpy as np


def _solve_cubic_in_domain(b, dt, eps=1e-12, iters=60):
    """
    Solve Y**3 - b*Y**2 - (1+dt)*Y + b = 0 for the unique root Y in (-1, 1),
    given P(-1) = +dt > 0 and P(+1) = -dt < 0 for every real b and dt > 0.

    Vectorized bisection over all paths simultaneously: guaranteed to converge
    to the in-domain root regardless of how large |b| is (i.e. regardless of
    how large the driving noise increment dW_n was).
    """
    lo = np.full_like(b, -1.0 + eps)
    hi = np.full_like(b, 1.0 - eps)

    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        P_mid = mid**3 - b * mid**2 - (1.0 + dt) * mid + b
        # P(lo) > 0 and P(hi) < 0 always hold (bracket invariant), so:
        # if P(mid) > 0, the sign change (root) lies in [mid, hi]
        # if P(mid) <= 0, the sign change (root) lies in [lo, mid]
        pos = P_mid > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)

    return 0.5 * (lo + hi)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    X_0 = 0.0
    eps = 1e-12

    X = np.full(num_paths, X_0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal(num_paths)
        dW = np.sqrt(dt) * Z
        b = X + dW
        X = _solve_cubic_in_domain(b, dt, eps=eps)

    # Floating-point guard: clamp strictly inside the open domain.
    X = np.clip(X, -1.0 + eps, 1.0 - eps)

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
    print(f"num_paths: {result['num_paths']}, dt: {result['dt']}, T: {result['T']}")
    print(f"All finite: {np.all(np.isfinite(X))}")
    print(f"Max |X(T)|: {np.max(np.abs(X)):.12f}")
    print(f"All |X(T)| < 1: {np.all(np.abs(X) < 1.0)}")
