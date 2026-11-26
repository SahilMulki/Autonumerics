"""Wright-Fisher diffusion with mutation: per-step conditional-moment-matched Beta draw.

dX = (a(1-X) - bX) dt + sigma sqrt(X(1-X)) dW,  X(0) = X_0 on [0, 1].

Each step draws X_{n+1} | X_n from the Beta law carrying the exact conditional mean and
variance of the transition over one step (sde_manual.md rule 0c).  The step generates its
own randomness, so it cannot consume a supplied dW (declared via exact_sampling_components).
"""
import numpy as np

X_0, A, B, SIGMA = 0.05, 0.25, 0.25, 1.0


def _conditional_moments(x, h, a=A, b=B, sigma=SIGMA):
    lam1 = a + b
    lam2 = 2.0 * (a + b) + sigma ** 2
    m_inf = a / (a + b)
    m2_inf = (2.0 * a + sigma ** 2) * m_inf / lam2
    e1 = np.exp(-lam1 * h)
    e2 = np.exp(-lam2 * h)
    mean = m_inf + (x - m_inf) * e1
    Bc = (2.0 * a + sigma ** 2) * (x - m_inf) / (lam1 + sigma ** 2)
    Cc = x * x - m2_inf - Bc
    second = m2_inf + Bc * e1 + Cc * e2
    return mean, second - mean * mean


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
    else:
        Nt = dW.shape[1]          # values of dW are not consumed: the Beta draw needs its own noise
    dt = T / Nt
    rng = np.random.default_rng(seed)

    X = np.full(num_paths, X_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        mean, var = _conditional_moments(X_prev, dt)
        mean = np.clip(mean, 1e-300, 1.0 - 1e-16)
        cap = (1.0 - 1e-12) * mean * (1.0 - mean)
        var = np.minimum(np.maximum(var, 1e-300), cap)
        nu = mean * (1.0 - mean) / var - 1.0
        X = rng.beta(mean * nu, (1.0 - mean) * nu)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=200000, dt=0.04, T=1.0, seed=42)
    print(f"Empirical mean:     {r['empirical_mean']:.6f}  (exact 0.227061)")
    print(f"Empirical variance: {r['empirical_variance']:.6f}  (exact 0.060993)")
    print(f"min/max: {r['terminal_paths'].min():.3e} {r['terminal_paths'].max():.6f}")
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
