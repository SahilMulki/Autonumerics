import numpy as np

X_0, KAPPA, THETA, EPS = 0.15, 22.84, 0.218, 8.56


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Drift-implicit Euler on ln V (Ito-corrected), exact positivity, weak order 1."""
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    V = np.full(num_paths, X_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    c = (KAPPA + 0.5 * EPS ** 2) * dt
    for n in range(Nt):
        V_prev = V
        dw = dW[:, n]
        # Ito on x = ln V:  dx = [kappa*theta - (kappa + eps^2/2) V] dt + eps sqrt(V) dW.
        # Explicit noise + constant drift part; the stiff restoring term is taken
        # implicitly at V_{n+1}:  x + c e^x = b  (monotone, convex -> Newton from the
        # right converges monotonically).  V = e^x > 0 by construction, no clipping.
        b = np.log(V_prev) + KAPPA * THETA * dt + EPS * np.sqrt(V_prev) * dw
        x = b.copy()
        for _ in range(25):
            ex = np.exp(x)
            x -= (x + c * ex - b) / (1.0 + c * ex)
        V = np.exp(x)
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt

    y = 1.0 / V
    ym = float(np.mean(y))
    yv = float(np.var(y, ddof=1))
    c = y - ym
    se_mean = float(np.std(y, ddof=1) / np.sqrt(num_paths))
    se_var = float(np.sqrt(max(np.mean(c ** 4) - yv ** 2, 0.0) / num_paths))

    out = {
        "terminal_paths": V,
        "empirical_mean": float(np.mean(V)),
        "empirical_variance": float(np.var(V, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"inv_var_mean": ym, "inv_var_variance": yv},
        "functional_se": {"inv_var_mean": se_mean, "inv_var_variance": se_var},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        r = solve_sde(num_paths=M, dt=0.004, T=0.1, seed=42)
        print(f"M={M} mean={r['empirical_mean']:.6f} var={r['empirical_variance']:.6f} "
              f"min V={r['terminal_paths'].min():.3e}")
        print("  functionals:", r["functionals"], "se:", r["functional_se"])
    dW = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook:", solve_sde(1000, 0.004, 0.1, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.004, 0.1,
          observables={"x": lambda a: 1.0 / a})["path_integrals"]["x"].shape)
