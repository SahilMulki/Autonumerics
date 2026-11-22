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

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    c = (KAPPA + 0.5 * EPS ** 2) * dt

    def run(inc, track):
        V = np.full(inc.shape[0], X_0, dtype=float)
        for n in range(Nt):
            V_prev = V
            dw = inc[:, n]
            # Ito on x = ln V:  dx = [kappa*theta - (kappa + eps^2/2) V] dt + eps sqrt(V) dW.
            # Restoring term implicit: x + c e^x = b (monotone, convex; Newton from the right).
            b = np.log(V_prev) + KAPPA * THETA * dt + EPS * np.sqrt(V_prev) * dw
            x = b.copy()
            for _ in range(25):
                ex = np.exp(x)
                x -= (x + c * ex - b) / (1.0 + c * ex)
            V = np.exp(x)
            if track:
                for k, phi in obs.items():
                    acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt
        return V

    V = run(dW, True)
    # Antithetic mirror on -dW feeds ONLY the reported functionals; terminal_paths,
    # empirical_mean/variance and path_integrals come from the supplied increments alone.
    Vm = run(-dW, False)
    y1, y2 = 1.0 / V, 1.0 / Vm
    ym = 0.5 * (float(np.mean(y1)) + float(np.mean(y2)))
    yv = 0.5 * (float(np.mean((y1 - ym) ** 2)) + float(np.mean((y2 - ym) ** 2)))
    yv *= 2 * num_paths / (2 * num_paths - 1)
    # pair-level influence functions give the SE of the combined estimators
    psi_m = 0.5 * ((y1 - ym) + (y2 - ym))
    psi_v = 0.5 * (((y1 - ym) ** 2 - yv) + ((y2 - ym) ** 2 - yv))
    se_mean = float(np.std(psi_m, ddof=1) / np.sqrt(num_paths))
    se_var = float(np.std(psi_v, ddof=1) / np.sqrt(num_paths))

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
