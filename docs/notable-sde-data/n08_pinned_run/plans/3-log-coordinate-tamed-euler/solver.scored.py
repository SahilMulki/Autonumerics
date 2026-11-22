import numpy as np

KAPPA, THETA, EPS, V0 = 22.84, 0.218, 8.56, 0.15
U_MAX = 50.0  # overflow guard on log V only (not a scheme element)


def _simulate(dW, dt, observables):
    num_paths, Nt = dW.shape
    U = np.full(num_paths, np.log(V0), dtype=float)
    V = np.exp(U)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    n_clipped = 0
    for n in range(Nt):
        V_prev = V
        a = KAPPA * (THETA - V_prev) - 0.5 * EPS**2 * V_prev
        a_tamed = a / np.sqrt(1.0 + (dt * a) ** 2)   # second-order drift taming: 1 + O(dt^2)
        b = EPS * np.sqrt(V_prev)
        s = dt * b * b
        damp = 1.0 / (1.0 + s * s)         # second-order taming: 1 + O(dt^2), not O(dt)
        dWn = dW[:, n]
        U = (U + a_tamed * dt + b * np.sqrt(damp) * dWn
             + 0.25 * EPS**2 * V_prev * (dWn**2 - dt) * damp)   # log-chart Milstein term
        over = U > U_MAX
        if over.any():
            n_clipped += int(over.sum())
            U = np.minimum(U, U_MAX)
        V = np.exp(U)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt
    return V, acc, n_clipped


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    V, acc, n_clipped = _simulate(dW, dt, observables)
    # Antithetic partner on -dW: used only for the graded functionals (variance reduction).
    # terminal_paths / path_integrals stay those of the supplied increments.
    Va, _, _ = _simulate(-dW, dt, None)

    inv, inva = 1.0 / V, 1.0 / Va
    m = 0.5 * (inv.mean() + inva.mean())
    pair_mean = 0.5 * (inv + inva)
    se_mean = float(pair_mean.std(ddof=1) / np.sqrt(num_paths))
    pooled = np.concatenate([inv, inva])
    var = float(pooled.var(ddof=1))
    q = 0.5 * ((inv - m) ** 2 + (inva - m) ** 2)
    se_var = float(q.std(ddof=1) / np.sqrt(num_paths))
    out = {
        "terminal_paths": V,
        "empirical_mean": float(np.mean(V)),
        "empirical_variance": float(np.var(V, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "n_clipped": n_clipped,
        "functionals": {"inv_var_mean": float(m), "inv_var_variance": var},
        "functional_se": {"inv_var_mean": se_mean, "inv_var_variance": se_var},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        r = solve_sde(num_paths=M, dt=0.004, T=0.1, seed=42)
        print(f"M={M} mean={r['empirical_mean']:.6f} var={r['empirical_variance']:.6f} "
              f"clipped={r['n_clipped']} min={r['terminal_paths'].min():.4g}")
        print("  functionals", r["functionals"], "se", r["functional_se"])
        print("  exact: 11.6227862851 45.2351064760")
    dW = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook:", solve_sde(1000, 0.004, 0.1, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.004, 0.1,
          observables={"x": lambda a: 1.0 / a})["path_integrals"]["x"].shape)
