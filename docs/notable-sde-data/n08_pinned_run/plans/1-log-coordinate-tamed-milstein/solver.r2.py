import numpy as np

X_0, KAPPA, THETA, EPS = 0.15, 22.84, 0.218, 8.56


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Milstein-type step for Y = 1/V (Ito chart change, V = 1/Y).

    dY = (a - b Y) dt - eps sqrt(Y) dW,  a = kappa + eps^2,  b = kappa*theta.
    Step: Crank-Nicolson on the linear drift, Milstein for the noise:
      (1 + b dt/2) Y' = (1 - b dt/2) Y + (a - eps^2/4) dt
                        + eps sqrt(Y) dB + eps^2/4 dB^2,   dB = -dW_n
    Falls back to the fully drift-implicit variant (positive by construction)
    on the rare path where the CN value would not be positive.
    One supplied Brownian increment per step; no transition law is sampled.
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    e2 = EPS ** 2
    a = KAPPA + e2
    b = KAPPA * THETA
    c0 = a - 0.25 * e2
    beta = 0.5 * b * dt
    Y = np.full(num_paths, 1.0 / X_0)
    V = 1.0 / Y
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    n_fallback = 0

    for n in range(Nt):
        V_prev = V
        w = -dW[:, n]
        s = np.sqrt(Y)
        core = Y + EPS * s * w + 0.25 * e2 * w * w + c0 * dt   # (sqrt(Y)+eps w/2)^2 + (a-e2/4)dt
        Y_cn = (core - beta * Y) / (1.0 + beta)
        Y_im = core / (1.0 + 2.0 * beta)
        bad = ~(Y_cn > 0.0)
        if bad.any():
            n_fallback += int(bad.sum())
            Y_cn = np.where(bad, Y_im, Y_cn)
        Y = Y_cn
        V = 1.0 / Y
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt

    M = num_paths
    R = Y  # = 1/V
    ym = float(R.mean())
    yv = float(R.var(ddof=1))
    mu4 = float(np.mean((R - ym) ** 4))
    out = {
        "terminal_paths": V,
        "empirical_mean": float(V.mean()),
        "empirical_variance": float(V.var(ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "n_fallback_steps": n_fallback,
        "functionals": {"inv_var_mean": ym, "inv_var_variance": yv},
        "functional_se": {"inv_var_mean": float(R.std(ddof=1) / np.sqrt(M)),
                          "inv_var_variance": float(np.sqrt(max(mu4 - yv ** 2, 0.0) / M))},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        r = solve_sde(num_paths=M, dt=0.004, T=0.1, seed=42)
        print(f"M={M} mean={r['empirical_mean']:.6f} var={r['empirical_variance']:.6f} "
              f"fallback={r['n_fallback_steps']} min={r['terminal_paths'].min():.3e}")
        f, se = r["functionals"], r["functional_se"]
        print(f"  inv_var_mean {f['inv_var_mean']:.4f} +/- {se['inv_var_mean']:.4f} "
              f"({f['inv_var_mean']/11.6227862851-1:+.2%}); inv_var_variance "
              f"{f['inv_var_variance']:.3f} +/- {se['inv_var_variance']:.3f} "
              f"({f['inv_var_variance']/45.2351064760-1:+.2%})")
    dW = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.004, 0.1, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.004, 0.1,
          observables={"x": lambda a: 1.0 / a})["path_integrals"]["x"].shape)
