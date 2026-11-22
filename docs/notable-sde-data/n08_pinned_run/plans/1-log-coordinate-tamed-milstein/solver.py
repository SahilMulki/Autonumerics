import numpy as np

X_0, KAPPA, THETA, EPS = 0.15, 22.84, 0.218, 8.56


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Strang-splitting step for Y = 1/V (Ito chart change, V = 1/Y).

    dY = (a - b Y) dt - eps sqrt(Y) dW,  a = kappa + eps^2,  b = kappa*theta.
    Generator split: L = L_drift + L_noise with
      drift: dY = (a - eps^2/4 - b Y) dt   (exact exponential flow)
      noise: dY = eps sqrt(Y) o dB          (exact flow (sqrt(Y)+eps B/2)^2)
    Step = half drift, noise with the supplied increment dB=-dW_n, half drift.
    Both sub-flows are exact, so the weak error is O(dt^2). Positive by
    construction; one supplied Brownian increment per step; no transition
    law is sampled.
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
    c0 = a - 0.25 * e2            # Stratonovich-corrected constant drift (>0)
    h = 0.5 * dt
    decay = np.exp(-b * h)
    shift = (c0 / b) * (1.0 - decay)
    def run(sign, obs):
        Y = np.full(num_paths, 1.0 / X_0)
        V = 1.0 / Y
        acc = {k: np.zeros(num_paths) for k in obs}
        for n in range(Nt):
            V_prev = V
            w = -sign * dW[:, n]
            Y = Y * decay + shift                      # exact half drift flow
            Y = (np.sqrt(Y) + 0.5 * EPS * w) ** 2      # exact noise flow
            Y = Y * decay + shift                      # half drift again (Strang)
            V = 1.0 / Y
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt
        return Y, V, acc

    Y, V, acc = run(1.0, observables or {})
    Ya, _, _ = run(-1.0, {})   # antithetic mirror (-dW), used ONLY for the functionals
    n_fallback = 0   # no fallback branch exists in the splitting scheme

    M = num_paths
    # antithetic pair-level estimators (the dW-driven run and its mirror are one pair)
    u = 0.5 * (Y + Ya)
    q = 0.5 * (Y ** 2 + Ya ** 2)
    um, qm = float(u.mean()), float(q.mean())
    ym = um
    yv = (qm - um ** 2) * (2.0 * M) / (2.0 * M - 1.0)
    infl = (q - qm) - 2.0 * um * (u - um)        # influence function of var per pair
    se_mean = float(u.std(ddof=1) / np.sqrt(M))
    se_var = float(infl.std(ddof=1) / np.sqrt(M))
    out = {
        "terminal_paths": V,
        "empirical_mean": float(V.mean()),
        "empirical_variance": float(V.var(ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "n_fallback_steps": n_fallback,
        "functionals": {"inv_var_mean": ym, "inv_var_variance": float(yv)},
        "functional_se": {"inv_var_mean": se_mean, "inv_var_variance": se_var},
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
