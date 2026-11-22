import numpy as np

KAPPA = 22.84
THETA = 0.218
EPS = 8.56
V0 = 0.15


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Explicit truncated Milstein in L = log V with a Heun (trapezoid) drift, for
    dV = kappa V (theta - V) dt + eps V^1.5 dW."""
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        # antithetic pairing inside the returned sample: paths i and i+half share |dW|
        half = (num_paths + 1) // 2
        base = np.sqrt(dt) * rng.standard_normal((half, Nt))
        dW = np.concatenate([base, -base[: num_paths - half]], axis=0)
        antithetic = True
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        antithetic = False

    R = dt ** -0.25                      # truncation radius, from the effective dt
    c0 = KAPPA * THETA
    c1 = KAPPA + 0.5 * EPS ** 2
    c2 = 0.25 * EPS ** 2

    L = np.full(num_paths, np.log(V0), dtype=float)
    V = np.exp(L)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    n_trunc = 0

    for n in range(Nt):
        V_prev = V
        dw = dW[:, n]
        Vt = np.minimum(V_prev, R)       # V itself is never modified
        n_trunc += int(np.count_nonzero(V_prev > R))
        mu = c0 - c1 * Vt
        sig = EPS * np.sqrt(np.maximum(Vt, 0.0))
        # noise + Milstein correction at the entering state
        L_pred = L + sig * dw + c2 * Vt * (dw ** 2 - dt)
        # drift by the explicit trapezoid (Heun) rule: second order in the stiff drift
        mu_pred = c0 - c1 * np.minimum(np.exp(L_pred), R)
        L = L_pred + 0.5 * dt * (mu + mu_pred)
        V = np.exp(L)
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt

    assert np.all(np.isfinite(L))

    inv = 1.0 / V
    m = float(np.mean(inv))
    var = float(np.var(inv, ddof=1))
    if antithetic and num_paths >= 4:
        # SE of the combined estimator: delete-one-pair jackknife over the (i, i+half) pairs
        half = (num_paths + 1) // 2
        npair = num_paths - half            # complete pairs
        a1, a2 = inv[:npair], inv[half:half + npair]
        extra = inv[npair:half]             # unpaired path when num_paths is odd
        pm = 0.5 * (a1 + a2)
        # pair-level estimators for the mean
        units = np.concatenate([pm, extra]) if extra.size else pm
        se_mean = float(np.std(units, ddof=1) / np.sqrt(units.size))
        # jackknife the sample variance by deleting one pair at a time (vectorised)
        N = float(num_paths)
        S1 = inv.sum(); S2 = np.sum(inv ** 2)
        d1 = a1 + a2; d2 = a1 ** 2 + a2 ** 2
        n1 = N - 2
        v_del = (S2 - d2 - (S1 - d1) ** 2 / n1) / (n1 - 1)
        se_var = float(np.sqrt((npair - 1) / npair * np.sum((v_del - v_del.mean()) ** 2)))
    else:
        se_mean = float(np.std(inv, ddof=1) / np.sqrt(num_paths))
        nb = 20 if num_paths >= 200 else max(2, num_paths // 2)
        bs = num_paths // nb
        bv = np.array([np.var(inv[i * bs:(i + 1) * bs], ddof=1) for i in range(nb)])
        se_var = float(np.std(bv, ddof=1) / np.sqrt(nb))

    out = {
        "terminal_paths": V,
        "empirical_mean": float(np.mean(V)),
        "empirical_variance": float(np.var(V, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "truncation_activations": n_trunc,
        "functionals": {"inv_var_mean": m, "inv_var_variance": var},
        "functional_se": {"inv_var_mean": se_mean, "inv_var_variance": se_var},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        r = solve_sde(M, 0.004, 0.1, seed=42)
        print(f"M={M} mean V={r['empirical_mean']:.6f} var V={r['empirical_variance']:.6f} "
              f"E[1/V]={r['functionals']['inv_var_mean']:.5f} (se {r['functional_se']['inv_var_mean']:.4f}) "
              f"Var[1/V]={r['functionals']['inv_var_variance']:.4f} (se {r['functional_se']['inv_var_variance']:.4f}) "
              f"trunc={r['truncation_activations']} minV={r['terminal_paths'].min():.4g}")
    dW = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook:", solve_sde(1000, 0.004, 0.1, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.004, 0.1,
                                 observables={"x": lambda a: 1.0 / a})["path_integrals"]["x"].shape)
