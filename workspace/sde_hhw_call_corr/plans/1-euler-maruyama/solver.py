"""Heston-Hull-White call, all three factors correlated -- Euler-Maruyama plan.

Scheme (per step n, all right-hand sides at time n):
    Z    = independent increments (already scaled by sqrt(dt)), correlated by Lc
    V+   = max(V, 0)                                  full truncation
    I   += R dt                                       left-point rate integral
    x   += (R - V+/2) dt + sqrt(V+) dW1               log-Euler, x = log(S/S0)
    V   += kappa (eta - V+) dt + sigma_v sqrt(V+) dW2
    R   += a_r (b_r - R) dt + sigma_r dW3

Square root at V = 0: every sqrt(V) is sqrt(max(V, 0)) (full truncation); the raw state V is
kept and may go negative.  The exact-sampling trade is resolved by NOT sampling V from its
noncentral-chi2 law: V is stepped with the W2 increment, so W1 and W3 are correlated with the
actual variance noise through the Cholesky factor, and every channel consumes the supplied dW.

Discounting: D_T = exp(-I) accumulated path by path.  Using the same left-point R_n in the
discount and in the log-S drift makes D_T S_T = S0 exp(sum sqrt(V+) dW1 - V+ dt / 2), a discrete
exact martingale with mean S0.  It and D_T (whose exact discrete-scheme mean is computed in
closed form, since Euler-OU and its Riemann sum are jointly Gaussian) are used as control
variates with coefficients estimated from the run itself.
"""
import numpy as np

# ---- parameters (problem_spec.json) -------------------------------------------------------
S0_MID = 100.0
SPOTS = {"price_S75": 75.0, "price_S100": 100.0, "price_S125": 125.0}
K = 100.0
KAPPA, ETA, SIGMA_V, V_0 = 0.5, 0.04, 1.0, 0.04
A_R, B_R, SIGMA_R, R_0 = 0.08, 0.10, 0.09, 0.10
RHO_SV, RHO_SR, RHO_VR = -0.9, 0.6, -0.7

CHUNK = 1 << 16  # paths per batch (cache-sized working set)


def _cholesky():
    l11 = np.sqrt(1.0 - RHO_SV ** 2)
    l21 = (RHO_VR - RHO_SR * RHO_SV) / l11
    l22 = np.sqrt(1.0 - RHO_SR ** 2 - l21 ** 2)
    return np.array([[1.0, 0.0, 0.0],
                     [RHO_SV, l11, 0.0],
                     [RHO_SR, l21, l22]])


def _discrete_bond_mean(Nt, dt):
    """Exact E[exp(-dt * sum_{n<Nt} R_n)] for the Euler-OU recursion used here."""
    phi = 1.0 - A_R * dt
    # mean of R_n: b + (R0 - b) phi^n
    n = np.arange(Nt)
    mean_I = dt * np.sum(B_R + (R_0 - B_R) * phi ** n)
    # sum_{n<Nt} (R_n - E R_n) = sum_{k=0}^{Nt-2} sigma_r eps_k (1 - phi^{Nt-1-k}) / (1 - phi)
    j = np.arange(1, Nt)  # j = Nt-1-k
    w = (1.0 - phi ** j) / (1.0 - phi)
    var_I = dt ** 2 * SIGMA_R ** 2 * dt * np.sum(w ** 2)
    return float(np.exp(-mean_I + 0.5 * var_I))


def _price_cv(P, Y, D, S0, ED):
    """Two-control regression estimator: controls Y = D S_T (mean S0) and D (mean ED)."""
    M = P.size
    C = np.stack([Y - S0, D - ED], axis=1)
    Cc = C - C.mean(axis=0)
    Pc = P - P.mean()
    beta = np.linalg.solve(Cc.T @ Cc, Cc.T @ Pc)
    adj = P - C @ beta
    price = float(adj.mean())
    se = float(adj.std(ddof=3) / np.sqrt(M))
    se_plain = float(P.std(ddof=1) / np.sqrt(M))
    return price, se, se_plain, beta


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    # dW, when given: INDEPENDENT increments (num_paths, Nt, 3), already scaled by sqrt(dt).
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim != 3 or dW.shape[0] != num_paths or dW.shape[2] != 3:
            raise ValueError(f"dW must have shape (num_paths, Nt, 3), got {dW.shape}")
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sqdt = np.sqrt(dt)
    Lc = _cholesky()
    l10, l11 = Lc[1, 0], Lc[1, 1]
    l20, l21, l22 = Lc[2, 0], Lc[2, 1], Lc[2, 2]
    obs = observables or {}

    x_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    R_all = np.empty(num_paths)
    I_all = np.empty(num_paths)
    acc = {k: np.zeros(num_paths) for k in obs}

    for i0 in range(0, num_paths, CHUNK):
        i1 = min(num_paths, i0 + CHUNK)
        m = i1 - i0
        x = np.zeros(m)
        V = np.full(m, V_0)
        R = np.full(m, R_0)
        I = np.zeros(m)
        Vp = np.empty(m)
        sq = np.empty(m)
        tmp = np.empty(m)
        dW2 = np.empty(m)
        dW3 = np.empty(m)
        zbuf = np.empty((3, m)) if rng is not None else None

        def state():
            return np.stack([S0_MID * np.exp(x), V, R], axis=1)

        X_prev_obs = state() if obs else None

        for n in range(Nt):
            if rng is not None:
                rng.standard_normal(out=zbuf)
                zbuf *= sqdt
                z0, z1, z2 = zbuf[0], zbuf[1], zbuf[2]
            else:
                zn = dW[i0:i1, n, :]
                z0, z1, z2 = zn[:, 0], zn[:, 1], zn[:, 2]
            # correlated increments: dW1 = z0, dW2 = l10 z0 + l11 z1, dW3 = l20 z0 + l21 z1 + l22 z2
            np.multiply(z0, l10, out=dW2); dW2 += l11 * z1
            np.multiply(z0, l20, out=dW3); dW3 += l21 * z1; dW3 += l22 * z2

            np.maximum(V, 0.0, out=Vp)
            np.sqrt(Vp, out=sq)
            # left-point discount integral
            I += R * dt
            # log-Euler asset: x += (R - Vp/2) dt + sq dW1
            np.multiply(Vp, -0.5, out=tmp); tmp += R; tmp *= dt; x += tmp
            np.multiply(sq, z0, out=tmp); x += tmp
            # full-truncation Euler variance: V += kappa (eta - Vp) dt + sigma_v sq dW2
            np.subtract(ETA, Vp, out=tmp); tmp *= KAPPA * dt; V += tmp
            np.multiply(sq, dW2, out=tmp); tmp *= SIGMA_V; V += tmp
            # Euler OU rate (uses R_n): R += a_r (b_r - R) dt + sigma_r dW3
            np.subtract(B_R, R, out=tmp); tmp *= A_R * dt; tmp += SIGMA_R * dW3; R += tmp

            if obs:
                X_new = state()
                for k, phi in obs.items():
                    acc[k][i0:i1] += 0.5 * (phi(X_prev_obs) + phi(X_new)) * dt
                X_prev_obs = X_new

        x_all[i0:i1] = x
        V_all[i0:i1] = V
        R_all[i0:i1] = R
        I_all[i0:i1] = I

    S1 = np.exp(x_all)              # S_T for S0 = 1
    D = np.exp(-I_all)              # path-accumulated discount factor
    DS1 = D * S1                    # discounted asset per unit spot (mean 1 under the scheme)
    ED = _discrete_bond_mean(Nt, dt)

    functionals, se, se_plain, betas = {}, {}, {}, {}
    for name, S0 in SPOTS.items():
        P = D * np.maximum(S0 * S1 - K, 0.0)
        p, s, s0, b = _price_cv(P, S0 * DS1, D, S0, ED)
        functionals[name] = p
        se[name] = s
        se_plain[name] = s0
        betas[name] = b.tolist()

    X = np.column_stack([S0_MID * S1, V_all, R_all])
    out = {
        "terminal_paths": X,
        "empirical_mean": X.mean(axis=0).tolist(),
        "empirical_variance": X.var(axis=0, ddof=1).tolist(),
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "discount_factor": D,       # per-path exp(-sum R_n dt), for the identity checks
        "diagnostics": {
            "price_se": se,
            "price_se_plain_mc": se_plain,
            "cv_beta": betas,
            "mean_D": float(D.mean()),
            "discrete_E_D": ED,
            "mean_DS_over_S0": float(DS1.mean()),
            "mean_V_plus": float(np.maximum(V_all, 0.0).mean()),
            "frac_V_negative": float(np.mean(V_all < 0.0)),
        },
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time

    for M in (1_000_000, 250_000):
        t0 = time.perf_counter()
        res = solve_sde(num_paths=M, dt=0.004, T=10, seed=42)
        el = time.perf_counter() - t0
        d = res["diagnostics"]
        print(f"=== num_paths={M}  dt={res['dt']}  time={el:.1f}s")
        print("Empirical mean:    ", res["empirical_mean"])
        print("Empirical variance:", res["empirical_variance"])
        for k, v in res["functionals"].items():
            print(f"  {k}: {v:.6f}  se={d['price_se'][k]:.6f} (plain {d['price_se_plain_mc'][k]:.6f})")
        for k in ("mean_D", "discrete_E_D", "mean_DS_over_S0", "mean_V_plus", "frac_V_negative"):
            print(f"  {k}: {d[k]:.6f}")
        print("  finite:", bool(np.all(np.isfinite(res["terminal_paths"]))),
              " S>0:", bool(np.all(res["terminal_paths"][:, 0] > 0)))

    # hooks
    rng = np.random.default_rng(0)
    dW = np.sqrt(0.04) * rng.standard_normal((1000, 250, 3))
    r = solve_sde(1000, 0.123, 10.0, dW=dW)
    print("dW hook: dt =", r["dt"], "shape", r["terminal_paths"].shape)
    r = solve_sde(1000, 0.04, 10.0, observables={"VR": lambda X: X[:, 1] * X[:, 2]})
    print("obs hook:", r["path_integrals"]["VR"].shape, float(np.mean(r["path_integrals"]["VR"])))
