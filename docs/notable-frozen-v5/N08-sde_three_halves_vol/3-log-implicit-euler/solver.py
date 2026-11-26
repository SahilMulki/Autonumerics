import numpy as np

KAPPA = 22.84
THETA = 0.218
EPS = 8.56
V0 = 0.15
C_COEF = KAPPA + 0.5 * EPS ** 2          # coefficient of the implicit e^L drift
KT = KAPPA * THETA                       # constant part of the L drift


def _implicit_log_step(L, dWn, dt):
    """Solve x + c*dt*e^x = r, r = L + kt*dt + eps*e^(L/2)*dW, by safeguarded Newton."""
    cdt = C_COEF * dt
    r = L + KT * dt + EPS * np.exp(0.5 * L) * dWn
    # F(x) = x + cdt*e^x - r is increasing, convex, F' >= 1; start at/right of the root.
    x = np.where(r > 1.0, np.minimum(r, np.log(np.maximum(r, 1.0) / cdt)), r)
    for _ in range(60):
        ex = np.exp(x)
        dx = (x + cdt * ex - r) / (1.0 + cdt * ex)
        x = x - dx
        if np.max(np.abs(dx)) < 1e-12:
            break
    return x


A1 = KAPPA + EPS ** 2                    # drift constant of Y = 1/V (CIR)


def _gen_y(Y):
    """Generator images L(Y), L(Y^2) for Y = 1/V:  dY = (A1 - KT*Y) dt - eps*sqrt(Y) dW."""
    f = A1 - KT * Y
    return f, 2.0 * Y * f + EPS ** 2 * Y


def _cv_estimates(Y, c1, c2):
    """Control-variate estimates of E[Y], Var[Y] (ddof=1) and their standard errors.

    c1, c2 are Dynkin martingale increments of Y and Y^2 along the simulated path (mean zero for
    the SDE itself); regressing Y and Y^2 on them removes their pathwise noise.
    """
    M = Y.shape[0]
    A = np.column_stack([np.ones(M), c1, c2])
    bm, *_ = np.linalg.lstsq(A, Y, rcond=None)
    bs, *_ = np.linalg.lstsq(A, Y * Y, rcond=None)
    e1 = Y - A @ bm
    e2 = Y * Y - A @ bs
    m, q = bm[0], bs[0]
    var = (q - m * m) * M / (M - 1)
    infl = e2 - 2.0 * m * e1
    return (float(m), float(var), float(np.std(e1, ddof=3) / np.sqrt(M)),
            float(np.std(infl, ddof=3) / np.sqrt(M) * M / (M - 1)))


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

    L = np.full(num_paths, np.log(V0), dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    V_prev = np.exp(L) if obs else None
    Y = np.exp(-L)
    f1, f2 = _gen_y(Y)
    I1 = np.zeros(num_paths)
    I2 = np.zeros(num_paths)
    Y0 = Y[0]

    for n in range(Nt):
        L = _implicit_log_step(L, dW[:, n], dt)
        Y = np.exp(-L)
        g1, g2 = _gen_y(Y)
        I1 += 0.5 * (f1 + g1) * dt          # trapezoid of L(Y) and L(Y^2) along the path
        I2 += 0.5 * (f2 + g2) * dt
        f1, f2 = g1, g2
        if obs:
            V_new = np.exp(L)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(V_prev) + phi(V_new)) * dt
            V_prev = V_new

    V = np.exp(L)
    c1 = Y - Y0 - I1                     # Dynkin martingale pieces, mean 0 for the SDE
    c2 = Y * Y - Y0 * Y0 - I2
    M = num_paths
    m = float(np.mean(Y))
    s2 = float(np.var(Y, ddof=1))
    m4 = float(np.mean((Y - m) ** 4))
    # graded functionals: control-variate estimators (plain sample moments kept as *_plain)
    m_cv, s2_cv, se_m, se_v = _cv_estimates(Y, c1, c2)
    out = {
        "terminal_paths": V,
        "empirical_mean": float(np.mean(V)),
        "empirical_variance": float(np.var(V, ddof=1)),
        "num_paths": M,
        "dt": dt,
        "T": T,
        "min_log_V_T": float(np.min(L)),   # underflow guard: must be > -700
        "functionals": {"inv_var_mean": m_cv, "inv_var_variance": s2_cv},
        "plain_functionals": {"inv_var_mean": m, "inv_var_variance": s2},
        "functional_se": {
            "inv_var_mean": se_m,
            "inv_var_variance": se_v,
        },
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    T = 0.1
    ref_m, ref_v = 11.622786285052442, 45.23510647598556
    for M in (200000, 50000):
        r = solve_sde(M, 0.004, T, seed=42)
        f = r["functionals"]
        print(f"M={M} Empirical mean: {r['empirical_mean']:.6f}  "
              f"Empirical variance: {r['empirical_variance']:.6f}  "
              f"min V={r['terminal_paths'].min():.3e} min L={r['min_log_V_T']:.2f}")
        print(f"   inv_var_mean={f['inv_var_mean']:.5f} (ref {ref_m:.5f}, "
              f"rel {f['inv_var_mean']/ref_m-1:+.4f})  "
              f"inv_var_variance={f['inv_var_variance']:.5f} (ref {ref_v:.5f}, "
              f"rel {f['inv_var_variance']/ref_v-1:+.4f})  se={r['functional_se']}")
    # CRN self-convergence at dt, dt/2, dt/4
    M = 50000
    Nf = 25 * 4
    dWf = np.sqrt(T / Nf) * np.random.default_rng(0).standard_normal((M, Nf))
    for agg in (4, 2, 1):
        d = dWf.reshape(M, -1, agg).sum(axis=2)
        r = solve_sde(M, 0.004, T, dW=d)
        f = r["functionals"]
        print(f"CRN dt={r['dt']:.5f} inv_mean={f['inv_var_mean']:.5f} inv_var={f['inv_var_variance']:.5f}")
    print("dW hook:", solve_sde(1000, 0.004, T, dW=dWf[:1000, :50])["dt"])
    o = solve_sde(1000, 0.004, T, observables={"x": lambda a: a})
    print("obs hook:", o["path_integrals"]["x"].shape)
