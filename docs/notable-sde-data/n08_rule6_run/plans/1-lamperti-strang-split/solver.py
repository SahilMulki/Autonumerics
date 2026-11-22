"""Lamperti transform + Strang splitting for the 3/2 stochastic-volatility SDE.

dV = kappa V (theta - V) dt + epsilon V^1.5 dW,  V(0) = 0.15.
X = V**-0.5 gives dX = (-0.5 a X + c / X) dt - sg dW  (constant additive noise).
Drift sub-flow is exact via u = X^2; noise sub-step is exact; reflection at X = 0.
"""
import numpy as np

KAPPA, THETA, EPS, V0 = 22.84, 0.218, 8.56, 0.15


def _cv_estimates(p, q, S, T):
    """Control-variate mean/variance of y with units p=mean(y), q=mean(y^2).

    Controls have exactly known expectation under the driving Brownian motion:
    c1 = S^2 - T, c2 = S^4 - 3T^2 (S = sum of increments, N(0,T)).
    Returns (mean, var, se_mean, se_var), var with ddof=1 correction.
    """
    m = p.size
    c1 = S * S - T
    c2 = S ** 4 - 3.0 * T * T
    C = np.column_stack([c1, c2])
    C = C - 0.0  # expectations are zero exactly; do not recentre
    def adj(z):
        Cc = C - C.mean(axis=0)
        beta, *_ = np.linalg.lstsq(Cc, z - z.mean(), rcond=None)
        return z - C @ beta
    pa, qa = adj(p), adj(q)
    P, Q = pa.mean(), qa.mean()
    f = (2.0 * m) / (2.0 * m - 1.0)
    se_mean = float(np.std(pa, ddof=1) / np.sqrt(m))
    infl = (qa - Q) - 2.0 * P * (pa - P)
    se_var = float(f * np.std(infl, ddof=1) / np.sqrt(m))
    return float(P), float(f * (Q - P * P)), se_mean, se_var


def _step(dW, dt, obs=None):
    num_paths, Nt = dW.shape
    a = KAPPA * THETA
    c = 0.5 * KAPPA + 0.375 * EPS ** 2
    sg = 0.5 * EPS
    u_star = 2.0 * c / a
    e_half = np.exp(-0.5 * a * dt)
    X = np.full(num_paths, V0 ** -0.5, dtype=float)
    obs = obs or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    V = X ** -2.0 if obs else None
    n_reflect = 0
    for n in range(Nt):
        V_prev = V
        u = X * X
        X = np.sqrt(u_star + (u - u_star) * e_half)
        X = X - sg * dW[:, n]
        neg = X < 0.0
        if neg.any():
            n_reflect += int(neg.sum())
            X = np.abs(X)
        u = X * X
        X = np.sqrt(u_star + (u - u_star) * e_half)
        if obs:
            V = X ** -2.0
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt
    return X, acc, n_reflect


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    antithetic = dW is None
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        m = num_paths // 2
        Z = rng.standard_normal((m, Nt))
        Z = np.concatenate([Z, -Z], axis=0)
        if num_paths % 2:
            Z = np.concatenate([Z, rng.standard_normal((1, Nt))], axis=0)
        dW = np.sqrt(dt) * Z
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    obs = observables or {}
    X, acc, n_reflect = _step(dW, dt, obs)
    Vt = X ** -2.0
    y = X * X  # = 1/V_T
    S = dW.sum(axis=1)
    if antithetic and num_paths >= 4:
        m = num_paths // 2
        p = 0.5 * (y[:m] + y[m:2 * m])
        q = 0.5 * (y[:m] ** 2 + y[m:2 * m] ** 2)
        Su = S[:m]
    else:
        p, q, Su = y, y * y, S
    mean_y, var_y, se_m, se_v = _cv_estimates(p, q, Su, T)

    out = {
        "terminal_paths": Vt,
        "empirical_mean": float(np.mean(Vt)),
        "empirical_variance": float(np.var(Vt, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"inv_var_mean": mean_y, "inv_var_variance": var_y},
        "functional_se": {"inv_var_mean": se_m, "inv_var_variance": se_v},
        "n_reflections": n_reflect,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        r = solve_sde(M, 0.004, 0.1, seed=42)
        print(f"M={M}: V mean {r['empirical_mean']:.6f} var {r['empirical_variance']:.6f}")
        print("  inv_var_mean", r["functionals"]["inv_var_mean"], "+-", r["functional_se"]["inv_var_mean"],
              "(exact 11.622786)")
        print("  inv_var_variance", r["functionals"]["inv_var_variance"], "+-", r["functional_se"]["inv_var_variance"],
              "(exact 45.235106)")
        print("  reflections:", r["n_reflections"], "min V", r["terminal_paths"].min())
    dW = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.004, 0.1, dW=dW)["dt"])
    o = solve_sde(1000, 0.004, 0.1, observables={"invV": lambda v: 1.0 / v})
    print("obs hook:", o["path_integrals"]["invV"].shape)
