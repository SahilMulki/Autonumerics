"""Heston double-no-touch: Euler-family scheme.

V : truncated scalar CIR Milstein step (Vp = max(V, 0) everywhere V enters; state not clipped)
y : log-Euler on ln(S/S0) with frozen per-step variance
Survival: per-step Brownian-bridge weight in log space (continuous monitoring),
          one (V, y) path set serves all three spots.
"""
import numpy as np

KAPPA, ETA, SIGMA, RHO = 2.58, 0.043, 1.0, -0.36
V0, S0_STATE = 0.114, 100.0
SPOTS = (75.0, 100.0, 125.0)
LOWER, UPPER = 50.0, 150.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sqdt = np.sqrt(dt)
    c = np.sqrt(1.0 - RHO ** 2)
    lU, lL = np.log(UPPER), np.log(LOWER)
    lnS0 = [np.log(s) for s in SPOTS]

    V = np.full(num_paths, V0, dtype=float)
    y = np.zeros(num_paths)
    W = [np.ones(num_paths) for _ in SPOTS]

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(V_, y_):
        return np.column_stack([S0_STATE * np.exp(y_), np.maximum(V_, 0.0)])

    X = state(V, y) if obs else None

    for n in range(Nt):
        if dW is None:
            Z = rng.standard_normal((num_paths, 2)) * sqdt   # per-step draw (memory)
            dZ1, dZ2 = Z[:, 0], Z[:, 1]
        else:
            dZ1, dZ2 = dW[:, n, 0], dW[:, n, 1]
        dW1 = dZ1
        dW2 = RHO * dZ1 + c * dZ2

        Vp = np.maximum(V, 0.0)
        sq = np.sqrt(Vp)
        V_new = (V + KAPPA * (ETA - Vp) * dt + SIGMA * sq * dW2
                 + 0.25 * SIGMA ** 2 * (dW2 ** 2 - dt))
        y_new = y - 0.5 * Vp * dt + sq * dW1
        s2 = Vp * dt
        pos = s2 > 0.0
        s2safe = np.where(pos, s2, 1.0)

        for k in range(len(SPOTS)):
            a = lnS0[k] + y
            b = lnS0[k] + y_new
            inside = (b > lL) & (b < lU)
            p_up = np.exp(-2.0 * (lU - a) * (lU - b) / s2safe)
            p_dn = np.exp(-2.0 * (a - lL) * (b - lL) / s2safe)
            surv = np.clip(1.0 - p_up - p_dn, 0.0, 1.0)
            surv = np.where(pos, surv, 1.0)
            surv = np.where(inside, surv, 0.0)
            W[k] *= surv

        V, y = V_new, y_new
        if obs:
            X_prev, X = X, state(V, y)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    term = state(V, y)
    out = {
        "terminal_paths": term,
        "empirical_mean": [float(m) for m in np.mean(term, axis=0)],
        "empirical_variance": [float(v) for v in np.var(term, axis=0, ddof=1)],
        "functionals": {f"no_touch_S{int(s)}": float(np.mean(W[k]))
                        for k, s in enumerate(SPOTS)},
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=400000, dt=0.002, T=1.0, seed=42)
    print("Empirical mean:     ", r["empirical_mean"])
    print("Empirical variance: ", r["empirical_variance"])
    print("Functionals:        ", r["functionals"])
    r2 = solve_sde(num_paths=100000, dt=0.002, T=1.0, seed=42)
    print("100k functionals:   ", r2["functionals"])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.002, 1.0, dW=dWt)["dt"])
    po = solve_sde(1000, 0.01, 1.0, observables={"V": lambda x: x[:, 1]})
    print("obs hook:", po["path_integrals"]["V"].shape)
