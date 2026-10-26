"""Full-truncation Euler-Maruyama for V; exact Bessel-3 step for R = 1/(c_lv*S)
on the frozen clock V^+ dt, with Brownian-bridge survival weights for both barriers.

Model (Ito, zero rate):
    dS = sqrt(V) c_lv S^2 dW1,  dV = kappa (eta - V) dt + sigma sqrt(V) dW2,
    d<W1,W2> = rho dt,  dW1 = dZ1,  dW2 = rho dZ1 + sqrt(1-rho^2) dZ2.
"""
import numpy as np

C_LV = 0.01
KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.114
L_BAR = 50.0
U_BAR = 150.0
SPOTS = (75.0, 100.0, 125.0)
MID = 1  # index of S0 = 100 in SPOTS (the spot whose terminal states are returned)
FUNC_NAMES = ("no_touch_S75", "no_touch_S100", "no_touch_S125")
CHUNK = 25000


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    if dW is not None:
        dW = np.asarray(dW, dtype=float)
        num_paths = dW.shape[0]
        Nt = dW.shape[1]
        dt = T / Nt
    else:
        Nt = max(1, round(T / dt))
        dt = T / Nt

    obs = observables or {}
    rho_c = np.sqrt(1.0 - RHO ** 2)
    a = 1.0 / (C_LV * U_BAR)   # lower R barrier (upper S barrier)
    b = 1.0 / (C_LV * L_BAR)   # upper R barrier (lower S barrier)
    R_init = np.array([1.0 / (C_LV * s) for s in SPOTS])[:, None]   # (3, 1)

    S_T = np.empty(num_paths)
    V_T = np.empty(num_paths)
    W_all = np.empty((len(SPOTS), num_paths))
    acc = {k: np.zeros(num_paths) for k in obs}

    for lo in range(0, num_paths, CHUNK):
        hi = min(lo + CHUNK, num_paths)
        m = hi - lo
        if dW is None:
            dz = np.sqrt(dt) * rng.standard_normal((m, Nt, 2))
        else:
            dz = dW[lo:hi]
        aux = rng.standard_normal((m, Nt, 2))      # Bessel-3 auxiliary Gaussians

        V = np.full(m, V0)
        R = np.repeat(R_init, m, axis=1)           # (3, m)
        w = np.ones((len(SPOTS), m))
        for n in range(Nt):
            dz1 = dz[:, n, 0]
            dz2 = dz[:, n, 1]
            Vp = np.maximum(V, 0.0)
            s = np.sqrt(Vp)
            tau = Vp * dt
            V_new = V + KAPPA * (ETA - Vp) * dt + SIGMA * s * (RHO * dz1 + rho_c * dz2)
            g2 = aux[:, n, 0]
            g3 = aux[:, n, 1]
            R_new = np.sqrt((R - s * dz1) ** 2 + tau * (g2 ** 2 + g3 ** 2))

            moving = tau > 0.0
            tau_safe = np.where(moving, tau, 1.0)
            fa = 1.0 - np.exp(-2.0 * (R - a) * (R_new - a) / tau_safe)
            fb = 1.0 - np.exp(-2.0 * (b - R) * (b - R_new) / tau_safe)
            f = np.where(moving, fa * fb, 1.0)
            inside = (R_new > a) & (R_new < b)
            w = w * np.where(inside, f, 0.0)

            if obs:
                X_prev = np.column_stack((1.0 / (C_LV * R[MID]), V))
                X_new = np.column_stack((1.0 / (C_LV * R_new[MID]), V_new))
                for k, phi in obs.items():
                    acc[k][lo:hi] += 0.5 * (phi(X_prev) + phi(X_new)) * dt
            R = R_new
            V = V_new

        S_T[lo:hi] = 1.0 / (C_LV * R[MID])
        V_T[lo:hi] = V
        W_all[:, lo:hi] = w

    X = np.column_stack((S_T, V_T))
    funcs = {}
    funcs_se = {}
    for i, name in enumerate(FUNC_NAMES):
        funcs[name] = float(np.mean(W_all[i]))
        funcs_se[name] = float(np.std(W_all[i], ddof=1) / np.sqrt(num_paths))

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": funcs,
        "functionals_se": funcs_se,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r4 = solve_sde(400000, 0.002, 1.0, seed=42)
    r1 = solve_sde(100000, 0.002, 1.0, seed=42)
    print("mean:", r4["empirical_mean"])
    print("variance:", r4["empirical_variance"])
    print("E[V_T] exact 0.048380, got", r4["empirical_mean"][1])
    print("min S_T:", float(r4["terminal_paths"][:, 0].min()))
    for k in FUNC_NAMES:
        d = abs(r4["functionals"][k] - r1["functionals"][k])
        bar = 2 * np.sqrt(r4["functionals_se"][k] ** 2 + r1["functionals_se"][k] ** 2)
        print(f"{k}: 400k={r4['functionals'][k]:.6f} (se {r4['functionals_se'][k]:.2e})  "
              f"100k={r1['functionals'][k]:.6f} (se {r1['functionals_se'][k]:.2e})  "
              f"|diff|={d:.2e} bar={bar:.2e}")
    # hook checks
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dWt)["dt"])
    ro = solve_sde(1000, 0.01, 1.0, observables={"v": lambda x: x[:, 1]})
    print("obs hook shape:", ro["path_integrals"]["v"].shape)
