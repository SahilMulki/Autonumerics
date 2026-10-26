import numpy as np

# Heston, r = 0:  dS = sqrt(V) S dW1,  dV = kappa (eta - V) dt + sigma sqrt(V) dW2,  corr = rho
X_0 = 100.0
V_0 = 0.164
KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Full-truncation Euler-Maruyama on V, log-Euler on S.

    dW, if given, has shape (num_paths, Nt, 2): independent channel increments
    (dZ1, dZ2), already scaled by sqrt(dt). Otherwise they are drawn one step at
    a time from default_rng(seed) (a full block would be 16 GB at 2e6 paths).
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        sdt = np.sqrt(dt)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
        sdt = 1.0

    c = np.sqrt(1.0 - RHO ** 2)
    logR = np.zeros(num_paths)          # log(S / S0)
    Vh = np.full(num_paths, V_0)        # auxiliary, unclipped variance
    Vp = np.empty(num_paths)
    sq = np.empty(num_paths)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(lr, vh):
        return np.column_stack([X_0 * np.exp(lr), np.maximum(vh, 0.0)])

    prev_vals = None
    if obs:
        st = state(logR, Vh)
        prev_vals = {k: phi(st) for k, phi in obs.items()}

    for n in range(Nt):
        if dW is None:
            Z = sdt * rng.standard_normal((num_paths, 2))
            z1, z2 = Z[:, 0], Z[:, 1]
        else:
            z1, z2 = dW[:, n, 0], dW[:, n, 1]
        np.maximum(Vh, 0.0, out=Vp)
        np.sqrt(Vp, out=sq)
        dW2 = RHO * z1 + c * z2
        logR += -0.5 * Vp * dt + sq * z1
        Vh += KAPPA * (ETA - Vp) * dt + SIGMA * sq * dW2
        if obs:
            st = state(logR, Vh)
            for k, phi in obs.items():
                cur = phi(st)
                acc[k] += 0.5 * (prev_vals[k] + cur) * dt
                prev_vals[k] = cur

    R = np.exp(logR)
    V_T = np.maximum(Vh, 0.0)
    terminal = np.column_stack([X_0 * R, V_T])
    functionals = {
        f"price_S{int(s)}": float(np.mean(np.maximum(s * R - STRIKE, 0.0)))
        for s in SPOTS
    }
    out = {
        "terminal_paths": terminal,
        "empirical_mean": [float(m) for m in np.mean(terminal, axis=0)],
        "empirical_variance": [float(v) for v in np.var(terminal, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=2000000, dt=0.002, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("Prices:", r["functionals"])
    r2 = solve_sde(num_paths=500000, dt=0.002, T=1.0, seed=42)
    print("Prices (5e5):", r2["functionals"])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dWt)["dt"])
    po = solve_sde(1000, 0.01, 1.0, observables={"v": lambda a: a[:, 1]})
    print("obs hook:", po["path_integrals"]["v"].shape)
