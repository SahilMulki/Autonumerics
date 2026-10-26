import numpy as np

X_0 = 0.07
Y_0 = 0.34
NU = 0.8
RHO = -0.6
K = {"K1": 0.051022538988, "K2": 0.07, "K3": 0.096035989138}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """SABR (beta=1/2, zero drift). State (S, sigma). EM on S with permanent
    absorption at 0; exact log-step on sigma. Prices via put-call parity."""
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        sq = np.sqrt(dt)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None

    c = np.sqrt(1.0 - RHO ** 2)
    S = np.full(num_paths, X_0, dtype=float)
    sig = np.full(num_paths, Y_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        if dW is None:
            inc = sq * rng.standard_normal((num_paths, 2))   # one generator, per step
            dW0, dW1 = inc[:, 0], inc[:, 1]
        else:
            dW0, dW1 = dW[:, n, 0], dW[:, n, 1]
        dWs = RHO * dW0 + c * dW1

        if obs:
            prev = np.column_stack((S, sig))
        # both updates use the state at the start of the step
        S_trial = S + sig * np.sqrt(np.maximum(S, 0.0)) * dW0
        alive = (S > 0.0) & (S_trial > 0.0)       # absorbed once: stays 0 (no reflection)
        S_new = np.where(alive, S_trial, 0.0)
        sig = sig * np.exp(NU * dWs - 0.5 * NU ** 2 * dt)
        S = S_new

        if obs:
            cur = np.column_stack((S, sig))
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(prev) + phi(cur)) * dt

    X = np.column_stack((S, sig))
    functionals = {}
    raw = {}
    for name, strike in K.items():
        put = float(np.mean(np.maximum(strike - S, 0.0)))
        functionals["price_" + name] = float((X_0 - strike) + put)
        raw["raw_price_" + name] = float(np.mean(np.maximum(S - strike, 0.0)))
    functionals.update(raw)

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = solve_sde(num_paths=1000000, dt=0.004, T=10.0, seed=42)
    print("time", round(time.time() - t0, 1))
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("functionals:", r["functionals"])
    print("absorbed frac:", float(np.mean(r["terminal_paths"][:, 0] == 0.0)))
    r2 = solve_sde(num_paths=250000, dt=0.004, T=10.0, seed=42)
    print("250k functionals:", {k: v for k, v in r2["functionals"].items() if k.startswith("price")})
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 1.0,
          observables={"s": lambda a: a[:, 0]})["path_integrals"]["s"].shape)
