import numpy as np

S_0, R, SIGMA, K = 100.0, 0.05, 0.2, 100.0


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

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    S = np.full(num_paths, S_0, dtype=float)
    log_int = np.zeros(num_paths)           # int ln S ds by trapezoid
    prev_log = np.log(S)
    for n in range(Nt):
        S_prev = S
        # plain Euler-Maruyama (no Milstein correction)
        S = S_prev + R * S_prev * dt + SIGMA * S_prev * dW[:, n]
        # positivity is surfaced, not clamped: a non-positive S gives NaN log
        # and poisons the path, which the kernel's finite/positivity gates catch.
        with np.errstate(invalid="ignore", divide="ignore"):
            cur_log = np.log(S)
        log_int += 0.5 * dt * (prev_log + cur_log)
        prev_log = cur_log
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(S_prev) + phi(S)) * dt

    A = np.exp(log_int / T)
    disc = np.exp(-R * T)
    payoff = disc * np.maximum(A - K, 0.0)
    M = num_paths
    call = float(np.mean(payoff))
    call_se = float(np.std(payoff, ddof=1) / np.sqrt(M))
    gvar = float(np.var(A, ddof=1))
    # delta-method s.e. of a sample variance: sqrt((m4 - s^4)/M)
    m4 = float(np.mean((A - np.mean(A)) ** 4))
    gvar_se = float(np.sqrt(max(m4 - gvar ** 2, 0.0) / M))

    out = {
        "terminal_paths": S,
        "empirical_mean": float(np.mean(S)),
        "empirical_variance": float(np.var(S, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": call, "geo_average_variance": gvar},
        "functional_se": {"call_price": call_se, "geo_average_variance": gvar_se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=400000, dt=0.04, T=1.0, seed=42)
    print(f"Empirical mean:     {res['empirical_mean']:.6f}")
    print(f"Empirical variance: {res['empirical_variance']:.6f}")
    print("functionals:", res["functionals"])
    print("functional_se:", res["functional_se"])
    print("positive/finite:", bool(np.all(res["terminal_paths"] > 0)),
          bool(np.all(np.isfinite(res["terminal_paths"]))))
    r2 = solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)
    print("100k functionals:", r2["functionals"])
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
          observables={"x": lambda a: a})["path_integrals"]["x"].shape)
