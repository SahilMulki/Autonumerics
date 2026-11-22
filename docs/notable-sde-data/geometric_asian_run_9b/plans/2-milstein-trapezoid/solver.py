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

    S = np.full(num_paths, S_0, dtype=float)
    logS = np.log(S)
    log_int = np.zeros(num_paths)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        S_prev = S
        w = dW[:, n]
        # Milstein: correction uses S_n (pre-step)
        S = S_prev + R * S_prev * dt + SIGMA * S_prev * w \
            + 0.5 * SIGMA ** 2 * S_prev * (w ** 2 - dt)
        if not (np.all(np.isfinite(S)) and np.all(S > 0.0)):
            raise FloatingPointError("Milstein step produced non-finite or non-positive S")
        logS_new = np.log(S)
        log_int += 0.5 * dt * (logS + logS_new)      # trapezoid on ln S
        logS = logS_new
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(S_prev) + phi(S)) * dt

    A = np.exp(log_int / T)
    payoff = np.exp(-R * T) * np.maximum(A - K, 0.0)
    n = num_paths
    # Control variate: S_T. The Milstein step is mean-exact, so E[S_T] = S_0 (1 + R dt)^Nt
    # exactly under the scheme; no extra discretization bias is introduced.
    Y = S
    EY = S_0 * (1.0 + R * dt) ** Nt
    yc = Y - Y.mean()
    beta = float(np.dot(yc, payoff - payoff.mean()) / np.dot(yc, yc))
    payoff_cv = payoff - beta * (Y - EY)
    var_A = float(np.var(A, ddof=1))
    d = A - A.mean()
    m4 = float(np.mean(d ** 4))
    se_var = float(np.sqrt(max(m4 - var_A ** 2, 0.0) / n))   # delta-method SE of sample variance

    out = {
        "terminal_paths": S,
        "empirical_mean": float(np.mean(S)),
        "empirical_variance": float(np.var(S, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": float(np.mean(payoff_cv)),
                        "geo_average_variance": var_A},
        "functional_se": {"call_price": float(np.std(payoff_cv, ddof=1) / np.sqrt(n)),
                          "geo_average_variance": se_var},
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
    r2 = solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)
    print("100k functionals:", r2["functionals"])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
