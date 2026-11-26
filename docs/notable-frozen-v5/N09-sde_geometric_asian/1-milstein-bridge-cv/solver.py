"""Milstein on S (GBM) + trapezoid/Brownian-bridge quadrature of int ln S dt
+ control variate on the discounted terminal asset, for the geometric-average Asian call.

    dS = r S dt + sigma S dW,   S(0) = X_0
    A  = exp( (1/T) int_0^T ln S dt )
    call_price           = e^{-rT} E[(A - K)^+]
    geo_average_variance = Var[A]  (ddof=1)
"""
import numpy as np

X_0 = 100.0
R = 0.05
SIGMA = 0.2
K = 100.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    bridge = dW is None
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))   # one block, path-major
        # fresh randomness for the Brownian-bridge term, drawn AFTER the dW block
        xi = rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        xi = None   # evaluator-supplied increments: no randomness of our own

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    S = np.full(num_paths, X_0, dtype=float)
    logS = np.log(S)
    I = np.zeros(num_paths)            # int_0^T ln S dt
    bridge_sd = SIGMA * np.sqrt(dt ** 3 / 12.0)

    for n in range(Nt):
        S_prev = S
        dWn = dW[:, n]
        # Milstein step; the correction uses the pre-step value S_n
        S = S_prev + R * S_prev * dt + SIGMA * S_prev * dWn \
            + 0.5 * SIGMA ** 2 * S_prev * (dWn ** 2 - dt)
        logS_new = np.log(S)
        I += 0.5 * (logS + logS_new) * dt
        if bridge:
            I += bridge_sd * xi[:, n]
        logS = logS_new
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(S_prev) + phi(S)) * dt

    A = np.exp(I / T)
    disc = np.exp(-R * T)

    # --- call price with control variate Y = e^{-rT} S_N ---
    P = disc * np.maximum(A - K, 0.0)
    Y = disc * S
    EY = disc * X_0 * (1.0 + R * dt) ** Nt          # exact mean of Y under this scheme
    Pc = P - P.mean()
    Yc = Y - Y.mean()
    beta = float(np.dot(Pc, Yc) / np.dot(Yc, Yc))
    call = float(P.mean() - beta * (Y.mean() - EY))
    call_se = float(np.std(P - beta * Y, ddof=1) / np.sqrt(num_paths))

    # --- Var[A], with delta-method standard error of the sample variance ---
    var_A = float(np.var(A, ddof=1))
    Ac = A - A.mean()
    m4 = float(np.mean(Ac ** 4))
    var_A_se = float(np.sqrt(max(m4 - np.mean(Ac ** 2) ** 2, 0.0) / num_paths))

    out = {
        "terminal_paths": S,
        "empirical_mean": float(np.mean(S)),
        "empirical_variance": float(np.var(S, ddof=1)),
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
        "functionals": {"call_price": call, "geo_average_variance": var_A},
        "functional_se": {"call_price": call_se, "geo_average_variance": var_A_se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    # exact references (problem_spec.json -> functional_reference)
    import math
    mu_A = math.log(X_0) + (R - 0.5 * SIGMA ** 2) * 1.0 / 2
    s2_A = SIGMA ** 2 / 3
    d2 = (mu_A - math.log(K)) / math.sqrt(s2_A)
    d1 = d2 + math.sqrt(s2_A)
    Phi = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    ref_call = math.exp(-R) * (math.exp(mu_A + s2_A / 2) * Phi(d1) - K * Phi(d2))
    ref_var = math.exp(2 * mu_A + s2_A) * (math.exp(s2_A) - 1)

    for M in (400000, 100000):
        res = solve_sde(num_paths=M, dt=0.04, T=1.0, seed=42)
        f, se = res["functionals"], res["functional_se"]
        print(f"[M={M}] Empirical mean:     {res['empirical_mean']:.6f}")
        print(f"[M={M}] Empirical variance: {res['empirical_variance']:.6f}")
        print(f"[M={M}] call_price = {f['call_price']:.5f} +/- {se['call_price']:.5f} "
              f"(ref {ref_call:.5f}, rel err {abs(f['call_price']/ref_call-1):.2e})")
        print(f"[M={M}] geo_average_variance = {f['geo_average_variance']:.4f} +/- "
              f"{se['geo_average_variance']:.4f} (ref {ref_var:.4f}, "
              f"rel err {abs(f['geo_average_variance']/ref_var-1):.2e})")

    # confirm both hooks work
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    po = solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a})["path_integrals"]["x"]
    print("obs hook shape:", po.shape)
