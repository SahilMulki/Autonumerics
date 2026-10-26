import numpy as np

X_0, Y_0 = 0.07, 0.34
NU, RHO = 0.8, -0.6
K1, K2, K3 = 0.051022538988, 0.07, 0.096035989138
PSI_C = 1.5


def _norm_cdf(x):
    """Standard normal CDF via Abramowitz-Stegun 7.1.26 erf (|err| < 1.5e-7), numpy only."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erf = 1.0 - poly * np.exp(-z * z)
    return 0.5 * (1.0 + np.sign(x) * erf)


def _qe_step(S, sig, Z_S, dt):
    """QE moment-matched step for S (absorbing at 0), using start-of-step (S, sigma)."""
    S_new = np.zeros_like(S)
    idx = np.flatnonzero(S > 0.0)
    if idx.size == 0:
        return S_new
    m = S[idx]
    psi = sig[idx] ** 2 * dt / m          # = s2 / m^2
    z = Z_S[idx]
    out = np.zeros(idx.size)

    q = psi <= PSI_C
    if q.any():
        pq = psi[q]
        r = 2.0 / pq
        b2 = r - 1.0 + np.sqrt(r) * np.sqrt(r - 1.0)
        a = m[q] / (1.0 + b2)
        out[q] = a * (np.sqrt(b2) + z[q]) ** 2
    e = ~q
    if e.any():
        pe = psi[e]
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m[e]
        U = _norm_cdf(z[e])
        val = np.log((1.0 - p) / np.maximum(1.0 - U, 1e-300)) / beta
        out[e] = np.where(U <= p, 0.0, val)
    S_new[idx] = out
    return S_new


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = None
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    sq = np.sqrt(dt)
    c = np.sqrt(1.0 - RHO ** 2)

    S = np.full(num_paths, X_0, dtype=float)
    sig = np.full(num_paths, Y_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        if dW is None:
            Z = rng.standard_normal((num_paths, 2))   # one generator, per-step draw
        else:
            Z = dW[:, n, :] / sq
        Z0, Z1 = Z[:, 0], Z[:, 1]
        Z_sig = RHO * Z0 + c * Z1

        if obs:
            X_prev = np.column_stack((S, sig))
        S_new = _qe_step(S, sig, Z0, dt)
        sig_new = sig * np.exp(NU * sq * Z_sig - 0.5 * NU ** 2 * dt)
        S, sig = S_new, sig_new
        if obs:
            X_new = np.column_stack((S, sig))
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X_new)) * dt

    X = np.column_stack((S, sig))
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    funcs, raw, se = {}, {}, {}
    for i, K in enumerate((K1, K2, K3), start=1):
        put = np.maximum(K - S, 0.0)
        funcs[f"price_K{i}"] = float((X_0 - K) + np.mean(put))
        raw[f"raw_price_K{i}"] = float(np.mean(np.maximum(S - K, 0.0)))
        se[f"se_K{i}"] = float(np.std(put, ddof=1) / np.sqrt(num_paths))
    out["functionals"] = funcs
    out.update(raw)
    out.update(se)
    out["absorbed_fraction"] = float(np.mean(S == 0.0))
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import sys, time
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    t0 = time.time()
    r = solve_sde(num_paths=M, dt=0.004, T=10.0, seed=42)
    print(f"[{M} paths, dt=0.004] time {time.time()-t0:.1f}s")
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("functionals:", r["functionals"])
    print("raw:", {k: r[k] for k in r if k.startswith("raw_")})
    print("se:", {k: r[k] for k in r if k.startswith("se_")})
    print("absorbed:", r["absorbed_fraction"], "S>=0:", bool((r["terminal_paths"][:, 0] >= 0).all()),
          "sigma>0:", bool((r["terminal_paths"][:, 1] > 0).all()))
    # hooks
    Nt = 100
    dWt = np.sqrt(10.0 / Nt) * np.random.default_rng(0).standard_normal((1000, Nt, 2))
    print("dW hook dt:", solve_sde(1000, 0.004, 10.0, dW=dWt)["dt"])
    o = solve_sde(1000, 0.05, 10.0, observables={"S": lambda a: a[:, 0]})
    print("obs hook:", o["path_integrals"]["S"].shape)
