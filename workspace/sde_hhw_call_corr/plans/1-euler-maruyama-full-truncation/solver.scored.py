"""Heston-Hull-White (S, V, R) -- Euler-Maruyama with full truncation for V, log-Euler for S,
Euler for R, left-point discount sharing R_n with the S drift, control variates exact for the
discrete scheme.  numpy + stdlib only.  No work at import."""
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

# Module-level parameters (a dict so the Black-Scholes degenerate limit can be run by editing it).
PARAMS = dict(kappa=0.5, eta=0.04, sigma_v=1.0, alpha=0.08, r_bar=0.10, sigma_r=0.09,
              rho_SV=-0.9, rho_SR=0.6, rho_VR=-0.7, K=100.0,
              S_0=100.0, V_0=0.04, R_0=0.10)
SPOTS = {"price_S75": 75.0, "price_S100": 100.0, "price_S125": 125.0}
CHUNK = 50_000


def _cholesky(p):
    rsv, rsr, rvr = p["rho_SV"], p["rho_SR"], p["rho_VR"]
    l22 = np.sqrt(1.0 - rsv ** 2)
    l32 = (rvr - rsv * rsr) / l22
    l33 = np.sqrt(1.0 - rsr ** 2 - l32 ** 2)
    return rsv, l22, rsr, l32, l33


def _discrete_mean_D(p, Nt, dt):
    """Exact E[D_N] for logD = -dt*sum_{n<N} R_n with R the Euler AR(1) recursion."""
    a = 1.0 - p["alpha"] * dt
    c = p["alpha"] * p["r_bar"] * dt
    m = p["R_0"]
    tot = 0.0
    for _ in range(Nt):
        tot += m
        m = a * m + c
    j = np.arange(Nt)
    if abs(1.0 - a) < 1e-300:
        coef = (Nt - 1 - j).astype(float)
    else:
        coef = (1.0 - a ** (Nt - 1 - j)) / (1.0 - a)
    var_sum = p["sigma_r"] ** 2 * dt * np.sum(coef ** 2)
    return float(np.exp(-dt * tot + 0.5 * dt * dt * var_sum))


def _run_chunk(p, M, Nt, dt, dW_chunk, rng, observables):
    kappa, eta, sv, alpha, rbar, sr = (p[k] for k in ("kappa", "eta", "sigma_v", "alpha", "r_bar", "sigma_r"))
    c21, c22, c31, c32, c33 = _cholesky(p)
    s0 = p["S_0"]
    sqdt = np.sqrt(dt)

    logS = np.zeros(M)           # log(S/S_0)
    V = np.full(M, p["V_0"])
    R = np.full(M, p["R_0"])
    logD = np.zeros(M)
    Vp = np.empty(M)
    sq = np.empty(M)
    t1 = np.empty(M)
    t2 = np.empty(M)
    w1 = np.empty(M)
    w2 = np.empty(M)
    w3 = np.empty(M)
    Z = np.empty((3, M))
    acc = {k: np.zeros(M) for k in (observables or {})}

    def state():
        return np.column_stack((s0 * np.exp(logS), V, R))

    X_prev = state() if observables else None

    for n in range(Nt):
        if dW_chunk is not None:
            Z[:] = dW_chunk[:, n, :].T          # already scaled by sqrt(dt)
        else:
            rng.standard_normal(out=Z)
            Z *= sqdt
        # correlate with the Cholesky factor (fused multiply-adds, no matmul)
        np.copyto(w1, Z[0])
        np.multiply(Z[0], c21, out=w2); np.multiply(Z[1], c22, out=t1); w2 += t1
        np.multiply(Z[0], c31, out=w3); np.multiply(Z[1], c32, out=t1); w3 += t1
        np.multiply(Z[2], c33, out=t1); w3 += t1

        np.maximum(V, 0.0, out=Vp)               # full truncation
        np.sqrt(Vp, out=sq)
        # log S:  (R_n - V+/2) dt + sqrt(V+) dW1
        np.multiply(Vp, -0.5, out=t1); t1 += R; t1 *= dt
        np.multiply(sq, w1, out=t2); t1 += t2
        logS += t1
        # V:  kappa (eta - V+) dt + sigma_v sqrt(V+) dW2
        np.subtract(eta, Vp, out=t1); t1 *= kappa * dt
        sq *= w2; sq *= sv; t1 += sq
        V += t1
        # discount: left-point R_n, same as in the S drift
        np.multiply(R, dt, out=t1); logD -= t1
        # R:  alpha (rbar - R) dt + sigma_r dW3
        np.subtract(rbar, R, out=t1); t1 *= alpha * dt
        w3 *= sr; t1 += w3
        R += t1
        if observables:
            X = state()
            for k, phi in observables.items():
                acc[k] += 0.5 * (np.asarray(phi(X_prev)) + np.asarray(phi(X))) * dt
            X_prev = X

    return logS, V, R, logD, acc


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    p = PARAMS
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
    else:
        dW = np.asarray(dW)
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]
    M = int(num_paths)
    K = p["K"]
    s0 = p["S_0"]

    bounds = [(lo, min(lo + CHUNK, M)) for lo in range(0, M, CHUNK)]
    seeds = np.random.SeedSequence(seed).spawn(len(bounds)) if dW is None else [None] * len(bounds)

    def work(i):
        lo, hi = bounds[i]
        rng = np.random.default_rng(seeds[i]) if dW is None else None
        chunk = dW[lo:hi] if dW is not None else None
        return _run_chunk(p, hi - lo, Nt, dt, chunk, rng, observables)

    nthreads = max(1, min(os.cpu_count() or 1, len(bounds)))
    if nthreads == 1:
        results = [work(i) for i in range(len(bounds))]
    else:
        with ThreadPoolExecutor(nthreads) as ex:
            results = list(ex.map(work, range(len(bounds))))

    logS = np.concatenate([r[0] for r in results])
    V = np.concatenate([r[1] for r in results])
    R = np.concatenate([r[2] for r in results])
    logD = np.concatenate([r[3] for r in results])
    S = s0 * np.exp(logS)             # exponentiate once, at T
    X = np.column_stack((S, V, R))

    # ---- discounted call prices, one simulation for all spots (S_T(S0) = S0/s0 * S_T(s0)) ----
    D = np.exp(logD)
    DS = D * S
    ED_disc = _discrete_mean_D(p, Nt, dt)
    functionals = {}
    for name, S0 in SPOTS.items():
        c = S0 / s0
        y = D * np.maximum(c * S - K, 0.0)
        # controls with means exact for the discrete scheme: E[D S_T] = s0 (so E[D c S_T] = S0), E[D]
        C = np.column_stack((c * DS - S0, D - ED_disc))
        if M > 10:
            Cc = C - C.mean(axis=0)
            b, *_ = np.linalg.lstsq(Cc, y - y.mean(), rcond=None)
            est = y.mean() - C.mean(axis=0) @ b
        else:
            est = y.mean()
        functionals[name] = float(est)

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": M,
        "dt": dt,
        "T": T,
        "functionals": functionals,
    }
    if observables:
        out["path_integrals"] = {k: np.concatenate([r[4][k] for r in results]) for k in observables}
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve_sde(num_paths=1_000_000, dt=0.004, T=10.0, seed=42)
    print(f"time {time.time() - t0:.1f}s")
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("functionals (1e6):", res["functionals"])
    res2 = solve_sde(num_paths=250_000, dt=0.004, T=10.0, seed=42)
    print("functionals (250k):", res2["functionals"])
    dWt = np.sqrt(0.1) * np.random.default_rng(0).standard_normal((1000, 100, 3))
    r3 = solve_sde(1000, 0.1, 10.0, dW=dWt)
    print("dW hook: dt =", r3["dt"], "mean", r3["empirical_mean"])
    r4 = solve_sde(2000, 0.1, 10.0, observables={"v": lambda a: a[:, 1]})
    print("obs hook:", r4["path_integrals"]["v"].shape)
