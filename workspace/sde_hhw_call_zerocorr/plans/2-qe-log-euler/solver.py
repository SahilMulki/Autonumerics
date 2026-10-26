"""Heston-Hull-White discounted call: Andersen QE for V, central log-S step, exact OU for R."""
import math
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import numpy as np

KAPPA, ETA, SIGMA_V = 0.5, 0.04, 1.0
KAPPA_R, THETA_R, SIGMA_R = 0.08, 0.10, 0.09
RHO = -0.9
V0, R0, STRIKE = 0.04, 0.10, 100.0
SPOTS = {"price_S75": 75.0, "price_S100": 100.0, "price_S125": 125.0}
P0T = 0.7968400988  # E[D_T], Hull-White closed form (control-variate mean only)
PSI_C = 1.5
CHUNK = 20000


def _erfc(x):
    """Vectorised erfc, fractional error < 1.2e-7 everywhere (Numerical Recipes erfcc)."""
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    ans = t * np.exp(-z * z + poly)
    return np.where(x >= 0.0, ans, 2.0 - ans)


def _simulate_chunk(n, Nt, dt, get_z, observables, mart=True):
    """Return (logS, V, R, I, acc) at T for n paths; logS starts at 0 (S0 = 100 reference).

    get_z(i) returns the standardised channels as an array of shape (3, n) or (n, 3)
    columns/rows (z1, z2, z3), i.e. increments divided by sqrt(dt).
    Each QE branch is evaluated only on the paths that use it.  With mart=True the
    log-S constant K0 is Andersen's martingale-corrected one (per step, per path).
    """
    s = math.sqrt(1.0 - RHO ** 2)
    E = math.exp(-KAPPA * dt)
    Er = math.exp(-KAPPA_R * dt)
    sdr = SIGMA_R * math.sqrt((1.0 - Er ** 2) / (2.0 * KAPPA_R))
    c_a = SIGMA_V ** 2 * E * (1.0 - E) / KAPPA
    c_b = ETA * SIGMA_V ** 2 * (1.0 - E) ** 2 / (2.0 * KAPPA)
    g = 0.5 * dt * (KAPPA * RHO / SIGMA_V - 0.5)
    K0c = -RHO * KAPPA * ETA * dt / SIGMA_V
    K1 = g - RHO / SIGMA_V
    K2 = g + RHO / SIGMA_V
    K3 = 0.5 * dt * (1.0 - RHO ** 2)
    K4 = K3
    Aco = K2 + 0.5 * K4            # exponent coefficient on V_{n+1} in the martingale condition
    Bco = K1 + 0.5 * K3
    inv_sqrt2 = 1.0 / math.sqrt(2.0)

    logS = np.zeros(n)
    V = np.full(n, V0)
    R = np.full(n, R0)
    I = np.zeros(n)
    acc = {k: np.zeros(n) for k in (observables or {})}
    for i in range(Nt):
        Z = get_z(i)
        z1, z2, z3 = Z[0], Z[1], Z[2]
        zv = RHO * z1 + s * z2
        zperp = s * z1 - RHO * z2
        if observables:
            Xprev = np.empty((n, 3))
            Xprev[:, 0] = STRIKE * np.exp(logS); Xprev[:, 1] = V; Xprev[:, 2] = R
        # --- V: QE ---
        m = ETA + (V - ETA) * E
        psi = (V * c_a + c_b) / (m * m)
        quad = psi <= PSI_C
        qi = np.flatnonzero(quad)
        ei = np.flatnonzero(~quad)
        Vn = np.zeros(n)
        lnM = np.zeros(n) if mart else None
        if qi.size:
            psq = psi[qi]
            r2 = 2.0 / psq - 1.0
            b2 = r2 + np.sqrt(2.0 / psq) * np.sqrt(np.maximum(r2, 0.0))
            a = m[qi] / (1.0 + b2)
            Vn[qi] = a * (np.sqrt(b2) + zv[qi]) ** 2
            if mart:
                d = 1.0 - 2.0 * Aco * a
                lnM[qi] = Aco * b2 * a / d - 0.5 * np.log(d)
        if ei.size:
            pe = (psi[ei] - 1.0) / (psi[ei] + 1.0)
            Q = 0.5 * _erfc(zv[ei] * inv_sqrt2)       # 1 - U, accurate in the tail
            pos = np.flatnonzero(Q < 1.0 - pe)         # U > p; otherwise V_{n+1} = 0 exactly
            if pos.size:
                om = 1.0 - pe[pos]
                me = m[ei[pos]]
                Vn[ei[pos]] = np.log(om / np.maximum(Q[pos], 1e-300)) * me / om
            if mart:
                beta = (1.0 - pe) / m[ei]
                lnM[ei] = np.log(pe + beta * (1.0 - pe) / (beta - Aco))
        # --- R: exact OU, trapezoid integral ---
        Rn = THETA_R + (R - THETA_R) * Er + sdr * z3
        half = 0.5 * (R + Rn) * dt
        # --- log S ---
        # martingale-corrected K0* = -ln M - (K1 + K3/2) V_n  (else the plain central K0)
        K0 = (-lnM - Bco * V) if mart else K0c
        logS = logS + half + K0 + K1 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * zperp
        I = I + half
        V, R = Vn, Rn
        if observables:
            Xn = np.empty((n, 3))
            Xn[:, 0] = STRIKE * np.exp(logS); Xn[:, 1] = V; Xn[:, 2] = R
            for k, phi in observables.items():
                acc[k] += 0.5 * (phi(Xprev) + phi(Xn)) * dt
    return logS, V, R, I, acc


def _cv_estimate(D, lr, S0):
    """Plain and regression control-variate estimates (with s.e.) of the discounted call."""
    Y = D * np.maximum(S0 * np.exp(lr) - STRIKE, 0.0)
    C = np.stack([D * S0 * np.exp(lr) - S0, D - P0T], axis=1)
    n = Y.shape[0]
    plain = float(Y.mean()); se_plain = float(Y.std(ddof=1) / math.sqrt(n))
    Yc = Y - plain; Cc = C - C.mean(axis=0)
    beta = np.linalg.solve(Cc.T @ Cc, Cc.T @ Yc)
    # estimator: mean(Y) - beta . mean(C)   (population mean of C is 0 by construction)
    cv = float(plain - beta @ C.mean(axis=0))
    resid = Yc - Cc @ beta
    se_cv = float(resid.std(ddof=2) / math.sqrt(n))
    return plain, se_plain, cv, se_cv


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
    else:
        Nt = dW.shape[1]
        dt = T / Nt
    sq = math.sqrt(dt)

    bounds = [(lo, min(num_paths, lo + CHUNK)) for lo in range(0, num_paths, CHUNK)]
    # one independent child stream per chunk, all spawned from default_rng(seed):
    # reproducible and independent of the number of worker threads
    rngs = default_rng_children(seed, len(bounds)) if dW is None else [None] * len(bounds)

    lr = np.empty(num_paths); VT = np.empty(num_paths)
    RT = np.empty(num_paths); IT = np.empty(num_paths)
    pint = {k: np.empty(num_paths) for k in (observables or {})}

    def work(j):
        lo, hi = bounds[j]
        n = hi - lo
        if dW is not None:
            def get_z(i):
                return np.asarray(dW[lo:hi, i, :], dtype=float).T / sq
        else:
            rng = rngs[j]
            def get_z(i):
                return rng.standard_normal((3, n))
        l, v, r, ii, acc = _simulate_chunk(n, Nt, dt, get_z, observables)
        lr[lo:hi], VT[lo:hi], RT[lo:hi], IT[lo:hi] = l, v, r, ii
        for k in acc:
            pint[k][lo:hi] = acc[k]

    nw = min(len(bounds), os.cpu_count() or 1)
    if nw > 1 and dW is None and not observables and _can_fork():
        # forked worker processes (numpy ufuncs on small arrays hold the GIL, so threads
        # do not scale); each worker simulates whole chunks from its own child stream
        global _JOB
        _JOB = (bounds, rngs, Nt, dt)
        ctx = mp.get_context("fork")
        with ctx.Pool(nw) as pool:
            for j, (l, v, r, ii) in enumerate(pool.imap(_fork_work, range(len(bounds)))):
                lo, hi = bounds[j]
                lr[lo:hi], VT[lo:hi], RT[lo:hi], IT[lo:hi] = l, v, r, ii
        _JOB = None
    elif nw > 1:
        with ThreadPoolExecutor(max_workers=nw) as ex:
            list(ex.map(work, range(len(bounds))))
    else:
        for j in range(len(bounds)):
            work(j)

    D = np.exp(-IT)
    ST = STRIKE * np.exp(lr)
    X = np.stack([ST, VT, RT], axis=1)

    functionals, diag = {}, {}
    for name, S0 in SPOTS.items():
        plain, se_p, cv, se_c = _cv_estimate(D, lr, S0)
        functionals[name] = cv
        diag[name] = {"plain": plain, "plain_se": se_p, "cv": cv, "cv_se": se_c}
    diag["mean_D"] = float(D.mean()); diag["mean_D_se"] = float(D.std(ddof=1) / math.sqrt(num_paths))
    diag["mean_DS"] = {f"S0={int(s0)}": float((D * s0 * np.exp(lr)).mean()) for s0 in (75, 100, 125)}
    diag["frac_V_zero"] = float(np.mean(VT == 0.0))

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in X.mean(axis=0)],
        "empirical_variance": [float(v) for v in X.var(axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "diagnostics": diag,
    }
    if observables:
        out["path_integrals"] = pint
    return out


_JOB = None


def _can_fork():
    return "fork" in mp.get_all_start_methods()


def _fork_work(j):
    bounds, rngs, Nt, dt = _JOB
    lo, hi = bounds[j]
    n = hi - lo
    rng = rngs[j]
    l, v, r, ii, _ = _simulate_chunk(n, Nt, dt, lambda i: rng.standard_normal((3, n)), None)
    return l, v, r, ii


def default_rng_children(seed, k):
    return np.random.default_rng(seed).spawn(k)


if __name__ == "__main__":
    import sys, time
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 1000000
    t0 = time.time()
    res = solve_sde(num_paths=M, dt=0.004, T=10.0, seed=42)
    print(f"num_paths={M} time={time.time()-t0:.1f}s")
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("functionals:", res["functionals"])
    import json
    print(json.dumps(res["diagnostics"], indent=1))
    # hooks
    dWt = np.sqrt(0.1) * np.random.default_rng(0).standard_normal((1000, 100, 3))
    print("dW hook dt:", solve_sde(1000, 0.004, 10.0, dW=dWt)["dt"])
    o = solve_sde(1000, 0.1, 10.0, observables={"logS": lambda a: np.log(a[:, 0]),
                                                "R": lambda a: a[:, 2]})
    print("obs hook:", {k: v.shape for k, v in o["path_integrals"].items()})
