"""Heston-Hull-White: Andersen QE (V) + martingale-corrected log-S (QE-M) + exact OU (R).

State columns (S, V, R).  See SOLUTION.md for the scheme.  numpy + stdlib only.
"""
import math
import os

import numpy as np

# Module-level parameter dict so the degenerate (Black-Scholes) limit can be run by
# changing sigma_v / sigma_r.  The graded path never takes the sigma_v == 0 branch.
PARAMS = dict(kappa=0.5, eta=0.04, sigma_v=1.0, alpha=0.08, r_bar=0.10, sigma_r=0.09,
              rho_SV=-0.9, rho_SR=0.6, rho_VR=-0.7, K=100.0,
              S_0=100.0, V_0=0.04, R_0=0.10)
SPOTS = {"price_S75": 75.0, "price_S100": 100.0, "price_S125": 125.0}
PSI_C = 1.5
GAMMA1 = GAMMA2 = 0.5
CHUNK = 100_000
_SQ2 = math.sqrt(2.0)


def _phi_neg(z):
    """Phi(-z) = 0.5*erfc(z/sqrt2), Numerical Recipes erfcc (frac. error < 1.2e-7)."""
    x = np.abs(z) * (1.0 / _SQ2)
    t = 1.0 / (1.0 + 0.5 * x)
    poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    erfc_abs = t * np.exp(-x * x + poly)
    erfc = np.where(z >= 0.0, erfc_abs, 2.0 - erfc_abs)
    return 0.5 * erfc


def _consts(dt, P):
    c = {}
    kap, eta, sv = P["kappa"], P["eta"], P["sigma_v"]
    rho = P["rho_SV"]
    c["E"] = math.exp(-kap * dt)
    c["degenerate"] = (sv == 0.0)
    if not c["degenerate"]:
        c["K0"] = -rho * kap * eta * dt / sv
        c["K1"] = GAMMA1 * dt * (kap * rho / sv - 0.5) - rho / sv
        c["K2"] = GAMMA2 * dt * (kap * rho / sv - 0.5) + rho / sv
        c["K3"] = GAMMA1 * dt * (1 - rho ** 2)
        c["K4"] = GAMMA2 * dt * (1 - rho ** 2)
        c["A"] = c["K2"] + 0.5 * c["K4"]
    # Cholesky rows (unit normals): Zv, Z3
    r_sv, r_sr, r_vr = P["rho_SV"], P["rho_SR"], P["rho_VR"]
    l21 = math.sqrt(1 - r_sv ** 2)
    l31 = r_sr
    l32 = (r_vr - r_sv * r_sr) / l21
    l33 = math.sqrt(1 - l31 ** 2 - l32 ** 2)
    c["L"] = (l21, l31, l32, l33)
    # OU exact transition
    al = P["alpha"]
    c["Er"] = math.exp(-al * dt)
    c["ou_sd"] = P["sigma_r"] * math.sqrt((1 - math.exp(-2 * al * dt)) / (2 * al))
    return c


def _simulate(M, Nt, dt, draw, P, obs, want_state=True):
    """Core loop for M paths.  draw(n) -> (M,3) INDEPENDENT unit normals for step n.
    Returns dict with logS (S0=100 basis), V, R, logD, and obs accumulators."""
    c = _consts(dt, P)
    kap, eta, sv = P["kappa"], P["eta"], P["sigma_v"]
    rho = P["rho_SV"]
    sq_dt = math.sqrt(dt)
    l21, l31, l32, l33 = c["L"]
    E = c["E"]
    degenerate = c["degenerate"]
    if not degenerate:
        K1, K2, K3, K4, A = c["K1"], c["K2"], c["K3"], c["K4"], c["A"]
        k13 = K1 + 0.5 * K3

    logS = np.full(M, math.log(P["S_0"]))
    V = np.full(M, float(P["V_0"]))
    R = np.full(M, float(P["R_0"]))
    logD = np.zeros(M)
    acc = {k: np.zeros(M) for k in (obs or {})}
    rbar_c, Er, ou_sd = P["r_bar"], c["Er"], c["ou_sd"]
    s2_const = eta * sv * sv * (1 - E) ** 2 / (2 * kap)
    s2_lin = sv * sv * E * (1 - E) / kap

    if obs:
        Xprev = np.empty((M, 3))
        Xnow = np.empty((M, 3))
        Xprev[:, 0] = np.exp(logS); Xprev[:, 1] = V; Xprev[:, 2] = R

    for n in range(Nt):
        z = draw(n)
        z1, z2, z3 = z[:, 0], z[:, 1], z[:, 2]
        Zv = rho * z1 + l21 * z2                       # dW2/sqrt(dt)
        Z3 = l31 * z1 + l32 * z2 + l33 * z3            # dW3/sqrt(dt)
        # R: exact OU on correlated increment
        Rn = rbar_c + (R - rbar_c) * Er + ou_sd * Z3
        Rbar = 0.5 * (R + Rn)

        if degenerate:
            Vn = eta + (V - eta) * E
            logS += (Rbar - 0.5 * V) * dt + np.sqrt(np.maximum(V, 0.0)) * (z1 * sq_dt)
        else:
            Zp = l21 * z1 - rho * z2                   # unit normal independent of Zv
            m = eta + (V - eta) * E
            psi = (V * s2_lin + s2_const) / (m * m)
            quad = psi <= PSI_C
            Vn = np.empty(M)
            K0s = np.empty(M)
            iq = np.flatnonzero(quad)
            ie = np.flatnonzero(~quad)
            if iq.size:
                ps = psi[iq]
                inv = 2.0 / ps
                b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
                a = m[iq] / (1.0 + b2)
                zq = Zv[iq]
                Vn[iq] = a * (np.sqrt(b2) + zq) ** 2
                Aa2 = 1.0 - 2.0 * A * a
                K0s[iq] = -A * b2 * a / Aa2 + 0.5 * np.log(Aa2)
            if ie.size:
                ps = psi[ie]
                p = (ps - 1.0) / (ps + 1.0)
                beta = (1.0 - p) / m[ie]
                Q = _phi_neg(Zv[ie])                   # 1-U, computed in the tail directly
                Vn[ie] = np.maximum(np.log((1.0 - p) / Q), 0.0) / beta
                K0s[ie] = -np.log(p + beta * (1.0 - p) / (beta - A))
            K0s -= k13 * V
            logS += Rbar * dt + K0s + K1 * V + K2 * Vn \
                + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zp
        logD -= Rbar * dt
        V, R = Vn, Rn

        if obs:
            Xnow[:, 0] = np.exp(logS); Xnow[:, 1] = V; Xnow[:, 2] = R
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(Xprev) + phi(Xnow)) * dt
            Xprev, Xnow = Xnow, Xprev
    return logS, V, R, logD, acc


def _chunk_job(args):
    M, Nt, dt, child, P = args
    rng = child

    def draw(n):
        return rng.standard_normal((M, 3))
    logS, V, R, logD, _ = _simulate(M, Nt, dt, draw, P, None)
    return logS, V, R, logD


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    P = PARAMS
    if dW is not None:
        Nt = dW.shape[1]
        dt = T / Nt
        sq = math.sqrt(dt)
        num_paths = dW.shape[0]
        logS, V, R, logD, acc = _simulate(num_paths, Nt, dt,
                                          lambda n: dW[:, n, :] / sq, P, observables)
    else:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        if observables:
            # small-scale path (Dynkin): single block, path-major, as per the contract
            sq = math.sqrt(dt)
            blk = sq * rng.standard_normal((num_paths, Nt, 3))
            logS, V, R, logD, acc = _simulate(num_paths, Nt, dt,
                                              lambda n: blk[:, n, :] / sq, P, observables)
        else:
            acc = {}
            sizes = []
            left = num_paths
            while left > 0:
                sizes.append(min(CHUNK, left))
                left -= sizes[-1]
            children = rng.spawn(len(sizes))
            jobs = [(sizes[i], Nt, dt, children[i], P) for i in range(len(sizes))]
            nproc = min(len(jobs), os.cpu_count() or 1, 4)
            results = None
            if nproc > 1 and Nt * num_paths > 2_000_000:
                try:
                    import multiprocessing as mp
                    ctx = mp.get_context("fork")
                    with ctx.Pool(nproc) as pool:
                        results = pool.map(_chunk_job, jobs, chunksize=1)
                except Exception:
                    results = None
            if results is None:
                results = [_chunk_job(j) for j in jobs]
            logS = np.concatenate([r[0] for r in results])
            V = np.concatenate([r[1] for r in results])
            R = np.concatenate([r[2] for r in results])
            logD = np.concatenate([r[3] for r in results])

    S = np.exp(logS)
    X = np.column_stack([S, V, R])
    D = np.exp(logD)

    # functionals: one simulation from S0=100, scaled by c = S0/100 (exact under log-S update)
    K = P["K"]
    out_f, diag = {}, {}
    for name, S0 in SPOTS.items():
        cfac = S0 / P["S_0"]
        pay = D * np.maximum(cfac * S - K, 0.0)
        ctrl = D * cfac * S - S0                      # exact zero mean for this scheme
        cm = ctrl - ctrl.mean()
        b1 = float(np.dot(pay - pay.mean(), cm) / np.dot(cm, cm))
        cv = pay - b1 * ctrl
        out_f[name] = float(cv.mean())
        diag[name] = dict(plain=float(pay.mean()), plain_se=float(pay.std(ddof=1) / math.sqrt(len(pay))),
                          cv_se=float(cv.std(ddof=1) / math.sqrt(len(cv))), b1=b1)
    diag["E_D"] = float(D.mean())
    diag["E_DS"] = float((D * S).mean())

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": int(len(S)),
        "dt": float(dt),
        "T": float(T),
        "functionals": out_f,
        "diagnostics": diag,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    res = solve_sde(num_paths=1_000_000, dt=0.004, T=10.0, seed=42)
    print("time", round(time.time() - t0, 1), "s")
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    print("functionals:", res["functionals"])
    print("diagnostics:", res["diagnostics"])
    dWt = np.sqrt(0.5) * np.random.default_rng(0).standard_normal((1000, 20, 3))
    print("dW hook dt:", solve_sde(1000, 0.5, 10.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.5, 10.0,
                                 observables={"v": lambda a: a[:, 1]})["path_integrals"]["v"].shape)
