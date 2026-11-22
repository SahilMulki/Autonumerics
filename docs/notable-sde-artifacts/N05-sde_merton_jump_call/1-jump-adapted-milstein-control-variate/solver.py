import math
import numpy as np

X_0, R, SIGMA, LAM = 100.0, 0.05, 0.15, 0.1
M_J, DELTA_J, STRIKE = -0.90, 0.45, 100.0
KAPPA = math.exp(M_J + DELTA_J ** 2 / 2) - 1.0
A = R - LAM * KAPPA          # compensated drift coefficient (> r here)


def _phi(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _cv_price(T):
    M = X_0 * math.exp(A * T)
    v = SIGMA ** 2 * T
    d2 = (math.log(M / STRIKE) - 0.5 * v) / math.sqrt(v)
    d1 = d2 + math.sqrt(v)
    return math.exp(-R * T) * (M * _phi(d1) - STRIKE * _phi(d2))


def _mil(x, h, dw):
    """Milstein step for the continuous (GBM) leg; correction uses the start value x."""
    return x + A * x * h + SIGMA * x * dw + 0.5 * SIGMA ** 2 * x * (dw ** 2 - h)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    # Jump stream first, and dt-independent: counts, times, sizes, bridge normals.
    counts = rng.poisson(LAM * T, size=num_paths)
    nj = int(counts.sum())
    jpath = np.repeat(np.arange(num_paths), counts)
    jtime = rng.uniform(0.0, T, size=nj)
    jsize = rng.normal(M_J, DELTA_J, size=nj)
    jxi = rng.standard_normal(nj)
    order = np.lexsort((jtime, jpath))
    jpath, jtime, jsize, jxi = jpath[order], jtime[order], jsize[order], jxi[order]
    jY = np.exp(jsize)

    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    jstep = np.minimum((jtime / dt).astype(np.int64), Nt - 1)
    # rank of each jump within its path
    if nj:
        first = np.r_[True, jpath[1:] != jpath[:-1]]
        start_idx = np.maximum.accumulate(np.where(first, np.arange(nj), 0))
        jrank = np.arange(nj) - start_idx
    else:
        jrank = np.zeros(0, dtype=np.int64)
    by_step = np.argsort(jstep, kind="stable")
    bounds = np.searchsorted(jstep[by_step], np.arange(Nt + 1))

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    X = np.full(num_paths, X_0, dtype=float)
    S = np.full(num_paths, X_0, dtype=float)

    for n in range(Nt):
        X_prev = X
        w = dW[:, n]
        S = _mil(S, dt, w)
        Xn = _mil(X_prev, dt, w)
        sel = by_step[bounds[n]:bounds[n + 1]]
        if sel.size:
            sel = sel[np.lexsort((jrank[sel], jpath[sel]))]
            paths = jpath[sel]
            ranks = jrank[sel]
            cur = X_prev[paths].copy()
            # per-path: remaining length L and remaining increment
            upaths, inv = np.unique(paths, return_inverse=True)
            xs = X_prev[upaths].copy()
            rem_w = w[upaths].copy()
            elapsed = np.zeros(upaths.size)          # time consumed within the step
            for rk in range(int(ranks.max()) + 1):
                m = ranks == rk
                jj = sel[m]
                ui = inv[m]
                L = dt - elapsed[ui]
                u = (jtime[jj] - (n * dt + elapsed[ui])) / L
                u = np.clip(u, 0.0, 1.0)
                dw1 = u * rem_w[ui] + np.sqrt(u * (1 - u) * L) * jxi[jj]
                xs[ui] = _mil(xs[ui], u * L, dw1) * jY[jj]
                rem_w[ui] -= dw1
                elapsed[ui] += u * L
            L = dt - elapsed
            xs = _mil(xs, L, rem_w)
            Xn = Xn.copy()
            Xn[upaths] = xs
        X = Xn
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    disc = math.exp(-R * T)
    pay_x = disc * np.maximum(X - STRIKE, 0.0)
    pay_s = disc * np.maximum(S - STRIKE, 0.0)
    beta = 1.0
    comb = pay_x - beta * pay_s
    price = float(np.mean(comb) + beta * _cv_price(T))
    se = float(np.std(comb, ddof=1) / math.sqrt(num_paths))

    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": price},
        "functional_se": {"call_price": se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=400000, dt=0.01, T=0.25, seed=42)
    print(f"Empirical mean:     {res['empirical_mean']:.6f}")
    print(f"Empirical variance: {res['empirical_variance']:.6f}")
    print(f"call_price: {res['functionals']['call_price']:.6f} +- {res['functional_se']['call_price']:.2e}")
    print(f"disc mean: {math.exp(-R*0.25)*res['empirical_mean']:.4f} (expect ~100); min X {res['terminal_paths'].min():.3f}")
    raw = math.exp(-R*0.25)*np.maximum(res['terminal_paths']-STRIKE,0)
    print(f"raw price {raw.mean():.4f} se {raw.std(ddof=1)/math.sqrt(400000):.2e}")
    r2 = solve_sde(num_paths=100000, dt=0.01, T=0.25, seed=42)
    print("100k price:", r2["functionals"]["call_price"])
    dW = np.sqrt(0.0025) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook dt:", solve_sde(1000, 0.01, 0.25, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 0.25, observables={"x": lambda a: a})["path_integrals"]["x"].shape)
    # CRN consistency: coarse vs fine on shared Brownian path
    rg = np.random.default_rng(1); fine = np.sqrt(0.25/100)*rg.standard_normal((20000, 100))
    ref = solve_sde(20000, 0, 0.25, dW=fine)["terminal_paths"]
    for g in (25, 50):
        cw = fine.reshape(20000, g, 100 // g).sum(2)
        print("strong err Nt=%d:" % g, np.sqrt(np.mean((solve_sde(20000, 0, 0.25, dW=cw)["terminal_paths"] - ref) ** 2)))
