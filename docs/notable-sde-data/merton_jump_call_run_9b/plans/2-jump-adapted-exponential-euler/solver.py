import numpy as np

X_0, R, SIGMA, LAM, M_J, DELTA_J, K = 100.0, 0.05, 0.15, 0.1, -0.90, 0.45, 100.0
KAPPA = float(np.exp(M_J + DELTA_J**2 / 2) - 1.0)
A = R - LAM * KAPPA  # compensated drift coefficient (> r here)


def _jumps(rng, n_units, T):
    """dt-independent jump stream: counts, then times, then sizes, then bridge draws."""
    counts = rng.poisson(LAM * T, n_units)
    tot = int(counts.sum())
    times = rng.uniform(0.0, T, tot)
    sizes = np.exp(rng.normal(M_J, DELTA_J, tot))
    xi = rng.standard_normal(tot)
    unit = np.repeat(np.arange(n_units), counts)
    return unit, times, sizes, xi


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    anti = dW is None
    if anti:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        half = num_paths // 2
        n_u = half + (num_paths % 2)
        dWu = np.sqrt(dt) * rng.standard_normal((n_u, Nt))
        dW = np.concatenate([dWu, -dWu[:half]], axis=0)
        unit, times, sizes, xi = _jumps(rng, n_u, T)
        # mirror: same jumps for antithetic partner, bridge draw negated
        mir = unit < half
        j_path = np.concatenate([unit, unit[mir] + n_u])
        times = np.concatenate([times, times[mir]])
        sizes = np.concatenate([sizes, sizes[mir]])
        xi = np.concatenate([xi, -xi[mir]])
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]
        j_path, times, sizes, xi = _jumps(rng, num_paths, T)

    # assign jumps to grid steps and sort by (step, path, time)
    pos = times / dt
    step = np.minimum(np.floor(pos).astype(int), Nt - 1)
    u = pos - step
    order = np.lexsort((u, j_path, step))
    j_path, step, u, sizes, xi = j_path[order], step[order], u[order], sizes[order], xi[order]
    bounds = np.searchsorted(step, np.arange(Nt + 1))

    c = A - 0.5 * SIGMA**2
    X = np.full(num_paths, X_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    sq = np.sqrt

    for n in range(Nt):
        X_prev = X
        X = X_prev * np.exp(c * dt + SIGMA * dW[:, n])
        lo, hi = bounds[n], bounds[n + 1]
        if hi > lo:
            jp, ju, js, jx = j_path[lo:hi], u[lo:hi], sizes[lo:hi], xi[lo:hi]
            # rank of each jump within its path (arrays sorted by path then time)
            first = np.r_[True, jp[1:] != jp[:-1]]
            start = np.maximum.accumulate(np.where(first, np.arange(len(jp)), 0))
            rank = np.arange(len(jp)) - start
            paths = jp[first]
            nj = np.bincount(np.cumsum(first) - 1)
            Xw = X_prev[paths].copy()
            Wrem = dW[paths, n].copy()
            cur = np.zeros(len(paths))
            for r in range(int(nj.max())):
                m = nj > r
                idx = np.nonzero(first)[0][m] + r
                uu, rem = ju[idx], 1.0 - cur[m]
                s = (uu - cur[m]) * dt
                L = rem * dt
                w1 = (s / L) * Wrem[m] + sq(np.maximum(s * (L - s) / L, 0.0)) * jx[idx]
                Xw[m] = Xw[m] * np.exp(c * s + SIGMA * w1) * js[idx]
                Wrem[m] -= w1
                cur[m] = uu
            s = (1.0 - cur) * dt
            Xw = Xw * np.exp(c * s + SIGMA * Wrem)
            X[paths] = Xw
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    payoff = np.exp(-R * T) * np.maximum(X - K, 0.0)
    if anti and num_paths > 1:
        half = num_paths // 2
        n_u = num_paths - half
        units = payoff[:n_u].copy()
        units[:half] = 0.5 * (payoff[:half] + payoff[n_u:n_u + half])
        se = float(np.std(units, ddof=1) / np.sqrt(n_u))
    else:
        se = float(np.std(payoff, ddof=1) / np.sqrt(num_paths))
    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": float(np.mean(payoff))},
        "functional_se": {"call_price": se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (400000, 100000):
        res = solve_sde(M, 0.01, 0.25, seed=42)
        print(M, f"mean {res['empirical_mean']:.6f} var {res['empirical_variance']:.6f}",
              f"call {res['functionals']['call_price']:.5f} se {res['functional_se']['call_price']:.5f}",
              f"martingale {np.exp(-R*0.25)*res['empirical_mean']:.4f}", f"minX {res['terminal_paths'].min():.3g}")
    print("exact mean 101.257845 var 147.946718")
    dW = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook:", solve_sde(1000, 0.01, 0.25, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 0.25, observables={"x": lambda a: a})["path_integrals"]["x"].shape)
    dW2 = np.sqrt(0.0025) * np.random.default_rng(1).standard_normal((200000, 100))
    dW1 = (dW2[:, 0::4] + dW2[:, 1::4] + dW2[:, 2::4] + dW2[:, 3::4])
    a, b = solve_sde(200000, 0, 0.25, dW=dW1), solve_sde(200000, 0, 0.25, dW=dW2)
    print("CRN level diff rms:", np.sqrt(np.mean((a["terminal_paths"] - b["terminal_paths"])**2)))
