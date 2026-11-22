import math

import numpy as np

X_0 = 100.0
R = 0.05
SIGMA = 0.15
LAM = 0.1
M_J = -0.90
DELTA_J = 0.45
STRIKE = 100.0
KAPPA = math.exp(M_J + DELTA_J**2 / 2) - 1.0   # E[Y] - 1 (negative)
A = R - LAM * KAPPA                            # compensated drift coefficient


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)

    # Jump stream first and dt-independent: counts, then times, then sizes.
    counts = rng.poisson(LAM * T, size=num_paths)
    total = int(counts.sum())
    jump_path = np.repeat(np.arange(num_paths), counts)
    jump_tau = rng.uniform(0.0, T, size=total)
    jump_Y = np.exp(M_J + DELTA_J * rng.standard_normal(total))

    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    # Bin each jump into the step it falls in; sort so each step is a slice.
    jump_step = np.minimum((jump_tau / dt).astype(np.int64), Nt - 1)
    order = np.argsort(jump_step, kind="stable")
    jump_path, jump_step, jump_Y = jump_path[order], jump_step[order], jump_Y[order]
    bounds = np.searchsorted(jump_step, np.arange(Nt + 1))

    obs = observables or {}
    X = np.full(num_paths, X_0, dtype=float)
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        X = X_prev + A * X_prev * dt + SIGMA * X_prev * dW[:, n]
        lo, hi = bounds[n], bounds[n + 1]
        if hi > lo:
            np.multiply.at(X, jump_path[lo:hi], jump_Y[lo:hi])
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    payoff = math.exp(-R * T) * np.maximum(X - STRIKE, 0.0)
    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"call_price": float(np.mean(payoff))},
        "functional_se": {"call_price": float(np.std(payoff, ddof=1) / np.sqrt(num_paths))},
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (400000, 100000):
        res = solve_sde(num_paths=M, dt=0.01, T=0.25, seed=42)
        print(f"M={M} mean={res['empirical_mean']:.6f} var={res['empirical_variance']:.6f} "
              f"call={res['functionals']['call_price']:.6f} se={res['functional_se']['call_price']:.6f} "
              f"min={res['terminal_paths'].min():.3f} "
              f"martingale={math.exp(-R*0.25)*res['empirical_mean']:.4f}")
    dW = np.sqrt(0.0125) * np.random.default_rng(0).standard_normal((1000, 20))
    print("dW hook dt:", solve_sde(1000, 0.01, 0.25, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.01, 0.25,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
