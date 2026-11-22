"""Kubo oscillator, Stratonovich form, stochastic implicit midpoint rule (Cayley map).

    dX = a J X dt + sigma J X o dW,   J = [[0, -1], [1, 0]]

The midpoint rule X_{n+1} = X_n + J (X_n + X_{n+1})/2 * (a dt + sigma dW_n) is linear in
X_{n+1} and solves in closed form as the Cayley transform, a rotation by 2*arctan(th)
with th = (a dt + sigma dW_n)/2. It conserves |X| to round-off.
"""
import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        dW = np.asarray(dW, dtype=float)
        if dW.ndim == 3:            # (num_paths, Nt, 1): single channel
            dW = dW[:, :, 0]
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.empty((num_paths, 2), dtype=float)
    X[:, 0] = X_0
    X[:, 1] = Y_0

    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        th = 0.5 * (A * dt + SIGMA * dW[:, n])
        den = 1.0 + th * th
        c = (1.0 - th * th) / den
        s = 2.0 * th / den
        X = np.empty_like(X_prev)
        X[:, 0] = c * X_prev[:, 0] - s * X_prev[:, 1]
        X[:, 1] = s * X_prev[:, 0] + c * X_prev[:, 1]
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=2000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:     ", res["empirical_mean"])
    print("Empirical variance: ", res["empirical_variance"])
    print("max ||X|-1|:", float(np.max(np.abs(np.linalg.norm(res["terminal_paths"], axis=1) - 1))))
    dWt = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((2000, 25))
    r2 = solve_sde(2000, 0.04, 1.0, dW=dWt)
    print("dW hook: dt =", r2["dt"], "mean =", r2["empirical_mean"], "var =", r2["empirical_variance"])
    r3 = solve_sde(2000, 0.04, 1.0, observables={"x": lambda X: X[:, 0]})
    print("obs hook:", r3["path_integrals"]["x"].shape)
    # path-count scan: relative SE of the sample variance (delta method, kurtosis form) and the
    # CI width ci_mult * SE_rel against the 0.05 resolution bar (ci_mult = 2)
    for M in (2000, 4000, 4200, 8000, 16000, 64000):
        r = solve_sde(M, 0.04, 1.0, seed=42)
        Z = r["terminal_paths"]
        w = []
        for j in range(2):
            z = Z[:, j] - Z[:, j].mean()
            m2, m4 = np.mean(z ** 2), np.mean(z ** 4)
            w.append(2.0 * np.sqrt((m4 / m2 ** 2 - (M - 3) / (M - 1)) / M))
        print(f"M={M:6d}: var CI rel width X={w[0]:.4f} Y={w[1]:.4f} (bar 0.05) "
              f"resolved={max(w) < 0.05}")
    # self-check vs exact pathwise solution on the same increments (diagnostic only)
    for Nt in (25, 50, 100):
        fine = np.sqrt(0.01) * np.random.default_rng(1).standard_normal((20000, 100))
        d = fine.reshape(20000, Nt, -1).sum(axis=2)
        r = solve_sde(20000, 1.0 / Nt, 1.0, dW=d)
        th = A * 1.0 + SIGMA * fine.sum(axis=1)
        ex = np.stack([np.cos(th), np.sin(th)], axis=1)
        err = np.sqrt(np.mean((r["terminal_paths"] - ex) ** 2)) / np.sqrt(np.mean(ex ** 2))
        print(f"Nt={Nt}: rel strong err {err:.4f}")
