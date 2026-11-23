import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Euler-Maruyama on the Ito form of the Kubo oscillator.

    X1' = m*X1 - theta*X2,  X2' = theta*X1 + m*X2,
    m = 1 - 0.5*sigma^2*dt, theta = a*dt + sigma*dW_n.
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    X = np.zeros((num_paths, 2))
    X[:, 0] = X_0
    X[:, 1] = Y_0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    m = 1.0 - 0.5 * SIGMA ** 2 * dt

    for n in range(Nt):
        X_prev = X
        theta = A * dt + SIGMA * dW[:, n]
        X = np.empty_like(X_prev)
        X[:, 0] = m * X_prev[:, 0] - theta * X_prev[:, 1]
        X[:, 1] = theta * X_prev[:, 0] + m * X_prev[:, 1]
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
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    dWt = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((2000, 25))
    r2 = solve_sde(2000, 0.04, 1.0, dW=dWt)
    th = 1.0 + dWt.sum(axis=1)
    err = np.sqrt(np.mean((r2["terminal_paths"] - np.c_[np.cos(th), np.sin(th)]) ** 2)) \
        / np.sqrt(np.mean(np.c_[np.cos(th), np.sin(th)] ** 2))
    print("dW hook dt:", r2["dt"], " rel strong err:", err)
    o = solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a[:, 0]})
    print("obs hook:", o["path_integrals"]["x"].shape)
    print("E|X|^2:", float(np.mean(np.sum(r["terminal_paths"] ** 2, axis=1))))
