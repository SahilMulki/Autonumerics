import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Milstein on the Ito form of the Kubo oscillator (one Brownian channel).

    X_{n+1} = [(1 - 0.5 s^2 dW^2) I + (a dt + s dW) J] X_n,  J = [[0,-1],[1,0]]
    (the +/-0.5 s^2 dt X terms of drift and correction cancel).
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    X = np.zeros((num_paths, 2))
    X[:, 0] = X_0
    X[:, 1] = Y_0
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        w = dW[:, n]
        theta = A * dt + SIGMA * w
        m = 1.0 - 0.5 * SIGMA**2 * w**2
        X = np.empty_like(X_prev)
        X[:, 0] = m * X_prev[:, 0] - theta * X_prev[:, 1]
        X[:, 1] = theta * X_prev[:, 0] + m * X_prev[:, 1]
        for k, phi in (observables or {}).items():
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
    r = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("E|X_T|^2:", float(np.mean(np.sum(r["terminal_paths"] ** 2, axis=1))))
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
          observables={"x": lambda a: a[:, 0]})["path_integrals"]["x"].shape)
