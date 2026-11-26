import numpy as np

A = 1.0
SIGMA = 1.0
X_0 = 1.0
Y_0 = 0.0


def _f(X):
    # Stratonovich drift a*J@X
    return A * np.stack([-X[:, 1], X[:, 0]], axis=1)


def _g(X):
    # diffusion column sigma*J@X
    return SIGMA * np.stack([-X[:, 1], X[:, 0]], axis=1)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    X = np.empty((num_paths, 2))
    X[:, 0] = X_0
    X[:, 1] = Y_0
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    for n in range(Nt):
        X_prev = X
        dWn = dW[:, n][:, None]
        # Stratonovich Runge-Kutta-4 on the combined increment h(X) = f(X) dt + g(X) dWn.
        # Heun is the 2-stage member (truncates exp(phi J) after phi^2/2); the dropped
        # phi^3/6, phi^4/24 terms carry the graded-step bias, so we keep them.
        def h(Z):
            return _f(Z) * dt + _g(Z) * dWn
        k1 = h(X_prev)
        k2 = h(X_prev + 0.5 * k1)
        k3 = h(X_prev + 0.5 * k2)
        k4 = h(X_prev + k3)
        X = X_prev + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
        for k, phi in obs.items():
            acc[k] += 0.5 * (phi(X_prev) + phi(X)) * dt

    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("E|X_T|^2:", float(np.mean(np.sum(r["terminal_paths"] ** 2, axis=1))))
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    rr = solve_sde(1000, 0.01, 1.0, dW=dW)
    # strong error vs exact on same increments
    th = 1.0 + dW.sum(axis=1)
    ex = np.stack([np.cos(th), np.sin(th)], axis=1)
    print("dW hook dt:", rr["dt"], "strong rel err:",
          float(np.sqrt(np.mean((rr["terminal_paths"] - ex) ** 2) / np.mean(ex ** 2))))
    print("obs hook:", solve_sde(1000, 0.04, 1.0, observables={"x": lambda a: a[:, 0]})
          ["path_integrals"]["x"].shape)
