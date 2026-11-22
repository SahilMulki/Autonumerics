import numpy as np

A = 1.0
SIGMA = 1.0
X1_0 = 1.0
X2_0 = 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Kubo oscillator (Stratonovich) via explicit Heun predictor-corrector."""
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
    X[:, 0] = X1_0
    X[:, 1] = X2_0
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        x1 = X_prev[:, 0]
        x2 = X_prev[:, 1]
        w = dW[:, n]
        # classical 4-stage explicit Stratonovich Runge-Kutta on the stated
        # Stratonovich coefficients (derivative-free; stage increments use the
        # whole-step dt and dW, i.e. RK4 along the control path)
        def inc(y1, y2):
            return (-A * y2 * dt - SIGMA * y2 * w,
                    A * y1 * dt + SIGMA * y1 * w)
        k1a, k1b = inc(x1, x2)
        k2a, k2b = inc(x1 + 0.5 * k1a, x2 + 0.5 * k1b)
        k3a, k3b = inc(x1 + 0.5 * k2a, x2 + 0.5 * k2b)
        k4a, k4b = inc(x1 + k3a, x2 + k3b)
        n1 = x1 + (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0
        n2 = x2 + (k1b + 2 * k2b + 2 * k3b + k4b) / 6.0
        X = np.column_stack((n1, n2))
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
    res = solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    amp = np.abs(np.sqrt(res["terminal_paths"][:, 0] ** 2 + res["terminal_paths"][:, 1] ** 2) - 1)
    print("max amplitude dev: ", float(amp.max()))
    print("exact mean: [0.32770991, 0.51037795]  exact var: [0.36444654, 0.26767402]")
    dW = np.sqrt(0.04) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.04, 1.0, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
          observables={"x": lambda a: a[:, 0]})["path_integrals"]["x"].shape)
