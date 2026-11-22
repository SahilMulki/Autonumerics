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
        # predictor: Euler on the Stratonovich coefficients
        p1 = x1 - A * x2 * dt - SIGMA * x2 * w
        p2 = x2 + A * x1 * dt + SIGMA * x1 * w
        # corrector: trapezoidal average of coefficients at X and at P
        n1 = x1 + 0.5 * (-A * x2 - A * p2) * dt + 0.5 * (-SIGMA * x2 - SIGMA * p2) * w
        n2 = x2 + 0.5 * (A * x1 + A * p1) * dt + 0.5 * (SIGMA * x1 + SIGMA * p1) * w
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
