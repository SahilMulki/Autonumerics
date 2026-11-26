import numpy as np

OMEGA, GAMMA, BETA = 4.0, 1.0, 1.0
Q0, P0 = 1.0, 0.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    """Stochastic trapezoidal rule (drift-implicit, theta=1/2) for underdamped Langevin.

    (I - h/2 A) Z_{n+1} = (I + h/2 A) Z_n + b dW_n.
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

    A = np.array([[0.0, 1.0], [-OMEGA**2, -GAMMA]])
    b = np.array([0.0, np.sqrt(2.0 * GAMMA / BETA)])
    I2 = np.eye(2)
    M = np.linalg.inv(I2 - 0.5 * dt * A)
    R = M @ (I2 + 0.5 * dt * A)
    Mb = M @ b
    R00, R01, R10, R11 = R[0, 0], R[0, 1], R[1, 0], R[1, 1]

    X = np.empty((num_paths, 2))
    X[:, 0] = Q0
    X[:, 1] = P0
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        w = dW[:, n]
        q, p = X_prev[:, 0], X_prev[:, 1]
        X = np.empty_like(X_prev)
        X[:, 0] = R00 * q + R01 * p + Mb[0] * w
        X[:, 1] = R10 * q + R11 * p + Mb[1] * w
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
    res = solve_sde(num_paths=100000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", res["empirical_mean"])
    print("Empirical variance:", res["empirical_variance"])
    nu = np.sqrt(OMEGA**2 - GAMMA**2 / 4)
    t = 1.0
    c, s = np.cos(nu * t), np.sin(nu * t)
    e = np.exp(-GAMMA * t)
    mq = np.exp(-GAMMA * t / 2) * ((c + GAMMA / (2 * nu) * s) * Q0 + s / nu * P0)
    mp = np.exp(-GAMMA * t / 2) * (-OMEGA**2 * s / nu * Q0 + (c - GAMMA / (2 * nu) * s) * P0)
    vq = (1 - e * (c + GAMMA / (2 * nu) * s)**2) / (BETA * OMEGA**2) - e * s**2 / (BETA * nu**2)
    vp = (1 - e * (c - GAMMA / (2 * nu) * s)**2) / BETA - OMEGA**2 * e * s**2 / (BETA * nu**2)
    print("Exact mean:        ", [mq, mp])
    print("Exact variance:    ", [vq, vp])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100))
    print("dW hook:", solve_sde(1000, 0.04, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.04, 1.0,
                                 observables={"qp": lambda a: a[:, 0] * a[:, 1]}
                                 )["path_integrals"]["qp"].shape)
