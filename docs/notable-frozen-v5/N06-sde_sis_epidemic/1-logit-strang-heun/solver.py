import numpy as np

# Stochastic SIS: dI = I[(beta N - mu - gamma) - beta I] dt + sigma I (N - I) dW
X_0, N, BETA, MU, GAMMA, SIGMA = 90.0, 100.0, 0.5, 20.0, 25.0, 0.03
A = BETA * N - MU - GAMMA          # 5
MG = MU + GAMMA                    # 45 (= N*beta - a)
LOGN = np.log(N)


def _to_I(Y):
    return N / (1.0 + np.exp(-Y))


def _logistic_half_flow(Y, E):
    """Exact flow of dI = I(a - beta I) over a half step (E = exp(-a h/2)), in logit coords."""
    logI0 = LOGN - np.logaddexp(0.0, -Y)
    I0 = np.exp(logI0)
    numer = N * (A - BETA * I0) * E + MG * I0          # = D*(N - I1)
    return np.log(A) + logI0 - np.log(numer)           # log(I1) - log(N - I1)


def _R(Y):
    return 0.5 * SIGMA**2 * N**2 * np.tanh(0.5 * Y)


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

    E = np.exp(-A * dt / 2.0)
    sN = SIGMA * N
    Y = np.full(num_paths, np.log(X_0) - np.log(N - X_0))
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    I = _to_I(Y) if obs else None

    for n in range(Nt):
        I_prev = I
        Y = _logistic_half_flow(Y, E)
        noise = sN * dW[:, n]
        R0 = _R(Y)
        Ys = Y + R0 * dt + noise
        Y = Y + 0.5 * (R0 + _R(Ys)) * dt + noise
        Y = _logistic_half_flow(Y, E)
        if obs:
            I = _to_I(Y)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(I_prev) + phi(I)) * dt

    X = _to_I(Y)
    out = {
        "terminal_paths": X,
        "empirical_mean": float(np.mean(X)),
        "empirical_variance": float(np.var(X, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    res = solve_sde(num_paths=200000, dt=0.02, T=0.5, seed=42)
    X = res["terminal_paths"]
    print(f"Empirical mean:     {res['empirical_mean']:.6f}")
    print(f"Empirical variance: {res['empirical_variance']:.6f}")
    print("range:", X.min(), X.max(), "finite:", bool(np.all(np.isfinite(X))))
    dW = np.sqrt(0.5 / 50) * np.random.default_rng(0).standard_normal((1000, 50))
    print("dW hook dt:", solve_sde(1000, 0.02, 0.5, dW=dW)["dt"])
    print("obs hook:", solve_sde(1000, 0.02, 0.5,
                                 observables={"x": lambda a: a})["path_integrals"]["x"].shape)
