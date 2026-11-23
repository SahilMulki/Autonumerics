import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None,
              dW_sub: np.ndarray | None = None) -> dict:
    """Plain Euler-Maruyama on the step increments only (Clark-Cameron control).

    State (S, U): dS = mu dt + dW2, dU = S dW1.  dW[..., 0]=dW1 drives U,
    dW[..., 1]=dW2 drives S.  dW_sub is accepted and deliberately unused:
    the Levy area I_21 is not added, so strong order is capped at 0.5.
    """
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt, 2))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    X = np.stack([S, U], axis=1)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}

    for n in range(Nt):
        X_prev = X
        dW1 = dW[:, n, 0]
        dW2 = dW[:, n, 1]
        S_next = X_prev[:, 0] + MU * dt + dW2
        U_next = X_prev[:, 1] + X_prev[:, 0] * dW1
        X = np.stack([S_next, U_next], axis=1)
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
    M, T = 50000, 1.0
    r = solve_sde(num_paths=M, dt=0.04, T=T, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    c = np.corrcoef(r["terminal_paths"][:, 0], r["terminal_paths"][:, 1])[0, 1]
    print("corr(S,U):", c, " (se ~", 1 / np.sqrt(M), ")")
    dWt = np.sqrt(0.04 / 2) * np.random.default_rng(0).standard_normal((1000, 50, 2))
    print("dW hook dt:", solve_sde(1000, 0.04, T, dW=dWt)["dt"])
    po = solve_sde(1000, 0.04, T, observables={"u2": lambda a: a[:, 1] ** 2})
    print("obs hook:", po["path_integrals"]["u2"].shape)
