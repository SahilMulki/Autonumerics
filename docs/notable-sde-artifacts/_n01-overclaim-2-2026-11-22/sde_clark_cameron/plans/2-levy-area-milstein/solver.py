"""Levy-area Milstein for the Clark-Cameron system.

dS = mu dt + dW2 (exact step), dU = S dW1, with I_{1,2} assembled from dW_sub.
Channel 0 = W1 (drives U), channel 1 = W2 (drives S).
"""
import numpy as np

S_0, U_0, MU = 1.0, 0.0, 1.0
_CHUNK = 1000


def _levy_I12(dW_sub):
    """I_{1,2} per path per step: sum_j (W2 displacement before sub-step j) * dW1_j."""
    out = np.empty(dW_sub.shape[:2])
    for a in range(0, dW_sub.shape[0], _CHUNK):
        blk = dW_sub[a:a + _CHUNK]
        d1 = blk[..., 0]
        d2 = blk[..., 1]
        w2 = np.cumsum(d2, axis=2) - d2          # exclusive prefix sum
        out[a:a + _CHUNK] = np.einsum("pnj,pnj->pn", w2, d1)
    return out


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None,
              dW_sub: np.ndarray | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt, 2))
        dW_sub = None
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        num_paths = dW.shape[0]

    if dW_sub is not None:
        I12 = _levy_I12(dW_sub)
    else:
        # conditional-mean area: E[I12 | dW] = 0.5 * dW1 * dW2 is the Stratonovich
        # half; for the Ito integral I_{1,2} = int (W2-W2_n) dW1 the conditional
        # mean given the step increments is 0.5 * dW1 * dW2.
        I12 = 0.5 * dW[:, :, 0] * dW[:, :, 1]

    S = np.full(num_paths, S_0, dtype=float)
    U = np.full(num_paths, U_0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}
    X = np.stack([S, U], axis=1)

    for n in range(Nt):
        X_prev = X
        S_new = S + MU * dt + dW[:, n, 1]
        U = U + S * dW[:, n, 0] + I12[:, n]
        S = S_new
        X = np.stack([S, U], axis=1)
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
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    r = solve_sde(num_paths=16000, dt=0.04, T=1.0, seed=42)
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    M, Nt, K = 500, 25, 16
    rng = np.random.default_rng(0)
    sub = np.sqrt(0.04 / K) * rng.standard_normal((M, Nt, K, 2))
    dWt = sub.sum(axis=2)
    print("dW hook:", solve_sde(M, 0.04, 1.0, dW=dWt, dW_sub=sub)["dt"])
    print("dW w/o sub:", solve_sde(M, 0.04, 1.0, dW=dWt)["dt"])
    print("obs hook:", solve_sde(M, 0.04, 1.0, dW=dWt, dW_sub=sub,
                                 observables={"u2": lambda a: a[:, 1] ** 2})["path_integrals"]["u2"].shape)
