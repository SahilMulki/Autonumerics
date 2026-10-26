"""Heston, Andersen QE scheme for V + central log-S step with martingale correction.

dS = sqrt(V) S dW1, dV = kappa (eta - V) dt + sigma sqrt(V) dW2, corr = rho, r = 0.
numpy + stdlib only.
"""
import numpy as np

KAPPA, ETA, SIGMA, RHO = 2.58, 0.043, 1.0, -0.36
V0, S0_STATE, KSTRIKE = 0.164, 100.0, 100.0
SPOTS = (75.0, 100.0, 125.0)
PSI_C = 1.5


def _ncdf_upper(x):
    """Q(x) = 1 - Phi(x) = 0.5*erfc(x/sqrt2), Numerical Recipes erfcc (rel err < 1.2e-7)."""
    z = np.abs(x) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = -z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (
        -0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (
            -0.82215223 + t * 0.17087277))))))))
    erfc_abs = t * np.exp(poly)                      # erfc(|x|/sqrt2)
    # erfc(z) for z>=0 is erfc_abs; for z<0 it is 2 - erfc_abs
    erfc = np.where(x >= 0, erfc_abs, 2.0 - erfc_abs)
    return 0.5 * erfc


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    kappa, eta, sigma, rho = KAPPA, ETA, SIGMA, RHO
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sdt = np.sqrt(dt)
    c = np.sqrt(1.0 - rho ** 2)

    E = np.exp(-kappa * dt)
    g1 = g2 = 0.5
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)
    A = K2 + 0.5 * K4
    c1 = sigma ** 2 * E * (1.0 - E) / kappa
    c2 = eta * sigma ** 2 * (1.0 - E) ** 2 / (2.0 * kappa)

    V = np.full(num_paths, V0, dtype=float)
    logR = np.zeros(num_paths)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(lr, v):
        return np.column_stack([S0_STATE * np.exp(lr), v])

    X_prev = state(logR, V) if obs else None

    for n in range(Nt):
        if dW is None:
            dZ = sdt * rng.standard_normal((num_paths, 2))
            z1, z2 = dZ[:, 0], dZ[:, 1]
        else:
            z1, z2 = dW[:, n, 0], dW[:, n, 1]
        Zv = (rho * z1 + c * z2) / sdt
        Zp = (c * z1 - rho * z2) / sdt

        m = eta + (V - eta) * E
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        quad = psi <= PSI_C
        Vn = np.empty(num_paths)
        K0s = np.empty(num_paths)

        iq = np.nonzero(quad)[0]
        if iq.size:
            ps = psi[iq]
            inv = 2.0 / ps
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
            a = m[iq] / (1.0 + b2)
            Vn[iq] = a * (np.sqrt(b2) + Zv[iq]) ** 2
            den = 1.0 - 2.0 * A * a
            assert np.all(den > 0)
            K0s[iq] = -A * b2 * a / den + 0.5 * np.log(den)
        ie = np.nonzero(~quad)[0]
        if ie.size:
            ps = psi[ie]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / m[ie]
            Q = _ncdf_upper(Zv[ie])                  # 1 - U
            above = Q < (1.0 - p)                    # U > p
            Qs = np.where(above, np.maximum(Q, 1e-300), 1.0 - p)
            Vn[ie] = np.where(above, np.log((1.0 - p) / Qs) / beta, 0.0)
            assert np.all(beta - A > 0)
            K0s[ie] = -np.log(p + beta * (1.0 - p) / (beta - A))
        K0s -= (K1 + 0.5 * K3) * V

        logR = logR + K0s + K1 * V + K2 * Vn + np.sqrt(K3 * V + K4 * Vn) * Zp
        V = Vn

        if obs:
            X_new = state(logR, V)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X_new)) * dt
            X_prev = X_new

    R = np.exp(logR)
    S = S0_STATE * R
    X = np.column_stack([S, V])
    functionals = {
        f"price_S{int(s0)}": float(np.mean(np.maximum(s0 * R - KSTRIKE, 0.0))) for s0 in SPOTS
    }
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "frac_V_T_zero": float(np.mean(V == 0.0)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = solve_sde(num_paths=2000000, dt=0.002, T=1.0, seed=42)
    print("time", round(time.time() - t0, 1), "s")
    print("Empirical mean:    ", r["empirical_mean"])
    print("Empirical variance:", r["empirical_variance"])
    print("functionals:", r["functionals"])
    print("frac V_T == 0:", r["frac_V_T_zero"])
    r2 = solve_sde(num_paths=500000, dt=0.002, T=1.0, seed=42)
    print("500k functionals:", r2["functionals"])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.01, 1.0, dW=dWt)["dt"])
    po = solve_sde(1000, 0.01, 1.0, observables={"v": lambda a: a[:, 1]})["path_integrals"]["v"]
    print("obs hook shape:", po.shape)
