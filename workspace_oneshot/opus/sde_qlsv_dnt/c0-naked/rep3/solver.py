import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt

    c = 0.01
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    srho = np.sqrt(1.0 - rho * rho)
    S0s = np.array([75.0, 100.0, 125.0])
    ylo, yhi = 1.0 / 150.0, 1.0 / 50.0   # Y = 1/S corridor

    N = num_paths
    V = np.full(N, V0)
    Y = np.repeat((1.0 / S0s)[:, None], N, axis=1)   # shape (3, N)
    w = np.ones((3, N))

    ekdt = np.exp(-kappa * dt)
    psi_c = 1.5
    c2 = c * c

    for _ in range(Nt):
        # ---- QE step for variance (Andersen) ----
        m = theta + (V - theta) * ekdt
        s2 = (V * sigma**2 * ekdt / kappa * (1.0 - ekdt)
              + theta * sigma**2 / (2.0 * kappa) * (1.0 - ekdt) ** 2)
        psi = s2 / (m * m)
        Zv = rng.standard_normal(N)
        U = rng.random(N)
        Vn = np.empty(N)
        lo = psi <= psi_c
        if np.any(lo):
            ip = 2.0 / psi[lo]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[lo] / (1.0 + b2)
            Vn[lo] = a * (np.sqrt(b2) + Zv[lo]) ** 2
        hi = ~lo
        if np.any(hi):
            p = (psi[hi] - 1.0) / (psi[hi] + 1.0)
            beta = (1.0 - p) / m[hi]
            u = U[hi]
            out = np.zeros(u.shape)
            pos = u > p
            out[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[hi] = out

        Vint = 0.5 * (V + Vn) * dt
        intW2 = Vn - V - kappa * theta * dt + kappa * Vint   # ∫ sqrt(V) dW2
        Z = rng.standard_normal(N)
        mart = rho * intW2 + srho * np.sqrt(Vint) * Z        # ∫ sqrt(V) dW1

        # ---- Y = 1/S:  dY = c^2 V / Y dt - c sqrt(V) dW1 ----
        Y0 = Y
        Y1 = Y0 + c2 * Vint[None, :] / Y0 - c * mart[None, :]
        Y1 = np.maximum(np.abs(Y1), 1e-12)

        # ---- continuous barrier monitoring via Brownian bridge ----
        var = np.maximum(c2 * Vint, 1e-300)[None, :]
        inside = (Y1 > ylo) & (Y1 < yhi)
        dlo0 = np.maximum(Y0 - ylo, 0.0)
        dlo1 = np.maximum(Y1 - ylo, 0.0)
        dhi0 = np.maximum(yhi - Y0, 0.0)
        dhi1 = np.maximum(yhi - Y1, 0.0)
        p_lo = np.exp(-2.0 * dlo0 * dlo1 / var)
        p_hi = np.exp(-2.0 * dhi0 * dhi1 / var)
        w = w * np.where(inside, (1.0 - p_lo) * (1.0 - p_hi), 0.0)

        Y = Y1
        V = Vn

    S100 = 1.0 / Y[1]
    terminal = np.column_stack([S100, V])
    probs = w.mean(axis=1)

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "no_touch_S75": float(probs[0]),
            "no_touch_S100": float(probs[1]),
            "no_touch_S125": float(probs[2]),
        },
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
