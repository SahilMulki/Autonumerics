import numpy as np

KAPPA = 2.58
THETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.164
L_BAR = 50.0
U_BAR = 150.0
SPOTS = (75.0, 100.0, 125.0)


def _qe_step(V, dt, rng, n):
    # Andersen Quadratic-Exponential step for CIR variance
    e = np.exp(-KAPPA * dt)
    m = THETA + (V - THETA) * e
    s2 = (V * SIGMA**2 * e * (1.0 - e) / KAPPA
          + THETA * SIGMA**2 * (1.0 - e) ** 2 / (2.0 * KAPPA))
    psi = s2 / (m * m)
    Zv = rng.standard_normal(n)
    U = rng.random(n)
    Vn = np.empty_like(V)

    q = psi <= 1.5
    if np.any(q):
        pq = psi[q]
        inv = 2.0 / pq
        b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(inv - 1.0)
        a = m[q] / (1.0 + b2)
        Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
    nq = ~q
    if np.any(nq):
        pe = psi[nq]
        p = (pe - 1.0) / (pe + 1.0)
        beta = (1.0 - p) / m[nq]
        u = U[nq]
        out = np.zeros_like(u)
        pos = u > p
        out[pos] = np.log((1.0 - p[pos]) / (1.0 - u[pos])) / beta[pos]
        Vn[nq] = out
    return Vn


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = num_paths

    V = np.full(n, V0, dtype=float)
    X = np.zeros(n)  # log(S/S0), independent of S0

    lu = np.log(U_BAR)
    ld = np.log(L_BAR)
    l0 = [np.log(s) for s in SPOTS]
    weights = [np.ones(n) for _ in SPOTS]

    sq = np.sqrt(1.0 - RHO**2)
    for _ in range(Nt):
        Vn = _qe_step(V, dt, rng, n)
        Z = rng.standard_normal(n)
        intV = 0.5 * (V + Vn) * dt
        # integral of sqrt(V) dW2 implied by the variance dynamics
        intdW2 = (Vn - V - KAPPA * THETA * dt + KAPPA * intV) / SIGMA
        Xn = (X - 0.5 * intV + RHO * intdW2
              + sq * np.sqrt(np.maximum(intV, 0.0)) * Z)

        denom = np.maximum(intV, 1e-300)
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            for k in range(len(SPOTS)):
                x0 = X + l0[k]
                x1 = Xn + l0[k]
                inside = (x1 < lu) & (x1 > ld) & (x0 < lu) & (x0 > ld)
                du0 = np.maximum(lu - x0, 0.0)
                du1 = np.maximum(lu - x1, 0.0)
                dd0 = np.maximum(x0 - ld, 0.0)
                dd1 = np.maximum(x1 - ld, 0.0)
                pu = np.exp(-2.0 * du0 * du1 / denom)
                pd = np.exp(-2.0 * dd0 * dd1 / denom)
                pu = np.where(intV > 0, pu, 0.0)
                pd = np.where(intV > 0, pd, 0.0)
                surv = np.clip((1.0 - pu) * (1.0 - pd), 0.0, 1.0)
                weights[k] *= np.where(inside, surv, 0.0)

        X = Xn
        V = Vn

    S100 = 100.0 * np.exp(X)
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(weights[0])),
        "no_touch_S100": float(np.mean(weights[1])),
        "no_touch_S125": float(np.mean(weights[2])),
    }

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1.0, seed=42)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
