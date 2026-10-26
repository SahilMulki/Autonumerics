import numpy as np

# Heston parameters
KAPPA = 2.58
THETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.164
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _simulate_chunk(n, Nt, dt, rng):
    """Andersen QE scheme for V with martingale-corrected log-S step.
    Returns X = log(S_T/S0) and V_T for n paths."""
    k, th, sg, rho = KAPPA, THETA, SIGMA, RHO
    E = np.exp(-k * dt)
    c1 = sg * sg * E * (1.0 - E) / k
    c2 = th * sg * sg * (1.0 - E) ** 2 / (2.0 * k)
    g1 = g2 = 0.5
    K1 = g1 * dt * (k * rho / sg - 0.5) - rho / sg
    K2 = g2 * dt * (k * rho / sg - 0.5) + rho / sg
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4
    psi_c = 1.5

    V = np.full(n, V0)
    X = np.zeros(n)
    Vn = np.empty(n)
    logM = np.empty(n)

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Zs = rng.standard_normal(n)

        m = th + (V - th) * E
        s2 = V * c1 + c2
        psi = s2 / (m * m)

        lo = psi <= psi_c
        hi = ~lo

        # Quadratic branch
        if lo.any():
            p_ = psi[lo]
            inv = 2.0 / p_
            b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
            a = m[lo] / (1.0 + b2)
            Vn[lo] = a * (np.sqrt(b2) + Zv[lo]) ** 2
            den = 1.0 - 2.0 * A * a
            logM[lo] = A * b2 * a / den - 0.5 * np.log(den)

        # Exponential branch (mass at zero)
        if hi.any():
            p_ = psi[hi]
            pp = (p_ - 1.0) / (p_ + 1.0)
            beta = (1.0 - pp) / m[hi]
            u = U[hi]
            vv = np.zeros_like(u)
            pos = u > pp
            vv[pos] = np.log((1.0 - pp[pos]) / (1.0 - u[pos])) / beta[pos]
            Vn[hi] = vv
            logM[hi] = np.log(pp + beta * (1.0 - pp) / (beta - A))

        var_term = np.maximum(K3 * V + K4 * Vn, 0.0)
        X += -logM - 0.5 * K3 * V + K2 * Vn + np.sqrt(var_term) * Zs
        V, Vn = Vn, V  # swap buffers (V now holds new values)
        Vn = Vn.copy() if Vn is V else Vn

    return X, V.copy()


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    chunk = 250000
    X_all = np.empty(num_paths)
    V_all = np.empty(num_paths)
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X, V = _simulate_chunk(n, Nt, dt, rng)
        X_all[start:start + n] = X
        V_all[start:start + n] = V
        start += n

    growth = np.exp(X_all)
    prices = {}
    for s0 in SPOTS:
        prices["price_S%d" % int(s0)] = float(np.mean(np.maximum(s0 * growth - STRIKE, 0.0)))

    S100 = 100.0 * growth
    terminal = np.column_stack([S100, V_all])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": prices,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=200000, dt=0.002, T=1.0)
    print(r["functionals"], r["empirical_mean"])
