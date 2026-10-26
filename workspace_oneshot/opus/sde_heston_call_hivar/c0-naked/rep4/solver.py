import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Heston parameters
    kappa = 2.58
    theta = 0.043
    eps = 1.0
    rho = -0.36
    V0 = 0.164
    K_strike = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt

    rng = np.random.default_rng(seed)

    # QE (Andersen 2008) constants
    ekd = np.exp(-kappa * dt)
    c1 = eps * eps * ekd * (1.0 - ekd) / kappa
    c2 = theta * eps * eps * (1.0 - ekd) ** 2 / (2.0 * kappa)
    psi_c = 1.5

    g1 = 0.5
    g2 = 0.5
    K1 = g1 * dt * (kappa * rho / eps - 0.5) - rho / eps
    K2 = g2 * dt * (kappa * rho / eps - 0.5) + rho / eps
    K3 = g1 * dt * (1.0 - rho * rho)
    K4 = g2 * dt * (1.0 - rho * rho)
    A = K2 + 0.5 * K4

    lx_all = np.empty(num_paths)
    V_all = np.empty(num_paths)

    chunk = 250000
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        V = np.full(n, V0)
        lx = np.zeros(n)
        Vn = np.empty(n)
        M = np.empty(n)
        for _ in range(Nt):
            Zv = rng.standard_normal(n)
            U = rng.random(n)
            Z = rng.standard_normal(n)

            m = theta + (V - theta) * ekd
            s2 = V * c1 + c2
            psi = s2 / (m * m)

            q = psi <= psi_c
            e = ~q

            # quadratic branch
            if q.any():
                psq = psi[q]
                mq = m[q]
                inv = 2.0 / psq
                b2 = inv - 1.0 + np.sqrt(inv * (inv - 1.0))
                a = mq / (1.0 + b2)
                Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
                den = 1.0 - 2.0 * A * a
                M[q] = np.exp(A * b2 * a / den) / np.sqrt(den)

            # exponential branch
            if e.any():
                pse = psi[e]
                me = m[e]
                p = (pse - 1.0) / (pse + 1.0)
                beta = (1.0 - p) / me
                Ue = U[e]
                vals = np.where(
                    Ue <= p, 0.0,
                    np.log((1.0 - p) / np.maximum(1.0 - Ue, 1e-300)) / beta)
                Vn[e] = vals
                M[e] = p + beta * (1.0 - p) / (beta - A)

            # martingale-corrected log-asset step
            var_inc = K3 * V + K4 * Vn
            lx += (-np.log(M) - 0.5 * K3 * V + K2 * Vn
                   + np.sqrt(np.maximum(var_inc, 0.0)) * Z)
            V = Vn.copy()

        lx_all[start:start + n] = lx
        V_all[start:start + n] = V
        start += n

    Y = np.exp(lx_all)  # S(T)/S0, exact mean 1
    Ym = Y.mean()
    Yc = Y - Ym
    varY = np.dot(Yc, Yc) / (num_paths - 1)

    functionals = {}
    for S0 in spots:
        pay = np.maximum(S0 * Y - K_strike, 0.0)
        pm = pay.mean()
        cov = np.dot(pay - pm, Yc) / (num_paths - 1)
        c = cov / varY if varY > 0 else 0.0
        price = pm - c * (Ym - 1.0)
        functionals["price_S%d" % int(S0)] = float(price)

    S_T = 100.0 * Y
    terminal = np.column_stack((S_T, V_all))

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(np.mean(terminal[:, 0])), float(np.mean(terminal[:, 1]))],
        "empirical_variance": [float(np.var(terminal[:, 0], ddof=1)),
                               float(np.var(terminal[:, 1], ddof=1))],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(out["functionals"], out["empirical_mean"])
