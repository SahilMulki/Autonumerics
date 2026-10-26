import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    kappa, theta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.164
    K = 100.0
    spots = (75.0, 100.0, 125.0)

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)
    n = int(num_paths)

    # Andersen QE scheme with martingale correction (gamma1 = gamma2 = 0.5)
    psic = 1.5
    g1 = g2 = 0.5
    e = np.exp(-kappa * dt)
    c1 = sigma ** 2 * e * (1.0 - e) / kappa
    c2 = theta * sigma ** 2 * (1.0 - e) ** 2 / (2.0 * kappa)
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho ** 2)
    K4 = g2 * dt * (1.0 - rho ** 2)
    A = K2 + 0.5 * K4

    V = np.full(n, V0)
    X = np.zeros(n)  # log(S/S0)
    Vn = np.empty(n)
    lnM = np.empty(n)
    Zv = np.empty(n)
    Zs = np.empty(n)
    U = np.empty(n)

    for _ in range(Nt):
        rng.standard_normal(n, out=Zv)
        rng.random(n, out=U)
        rng.standard_normal(n, out=Zs)

        m = theta + (V - theta) * e
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        quad = psi <= psic
        ex = ~quad

        # quadratic branch
        if quad.any():
            mq = m[quad]
            ip = 1.0 / psi[quad]
            b2 = 2.0 * ip - 1.0 + np.sqrt(2.0 * ip) * np.sqrt(np.maximum(2.0 * ip - 1.0, 0.0))
            a = mq / (1.0 + b2)
            Vn[quad] = a * (np.sqrt(b2) + Zv[quad]) ** 2
            d = 1.0 - 2.0 * A * a
            lnM[quad] = A * b2 * a / d - 0.5 * np.log(d)

        # exponential branch
        if ex.any():
            me = m[ex]
            pe = psi[ex]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / me
            Ue = U[ex]
            vv = np.zeros_like(me)
            pos = Ue > p
            vv[pos] = np.log((1.0 - p[pos]) / (1.0 - Ue[pos])) / beta[pos]
            Vn[ex] = vv
            lnM[ex] = np.log(p + beta * (1.0 - p) / (beta - A))

        X += (-lnM - 0.5 * K3 * V + K2 * Vn
              + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs)
        V, Vn = Vn, V

    eX = np.exp(X)
    functionals = {}
    for S0 in spots:
        ST = S0 * eX
        pay = np.maximum(ST - K, 0.0)
        # control variate: E[S_T] = S0 exactly
        cy = ST - S0
        varY = np.dot(cy, cy) / n - (cy.mean()) ** 2
        if varY > 0:
            cov = np.dot(pay - pay.mean(), cy - cy.mean()) / n
            b = cov / varY
        else:
            b = 0.0
        price = float(pay.mean() - b * cy.mean())
        functionals["price_S%d" % int(S0)] = price

    S100 = 100.0 * eX
    terminal = np.column_stack((S100, V))
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(v) for v in terminal.mean(axis=0)],
        "empirical_variance": [float(v) for v in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=200000, dt=0.002, T=1, seed=42)
    print(out["empirical_mean"], out["empirical_variance"], out["functionals"])
