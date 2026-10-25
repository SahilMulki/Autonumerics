import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    kappa, eta, sigma, rho = 2.58, 0.043, 1.0, -0.36
    V0 = 0.114
    spots = (75.0, 100.0, 125.0)
    lo_b, hi_b = 50.0, 150.0

    n = num_paths
    # Andersen QE constants
    ekd = np.exp(-kappa * dt)
    c1 = sigma**2 * ekd * (1.0 - ekd) / kappa
    c2 = eta * sigma**2 * (1.0 - ekd)**2 / (2.0 * kappa)
    psi_c = 1.5
    g1 = g2 = 0.5
    K0 = -rho * kappa * eta / sigma * dt
    K1 = g1 * dt * (kappa * rho / sigma - 0.5) - rho / sigma
    K2 = g2 * dt * (kappa * rho / sigma - 0.5) + rho / sigma
    K3 = g1 * dt * (1.0 - rho**2)
    K4 = g2 * dt * (1.0 - rho**2)

    V = np.full(n, V0)
    Y = np.zeros(n)  # log(S/S0), common to all spots
    ups = [np.log(hi_b / s) for s in spots]
    lws = [np.log(lo_b / s) for s in spots]
    surv = [np.ones(n) for _ in spots]

    for _ in range(Nt):
        Zv = rng.standard_normal(n)
        U = rng.random(n)
        Zs = rng.standard_normal(n)

        m = eta + (V - eta) * ekd
        s2 = V * c1 + c2
        psi = s2 / (m * m)
        Vn = np.empty(n)

        q = psi <= psi_c
        if np.any(q):
            ip = 2.0 / psi[q]
            b2 = ip - 1.0 + np.sqrt(ip) * np.sqrt(ip - 1.0)
            a = m[q] / (1.0 + b2)
            Vn[q] = a * (np.sqrt(b2) + Zv[q])**2
        e = ~q
        if np.any(e):
            pe = (psi[e] - 1.0) / (psi[e] + 1.0)
            beta = (1.0 - pe) / m[e]
            Ue = U[e]
            out = np.zeros(Ue.shape)
            pos = Ue > pe
            out[pos] = np.log((1.0 - pe[pos]) / (1.0 - Ue[pos])) / beta[pos]
            Vn[e] = out

        Y0 = Y
        Y = Y0 + K0 + K1 * V + K2 * Vn + np.sqrt(np.maximum(K3 * V + K4 * Vn, 0.0)) * Zs

        # Brownian-bridge crossing probability in log space
        var = np.maximum(0.5 * (V + Vn) * dt, 1e-300)
        inv = -2.0 / var
        for k in range(len(spots)):
            u, l = ups[k], lws[k]
            with np.errstate(over="ignore", invalid="ignore"):
                pu = np.exp(np.minimum(inv * (u - Y0) * (u - Y), 50.0))
                pl = np.exp(np.minimum(inv * (Y0 - l) * (Y - l), 50.0))
            p = np.clip(pu + pl, 0.0, 1.0)
            p = np.where((Y >= u) | (Y <= l), 1.0, p)
            surv[k] *= (1.0 - p)

        V = Vn

    S100 = 100.0 * np.exp(Y)
    X = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(surv[0])),
        "no_touch_S100": float(np.mean(surv[1])),
        "no_touch_S125": float(np.mean(surv[2])),
    }

    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)),
                               float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"])
    print(r["empirical_mean"], r["empirical_variance"])
