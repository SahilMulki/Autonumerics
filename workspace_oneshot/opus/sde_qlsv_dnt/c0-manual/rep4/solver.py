import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # Model parameters
    c = 0.01            # f(s) = c * s^2
    kappa = 2.58
    eta = 0.043
    sig = 1.0
    V0 = 0.114
    rho = -0.36
    srho = np.sqrt(1.0 - rho * rho)
    L_S, U_S = 50.0, 150.0
    # Y = 1/S :  dY = c^2 V / Y dt - c sqrt(V) dW1   (additive noise given V)
    # Corridor in Y: (1/150, 1/50)
    L_Y, U_Y = 1.0 / U_S, 1.0 / L_S

    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    spots = [75.0, 100.0, 125.0]
    Ys = [np.full(num_paths, 1.0 / s0) for s0 in spots]
    alive = [np.ones(num_paths) for _ in spots]

    V = np.full(num_paths, V0)

    ek = np.exp(-kappa * dt)
    one_m_ek = 1.0 - ek
    psi_c = 1.5
    Y_floor = 1e-6

    for n in range(Nt):
        Zv = rng.standard_normal(num_paths)
        Uv = rng.random(num_paths)
        Zp = rng.standard_normal(num_paths)

        # --- Andersen QE step for variance ---
        m = eta + (V - eta) * ek
        s2 = (V * sig * sig * ek * one_m_ek / kappa
              + eta * sig * sig * one_m_ek * one_m_ek / (2.0 * kappa))
        m_safe = np.maximum(m, 1e-300)
        psi = s2 / (m_safe * m_safe)
        Vn = np.empty_like(V)

        q = psi <= psi_c
        if np.any(q):
            pq = psi[q]
            inv = 2.0 / pq
            b2 = inv - 1.0 + np.sqrt(inv) * np.sqrt(np.maximum(inv - 1.0, 0.0))
            a = m[q] / (1.0 + b2)
            Vn[q] = a * (np.sqrt(b2) + Zv[q]) ** 2
        e = ~q
        if np.any(e):
            pe = psi[e]
            p = (pe - 1.0) / (pe + 1.0)
            beta = (1.0 - p) / m_safe[e]
            u = Uv[e]
            val = np.where(u <= p, 0.0,
                           np.log(np.maximum((1.0 - p) / np.maximum(1.0 - u, 1e-300), 1.0)) / beta)
            Vn[e] = val

        # Integrated variance over step (trapezoid) and int sqrt(V) dW2
        I = 0.5 * (V + Vn) * dt
        intW2 = (Vn - V - kappa * eta * dt + kappa * I) / sig
        intW1 = rho * intW2 + srho * np.sqrt(I) * Zp
        dM = -c * intW1                      # martingale increment of Y
        var_step = np.maximum(c * c * I, 1e-300)  # bridge variance for Y

        for k in range(3):
            Y0 = Ys[k]
            Yd = np.maximum(Y0, Y_floor)
            Y1 = Y0 + c * c * I / Yd + dM
            Y1 = np.abs(Y1)                  # Bessel(3)-like: reflect (only matters for dead paths)
            Y1 = np.maximum(Y1, Y_floor)

            inside = (Y1 > L_Y) & (Y1 < U_Y)
            a_up = np.maximum((U_Y - Y0) * (U_Y - Y1), 0.0)
            a_dn = np.maximum((Y0 - L_Y) * (Y1 - L_Y), 0.0)
            p_up = np.exp(-2.0 * a_up / var_step)
            p_dn = np.exp(-2.0 * a_dn / var_step)
            surv = (1.0 - p_up) * (1.0 - p_dn)
            alive[k] = alive[k] * np.where(inside, surv, 0.0)
            Ys[k] = Y1

        V = Vn

    S100 = 1.0 / Ys[1]
    terminal = np.column_stack([S100, V])

    functionals = {
        "no_touch_S75": float(np.mean(alive[0])),
        "no_touch_S100": float(np.mean(alive[1])),
        "no_touch_S125": float(np.mean(alive[2])),
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
    res = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(res["functionals"])
    print(res["empirical_mean"], res["empirical_variance"])
