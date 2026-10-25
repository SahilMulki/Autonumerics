import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, int(round(T / dt)))
    dt = T / Nt
    sqdt = np.sqrt(dt)

    kappa, theta, sigma = 0.5, 0.04, 1.0
    a_r, b_r, sig_r = 0.08, 0.10, 0.09
    V0, R0 = 0.04, 0.10
    K = 100.0
    spots = [75.0, 100.0, 125.0]

    C = np.array([[1.0, -0.9, 0.6],
                  [-0.9, 1.0, -0.7],
                  [0.6, -0.7, 1.0]])
    L = np.linalg.cholesky(C)

    rng = np.random.default_rng(seed)

    chunk = 50000
    S_term = np.empty(num_paths)
    V_term = np.empty(num_paths)
    R_term = np.empty(num_paths)
    pay_sums = np.zeros(len(spots))

    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        X = np.zeros(n)          # log(S/S0)
        V = np.full(n, V0)
        R = np.full(n, R0)
        I = np.zeros(n)          # integral of R (left point, consistent with S drift)
        for _ in range(Nt):
            Z = rng.standard_normal((3, n))
            dW1 = (L[0, 0] * Z[0]) * sqdt
            dW2 = (L[1, 0] * Z[0] + L[1, 1] * Z[1]) * sqdt
            dW3 = (L[2, 0] * Z[0] + L[2, 1] * Z[1] + L[2, 2] * Z[2]) * sqdt
            Vp = np.maximum(V, 0.0)
            sV = np.sqrt(Vp)
            X += (R - 0.5 * Vp) * dt + sV * dW1
            I += R * dt
            V = V + kappa * (theta - Vp) * dt + sigma * sV * dW2
            R = R + a_r * (b_r - R) * dt + sig_r * dW3
        D = np.exp(-I)
        eX = np.exp(X)
        for j, s0 in enumerate(spots):
            pay_sums[j] += np.sum(D * np.maximum(s0 * eX - K, 0.0))
        S_term[start:start + n] = 100.0 * eX
        V_term[start:start + n] = np.maximum(V, 0.0)
        R_term[start:start + n] = R
        start += n

    prices = pay_sums / num_paths
    terminal = np.column_stack([S_term, V_term, R_term])

    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": {
            "price_S75": float(prices[0]),
            "price_S100": float(prices[1]),
            "price_S125": float(prices[2]),
        },
    }


if __name__ == "__main__":
    out = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(out["functionals"], out["empirical_mean"])
