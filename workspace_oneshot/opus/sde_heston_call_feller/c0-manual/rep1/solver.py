import numpy as np

KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.114
STRIKE = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _erfc(x):
    # Numerical Recipes erfcc, fractional error < 1.2e-7 everywhere
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.5 * z)
    ans = t * np.exp(-z * z - 1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
          t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
          t * (-0.82215223 + t * 0.17087277)))))))))
    return np.where(x >= 0.0, ans, 2.0 - ans)


def _ncdf(x):
    return 0.5 * _erfc(-x / np.sqrt(2.0))


def _bs_call(F, K, sv):
    sv = np.maximum(sv, 1e-12)
    d1 = (np.log(F / K) + 0.5 * sv * sv) / sv
    d2 = d1 - sv
    return F * _ncdf(d1) - K * _ncdf(d2)


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    # Exact CIR transition (noncentral chi-square), valid at and near V = 0
    ekdt = np.exp(-KAPPA * dt)
    c = SIGMA ** 2 * (1.0 - ekdt) / (4.0 * KAPPA)
    df = 4.0 * KAPPA * ETA / SIGMA ** 2

    V = np.full(num_paths, V0, dtype=float)
    IV = np.zeros(num_paths)
    for _ in range(Nt):
        V_prev = V
        nonc = V_prev * ekdt / c
        V = c * rng.noncentral_chisquare(df, nonc)
        IV += 0.5 * (V_prev + V) * dt

    # int sqrt(V) dW2 recovered from the variance dynamics
    M = (V - V0 - KAPPA * ETA * T + KAPPA * IV) / SIGMA
    sv = np.sqrt((1.0 - RHO ** 2) * IV)
    growth = np.exp(RHO * M - 0.5 * RHO ** 2 * IV)

    functionals = {}
    for S0 in SPOTS:
        F = S0 * growth
        C = _bs_call(F, STRIKE, sv)
        # control variate: E[F] = S0 (martingale)
        Fc = F - F.mean()
        var_F = np.mean(Fc * Fc)
        b = np.mean(Fc * (C - C.mean())) / var_F if var_F > 0 else 0.0
        price = float(C.mean() - b * (F.mean() - S0))
        functionals["price_S%d" % int(S0)] = price

    Z = rng.standard_normal(num_paths)
    S = 100.0 * growth * np.exp(-0.5 * sv * sv + sv * Z)

    X = np.column_stack([S, V])
    return {
        "terminal_paths": X,
        "empirical_mean": [float(v) for v in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    r = solve_sde(num_paths=100000, dt=0.002, T=1, seed=42)
    print(r["functionals"], r["empirical_mean"])
