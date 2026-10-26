import numpy as np


def _norm_cdf(x):
    # Phi(x) = 0.5*erfc(-x/sqrt(2)), erfc via Chebyshev fit (rel. err < 1.2e-7)
    y = -x / np.sqrt(2.0)
    z = np.abs(y)
    t = 1.0 / (1.0 + 0.5 * z)
    poly = (-1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 +
            t * (-0.18628806 + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 +
            t * (-0.82215223 + t * 0.17087277)))))))))
    ans = t * np.exp(-z * z + poly)
    erfc = np.where(y >= 0.0, ans, 2.0 - ans)
    return 0.5 * erfc


def _simulate_chunk(rng, n, Nt, dt, S0, sig0, nu, rho):
    S = np.full(n, S0)
    sig = np.full(n, sig0)
    rc = np.sqrt(1.0 - rho * rho)
    sq = np.sqrt(dt)
    nu2dt = nu * nu * dt
    # E[int sigma^2 ds | sigma_n] = sigma_n^2 * dt * (exp(nu^2 dt)-1)/(nu^2 dt)
    cfac = dt * np.expm1(nu2dt) / nu2dt
    psic = 1.5
    for _ in range(Nt):
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        Z = rho * Z1 + rc * Z2
        v = sig * sig * cfac
        with np.errstate(divide="ignore", invalid="ignore"):
            psi = v / S  # inf where S == 0
        # Quadratic branch (moment-matched: mean S, variance S*v)
        pl = np.minimum(psi, psic)
        tt = 2.0 / pl
        b2 = tt - 1.0 + np.sqrt(tt * (tt - 1.0))
        a = S / (1.0 + b2)
        Snew = a * (np.sqrt(b2) + Z) ** 2
        # Exponential branch with point mass at zero (absorption)
        idx = np.nonzero((psi > psic) & (S > 0.0))[0]
        if idx.size:
            ps = psi[idx]
            Si = S[idx]
            p = (ps - 1.0) / (ps + 1.0)
            beta = (1.0 - p) / Si
            tail = _norm_cdf(-Z[idx])  # 1 - U
            q = 1.0 - p
            with np.errstate(divide="ignore", invalid="ignore"):
                val = np.log(q / tail) / beta
            Snew[idx] = np.where(tail >= q, 0.0, val)
        S = Snew
        sig = sig * np.exp(nu * sq * Z1 - 0.5 * nu2dt)
    return S, sig


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    S0 = 0.07
    sig0 = 0.34
    nu = 0.8
    rho = -0.6

    rng = np.random.default_rng(seed)
    chunk = 250000
    S_all = np.empty(num_paths)
    sig_all = np.empty(num_paths)
    start = 0
    while start < num_paths:
        n = min(chunk, num_paths - start)
        S, sig = _simulate_chunk(rng, n, Nt, dt, S0, sig0, nu, rho)
        S_all[start:start + n] = S
        sig_all[start:start + n] = sig
        start += n

    terminal = np.column_stack([S_all, sig_all])
    K1 = 0.051022538988
    K2 = 0.07
    K3 = 0.096035989138
    functionals = {
        "price_K1": float(np.mean(np.maximum(S_all - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S_all - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S_all - K3, 0.0))),
    }
    return {
        "terminal_paths": terminal,
        "empirical_mean": [float(x) for x in terminal.mean(axis=0)],
        "empirical_variance": [float(x) for x in terminal.var(axis=0, ddof=1)],
        "functionals": functionals,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=100000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["empirical_variance"], res["functionals"])
