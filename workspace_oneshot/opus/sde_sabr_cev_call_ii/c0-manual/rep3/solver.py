import numpy as np

RHO = -0.6
NU = 0.8
S0 = 0.07
SIG0 = 0.4
STRIKES = {"price_K1": 0.051022538988, "price_K2": 0.07, "price_K3": 0.096035989138}
PSI_C = 1.5


def _norm_sf(z):
    """Upper tail 1 - Phi(z), Abramowitz-Stegun 7.1.26 erfc approximation."""
    x = np.abs(z) / np.sqrt(2.0)
    t = 1.0 / (1.0 + 0.3275911 * x)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    erfc = poly * np.exp(-x * x)
    half = 0.5 * erfc
    return np.where(z >= 0, half, 1.0 - half)


def _simulate_chunk(n, Nt, dt, rng):
    S = np.full(n, S0)
    sig = np.full(n, SIG0)
    rhob = np.sqrt(1.0 - RHO * RHO)
    sqdt = np.sqrt(dt)
    sdrift = -0.5 * NU * NU * dt
    for _ in range(Nt):
        Z1 = rng.standard_normal(n)
        Z2 = rng.standard_normal(n)
        Zs = RHO * Z1 + rhob * Z2
        # Andersen QE step for dS = sig sqrt(S) dW with sig frozen over the step:
        # m = S, s^2 = sig^2 S dt, psi = sig^2 dt / S. Exact martingale, absorbing at 0.
        v = sig * sig * dt
        Spos = np.maximum(S, 1e-300)
        psi = v / Spos
        psic = np.minimum(psi, PSI_C)
        tq = 2.0 / psic
        b2 = tq - 1.0 + np.sqrt(tq * (tq - 1.0))
        a = S / (1.0 + b2)
        Snew = a * (np.sqrt(b2) + Zs) ** 2
        lo = np.flatnonzero((psi > PSI_C) & (S > 0.0))
        if lo.size:
            ps = psi[lo]
            p = (ps - 1.0) / (ps + 1.0)
            m = S[lo]
            beta = (1.0 - p) / m
            sf = _norm_sf(Zs[lo])            # 1 - U
            sf = np.maximum(sf, 1e-300)
            val = np.log((1.0 - p) / sf) / beta
            Snew[lo] = np.where(sf >= 1.0 - p, 0.0, np.maximum(val, 0.0))
        S = Snew
        # exact lognormal volatility
        sig = sig * np.exp(NU * sqdt * Z1 + sdrift)
    return S, sig


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    chunk = 100000
    S_all = np.empty(num_paths)
    sig_all = np.empty(num_paths)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        start = 0
        while start < num_paths:
            n = min(chunk, num_paths - start)
            S, sig = _simulate_chunk(n, Nt, dt, rng)
            S_all[start:start + n] = S
            sig_all[start:start + n] = sig
            start += n

    functionals = {}
    meanS = float(np.mean(S_all))
    varS = float(np.var(S_all, ddof=1))
    for name, K in STRIKES.items():
        pay = np.maximum(S_all - K, 0.0)
        price = float(np.mean(pay))
        # martingale control variate: E[S(T)] = S0
        if varS > 0:
            cov = float(np.mean((pay - price) * (S_all - meanS))) * num_paths / (num_paths - 1)
            b = cov / varS
            price_cv = price - b * (meanS - S0)
        else:
            price_cv = price
        functionals[name] = float(price_cv)
    functionals["price_K1_raw_mean_S"] = meanS

    X = np.column_stack([S_all, sig_all])
    return {
        "terminal_paths": X,
        "empirical_mean": [float(np.mean(X[:, 0])), float(np.mean(X[:, 1]))],
        "empirical_variance": [float(np.var(X[:, 0], ddof=1)), float(np.var(X[:, 1], ddof=1))],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }


if __name__ == "__main__":
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["functionals"])
