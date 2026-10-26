import numpy as np

S0 = 0.07
SIG0 = 0.34
ALPHA = 0.8
RHO = -0.6
STRIKES = {
    "price_K1": 0.051022538988,
    "price_K2": 0.07,
    "price_K3": 0.096035989138,
}
# Below this Poisson mean (mu = 2 S / v) the exact absorbing transition is sampled;
# above it the absorption probability exp(-mu) is negligible and a quadratic
# (Andersen QE) moment-matched sample is used.
MUC = 20.0


def _simulate_chunk(n, Nt, dt, rng):
    rhobar2 = 1.0 - RHO * RHO
    sq = np.sqrt(dt)
    sig_drift = -0.5 * ALPHA * ALPHA * dt
    sig_vol = ALPHA * sq
    corr_coef = RHO / ALPHA

    S = np.full(n, S0, dtype=float)
    sig = np.full(n, SIG0, dtype=float)

    for _ in range(Nt):
        Z = rng.standard_normal((2, n))
        # exact lognormal step for sigma
        sig_new = sig * np.exp(sig_vol * Z[0] + sig_drift)
        # correlated part: rho * sqrt(S) * int sigma dW_sigma, with int sigma dW_sigma = dsigma / alpha
        Sp = S + corr_coef * np.sqrt(S) * (sig_new - sig)
        np.maximum(Sp, 0.0, out=Sp)
        # uncorrelated part: exact zero-dimensional squared-Bessel (absorbing CEV beta=1/2)
        v = rhobar2 * 0.5 * (sig * sig + sig_new * sig_new) * dt
        mu = 2.0 * Sp / v
        mq = np.maximum(mu, MUC)
        b2 = mq - 1.0 + np.sqrt(mq * (mq - 1.0))
        Snew = Sp / (1.0 + b2) * (np.sqrt(b2) + Z[1]) ** 2

        small = mu < MUC
        if small.any():
            Snew[small] = 0.0
            idx = np.flatnonzero(small & (Sp > 0.0))
            if idx.size:
                N = rng.poisson(mu[idx])
                m = N > 0
                if m.any():
                    ii = idx[m]
                    Snew[ii] = 0.5 * v[ii] * rng.gamma(N[m].astype(float))
        S = Snew
        sig = sig_new
    return S, sig


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S_all = np.empty(num_paths, dtype=float)
    sig_all = np.empty(num_paths, dtype=float)
    chunk = 100000
    for start in range(0, num_paths, chunk):
        n = min(chunk, num_paths - start)
        S, sig = _simulate_chunk(n, Nt, dt, rng)
        S_all[start:start + n] = S
        sig_all[start:start + n] = sig

    X = np.column_stack([S_all, sig_all])

    # Call prices with martingale control variate (E[S_T] = S0 exactly)
    Sm = S_all.mean()
    Sc = S_all - Sm
    varS = np.mean(Sc * Sc)
    functionals = {}
    for name, K in STRIKES.items():
        pay = np.maximum(S_all - K, 0.0)
        pm = pay.mean()
        if varS > 0:
            beta = np.mean((pay - pm) * Sc) / varS
        else:
            beta = 0.0
        functionals[name] = float(pm - beta * (Sm - S0))
    functionals["raw_mean_S"] = float(Sm)
    functionals["absorbed_fraction"] = float(np.mean(S_all == 0.0))

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
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["empirical_variance"])
    print(res["functionals"])
