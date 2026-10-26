import numpy as np

S0 = 0.07
SIG0 = 0.34
NU = 0.8
RHO = -0.6
STRIKES = {
    "price_K1": 0.051022538988,
    "price_K2": 0.07,
    "price_K3": 0.096035989138,
}


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    """
    SABR (beta = 1/2) simulation.

    sigma : exact lognormal update.
    S     : splitting scheme in X = 2 sqrt(S) coordinates.
            - correlated part: X shifted by rho * (sigma_{n+1} - sigma_n) / nu
              (exact, since sigma dW_sigma = dsigma / nu), with its Ito drift
              -rho^2 sigma^2 dt / 4 applied in S; crossing zero => absorbed.
            - orthogonal part: exact squared-Bessel (dimension 0) transition
              (Poisson-mixed Gamma), which absorbs at 0 exactly; a normal
              approximation is used only when the Poisson mean is very large.
    Prices use a control variate on S(T) (E[S(T)] = S0 exactly).
    """
    rng = np.random.default_rng(seed)
    Nt = max(1, round(T / dt))
    dt = T / Nt

    rho = RHO
    rbar2 = 1.0 - rho * rho
    nu = NU
    sq = np.sqrt(dt)
    sig_drift = -0.5 * nu * nu * dt
    shift_c = rho / (2.0 * nu)
    lam_big = 400.0

    S = np.full(num_paths, S0, dtype=float)
    sig = np.full(num_paths, SIG0, dtype=float)
    alive = np.arange(num_paths)

    for _ in range(Nt):
        Z1 = rng.standard_normal(num_paths)
        sig_new = sig * np.exp(sig_drift + nu * sq * Z1)

        if alive.size:
            s0 = sig[alive]
            s1 = sig_new[alive]
            Sa = S[alive]

            v = 0.5 * (s0 * s0 + s1 * s1) * dt          # ~ int sigma^2 dt
            Xt = np.sqrt(Sa) + shift_c * (s1 - s0)        # half of shifted X
            m = Xt * Xt - 0.25 * rho * rho * v
            m = np.where(Xt > 0.0, np.maximum(m, 0.0), 0.0)

            tau = 0.25 * rbar2 * v
            lam = m / tau

            Snew = np.zeros_like(m)
            big = lam > lam_big
            nb = int(np.count_nonzero(big))
            if nb:
                mb = m[big]
                Z2 = rng.standard_normal(nb)
                Snew[big] = np.maximum(mb + np.sqrt(4.0 * mb * tau[big]) * Z2, 0.0)
            small = ~big
            if np.any(small):
                lam_s = lam[small]
                tau_s = tau[small]
                N = rng.poisson(0.5 * lam_s)
                vals = np.zeros_like(lam_s)
                pos = N > 0
                if np.any(pos):
                    vals[pos] = 2.0 * tau_s[pos] * rng.standard_gamma(N[pos].astype(float))
                Snew[small] = vals

            S[alive] = Snew
            alive = alive[Snew > 0.0]

        sig = sig_new

    terminal = np.column_stack((S, sig))

    functionals = {}
    S_mean = float(np.mean(S))
    S_var = float(np.var(S, ddof=1))
    for name, K in STRIKES.items():
        P = np.maximum(S - K, 0.0)
        Pm = float(np.mean(P))
        if S_var > 0.0:
            cov = float(np.mean((P - Pm) * (S - S_mean))) * num_paths / (num_paths - 1)
            b = cov / S_var
            price = Pm - b * (S_mean - S0)
        else:
            price = Pm
        functionals[name] = float(price)

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
    res = solve_sde(num_paths=20000, dt=0.004, T=10, seed=42)
    print(res["empirical_mean"], res["empirical_variance"])
    print(res["functionals"])
