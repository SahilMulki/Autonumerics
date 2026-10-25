import numpy as np


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict:
    # SABR with beta=1/2, rho=0:
    #   dS = sigma sqrt(S) dW_S,  dsigma = nu sigma dW_sigma  (independent)
    # sigma is simulated exactly (log-normal steps). Given the sigma path,
    # S is a time-changed squared Bessel process of dimension 0 (BESQ^0) with
    # clock tau = int_0^T sigma^2/4 dt, absorbed at zero. Its exact law:
    #   N ~ Poisson(S0 / (2 tau)),  S(T) = 2 tau * Gamma(N, 1)   (S(T)=0 if N=0)
    # This preserves absorption at 0 and the martingale property exactly;
    # the only discretisation is trapezoidal quadrature of int sigma^2 dt.
    S0, sig0, nu = 0.5, 0.5, 0.4
    Nt = max(1, round(T / dt))
    dt = T / Nt
    rng = np.random.default_rng(seed)

    S_T = np.empty(num_paths)
    sig_T = np.empty(num_paths)

    chunk = 1_000_000
    sqdt = np.sqrt(dt)
    drift = -0.5 * nu * nu * dt
    for start in range(0, num_paths, chunk):
        m = min(chunk, num_paths - start)
        sig = np.full(m, sig0)
        v_prev = sig * sig
        integ = np.zeros(m)
        for _ in range(Nt):
            z = rng.standard_normal(m)
            sig = sig * np.exp(nu * sqdt * z + drift)
            v = sig * sig
            integ += 0.5 * (v_prev + v) * dt
            v_prev = v
        tau = np.maximum(0.25 * integ, 1e-300)
        N = rng.poisson(S0 / (2.0 * tau))
        g = rng.gamma(np.maximum(N, 1).astype(float), 1.0)
        S = np.where(N > 0, 2.0 * tau * g, 0.0)
        S_T[start:start + m] = S
        sig_T[start:start + m] = sig

    K1, K2, K3 = 0.434061722697, 0.5, 0.575954955084
    functionals = {
        "price_K1": float(np.mean(np.maximum(S_T - K1, 0.0))),
        "price_K2": float(np.mean(np.maximum(S_T - K2, 0.0))),
        "price_K3": float(np.mean(np.maximum(S_T - K3, 0.0))),
    }

    X = np.column_stack((S_T, sig_T))
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
    r = solve_sde(num_paths=200000, dt=0.005, T=2, seed=42)
    print(r["empirical_mean"], r["empirical_variance"])
    print(r["functionals"])
