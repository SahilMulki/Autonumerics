"""Heston (r = 0, Feller violated): truncated Milstein for V + log-Euler for S.

V step (Hefter-Herzwurm truncated Milstein, equals Milstein + two truncations):
    V' = max( max(sqrt(V) + 0.5*sigma*dW2, 0)**2 + (kappa*eta - sigma^2/4 - kappa*V)*dt, 0 )
S step: log R += -0.5*V*dt + sqrt(V)*dW1  (V from the start of the step), R = S/S0.

Mean-preserving truncation. Both truncations remove or add mean (the inner max loses
E[((sqrt(V)+sigma*dW2/2)^-)^2], the outer max adds back the clipped negative part), which
biases E[V_T] by -2.4% and S-option prices by -0.5%. Given V_n, the one-step conditional mean
of each truncation is a closed form in Phi/phi of a = sqrt(V_n)/(sigma*sqrt(dt)/2). So:
  1. add the inner-truncation deficit D1(V_n) to the drift,
  2. after the outer max, rescale V_{n+1} by target/(target + E[outer clip gain]) with
     target = V_n + kappa*(eta - V_n)*dt, the exact one-step mean of the Milstein step.
The rescale is a factor in (0, 1], so V >= 0 holds by construction with no extra clip. Only the
conditional law of the update changes by O(dt) terms near zero; strong order is unaffected.
"""
import numpy as np

KAPPA = 2.58
ETA = 0.043
SIGMA = 1.0
RHO = -0.36
V0 = 0.164
S0 = 100.0
K = 100.0
SPOTS = (75.0, 100.0, 125.0)


def _erfc(x):
    # Abramowitz-Stegun 7.1.26, abs error < 1.5e-7 (numpy only, no scipy)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
        + t * (-1.453152027 + t * 1.061405429)))) * np.exp(-z * z)
    return np.where(x >= 0, y, 2.0 - y)


def _Phi(x):
    return 0.5 * _erfc(-x / np.sqrt(2.0))


def _pdf(x):
    return np.exp(-0.5 * x * x) / np.sqrt(2.0 * np.pi)


def _inner_deficit(a, dt):
    # E[ (dt/4) * ((a+Z)^-)^2 ],  Z ~ N(0,1): mean lost by the inner max(.,0)
    return 0.25 * dt * ((a * a + 1.0) * _Phi(-a) - a * _pdf(a))


def _outer_gain(a, e, dt):
    # E[ (max(raw,0) - raw) ] for raw = (dt/4)*((a+Z)^+)^2 - e*dt, e > 0 clipped part
    e = np.maximum(e, 0.0)
    b = 2.0 * np.sqrt(e)
    zh, zl = b - a, -a
    J = ((_Phi(zh) - zh * _pdf(zh)) - (_Phi(zl) - zl * _pdf(zl))
         - 2.0 * a * (_pdf(zh) - _pdf(zl)) + a * a * (_Phi(zh) - _Phi(zl)))
    return e * dt * _Phi(zh) - 0.25 * dt * J


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    # dW: shape (num_paths, Nt, 2), independent channel increments (dZ1, dZ2),
    # already scaled by sqrt(dt).
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
    else:
        Nt = dW.shape[1]
        dt = T / Nt
        rng = None
    sqdt = np.sqrt(dt)
    c_perp = np.sqrt(1.0 - RHO ** 2)
    c0 = KAPPA * ETA - 0.25 * SIGMA ** 2

    logR = np.zeros(num_paths)
    V = np.full(num_paths, V0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    def state(lr, v):
        return np.column_stack([S0 * np.exp(lr), v])

    for n in range(Nt):
        if dW is None:
            Z = sqdt * rng.standard_normal((num_paths, 2))  # one draw per step
            dZ1, dZ2 = Z[:, 0], Z[:, 1]
        else:
            dZ1, dZ2 = dW[:, n, 0], dW[:, n, 1]
        dW1 = dZ1
        dW2 = RHO * dZ1 + c_perp * dZ2

        if obs:
            X_prev = state(logR, V)

        sq = np.sqrt(V)                       # V >= 0 by construction
        logR = logR - 0.5 * V * dt + sq * dW1
        inner = np.maximum(sq + 0.5 * SIGMA * dW2, 0.0)
        a = sq / (0.5 * SIGMA * sqdt)
        drift = c0 - KAPPA * V
        target = V + KAPPA * (ETA - V) * dt
        # corrections are < 1e-9 relative once a > 2*sqrt(e) + 7 (tail of Phi); skip those paths
        near = a < 2.0 * np.sqrt(0.2 + KAPPA * V) + 7.0
        D1 = np.zeros(num_paths)
        gain = np.zeros(num_paths)
        an = a[near]
        D1[near] = _inner_deficit(an, dt)
        gain[near] = _outer_gain(an, -(drift[near] + D1[near] / dt), dt)
        Vn = np.maximum(inner * inner + drift * dt + D1, 0.0)
        V = Vn * (target / (target + gain))

        if obs:
            X_new = state(logR, V)
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(X_prev) + phi(X_new)) * dt

    R = np.exp(logR)
    X = np.column_stack([S0 * R, V])
    functionals = {
        f"price_S{int(s0)}": float(np.mean(np.maximum(s0 * R - K, 0.0)))
        for s0 in SPOTS
    }
    out = {
        "terminal_paths": X,
        "empirical_mean": [float(m) for m in np.mean(X, axis=0)],
        "empirical_variance": [float(v) for v in np.var(X, axis=0, ddof=1)],
        "functionals": functionals,
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
    }
    if obs:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (2_000_000, 500_000):
        res = solve_sde(num_paths=M, dt=0.002, T=1.0, seed=42)
        print(f"M={M} mean={res['empirical_mean']} var={res['empirical_variance']}")
        print("  prices:", res["functionals"])
    dWt = np.sqrt(0.01) * np.random.default_rng(0).standard_normal((1000, 100, 2))
    print("dW hook dt:", solve_sde(1000, 0.002, 1.0, dW=dWt)["dt"])
    print("obs hook shape:", solve_sde(1000, 0.01, 1.0,
          observables={"v": lambda a: a[:, 1]})["path_integrals"]["v"].shape)
