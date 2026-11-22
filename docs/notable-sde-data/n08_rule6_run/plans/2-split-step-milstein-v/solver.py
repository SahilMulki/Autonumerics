"""Strang splitting: exact logistic drift half-flows + exact Stratonovich noise flow (Z=V^-1/2).

dV = kappa*V*(theta - V) dt + epsilon*V**1.5 dW,  V(0) = 0.15
"""
import numpy as np

KAPPA = 22.84
THETA = 0.218
EPS = 8.56
V0 = 0.15


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    if dW is None:
        Nt = max(1, round(T / dt))
        dt = T / Nt
        rng = np.random.default_rng(seed)
        dW = np.sqrt(dt) * rng.standard_normal((num_paths, Nt))
    else:
        Nt = dW.shape[1]
        dt = T / Nt

    # Stratonovich form: dV = [r V - c V^2] dt + eps V^1.5 o dW, r = kappa*theta,
    # c = kappa + 0.75 eps^2.  Both vector fields have closed-form flows:
    #   drift : logistic, linear in u = 1/V :  u' = c - r u
    #   noise : Z = V^-1/2 obeys dZ = -(eps/2) o dW (additive)
    # The step consumes dW[:, n] once; no transition law is sampled.
    r = KAPPA * THETA
    c = KAPPA + 0.75 * EPS * EPS
    a_h = np.exp(-0.5 * r * dt)
    u_inf = c / r

    V = np.full(num_paths, V0, dtype=float)
    obs = observables or {}
    acc = {k: np.zeros(num_paths) for k in obs}

    u = 1.0 / V
    for n in range(Nt):
        V_prev = 1.0 / u
        u = u_inf + (u - u_inf) * a_h                  # drift half-flow
        z = np.sqrt(u) - 0.5 * EPS * dW[:, n]          # noise flow, full dt
        u = z * z
        u = u_inf + (u - u_inf) * a_h                  # drift half-flow
        if obs:
            V = 1.0 / u
            for k, phi in obs.items():
                acc[k] += 0.5 * (phi(V_prev) + phi(V)) * dt
    V = 1.0 / u

    M = num_paths
    inv = 1.0 / V
    m1 = float(np.mean(inv))
    s2 = float(np.var(inv, ddof=1))
    c = inv - m1
    m4 = float(np.mean(c ** 4))
    var_se = float(np.sqrt(max(m4 - s2 * s2 * (M - 3) / (M - 1), 0.0) / M))

    out = {
        "terminal_paths": V,
        "empirical_mean": float(np.mean(V)),
        "empirical_variance": float(np.var(V, ddof=1)),
        "num_paths": num_paths,
        "dt": dt,
        "T": T,
        "functionals": {"inv_var_mean": m1, "inv_var_variance": s2},
        "functional_se": {"inv_var_mean": float(np.std(inv, ddof=1) / np.sqrt(M)),
                          "inv_var_variance": var_se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for M in (200000, 50000):
        res = solve_sde(num_paths=M, dt=0.004, T=0.1, seed=42)
        print(f"M={M}")
        print(f"Empirical mean:     {res['empirical_mean']:.6f}")
        print(f"Empirical variance: {res['empirical_variance']:.6f}")
        print("functionals:", res["functionals"], "se:", res["functional_se"])
        print("min V:", res["terminal_paths"].min(), "max V:", res["terminal_paths"].max())
    dWt = np.sqrt(0.004) * np.random.default_rng(0).standard_normal((1000, 25))
    print("dW hook dt:", solve_sde(1000, 0.004, 0.1, dW=dWt)["dt"])
    print("obs hook:", solve_sde(1000, 0.004, 0.1,
                                 observables={"inv": lambda a: 1.0 / a})["path_integrals"]["inv"].shape)
