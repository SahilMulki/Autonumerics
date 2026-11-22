"""Log-exact stepping of ln S + Brownian-bridge-conditioned geometric average (antithetic)."""
import math
import numpy as np

S_0, R, SIGMA, K = 100.0, 0.05, 0.2, 100.0
_erfc = np.vectorize(math.erfc, otypes=[float])


def _Phi(x):
    return 0.5 * _erfc(-np.asarray(x, dtype=float) / math.sqrt(2.0))


def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42,
              dW: np.ndarray | None = None,
              observables: dict | None = None) -> dict:
    antithetic = False
    if dW is None:
        rng = np.random.default_rng(seed)
        Nt = max(1, round(T / dt))
        dt = T / Nt
        half = (num_paths + 1) // 2
        Z = rng.standard_normal((half, Nt))
        dW = np.sqrt(dt) * np.concatenate([Z, -Z], axis=0)[:num_paths]
        antithetic = True
    else:
        dW = np.asarray(dW, dtype=float)
        num_paths, Nt = dW.shape[0], dW.shape[1]
        dt = T / Nt

    drift = (R - 0.5 * SIGMA ** 2) * dt
    L = np.full(num_paths, math.log(S_0))
    G = np.zeros(num_paths)                      # trapezoid average of ln S
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    S = np.exp(L)
    for n in range(Nt):
        L_prev, S_prev = L, S
        L = L_prev + drift + SIGMA * dW[:, n]
        S = np.exp(L)
        G += 0.5 * dt * (L_prev + L)
        for k, phi in (observables or {}).items():
            acc[k] += 0.5 * (phi(S_prev) + phi(S)) * dt
    G /= T

    # integrate out the within-step Brownian-bridge integral: ln A | G ~ N(G, v)
    v = SIGMA ** 2 * dt ** 2 / (12.0 * T)
    sv = math.sqrt(v)
    dd1 = (G - math.log(K) + v) / sv
    dd2 = dd1 - sv
    EA = np.exp(G + 0.5 * v)
    call = EA * _Phi(dd1) - K * _Phi(dd2)        # E[(A-K)^+ | G]
    EA2 = np.exp(2 * G + 2 * v)
    disc = math.exp(-R * T)

    n = num_paths
    if antithetic and n % 2 == 0:
        P = n // 2
        pc = 0.5 * (call[:P] + call[P:])
        pa = 0.5 * (EA[:P] + EA[P:])
        pb = 0.5 * (EA2[:P] + EA2[P:])
    else:
        P, pc, pa, pb = n, call, EA, EA2
    call_price = disc * float(np.mean(call))
    call_se = disc * float(np.std(pc, ddof=1) / math.sqrt(P))
    ma, mb = float(np.mean(EA)), float(np.mean(EA2))
    geo_var = (n / (n - 1.0)) * (mb - ma ** 2)
    infl = pb - 2.0 * ma * pa
    var_se = (n / (n - 1.0)) * float(np.std(infl, ddof=1) / math.sqrt(P))

    out = {
        "terminal_paths": S,
        "empirical_mean": float(np.mean(S)),
        "empirical_variance": float(np.var(S, ddof=1)),
        "num_paths": int(num_paths),
        "dt": float(dt),
        "T": float(T),
        "functionals": {"call_price": call_price, "geo_average_variance": geo_var},
        "functional_se": {"call_price": call_se, "geo_average_variance": var_se},
    }
    if observables:
        out["path_integrals"] = acc
    return out


if __name__ == "__main__":
    for npaths in (400000, 100000):
        r = solve_sde(npaths, 0.04, 1.0, 42)
        print(npaths, "mean", r["empirical_mean"], "var", r["empirical_variance"], "dt", r["dt"])
        print("  functionals", r["functionals"], "se", r["functional_se"])
    rng = np.random.default_rng(1)
    dW = np.sqrt(1 / 50) * rng.standard_normal((20000, 50))
    r = solve_sde(20000, 0.04, 1.0, dW=dW, observables={"X": lambda x: x})
    print("dW run: dt", r["dt"], "mean", r["empirical_mean"], r["functionals"],
          "int X", r["path_integrals"]["X"].mean())
