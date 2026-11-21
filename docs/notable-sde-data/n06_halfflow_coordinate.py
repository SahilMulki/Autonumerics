"""N06 (`sde_sis_epidemic`): does the *coordinate of the logistic half-flows* explain the
1.32x that survived §8h's polish cycle?

§8h's accepted solver removed the mean bias by taking the **noise** sub-step in the logit
coordinate, but kept the exact logistic half-flows in `I`, round-tripping `I -> u -> I`
around that sub-step every step. §8g's variant and C0 instead stay in the logit coordinate
for the whole composition. The two are different symmetric compositions, and §8h recorded
the coordinate as the leading *hypothesis* for the residual gap, explicitly not a
measurement. This script is the measurement.

Three variants, identical Brownian increments (paired), at the graded contract
(`num_paths=200000`, `dt=0.02`, `T=0.5`):

  V3  shipped rep 2      half-flows in I,     Milstein noise sub-step in I
  V1  §8h accepted       half-flows in I,     Heun noise sub-step in logit (round-trip)
  V2  C0 / §8g variant   half-flows in logit, Heun noise sub-step in logit (no round-trip)

Reported against the entry's own reference truth, and over `k` seeds rather than one,
because §8g measured that a single seed on this suite reports a lottery (medians are stable
from about k=16).

    uv run python docs/notable-sde-data/n06_halfflow_coordinate.py
    uv run python docs/notable-sde-data/n06_halfflow_coordinate.py --seeds 32
"""

import argparse

import numpy as np

X_0, N_POP, BETA, MU, GAMMA, SIGMA = 90.0, 100.0, 0.5, 20.0, 25.0, 0.03
A = BETA * N_POP - MU - GAMMA          # 5.0
K_END = A / BETA                       # 10.0, the endemic level
C_LAM = 0.5 * SIGMA**2 * N_POP**2      # Ito-correction amplitude in the logit coordinate
S_NOISE = SIGMA * N_POP                # additive noise amplitude there

# The harness's reference truth for this entry (`verify.components[0]`, gt_kind=reference).
EXACT_MEAN, EXACT_VAR = 6.601725075907273, 39.79884248432973
MEAN_TOL, VAR_TOL = 0.05, 0.10         # N06 takes the suite defaults

# The two recorded single-seed results this script must reproduce before it is believed.
RECORDED = {"V3": (0.006529, 0.002524), "V1": (0.0036021193660953424,
                                               0.0072647693448175095)}


def _flow_I(x, h):
    """Exact flow of dI/dt = I*(A - BETA*I) over h, in the I coordinate (expm1 form)."""
    em = np.expm1(A * h)
    return x * (1.0 + em) / (1.0 + (BETA / A) * x * em)


def _flow_u(u, h):
    """The same exact flow, written in u = log(I/(N-I)) -- no round-trip through I."""
    E = np.exp(-A * h)
    log_a = np.log((N_POP - K_END) * (1.0 - E))
    log_b = np.log(K_END * E)
    return np.log(K_END) - np.logaddexp(log_a, log_b - u)


def _heun_u(u, w, dt):
    """Heun on the bounded Ito drift C*tanh(u/2), with additive noise S_NOISE*w."""
    b0 = C_LAM * np.tanh(0.5 * u)
    noise = S_NOISE * w
    u1 = u + b0 * dt + noise
    return u + 0.5 * (b0 + C_LAM * np.tanh(0.5 * u1)) * dt + noise


def v3_shipped(dW, dt):
    """Strang in I, Milstein noise sub-step in I. Rep 2 as shipped."""
    X = np.full(dW.shape[0], X_0)
    for n in range(dW.shape[1]):
        Y = _flow_I(X, 0.5 * dt)
        w = dW[:, n]
        g = SIGMA * Y * (N_POP - Y)
        dg = SIGMA * (N_POP - 2.0 * Y)
        X = _flow_I(Y + g * w + 0.5 * g * dg * (w * w - dt), 0.5 * dt)
    return X


def v1_accepted(dW, dt):
    """Half-flows in I, Heun noise sub-step in logit: I -> u -> I every step (§8h)."""
    X = np.full(dW.shape[0], X_0)
    for n in range(dW.shape[1]):
        Y = _flow_I(X, 0.5 * dt)
        u2 = _heun_u(np.log(Y / (N_POP - Y)), dW[:, n], dt)
        X = _flow_I(N_POP / (1.0 + np.exp(-u2)), 0.5 * dt)
    return X


def v2_logit_flows(dW, dt):
    """Everything in the logit coordinate; I is formed once, at T (C0 / §8g)."""
    u = np.full(dW.shape[0], np.log(X_0 / (N_POP - X_0)))
    for n in range(dW.shape[1]):
        u = _flow_u(u, 0.5 * dt)
        u = _heun_u(u, dW[:, n], dt)
        u = _flow_u(u, 0.5 * dt)
    return N_POP * 0.5 * (1.0 + np.tanh(0.5 * u))


VARIANTS = (("V3", "half-flows I  + Milstein I   (shipped)", v3_shipped),
            ("V1", "half-flows I  + Heun logit   (§8h)    ", v1_accepted),
            ("V2", "half-flows logit + Heun logit (C0)    ", v2_logit_flows))


def grade(X):
    """Relative errors against the reference truth, and the worst moment / its own gate."""
    m = abs(np.mean(X) / EXACT_MEAN - 1.0)
    v = abs(np.var(X, ddof=1) / EXACT_VAR - 1.0)
    return m, v, max(m / MEAN_TOL, v / VAR_TOL)


def run(seed, n, dt, T):
    Nt = max(1, round(T / dt))
    dW = np.sqrt(T / Nt) * np.random.default_rng(seed).standard_normal((n, Nt))
    return {tag: grade(fn(dW, T / Nt)) for tag, _, fn in VARIANTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--paths", type=int, default=200_000)
    args = ap.parse_args()
    dt, T = 0.02, 0.5

    print(f"N06 half-flow coordinate study  (paths={args.paths}, dt={dt}, T={T})\n")

    r42 = run(42, args.paths, dt, T)
    print("seed 42 -- reproduction of the two recorded results")
    ok = True
    for tag, label, _ in VARIANTS:
        m, v, f = r42[tag]
        line = f"  {tag}  {label}  mean {m*100:7.4f}%  var {v*100:7.4f}%  frac {f:.4f}"
        if tag in RECORDED:
            em, ev = RECORDED[tag]
            hit = abs(m - em) < 2e-5 and abs(v - ev) < 2e-5
            ok &= hit
            line += "   [ok]" if hit else f"   MISMATCH (recorded {em*100:.4f}%/{ev*100:.4f}%)"
        print(line)
    print(f"\n  reproduction: {'all recorded values reproduced' if ok else 'FAILED'}\n")

    seeds = list(range(1000, 1000 + args.seeds))
    acc = {tag: [] for tag, _, _ in VARIANTS}
    for sd in seeds:
        for tag, trip in run(sd, args.paths, dt, T).items():
            acc[tag].append(trip)

    print(f"median over k={len(seeds)} independent seeds")
    med = {}
    for tag, label, _ in VARIANTS:
        arr = np.array(acc[tag])
        med[tag] = np.median(arr[:, 2])
        # p90/p10, not max/min: a range statistic grows with k (§8g).
        sp = np.percentile(arr[:, 2], 90) / np.percentile(arr[:, 2], 10)
        print(f"  {tag}  {label}  mean {np.median(arr[:,0])*100:7.4f}%  "
              f"var {np.median(arr[:,1])*100:7.4f}%  frac {med[tag]:.4f}  (p90/p10 {sp:.2f}x)")

    print(f"\n  V1 -> V2 (the coordinate alone): {med['V1']/med['V2']:.2f}x")
    print(f"  V3 -> V1 (§8h's cycle):          {med['V3']/med['V1']:.2f}x")
    print(f"  V3 -> V2 (both together):        {med['V3']/med['V2']:.2f}x")
    print("\n  C0's own recorded seed-42 frac is 0.0550; V2 is the reimplementation of that "
          "scheme,\n  so V2 near 0.055 at seed 42 is the check that this script measures what "
          "it claims.")


if __name__ == "__main__":
    main()
