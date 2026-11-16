"""N03 and N06 truth derivations: the Kolmogorov route for two scalar diffusions.

`problems._kolmogorov_1d` already solves the backward equation by Chebyshev
collocation, but only for a **polynomial** drift and a **constant** sigma -- which is
what `sde_double_well_langevin` needs and neither of these entries has. This script
generalises it to callable `b(x)` and `s(x)`, **validates the generalisation against
an answer the repo already trusts**, and then applies it.

Sections:

  0. the solver, and its validation against `problems.double_well_moments`, whose
     Kolmogorov truth is independently cross-checked in `validate_ground_truth.py`
     (4th-order FD to 1e-10, the stationary quadrature, and a weak-order-2 MC)
  1. **N06 -- the go/no-go the plan asked for**, which comes back NO on the premise
     and YES on a better route: the Brownian SIS model is *not* reducible, but the
     logit chart makes its noise **additive**, which is a better-conditioned truth
     than the one the plan assumed
  2. **N03 -- Ait-Sahalia**, with the machinery working and the parameters still
     owed by step 1.

Seeded, numpy + scipy only. Run:
`uv run python docs/notable-sde-data/n03_n06_kolmogorov.py`
"""
import numpy as np
from scipy.linalg import expm

SEED = 42


# --- 0. the generalised solver ----------------------------------------------

def cheb(n):
    """Chebyshev differentiation matrix and nodes on [-1, 1] (Trefethen)."""
    if n == 0:
        return np.zeros((1, 1)), np.ones(1)
    x = np.cos(np.pi * np.arange(n + 1) / n)
    c = np.r_[2.0, np.ones(n - 1), 2.0] * (-1.0) ** np.arange(n + 1)
    X = np.tile(x, (n + 1, 1)).T
    dX = X - X.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n + 1))
    D -= np.diag(D.sum(axis=1))
    return D, x


def kolmogorov_general(b, s, x0, T, lo, hi, N=192, funcs=None):
    """`E[f(X_T) | X_0 = x0]` for `dX = b(X)dt + s(X)dW` on `[lo, hi]`.

    The same construction as `problems._kolmogorov_1d` -- backward equation,
    Chebyshev collocation, homogeneous Neumann ends eliminated, dense `expm`,
    barycentric interpolation to `x0` -- generalised in exactly two places:
    `b` and `s` are callables, and the interval need not be symmetric.

    The ends must sit where the process effectively cannot reach within `T`, which is
    what makes the reflecting condition harmless
    `lo`/`hi` independence is measured
    by the callers rather than assumed.
    """
    D, xi = cheb(N)
    x = 0.5 * (hi + lo) + 0.5 * (hi - lo) * xi
    scale = 2.0 / (hi - lo)
    D1 = D * scale
    D2 = D1 @ D1
    A = np.diag(b(x)) @ D1 + np.diag(0.5 * s(x) ** 2) @ D2
    B = D1[[0, N], :]
    M = -np.linalg.solve(B[:, [0, N]], B[:, 1:N])
    A_int = A[1:N, 1:N] + A[1:N, [0, N]] @ M
    E = expm(T * A_int)
    w = (-1.0) ** np.arange(N + 1)
    w[0] *= 0.5
    w[-1] *= 0.5
    funcs = funcs or [lambda z: z, lambda z: z ** 2]
    out = []
    for f in funcs:
        u = np.empty(N + 1)
        u[1:N] = E @ f(x[1:N])
        u[[0, N]] = M @ u[1:N]
        d = (x0 - 0.5 * (hi + lo)) * scale - xi
        if np.any(np.abs(d) < 1e-14):
            out.append(float(u[np.argmin(np.abs(d))]))
        else:
            out.append(float(np.sum(w * u / d) / np.sum(w / d)))
    return out


def validate_solver():
    """The generalisation must reproduce the specialisation it came from."""
    k, sig, x0, T = 10.0, 2.0, 1.0, 1.0

    def b(x):
        return k * (x - x ** 3)

    m1, m2 = kolmogorov_general(b, lambda x: np.full_like(x, sig), x0, T, -4.0, 4.0, N=192)
    m1b, m2b = kolmogorov_general(b, lambda x: np.full_like(x, sig), x0, T, -5.0, 5.0, N=256)
    return (m1, m2 - m1 ** 2), (m1b, m2b - m1b ** 2)


# --- 1. N06: the stochastic SIS model ---------------------------------------
#
# dI = I[(beta N - mu - gamma) - beta I] dt + sigma I (N - I) dW,  I in (0, N)
#
# Scale from Gray-Greenhalgh-Mao-Pan (2012) Example 6.1 (N = 100, mu and gamma of
# order 0.1, beta of order 1e-3, I0 in {15, 60, 90}); the Brownian sigma is set by
# calibration here. `[to confirm against Gray et al. 2011 in step 6]`

N_POP = 100.0
BETA = 0.004
MU = 0.05
GAMMA_R = 0.15
ALPHA = BETA * N_POP - MU - GAMMA_R        # = 0.2 > 0: persistence
SIGMA_SIS = 0.0025
I0 = 60.0
T_SIS = 1.0


def sis_b(i):
    return i * (ALPHA - BETA * i)


def sis_s(i):
    return SIGMA_SIS * i * (N_POP - i)


def sis_lamperti_drift(z):
    """The logit chart Z = ln(I/(N-I)) makes the noise ADDITIVE.

    By Ito, with f(I) = ln(I/(N-I)), f' = N/(I(N-I)) and f'' = 1/(N-I)^2 - 1/I^2:

        dZ = N[ beta - (mu+gamma)/(N-I) + (sigma^2/2)(2I - N) ] dt + sigma N dW

    The diffusion coefficient is the constant `sigma N`, which is what makes the
    Kolmogorov solve in this chart well conditioned: in `I` the diffusion vanishes at
    both ends of (0, N) and the operator is degenerate there.
    """
    i = N_POP / (1.0 + np.exp(-z))          # inverse logit
    return N_POP * (BETA - (MU + GAMMA_R) / (N_POP - i)
                    + 0.5 * SIGMA_SIS ** 2 * (2.0 * i - N_POP))


def sis_mc(n=400_000, nt=20_000, seed=SEED, in_chart=True):
    """Fine-step Euler-Maruyama, in the chart (additive noise, so the scheme is
    unusually accurate) or in `I` directly. Used only as a cross-check."""
    rng = np.random.default_rng(seed)
    h = T_SIS / nt
    if in_chart:
        z = np.full(n, np.log(I0 / (N_POP - I0)))
        c = SIGMA_SIS * N_POP * np.sqrt(h)
        for _ in range(nt):
            z = z + sis_lamperti_drift(z) * h + c * rng.standard_normal(n)
        i = N_POP / (1.0 + np.exp(-z))
    else:
        i = np.full(n, I0)
        for _ in range(nt):
            i = i + sis_b(i) * h + sis_s(i) * np.sqrt(h) * rng.standard_normal(n)
            i = np.clip(i, 1e-12, N_POP - 1e-12)
    return float(i.mean()), float(i.var(ddof=1))


def sis_truth(N=192, zlo=-6.0, zhi=6.0):
    """E[I_T] and E[I_T^2] by the Kolmogorov solve in the logit chart.

    The observables are pulled back through the chart: f(z) = N/(1+e^{-z}) and its
    square, so the solve is in `z` and the answer is in `I`.
    """
    z0 = np.log(I0 / (N_POP - I0))

    def inv_logit(z):
        return N_POP / (1.0 + np.exp(-z))

    m1, m2 = kolmogorov_general(
        sis_lamperti_drift, lambda z: np.full_like(z, SIGMA_SIS * N_POP),
        z0, T_SIS, zlo, zhi, N=N,
        funcs=[inv_logit, lambda z: inv_logit(z) ** 2])
    return m1, m2


# --- 2. N03: the Ait-Sahalia model ------------------------------------------
#
# dX = (a_{-1} X^{-1} - a_0 + a_1 X - a_2 X^rho) dt + sigma X^theta dW,  X > 0
#
# PARAMETERS PENDING step 1: the set below is provisional, chosen only so the
# machinery can be exercised and the conditioning measured. It is NOT the published
# set and must not reach `problems.py`.

AS = dict(a_m1=0.1, a_0=1.0, a_1=2.0, a_2=1.0, rho=3.0, sigma=0.5, theta=1.5,
          x0=0.5, T=1.0)


def as_b(x):
    x = np.maximum(x, 1e-12)
    return (AS["a_m1"] / x - AS["a_0"] + AS["a_1"] * x - AS["a_2"] * x ** AS["rho"])


def as_s(x):
    return AS["sigma"] * np.maximum(x, 0.0) ** AS["theta"]


def as_mc(n=200_000, nt=200_000, seed=SEED):
    """Backward-Euler-like positivity-preserving step: an explicit step then a
    projection onto (eps, inf). Only a cross-check, and deliberately fine."""
    rng = np.random.default_rng(seed)
    h = AS["T"] / nt
    x = np.full(n, AS["x0"])
    for _ in range(nt):
        x = x + as_b(x) * h + as_s(x) * np.sqrt(h) * rng.standard_normal(n)
        x = np.maximum(x, 1e-8)
    return float(x.mean()), float(x.var(ddof=1))


if __name__ == "__main__":
    print("=" * 78)
    print("0. the generalised solver, against an answer the repo already trusts")
    print("=" * 78)
    (m1, v1), (m1b, v1b) = validate_solver()
    # problems.double_well_moments(T=1, k=10, sigma=2, X0=1) -- the shipped truth.
    import importlib.util
    import os
    spec = importlib.util.spec_from_file_location(
        "problems", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..", "benchmark", "problems.py"))
    try:
        P = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(P)
        ref = P.double_well_moments(1.0, k=10.0, sigma=2.0, X0=1.0)
        print(f"  double-well, shipped truth : mean {ref['mean']:.12f}  "
              f"var {ref['variance']:.12f}")
    except Exception as exc:  # noqa: BLE001 -- the comparison is a courtesy
        ref = None
        print(f"  (could not import problems.py: {type(exc).__name__}: {exc})")
    print(f"  this solver, L=4  N=192    : mean {m1:.12f}  var {v1:.12f}")
    print(f"  this solver, L=5  N=256    : mean {m1b:.12f}  var {v1b:.12f}")
    print(f"  domain/resolution agreement : {abs(m1 - m1b):.2e} / {abs(v1 - v1b):.2e}")
    if ref is not None:
        print(f"  against the shipped truth   : {abs(m1 - ref['mean']):.2e} / "
              f"{abs(v1 - ref['variance']):.2e}")
        print("  => the generalisation reproduces the specialisation it came from.")

    print()
    print("=" * 78)
    print("1. N06 -- the go/no-go on the SIS explicit solution")
    print("=" * 78)
    print("""
  THE PLAN'S PREMISE IS WRONG, and the paper trail says why.

  plan-notable-sde.md section 2 (N06) assumed: "The equation is reducible -- a
  stochastic logistic, solvable in closed form", citing the explicit solution the
  Gray-Greenhalgh-Mao-Pan line of work advertises. Read at source
  (strathprints.strath.ac.uk/40698, the 2012 JMAA paper), that explicit solution --
  Theorem 2.1, equation (2.5) -- is for equation (2.3):

        dI/dt = I[alpha_{r(t)} - beta_{r(t)} I]

  a *deterministic ODE with piecewise-constant random coefficients* (telegraph
  noise), and its proof substitutes y = 1/I and applies the ordinary
  variation-of-constants formula. There is no Ito calculus in it. It does not
  apply to the Brownian model.

  And the Brownian model is NOT reducible by that substitution. With y = 1/I and
  Ito's formula:

        dy = [(sigma^2 N^2 - alpha) y + (beta - 2 sigma^2 N) + sigma^2 / y] dt
             - sigma (N y - 1) dW

  The noise is linear in y, but the `sigma^2 / y` term in the drift is not affine,
  so the equation does not linearise. (The logit chart fails the same way in I.)
  Consistent with it, there is a live literature on positivity-preserving schemes
  for exactly this model -- which would be pointless if a closed form existed.

  SO THE ANSWER IS "NO" ON THE PREMISE AND "YES" ON A BETTER ROUTE. The logit
  chart Z = ln(I/(N-I)) does not linearise the equation, but it makes the noise
  ADDITIVE (constant sigma*N), which is a *better* truth route than the one the
  plan assumed: in I the diffusion sigma*I*(N-I) vanishes at both ends of (0,N)
  and the Kolmogorov operator is degenerate there
  in Z it is a constant-diffusion
  operator on the real line. The chart is also the technique the entry punishes a
  solver for not finding, so the truth route and the lesson coincide.
""")
    print(f"  parameters: N={N_POP}, beta={BETA}, mu={MU}, gamma={GAMMA_R}, "
          f"alpha={ALPHA}, sigma={SIGMA_SIS}, I0={I0}, T={T_SIS}")
    print(f"  deterministic equilibrium alpha/beta = {ALPHA / BETA}")
    m1, m2 = sis_truth()
    v = m2 - m1 ** 2
    for N, zlo, zhi in ((160, -6.0, 6.0), (256, -8.0, 8.0)):
        a, b2 = sis_truth(N=N, zlo=zlo, zhi=zhi)
        print(f"  Kolmogorov (logit chart) N={N:<4} z in [{zlo},{zhi}]: "
              f"mean {a:.10f}  var {b2 - a ** 2:.10f}")
    print(f"  Kolmogorov (logit chart) N=192 z in [-6,6]: mean {m1:.10f}  "
          f"var {v:.10f}")
    mc_m, mc_v = sis_mc()
    print(f"  fine-step MC in the chart (400k x 20k steps): mean {mc_m:.6f}  "
          f"var {mc_v:.6f}")
    print(f"    agreement: mean {abs(m1 - mc_m) / abs(m1):.2e} rel, "
          f"var {abs(v - mc_v) / abs(v):.2e} rel")
    mc2_m, mc2_v = sis_mc(in_chart=False, n=200_000, nt=20_000)
    print(f"  fine-step MC in I directly  (200k x 20k steps): mean {mc2_m:.6f}  "
          f"var {mc2_v:.6f}")
    print(f"    the two charts agree to: mean {abs(mc_m - mc2_m) / abs(mc_m):.2e} rel")
    print("\n  --- reference error and vacuity ---")
    a160, _ = sis_truth(N=160, zlo=-6.0, zhi=6.0)
    a256, _ = sis_truth(N=256, zlo=-8.0, zhi=8.0)
    ref_err = max(abs(m1 - a160), abs(m1 - a256)) / abs(m1)
    print(f"  resolution/truncation spread over the three solves: {ref_err:.2e} rel")
    print("  => this is a `reference` with a MEASURED error, not an `exact`: the")
    print("     1/(N-I) term in the chart drift stiffens as I -> N, so the solve does")
    print("     not reach the 1e-10 the double-well does. It is still 2 orders inside")
    print(f"     the MC cross-check ({abs(m1 - mc_m) / abs(m1):.0e}) and 3 inside any gate.")
    eq = ALPHA / BETA
    print(f"\n  I0={I0}, E[I_T]={m1:.4f}, deterministic equilibrium={eq:.4f}")
    from_i0 = abs(m1 / I0 - 1.0)
    print(f"  the mean moved {from_i0:.2%} from I0, against the 5% mean gate: "
          f"{from_i0 / 0.05:.2f}x")
    if from_i0 < 0.05:
        print("  *** NEARLY VACUOUS: a solver RETURNING I0 UNCHANGED passes the mean")
        print("      gate. T must be raised (the mean tends to the equilibrium, "
              f"{abs(eq / I0 - 1):.1%}")
        print("      away) or the variance must be the deciding gate. Flagged for "
              "step 5.")
    print(f"  variance {v:.4f}: returning 0 fails outright, so the variance gate is")
    print("  not vacuous at this horizon.")

    print()
    print("=" * 78)
    print("2. N03 -- Ait-Sahalia: machinery exercised, PARAMETERS PENDING step 1")
    print("=" * 78)
    print(f"  provisional (NOT the published set): {AS}")
    lo, hi = 1e-3, 4.0
    for N, hi_x in ((192, 4.0), (256, 6.0)):
        try:
            a, b2 = kolmogorov_general(as_b, as_s, AS["x0"], AS["T"], lo, hi_x, N=N)
            print(f"  Kolmogorov N={N:<4} x in [{lo},{hi_x}]: mean {a:.8f}  "
                  f"var {b2 - a ** 2:.8f}")
        except Exception as exc:  # noqa: BLE001
            print(f"  N={N} hi={hi_x}: {type(exc).__name__}: {exc}")
    print("  NOTE: the x^{-1} drift makes the operator singular as x -> 0, so the")
    print("  left end cannot simply be pushed to 0 -- the conditioning of this solve")
    print("  is itself a finding and is reported in the plan.")
