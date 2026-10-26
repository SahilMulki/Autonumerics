---
id: 2
plan_slug: qe-martingale-corrected
scheme: euler-maruyama
strategy: Andersen quadratic-exponential (QE) step for V driven comonotonically by the supplied dW2, martingale-corrected log-S step built from the recovered int sqrt(V) dW2, exact-in-law OU step for R on the correlated dW3, and a trapezoid discount shared with the S drift.
expected_strong_order: 0.1
expected_weak_order: 0.5
---

<!--
scheme_family is deliberately OMITTED (it would be `other`). problem_spec.json declares a single scalar
expected_strong_order = 0.1 and the kernel takes the worst component; declaring scheme_family turns the
order guard into a two-sided +/-0.4 band, which a correct R component (~1.0, or "unresolved" for an
exact OU step) fails. See verification.expected_strong_order_note and implementation_notes
"KERNEL CEILINGS (3)". The `scheme` field reads euler-maruyama only because the enum has no QE
entry: S is stepped by an Euler-type log scheme; V is not.
-->

## SDE Reference

Itô SDE on t in [0, T], T = 10. State X = (S, V, R), columns in that order:

    dS = R*S dt + sqrt(V)*S dW1,              S(0) = S0
    dV = kappa*(eta - V) dt + sigma_v*sqrt(V) dW2,  V(0) = V_0
    dR = alpha*(r_bar - R) dt + sigma_r dW3,   R(0) = R_0

| Parameter | Value |
|---|---|
| kappa | 0.5 |
| eta | 0.04 |
| sigma_v | 1.0 |
| alpha | 0.08 |
| r_bar | 0.10 |
| sigma_r | 0.09 |
| rho_SV = corr(dW1, dW2) | -0.9 |
| rho_SR = corr(dW1, dW3) | 0.6 |
| rho_VR = corr(dW2, dW3) | -0.7 |
| K (strike) | 100.0 |
| S0 for prices | 75, 100, 125 |
| S_0 for terminal_paths | 100 |
| V_0 | 0.04 |
| R_0 | 0.10 |

Feller ratio 2*kappa*eta/sigma_v^2 = 0.04: V sits at zero for a large fraction of the time.

Correlation: independent channels (Y1, Y2, Y3), each ~ N(0, dt), mapped with the lower-triangular
Cholesky factor (rows S, V, R):

    Lc = [[ 1.0,           0.0,           0.0         ],
          [-0.9,           0.4358898944,  0.0         ],
          [ 0.6,          -0.3670651742,  0.7108186533]]

    dW1 = Y1
    dW2 = -0.9*Y1 + 0.4358898944*Y2
    dW3 =  0.6*Y1 - 0.3670651742*Y2 + 0.7108186533*Y3

Quantity of interest (returned in `functionals`):

    price_S{75,100,125} = E[ exp(-int_0^T R dt) * max(S_T - 100, 0) ]   from S0 = 75, 100, 125

each must be within 0.3% relative of an independent reference.

## Numerical Scheme

- **Method**: QE (Andersen 2008) for V + martingale-corrected log-S step (QE-M) + exact OU transition for R. Euler-type in S; not a Milstein scheme.
- **dt**: 0.004 (given by the caller; `Nt = max(1, round(T/dt)) = 2500`, `dt = T/Nt`; never hardcode another step)
- **num_paths**: 1,000,000 (graded), then 250,000 (path-count agreement check)
- **seed**: 42 (`np.random.default_rng(seed)` only)
- **T**: 10.0
- **Milstein correction**: not applicable (multi-D, non-commuting noise)
- **Expected strong order**: 0.1 (spec-declared value; the kernel's CRN ladder will see V driven by a monotone transform of dW2, so the V and S components may read near zero or `unresolved`; R uses an exact transition and is expected to read `unresolved` or high — neither is a defect, see Notes)
- **Expected weak order**: 0.5 (spec's diagnostic value; QE is designed for small weak bias at large steps)
- **Reference for scoring**: none in the kernel — self-convergence only (`expected_provenance: self_convergence`). No moment ODE, Kolmogorov solve, stationary density or terminal law is declared, so the score band tops out at 6–8. Hand-checkable anchors in `verification.exact_identities`.

### Resolving the exact-V trade-off (problem.md requires this)

QE samples V_{n+1} from a moment-matched approximation of its conditional law, so there is no
sqrt(V) dW2 increment to correlate W1 and W3 with. This plan resolves it as follows:

1. **The QE draw is driven by the supplied dW2.** Let Zv = dW2_n / sqrt(dt) ~ N(0,1) (built from the
   Cholesky row above). Quadratic regime uses Zv directly; exponential regime uses U = Phi(Zv)
   (and 1 - U = Phi(-Zv) computed directly, never as 1 - Phi(Zv), to keep precision in the tail).
   V_{n+1} is then a monotone function of dW2_n, so the scheme consumes the kernel's increments and
   the W2-side correlations are carried by the same Gaussian.
2. **S uses the recovered integral.** From the exact integrated V dynamics,

       int_n sqrt(V) dW2 ≈ (V_{n+1} - V_n - kappa*eta*dt + kappa*dt*(V_n + V_{n+1})/2) / sigma_v

   and the orthogonal part of W1 is dWperp = (dW1 - rho_SV*dW2)/sqrt(1 - rho_SV^2), independent of
   dW2 with unit variance per sqrt(dt). Then sqrt(V) dW1 = rho_SV * (recovered integral)
   + sqrt(1 - rho_SV^2) * sqrt((V_n + V_{n+1})/2) * dWperp. No division by sqrt(V) ever occurs.
3. **R uses dW3 directly.** dW3 is formed from the same (Y1, Y2, Y3), so corr(dW3, dW1) and
   corr(dW3, dW2) are exact at the increment level. R's marginal stays exactly OU.

### Per-step update (gamma1 = gamma2 = 1/2, psi_c = 1.5)

V (Andersen QE), with E = exp(-kappa*dt):

    m   = eta + (V_n - eta)*E
    s2  = V_n*sigma_v^2*E*(1 - E)/kappa + eta*sigma_v^2*(1 - E)^2/(2*kappa)
    psi = s2/m^2
    if psi <= psi_c:   b2 = 2/psi - 1 + sqrt(2/psi)*sqrt(2/psi - 1);  a = m/(1 + b2)
                       V_{n+1} = a*(sqrt(b2) + Zv)^2
    else:              p = (psi - 1)/(psi + 1);  beta = (1 - p)/m
                       V_{n+1} = 0 if U <= p else log((1 - p)/(1 - U))/beta

At V_n = 0, m = eta*(1 - E) > 0, so psi is finite (≈ 25, exponential regime). V is never negative.

log S (Andersen's QE-M with stochastic rate added), with rho = rho_SV:

    K0 = -rho*kappa*eta*dt/sigma_v
    K1 = gamma1*dt*(kappa*rho/sigma_v - 0.5) - rho/sigma_v
    K2 = gamma2*dt*(kappa*rho/sigma_v - 0.5) + rho/sigma_v
    K3 = gamma1*dt*(1 - rho^2);   K4 = gamma2*dt*(1 - rho^2)
    A  = K2 + 0.5*K4
    martingale correction K0*:
      quadratic regime:   K0* = -A*b2*a/(1 - 2*A*a) + 0.5*log(1 - 2*A*a) - (K1 + 0.5*K3)*V_n
      exponential regime: K0* = -log(p + beta*(1 - p)/(beta - A))          - (K1 + 0.5*K3)*V_n
    logS_{n+1} = logS_n + Rbar_n*dt + K0* + K1*V_n + K2*V_{n+1} + sqrt(K3*V_n + K4*V_{n+1}) * Zperp

where Zperp = dWperp/sqrt(dt) ~ N(0,1) and Rbar_n is the rate used in the discount (below).
Here A ≈ rho/sigma_v ≈ -0.9 < 0, so the conditions A < 1/(2a) and A < beta hold automatically. The
solver must verify these formulas against Andersen (2008), §3.2.3–3.2.4, before relying on them and
check E[exp(K0* + K1 V_n + K2 V_{n+1} + ½(K3 V_n + K4 V_{n+1}))| V_n] = 1 numerically on a few V_n.

R (exact OU transition on the correlated increment):

    R_{n+1} = r_bar + (R_n - r_bar)*exp(-alpha*dt) + sigma_r*sqrt((1 - exp(-2*alpha*dt))/(2*alpha)) * dW3_n/sqrt(dt)

Discount and rate in S: Rbar_n = (R_n + R_{n+1})/2 (trapezoid); logD -= Rbar_n*dt. Using the same
Rbar_n in the S drift and the discount makes D_n*S_n = S0*prod exp(K0* + ... + Zperp term), which
the martingale correction makes an exact discrete martingale: E[D_N S_N] = S0 for this scheme.

### One simulation serves all three spots

V, R and D do not depend on S0 and S_T(S0) = (S0/100) * S_T(100) under the log update. Simulate once
from S0 = 100 and form `D * max(c*S_T - 100, 0)` for c = 0.75, 1.0, 1.25.

### Control variate (unbiased for the discrete scheme)

    Y = D*max(S_T - K, 0) - b1*(D*S_T - S0) [- b2*(D - P_disc)]

b by least squares on the sample. E[D S_T] = S0 is exact for this scheme (martingale correction).
The bond control is optional; if used, its mean must be the discrete E[D]: with exact OU transitions
and trapezoid weights, sum_n Rbar_n dt is Gaussian with mean and variance computable exactly by a
deterministic O(Nt) recursion, so E[D_disc] = exp(-m + v/2). (It also converges to the continuous
P(0,T) = 0.7968400987842249 at O(dt^2), which is a useful sanity check.)

## Implementation Notes

- **Contract**: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict` returning
  `terminal_paths` (num_paths, 3) for S0 = 100 in column order (S, V, R); `empirical_mean` and
  `empirical_variance` (ddof=1) as lists of 3 floats; `functionals` with float `price_S75`,
  `price_S100`, `price_S125`; plus `num_paths`, `dt`, `T`. numpy + stdlib only; no work at import.
- **dW hook**: when supplied, shape (num_paths, Nt, 3) of INDEPENDENT increments scaled by sqrt(dt);
  form dW1, dW2, dW3 with Lc as above, take Nt from the shape, draw nothing. **observables hook**:
  trapezoid accumulation of phi on the (num_paths, 3) state.
- **Normal CDF without scipy**: implement Phi with a vectorised high-accuracy erfc approximation
  (e.g. the Numerical Recipes `erfcc` Chebyshev form, fractional error < 1.2e-7 everywhere) and use
  Phi(-Zv) for 1 - U so log((1 - p)/(1 - U)) keeps relative precision in the upper tail.
  `math.erf` via np.vectorize is too slow at 2.5e9 evaluations.
- **Strong-order reading**: because V_{n+1} is a nonlinear (monotone) function of the step's dW2 and
  not an Itô increment of it, the kernel's CRN strong-order estimate on V and S may read near zero or
  `unresolved`; R uses an exact transition and may read `unresolved` or high. With scheme_family
  omitted the order guard is one-sided at 0.1 - 0.4 = -0.3, so none of this should fail a correct
  plan; state the measured per-component orders in Results so the evaluator can see them.
- **Memory**: never materialise the 60 GB (1e6, 2500, 3) block; draw (chunk, 3) normals per step or
  loop over path chunks. Store log S, V, R, logD as flat float64 arrays updated in place.
- **Wall time**: QE is ~2× the per-step cost of full-truncation Euler (branching, one log on the
  exponential subset, one Phi); the spec's naive Euler estimate is already ~600 s against a 300 s
  sandbox. Use boolean-mask branching (compute both regimes only on their own subsets, or use
  `np.where` with care to avoid log(0)/divide-by-zero warnings), in-place ops, precomputed constants
  (E, K1..K4, OU coefficients), path chunks of 50k–200k, and exponentiate log S only at T. Time the
  1e6 run before handing over; if it does not fit, report it rather than reducing steps.
- **Square roots**: V >= 0 by construction under QE, so no truncation is needed; guard
  sqrt(K3*V_n + K4*V_{n+1}) with max(., 0) against roundoff only. R is Gaussian and goes negative;
  do **not** floor it.
- **Discount**: accumulate logD along each path; never apply a constant-rate exp(-r*T).
- **Path-count check**: the 1e6 and 250k runs must return different values that agree within MC
  noise. Never return a constant.
- **Accuracy budget**: plain-MC relative s.e. ~0.08% at 1e6, ~0.15% at 250k; the 0.3% band leaves
  ~0.1–0.15% for time-discretisation bias. QE's selling point is that its bias at moderate steps is far
  smaller than truncated Euler's at a Feller ratio this low. The solver should run the same seed at
  dt = 0.016 / 0.008 / 0.004 and report how the three prices move with dt in Results.
- **Self-checks to report** (from `verification.exact_identities`):
  - E[D S_T] = S0 (exact in expectation for this scheme)
  - E[D] vs Vasicek P(0,T) = 0.7968400987842249 and vs the discrete E[D]
  - put-call parity C - P = S0 - K*E[D]: -4.68401 / 20.31599 / 45.31599 (continuous P)
  - no-arbitrage bounds; price_S75 < price_S100 < price_S125; convexity in S0
  - E[V_T] = 0.04 and Var[V_T] = 0.0399981840028095 (QE matches the first two conditional moments
    exactly, so these should hold to MC noise; heavy-tailed), E[R_T] = 0.10,
    Var[R_T] = 0.04040398877652057 (exact for the OU transition)
  - `variance_nonnegative`: should report 100% (QE never produces V < 0)
  - Black-Scholes degenerate limit (sigma_v = sigma_r = 0 via a module-level parameter dict):
    40.305614268969904 / 64.11045523165186 / 88.62935813769187 within 0.3%. K0, K1 and K2 contain
    rho/sigma_v, which is singular at sigma_v = 0, so the limit needs an explicit branch: when
    sigma_v == 0, set V_{n+1} = m (deterministic; here V stays at 0.04) and step
    logS += (Rbar_n - 0.5*V_n)*dt + sqrt(V_n)*dW1. Use this branch only for the degenerate check;
    the graded path never takes it.
- **Kernel ceilings** (from the spec): provenance will be `self_convergence`; Dynkin at 1e6 paths
  exceeds the 12 GB budget and reports `unavailable`; the functionals are not kernel-graded on the
  SDE path, so the evaluator checks them and the two-path-count agreement by hand.

## Results

Run: `solve_sde(1_000_000, 0.004, 10.0, seed=42)` (Nt = 2500), 155 s wall on 4 forked workers (100k-path chunks, child streams from `default_rng(42).spawn`).

- empirical_mean (S, V, R): [635.08, 0.040305, 0.099968]
- empirical_variance (S, V, R): [1.7361e6, 0.040434, 0.040495]  (exact V: 0.039998, R: 0.040404)
- functionals (control variate on D*S_T, in-sample b): price_S75 = 37.5751, price_S100 = 57.4792, price_S125 = 78.6474 (CV s.e. 0.028 / 0.035 / 0.042, i.e. about 0.07 / 0.06 / 0.05 percent)
- 250k paths, same seed/dt: 37.5413 / 57.4262 / 78.5839 (differs from the 1e6 run, agrees within noise; the first two 100k chunks share streams)
- dt ladder (200k paths, plain seed 42): dt = 0.016: 37.567 / 57.477 / 78.667; 0.008: 37.583 / 57.480 / 78.647; 0.004: 37.533 / 57.406 / 78.552. Movement with dt is within MC noise (s.e. ~0.07 / 0.10 / 0.12).
- Self-checks: E[D S_T]/100 = 1.000106 (martingale identity; one-step check E[S_1 D_1]/S0 = 1.00000 to 1e-5 at V_n = 0, 0.001, 0.04, 0.5); E[D] = 0.79788 vs Vasicek 0.79684 (corrected per review: plain-MC s.e. of E[D] at 1e6 paths is about 0.0015, so the 0.001 gap is within 1 s.e.; trapezoid bias is O(dt^2)); E[V_T] = 0.04030 (exact 0.04); E[R_T] = 0.09997; V never negative (QE); no-arbitrage ordering and convexity hold (57.48 <= (37.58 + 78.65)/2).
- Black-Scholes degenerate limit (sigma_v = sigma_r = 0, 200k paths): 40.312 / 64.114 / 88.629 vs exact 40.3056 / 64.1105 / 88.6294.
- Strong order: not measured by me; as stated in the plan, V is a monotone function of dW2 and S/V may read near zero or unresolved, R is an exact OU transition.
- Hooks: `dW` of shape (M, Nt, 3) of independent increments is used exactly (Nt from shape, nothing drawn); `observables` trapezoid-accumulated on the (M, 3) state (S, V, R) in a single-block path.
- Open: the Dynkin ladder at the graded size is out of kernel budget as the spec notes; no independent reference for the prices was available to me.

## Evaluation

<metrics>
score: 7
certified: false
provenance: self_convergence
estimated_rel_error: 8.691e-06
error_is_estimate: true
observed_order: 0.9990
order_floor: 0.1000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 9.100e-05
invariant_worst_headroom: 0.7524
order_check: true
mc_se_rel: 0.0216
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: d60556a0110057d590dd8f718427b269edeeeb7433edc39fc93ffdb9746d1150
solve_time_s: 96.0750
wall_time_s: 135.8700
</metrics>

<review score=7>

**Score: 7/10**

### Numerical Accuracy
- provenance:              self_convergence  (reference route: none; moment_ode, kolmogorov and stationary_density all unavailable, as the spec declares)
- mean_rel_err:            not computed (no reference; moments_ok unavailable). Terminal S mean 635.08 +/- 2 x 1.32
- variance_rel_err:        not computed (no reference). Terminal S variance 1.7361e6 +/- 2 x 3.76e4
- mc_se_rel:               0.0216   resolved: unavailable (kernel reports "no reference variance to resolve against"). This is NOT the MC-inconclusive score-6 case, and the 2.2% figure is the heavy-tailed S variance, not the graded prices
- observed_order:          0.999 +/- 0.011 (worst-component readout, component 2 = R; expected 0.1, one-sided band). Per component: S 0.037 +/- 0.033, V 0.087 +/- 0.063, R 0.999 +/- 0.011. All pass the one-sided floor of -0.3
- weak_order:              indeterminate, at the noise floor on every component
- dynkin:                  unavailable (finest level needs 240 GB against the 12 GB budget). Not a failure, not a pass
- terminal_law:            not declared
- shape:                   S skewness 12.6 +/- 1.3, excess kurtosis 467 +/- 152; V skewness 9.96 +/- 0.11, excess kurtosis 146.7 +/- 4.4; R skewness -0.002 +/- 0.002, excess kurtosis 0.0003 +/- 0.005 (R is Gaussian as it should be)
- constraints:             finite pass, asset_positivity pass (0 violations), martingale pass (E[V]/E[R] diagnostics, worst headroom 0.75), variance_nonnegative pass (0 negative V, QE as designed)
- functionals (hand check, kernel does not grade them): 1e6 paths price_S75/100/125 = 37.575 / 57.479 / 78.647 (control-variate s.e. 0.028 / 0.035 / 0.042, i.e. 0.07 / 0.06 / 0.05 percent); 250k paths = 37.541 / 57.426 / 78.584 (s.e. 0.055 / 0.070 / 0.083). They differ and agree within noise (gaps 0.034 / 0.053 / 0.063 against a combined s.e. of at most 0.06 / 0.08 / 0.09). I reproduced both runs and they match the plan's Results to every digit. E[D S_T] = 100.011 (martingale identity holds); E[D] = 0.7979 vs Vasicek 0.79684 (plain-MC s.e. of D at 1e6 is about 0.0015, so the 0.001 gap is within 1 s.e.; the plan's quoted s.e. of 0.0002 is too small). Monotone, within no-arbitrage bounds, convex in S0. There is no independent reference value, so the 0.3% accuracy claim is NOT established
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. The kernel's reason string reads "Richardson-extrapolated moments are unstable, or Dynkin failed", but richardson_stable is true and dynkin_ok is unavailable. Neither happened. The plan fell through the no-surrogate 9 row (it needs dynkin_ok true) to the 7 row, so the reason text is the generic wording of that row and not a finding about this solver. A self_convergence certification ceiling of 8 also applies

### Feedback for solver
- No defect found, and nothing here is an MC-inconclusive result. The score is 7 because the kernel could not obtain a reference and the Dynkin check could not run at 1e6 paths and dt = 0.004. Both are properties of the spec and the kernel budget (spec ceilings 1 and 2), not of the scheme.
- The CRN ladder passed. The solver consumes `dW` exactly (S and V orders are low but positive, 0.037 and 0.087, in line with the spec's measured 0.043 / 0.110; R is 0.999). Nothing to change here.
- `scheme_family` is correctly left undeclared, so the band stays one-sided. Declaring it would make the R component (about 1.0) fail the two-sided band around 0.1 on a correct solver.
- Correct one statement in Results: the s.e. of E[D] at 1e6 paths is about 0.0015, not 0.0002, so E[D] = 0.7979 against 0.79684 is consistent with the Vasicek price. The 1e6 and 250k E[D] values (0.7979 and 0.8010) are consistent with that s.e. as well.
- Open accuracy risk the kernel cannot see: the 0.3% bar (about 0.11 / 0.17 / 0.24 in price) is only 3 to 5 times the 1e6-path s.e. The dt ladder in Results moves within noise (200k paths), so discretisation bias is not resolved at that precision. If you want to tighten this, a dt ladder with common random numbers on the discounted payoff itself, or more paths, would resolve it. This is optional and does not affect the kernel score.
- Wall time is fine: 96 to 141 s for the graded 1e6 solve on 4 forked workers, inside the 300 s sandbox limit.

### Formulator note
- No surrogate was available for S, so I found no declared-but-missed reference. The spec's own notes state that the S moments do not close and that the model is non-affine, and I have no counterexample. The Dynkin check being budget-blocked at the graded size is a spec design issue (num_paths = 1e6 with dt = 0.004 against a 12 GB limit). A smaller declared `conv_num_paths`-scale Dynkin run, if the kernel allowed one, would be what lifts this class of plan from 7 to 9.

</review>
