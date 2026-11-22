---
id: 2
plan_slug: stratonovich-heun
scheme: stratonovich-heun (explicit predictor-corrector, equivalently Stratonovich-Milstein)
scheme_family: milstein
strategy: Advance the state with an explicit Stratonovich Heun predictor-corrector on the stated coefficients — derivative-free, strong order 1, and an independent cross-check on any implicit or correction-term scheme.
---

`scheme_family: milstein` is declared to select the spec's **order-1** strong-order
floor (`verification.expected_strong_order_milstein: 1.0`), which is this scheme's
actual order: `milstein_correction` in the spec says in as many words that "a Heun /
midpoint (Stratonovich) predictor-corrector reaches the same strong order 1 without
forming Dg.g explicitly". Declaring a family the spec has no order key for would
return the 0.5 default with a *two-sided* band and fail a correct order-1 scheme
measuring 1.0. No `exact_sampling_components` is declared, and must not be —
`implementation_notes` (4) and requirement R19 say both components must be stepped
from the supplied `dW`, so a near-zero measured strong order here would be a real
defect rather than a waivable one. There is no declared barrier or absorbing boundary,
so no `boundary_treatment` line.

## SDE Reference

**Stratonovich form** (the form `problem.md` states, `a = 1`, `sigma = 1`):

```
dX1 = -a*X2 dt - sigma*X2 o dW
dX2 =  a*X1 dt + sigma*X1 o dW,     X(0) = (1, 0)
```

**Equivalent Ito form** (Stratonovich-to-Ito correction `0.5*(Dg)g = -0.5*sigma**2*X`):

```
dX1 = (-a*X2 - 0.5*sigma**2*X1) dt - sigma*X2 dW
dX2 = ( a*X1 - 0.5*sigma**2*X2) dt + sigma*X1 dW
```

Both are driven by the **same single** scalar Brownian motion `W` (noise_dimension = 1).

Parameters: `a = 1.0`, `sigma = 1.0`, `X1_0 = 1.0`, `X2_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.

Coefficients, with state `X` of shape `(num_paths, 2)`:

```
# Stratonovich drift -- this is what Heun must use
f0_strat = -a*X[:, 1]
f1_strat =  a*X[:, 0]
# Ito drift (same, plus the correction terms)
f0 = -a*X[:, 1] - 0.5*sigma**2*X[:, 0]
f1 =  a*X[:, 0] - 0.5*sigma**2*X[:, 1]
# diffusion, single channel (identical in both conventions)
g0 = -sigma*X[:, 1]
g1 =  sigma*X[:, 0]
# Dg = sigma*[[0, -1], [1, 0]]  (constant);  Dg.g = sigma**2*(-X[:, 0], -X[:, 1])
```

Analytic moments at `t = T = 1` (from `analytic_moments`):

```
mean_X1 = np.cos(a*t) * np.exp(-0.5*sigma**2*t)                     = 0.32770991402
mean_X2 = np.sin(a*t) * np.exp(-0.5*sigma**2*t)                     = 0.51037795154
var_X1  = 0.5*(1 + np.cos(2*a*t)*np.exp(-2*sigma**2*t)) - np.cos(a*t)**2 * np.exp(-sigma**2*t) = 0.36444653726
var_X2  = 0.5*(1 - np.cos(2*a*t)*np.exp(-2*sigma**2*t)) - np.sin(a*t)**2 * np.exp(-sigma**2*t) = 0.26767402157
E[X1*X2] = 0.06153001240
```

## Numerical Scheme

- **Method**: explicit Stratonovich Heun predictor-corrector, on the **Stratonovich**
  coefficients (no `-0.5*sigma**2*X` drift anywhere in this scheme).

  Cycle 2 change: the corrector stage was raised from the 2-stage Heun
  trapezoid to a classical 4-stage explicit Stratonovich Runge-Kutta. Per step `n`,
  with `dWn = dW[:, n]` and `F(Y) = (-a*Y2*dt - sigma*Y2*dWn, a*Y1*dt + sigma*Y1*dWn)`:

  ```
  k1 = F(X);  k2 = F(X + k1/2);  k3 = F(X + k2/2);  k4 = F(X + k3)
  X_new = X + (k1 + 2*k2 + 2*k3 + k4)/6
  ```

  Still derivative-free: no derivative of `g`, no `dW**2` term written by hand. The averaging
  of the coefficients over the predictor is what supplies the Stratonovich correction
  automatically, and it is also what makes this the honest cross-check on a scheme
  that forms the correction explicitly.

- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 50000 for the kernel's moment checks
  (`evaluation_thresholds.num_paths`); the contract call is
  `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)` and the solver must honour
  whatever `num_paths` it is handed
- **T**: 1.0
- **seed**: 42 (`np.random.default_rng(seed)`; only used when `dW is None`)
- **Milstein correction**: not formed explicitly — but it is what this scheme is
  reproducing. For this system Heun collapses algebraically to
  `X_{n+1} = X_n + theta_n*J@X_n - 0.5*theta_n**2*X_n` with
  `theta_n = a*dt + sigma*dWn` and `J = [[0, -1], [1, 0]]`, which is the second-order
  Taylor truncation of the exact rotation. The spec's explicit alternative is
  `0.5*(Dg.g)*(dW**2 - dt) = -0.5*sigma**2*X_n*(dW**2 - dt)` added to the **Ito**
  Euler step; the two differ only in `O(dt**1.5)` terms and both are strong order 1.
  **Implement the predictor-corrector as written above**, not the collapsed form —
  the collapsed form is given here only as the algebraic check that the scheme is
  order 1, and writing it directly in terms of `theta_n` invites drifting toward the
  closed-form rotation.
- **Expected strong order**: **1.0** (the spec's `expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: **analytic moments** (`has_analytic_solution: true`),
  cross-checked by the evaluator against `verification.moment_ode`, which
  `closes_exactly`. Provenance should come back `analytic`.

## Implementation Notes

1. **The Stratonovich-to-Ito correction is the whole problem**, and this scheme
   handles it by being a genuinely Stratonovich scheme applied to the stated
   coefficients. The one fatal mistake is to use the **Ito** drift
   (`-0.5*sigma**2*X` terms included) *inside* the Heun averaging — that
   double-counts the correction. Equally fatal in the other direction: a plain Euler
   step on the stated coefficients with no corrector integrates a different equation
   whose amplitude grows like `exp(0.5*sigma**2*t) = 1.65` over `[0, 1]`, breaching
   the gated `amplitude_conservation` constraint.

2. **Amplitude behaviour.** This scheme is *not* exactly norm-preserving: the
   collapsed form gives `|X_{n+1}|**2 = |X_n|**2 * (1 + theta_n**4/4)`, so the
   amplitude drifts upward by about `0.75*dt` per unit time — roughly `0.03` at
   `dt = 0.04`, comfortably inside the gated bound of `0.25`, and shrinking linearly
   in `dt`. Report `max(abs(sqrt(X1**2 + X2**2) - 1))` over paths in Results; it
   should be a few percent, not a few tenths. If it comes back near `0.65`, the
   corrector is missing or the convention was read as Ito.

3. **Why this is a discretisation and not the closed form.** Required reading:
   `implementation_notes` (4) and (5). The per-step matrix exponential
   `exp(theta_n*J)` is exact for this flow and is at risk of **exclusion** from the
   comparison, not merely a poor score. Heun truncates the rotation at second order
   in `theta_n`, leaving a real `theta_n**3/6` per-step error (~1.6e-3 at this `dt`,
   where `theta_n` has standard deviation 0.2), so terminal states genuinely differ
   at `dt`, `dt/2` and `dt/4`. Do **not** call `scipy.linalg.expm`, and do not
   evaluate `np.cos`/`np.sin` of the accumulated angle anywhere in the step
   (`numpy` and the standard library only, per R17).

4. **Strong order 1 is reachable at `state_dimension = 2`** because
   `noise_dimension = 1`: with one channel no Levy area arises, which is why
   `milstein_eligible` is `true` here (see `diffusion_derivative`). A measured strong
   order near 0.5 means the corrector stage is not actually being applied — the usual
   cause is building the predictor from the already-updated state.

5. **Consume the supplied `dW` exactly.** `dW` is always supplied, shape
   `(num_paths, Nt)`, **already scaled by `sqrt(dt)`**. Take `Nt = dW.shape[1]` and
   `dt = T / Nt`; draw no randomness of your own for the Brownian path. When
   `dW is None` (the standalone `__main__` smoke test), build it as
   `np.sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt))` with
   `Nt = max(1, round(T / dt))` and `dt = T / Nt`. Ignoring `dW` makes the pathwise
   grading meaningless and makes the CRN order ladder read a strong order near zero.

6. **Return contract** (R12/R13): `terminal_paths` a `(num_paths, 2)` float array with
   columns `(X1, X2)` in that order; `empirical_mean` a list of 2 floats;
   `empirical_variance` a list of 2 floats with `ddof=1`. Keep `solve_sde` importable —
   no module-level work outside `if __name__ == "__main__"`.

7. **No variance reduction is proposed.** The spec declares no payoff functional and
   the problem is graded path by path against the exact solution driven by the *given*
   increments, so an antithetic pair would reduce nothing that is scored and would not
   change the pathwise dynamics the order ladder measures.

8. **Numerics are benign**: the state is bounded near 1, no stiffness, no positivity
   constraint, nothing to guard.

9. The Dynkin check will be built from `verification.generator` (the **Ito** drift and
   the 2x1 diffusion matrix), with test functions `X1`, `X1**2 + X2**2`, `X1*X2`,
   `tanh(X1)`. The second has `L.phi = 0` identically; this scheme's small amplitude
   drift is the only thing that shows up against it, at order `dt`.

10. **Cycle 2 (moment-bias refinement).** The bias sat in the corrector stage: Heun
   truncates the step rotation at second order in `theta_n`, so
   `|X_{n+1}|**2 = |X_n|**2 (1 + theta_n**4/4)` and the amplitude drifts at O(dt)
   (mean bias 1.9%, variance 3.1% at dt = 0.04). Replaced by an explicit 4-stage RK
   (stages evaluated on the stated Stratonovich coefficients, no `expm`, no
   `cos`/`sin`, no exponential series written by hand), which stays inside the
   explicit derivative-free family and remains independent of plan 1's Cayley step.
   Measured at dt = 0.04, 50000 paths: max amplitude deviation 0.0077 (was 0.155),
   mean [0.32738, 0.51359] vs exact [0.32771, 0.51038], variance
   [0.36375, 0.26494] vs [0.36445, 0.26767]. `scheme_family` stays `milstein`
   (unchanged, was `milstein`): it is used only to select the spec's order-1 floor;
   the scheme is now nominally higher order on this linear single-noise flow, so
   the measured strong order may overshoot 1.0 (one-sided floor, flagged not failed).
   The title `scheme:` line still says Heun and is historical.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt = 25), explicit RK4 corrector.

- Empirical mean (X1, X2):     [0.327377, 0.513593]  (exact 0.327710, 0.510378)
- Empirical variance (X1, X2): [0.363754, 0.264943]  (exact 0.364447, 0.267674)
- max | |X| - 1 | over paths: 0.0077 (gated bound 0.25; was 0.155 with Heun)
- `dW` hook and `observables` hook both confirmed working.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 0.0102
error_is_estimate: false
observed_order: 1.9211
observed_order_se: 0.0131
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 8.582e-03
invariant_worst_headroom: 0.0664
invariants_not_reported: ['second_moment_sum']
reference_outcome: validated
mean_rel_err: 6.298e-03
variance_rel_err: 0.0102
moment_bias_binding: mean
moment_bias_frac_tol: 1.838e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.950e-03
resolved: true
dynkin_z: 5.0943
dynkin_bar: 2.5310
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: e16a4e93e5ea478c70c536d68f55b3d6b3cc322ee6440304ff52eadec0dcb843
solve_time_s: 0.1460
wall_time_s: 18.8800
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, reference validated; Tier A)
- mean_rel_err:            6.3e-03 (worst component) +/- 2.0 x se; component 1 mean 0.51359 vs 0.51038, se 2.30e-03; tol 5e-02
- variance_rel_err:        1.02e-02 (worst, component 1) +/- 2.0 x se; 0.26494 vs 0.26767, se 1.84e-03; tol 1e-01
- mc_se_rel:               6.95e-03   resolved: true (CI widths 0.0100 and 0.0138 against bar 0.05, M = 50000)
- functional_rel_error:    not declared (no gated functional; moments are the accuracy claim)
- observed_order:          1.921 +/- 0.013 (median 1.77), expected 1.0, one-sided below the declared floor; strong_order_above_band: true
- weak_order:              2.03 (out of band, reported not scored); the dt bias is tiny: mean 9.2e-05 and variance 1.6e-04 relative, i.e. 0.18% and 0.16% of their tolerances, not actionable
- dynkin:                  UNRESOLVED (score 9, not 10). One seed, z = -5.09 against bar 2.53 on `X0^2+X1^2`; replicated: false, because an unresolved outcome is not a verdict and more seeds cannot change it. The other three functions pass (z = -0.99, -1.37, -1.37)
- terminal_law:            not declared
- shape:                   skewness -0.682 +/- 0.007 / -1.159 +/- 0.009, excess kurtosis -0.758 +/- 0.014 / 0.415 +/- 0.027 (components 0 / 1)
- constraints:             finite pass, amplitude_conservation pass, estimator_stability pass at drift 8.58e-03 against bar 1e-2 (headroom 0.066 decades), second_moment_sum not_reported
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: moments, MC resolution and constraints all pass; the generator check ran and did not reach a verdict

### What the Dynkin check did this time
The spec repair worked. The check now reads `x0 = [1, 0]`, deterministic, and runs on four test functions over four levels (it re-measured on 4 levels after a 3-level failure). It did not reach a verdict because of one function, `X0^2+X1^2`. The generator residual for that function is essentially zero in absolute terms (-5.3e-06 at the finest level, extrapolated -9.5e-08), because the Kubo flow conserves the radius and the scheme conserves it to high order. Two things follow.

- The extrapolated limit moves from -6.9e-07 to -9.5e-08 between the two level windows, 5.5 paired standard errors, so the levels are not in one asymptotic regime. The measured rate on this function is 2.03, outside the band the extrapolation assumes (first order), and the paired se (1.9e-08) is tiny only because the quantity is nearly deterministic. A z of -5.1 here is a ratio of two vanishing numbers, not evidence the drift is wrong. The conclusion that the Stratonovich-to-Ito correction is right comes from the other three functions (|z| <= 1.4, residual consistent with zero) and from the moments matching the exact Stratonovich values; it is not established by the kernel's verdict, so the score stays at 9.
- This is a property of the estimator meeting a higher-order scheme, not of the solver. Nothing in `solver.py` needs to change for it. Note also what it does not establish: the generator is built from the spec's own drift and diffusion, so a mis-transcribed coefficient would fool it.

### Estimator stability, re-measured
Drift 8.58e-03 against the 1e-2 bar, headroom 0.066 decades, against 9.2e-03 and 0.038 decades last time. It widened, but only slightly, and it is still the tightest check on the plan. The dt bias is not what binds it: the moment dt biases are below 2e-4 relative. A residual drift of order 1e-2 that barely moved while `max||X|-1|` fell 20x suggests the statistic is dominated by Monte Carlo noise at M = 50000, not by the scheme. I have not isolated that by changing the path count, so treat it as the likely cause, not a measurement. Do not try to buy headroom by changing the scheme.

### The two questions left for judgement
1. **`scheme_family: milstein` is a stale label.** The code is a classical 4-stage explicit Stratonovich Runge-Kutta step on the stated Stratonovich coefficients: derivative-free, no `Dg.g` is formed, and no Milstein correction term exists in the source. It is neither Heun nor Milstein. The frontmatter `scheme:` line still says Heun, and the body text of SOLUTION.md still describes a predictor-corrector. The spec marks `milstein_eligible: true` with `milstein_ineligible_reason: null`, so the label draws no `milstein_claim` cap and cost nothing in score here. What it does do is select the order-1 floor with a one-sided band, which is the effect the solver wanted, but it does it by describing the scheme as something it is not. This is a documentation finding, not a numerical one. Rewrite the `scheme:` line and the strategy sentence to name the 4-stage Stratonovich RK. Whether to change the family to `other` is a separate decision: I cannot confirm from the evaluation what floor or band `other` selects on this spec, and switching could change the order check, so only change it if you can read the resolved floor from the kernel's `config` first. If it stays `milstein`, say in the notes that the label is chosen for its order floor.
2. **Strong order 1.92 overshoots 1.0.** The band is one-sided below the floor of 1.0 (`order_band`), so an overshoot is flagged (`strong_order_above_band: true`) and not failed, and `orders_ok` is true. The measured rate is credible: se 0.013, the median estimator reads 1.77, and the weak order (2.03) agrees with it. The likely reason is that this is a linear flow driven by a single noise, so the stochastic Taylor terms collapse to the deterministic ones and a 4-stage method behaves like a higher-order one. That is a property of this problem, not evidence the scheme is order 2 in general. Do not quote 1.92 as the scheme's order in the report; quote 1.0 as the design claim and 1.92 as what this problem showed.

### Feedback for solver
- Nothing numerical needs changing. Moments are inside tolerance with the Monte Carlo resolved, the constraints pass, the order check passes, and the dt bias is under 0.2% of tolerance. The 10 is unavailable only because the Dynkin check did not reach a verdict, and that is a property of the check meeting a near-conserved quantity, not of the solver.
- Fix the documentation: the `scheme:` frontmatter line and the strategy sentence still say Heun, and the code is a 4-stage Stratonovich RK. Keep or change `scheme_family` as described above, and record the reason in the Implementation Notes.
- `estimator_stability` headroom is 0.066 decades. If you want to know whether it is sampling-bound, evaluate it at a larger path count in your own run; a smaller `dt` will not move it.

</review>
