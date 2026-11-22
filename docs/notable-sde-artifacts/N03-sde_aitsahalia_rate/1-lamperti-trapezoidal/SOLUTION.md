---
id: 1
plan_slug: lamperti-trapezoidal
scheme: lamperti-transform drift-implicit trapezoidal (theta = 1/2) step on Y = X^(-1/2)
scheme_family: implicit
strategy: Transform to Y = X^(-1/2), where the noise is additive, step Y with a theta = 1/2 drift-implicit (trapezoidal) scheme whose implicit map is a bijection of (0, inf) onto R, and map back X = Y^(-2), so every path stays strictly positive and the coarse-dt bias is second order in the drift.

## SDE Reference

Scalar Ito SDE on X > 0 (Ait-Sahalia short-rate model):

    dX = ( a_m1 / X - a_0 + a_1 X - a_2 X^2 ) dt + sigma X^{3/2} dW,    X(0) = X_0

| parameter | value |
|---|---|
| X_0 | 1.0 |
| a_m1 | 2.0 |
| a_0 | 3.0 |
| a_1 | 4.0 |
| a_2 | 7.0 |
| sigma | sqrt(2) = 1.4142135623730951 |
| T | 0.1 (T0 = 0) |

- drift `f(X) = a_m1/X - a_0 + a_1*X - a_2*X**2`, `f'(X) = -a_m1/X**2 + a_1 - 2*a_2*X`
- diffusion `g(X) = sigma*X**1.5`, `g'(X) = 1.5*sigma*X**0.5`
- Milstein correction `0.5*g*g'*(dW**2 - dt) = 0.75*sigma**2*X_n**2*(dW**2 - dt)`
- noise: multiplicative, scalar; `stiff: true`; `milstein_eligible: true`
- No closed-form moments (`has_analytic_solution: false`, `terminal_law: null`).
- Reference moments (backward-Kolmogorov surrogate in the spec): E[X_T] ~ 0.74887, Var[X_T] ~ 0.0494.
- Gates: mean rel. error < 5%, variance rel. error < 10%; every terminal state finite and strictly
  positive (`verification.constraints`: `X > 0`, `np.isfinite(X)`, both gated).
- Graded call: `solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)` -> Nt = 5. Development and
  kernel runs use num_paths = 50000, seed = 42, dt = 0.02 (dt0 = 0.02, 3 CRN levels).

### Lamperti transform (derived for this plan)

With `Y = X^(-1/2)` (so `X = Y^(-2)`), Ito's formula gives additive noise:

    dY = h(Y) dt - (sigma/2) dW
    h(Y) = -(a_m1/2) Y^5 + (a_0/2) Y^3 - (a_1/2) Y + (a_2/2 + 3 sigma^2/8) / Y
         = -Y^5 + 1.5 Y^3 - 2 Y + 4.25 / Y            (with the given parameters)
    h'(Y) = -5 Y^4 + 4.5 Y^2 - 2 - 4.25 / Y^2  < 0  for all Y > 0
    Y(0) = X_0^(-1/2) = 1

`-5Y^4 + 4.5Y^2 - 2` has negative discriminant, so `h' < 0` everywhere on (0, inf). Note the
**minus sign** on the noise: the same `dW` the kernel supplies drives Y as `-(sigma/2) dW`.
A wrong sign leaves the moments unchanged (W and -W have the same law), so the moment checks would
not catch it, but it makes this plan's paths disagree with the original equation driven by the
same Brownian path. Get it right.

## Numerical Scheme

- **Method**: theta-method with theta = 1/2 in the drift, additive noise explicit, on Y:

      Y_{n+1} - (dt/2) h(Y_{n+1}) = Y_n + (dt/2) h(Y_n) - (sigma/2) dW_n

  Then `X_n = Y_n**(-2)`.
- **Root solve**: the map `y -> y - (dt/2) h(y)` is strictly increasing on (0, inf) (since h' < 0)
  and runs from -inf (y -> 0+, the 4.25/y term) to +inf, so there is **exactly one positive root
  for every real right-hand side r**. Equivalently (multiply by y, theta*dt = dt/2 =: c):
  `c y^6 - 1.5 c y^4 + (1 + 2c) y^2 - r y - 4.25 c = 0`. Vectorised safeguarded Newton from
  `y0 = Y_n`, Newton derivative `1 - c h'(y)`, keep iterates > 0 (e.g. halve toward the lower
  bracket instead of crossing zero). 30–60 iterations is ample; check convergence explicitly.
- **dt**: 0.02 (Nt = 5; the contract fixes it — do not change it)
- **num_paths**: 50000 for development / kernel; graded at 200000
- **seed**: 42
- **T**: 0.1
- **Milstein correction**: none needed — the noise in Y is additive (g_Y' = 0), so this scheme is
  already strong order 1 in Y, and X = Y^(-2) is a smooth map on the range of paths.
- **Expected strong order**: 1.0 (additive noise in the transformed variable). The declared
  `scheme_family: implicit` selects the spec's 0.5 floor; the gate is one-sided, so a measured
  ~1.0 is fine and is the expected reading.
- **Expected weak order**: 1.0 (formally 2 for the trapezoidal drift on smooth problems, but do not
  claim more than 1.0).
- **Reference for scoring**: Kolmogorov solve (`verification.kolmogorov`).

## Implementation Notes

### Contract (all plans)

- Signature: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. The graded
  harness calls only the first four arguments; the kernel additionally passes `dW` (shape
  `(num_paths, Nt)`, already scaled by sqrt(dt)) for the CRN order study and `observables` for the
  Dynkin check (`dynkin_test_functions`: `X`, `X**2`, `np.exp(-X)`). When `dW` is given, take `Nt`
  from its shape, set `dt = T/Nt`, and draw no randomness of your own.
- When `dW` is None: `Nt = max(1, round(T/dt))`, `dt = T/Nt`, `rng = np.random.default_rng(seed)`,
  `dW = sqrt(dt)*rng.standard_normal((num_paths, Nt))`. numpy + stdlib only; no work at import.
- Return `terminal_paths` (1-D, shape `(num_paths,)`), `empirical_mean`, `empirical_variance`
  (ddof=1), `num_paths`, `dt`, `T`, and `path_integrals` (trapezoid rule on X) when observables are given.
- **Never clip the state** (`np.maximum` on X after an update): the spec flags it. Positivity must
  come from the scheme itself; a lower guard *inside* the root-finder's iterate is fine.
- The implicit root solve must be vectorised over paths and converged tightly (|residual| < 1e-12
  relative); use safeguarded Newton (fall back to bisection on the bracket where a Newton step leaves
  it). The implicit map is strictly increasing on (0, inf), so the bracket always exists.
- Explicit Euler-Maruyama is not an option: at X ~ 3 the drift is ~ -53 and dt*f ~ -1.07, so the
  step leaves the domain and fails the positivity gate.
- Reference for scoring: **Kolmogorov solve** (`verification.kolmogorov`, bounds [0.05, 6],
  payoffs x and x**2). There is no moment ODE and no stationary density route.
- No barrier/absorbing boundary is declared and the problem prices no payoff, so no
  `boundary_treatment` and no variance-reduction wrapper is required.

### Plan-specific

- Step entirely in Y; convert to X only for the observables and the return values. For
  `path_integrals`, evaluate `phi(Y**-2)` at both ends of each step (trapezoid in time).
- The explicit half-drift `(dt/2) h(Y_n)` can be large for rare extreme Y_n (Y_n = 3 gives
  ~ -2.1), but that only moves r; the implicit map still has a unique positive root for any r, so
  positivity and finiteness are structural, not luck. Do not add any clipping of Y or X.
- Expected outcome (plan-creator's scratch estimate at seed 42, 200000 paths, Nt = 5): mean ~0.7487,
  variance ~0.0496 — both within 0.5% of the Kolmogorov reference, minimum path ~0.32. If the
  solver's numbers are far from this, suspect the Ito correction term `3 sigma^2/8` in h or the
  noise coefficient `sigma/2`.
- Caution observed in scratch: the fully implicit (theta = 1) variant of this same transform reads
  variance ~0.0452 at Nt = 5 (8.5% low, close to the 10% gate). The theta = 1/2 choice is what makes
  this plan accurate at the contract's coarse dt; keep it.

## Results

Run: `solve_sde(num_paths=200000, dt=0.02, T=0.1, seed=42)` (Nt = 5), the graded call.

| quantity | solver | Kolmogorov reference | rel. error |
|---|---|---|---|
| empirical mean | 0.748688 | 0.74887 | 0.02% |
| empirical variance (ddof=1) | 0.049581 | 0.04940 | 0.37% |

- Terminal range: min 0.3212, max 4.4054 -- every path finite and strictly positive, with no
  clipping of the state (positivity comes from the implicit root being unique on (0, inf)).
- Hooks confirmed: explicit `dW` (shape (1000, 10)) gives `dt = T/10 = 0.01`; a one-entry
  `observables` dict returns `path_integrals["x"]` of shape (1000,).
- Implementation: Lamperti variable Y = X^(-1/2), noise coefficient `-(sigma/2) dW` (minus sign
  kept), theta = 1/2 drift-trapezoidal step, vectorised safeguarded Newton/bisection root solve with
  an explicit convergence check (raises if residual > 1e-10 relative). No `dW_sub` argument: the
  spec declares no `sub_increments_per_step`.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: surrogate
estimated_rel_error: 2.840e-03
error_is_estimate: true
observed_order: 0.9879
observed_order_se: 0.0118
order_floor: 0.5000
order_band: one-sided (scheme family undeclared in the plan)
design_order: 0.5000
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 1.786e-03
invariant_worst_headroom: 0.7482
mean_rel_err: 6.329e-05
variance_rel_err: 2.840e-03
order_check: true
mc_se_rel: 0.0130
resolved: true
dynkin_z: 1.1527
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 7363e79ad84d933bcd82636fc45656b9214d46af2c62c00d8ea458102dc0e1bd
solve_time_s: 0.5190
wall_time_s: 6.6900
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              surrogate  (reference route: kolmogorov; the moment ODE was attempted and does not close; no stationary density declared)
- mean_rel_err:            6.33e-05 +/- 2.0 x 9.95e-04 absolute se (0.7489 vs reference 0.74887; tolerance 5%). Inside the interval by a wide margin.
- variance_rel_err:        2.84e-03 +/- 2.0 x 6.46e-04 absolute se (0.049545 vs reference 0.049404; tolerance 10%).
- mc_se_rel:               0.0130 (CI width 0.0262 against a 0.05 bar)   resolved: true, at 50 000 paths. Graded count is 200 000.
- functional_rel_error:    not applicable. The spec declares no gated functional, so the kernel writes none of the functional fields.
- observed_order:          0.988 +/- 0.012 (median-based 1.007; expected 0.5, one-sided band, floor 0.5). The ladder is clean: s0 = 6.78e-03, s1 = 3.42e-03, a factor of 1.98 per halving, i.e. strong order 1, as additive noise in the Lamperti variable predicts.
- weak_order:              indeterminate. Only the tanh function survived the noise-floor filter and read 0.24 (against an expected 1.0). Recorded as `weak_order_out_of_band` and never scored on. `x` and `x**2` were at the noise floor (w1 of 7.3e-05 against se 3.3e-05, and 1.2e-04 against 7.2e-05).
- dynkin:                  ok. Single seed, z = -0.74 / -1.15 / +0.28 on X, X**2, exp(-X) against a family-wise bar of 2.43. `dynkin_replicated: false`, meaning the plan was not marginal: |z - bar| = 1.28 is outside the 0.7 margin, so one seed settled it. The measured extrapolation order was 1.05 on X and 1.64 on exp(-X). On X**2 the finer difference was inside its own MC noise, so p = 1 was assumed and recorded.
- terminal_law:            not declared (`terminal_law: null`)
- shape:                   skewness 1.759 +/- 0.062, excess kurtosis 6.513 +/- 0.734. The spec's note says about 1.6 and 5.4, so the plan's terminal law is right-skewed and heavy-tailed as the problem predicts. Kurtosis sits about 1.5 se above the spec's rough figure. That is a rough prior, not a gate.
- constraints:             strict_positivity (gate) ok, drift 0.0, min path 0.32; finite (gate) ok; estimator_stability (non-gate) ok at 1.79e-03 against a 0.01 bar (headroom 0.75 decades). `neg_fraction` is 0. `gate_violation` is false.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. A surrogate reference passed its trust guard, the moments match it, and every certifying condition held. Tier A', `certified: true`, `agent_cap` none.

Other checks: `reported_moments_agree` is true, with rel_diff 0.0, so the solver's own arithmetic (ddof=1) matches the kernel's. `provenance_as_expected` is true (spec predicted surrogate). `richardson_stable` is true, although the Richardson shift is 2.2 se (p_used 0.5). `cross_plan_agreement` is unavailable because there was no sibling plan to compare against. That is a skip and not a failure.

Source inspection, which no kernel check covers: `solver.py` implements what `SOLUTION.md` describes. It steps in Y = X^(-1/2) with the theta = 1/2 trapezoidal drift and the `-(sigma/2) dW` noise, with the minus sign kept. It uses the supplied `dW` and takes `Nt` from `dW.shape[1]`. It does not clip the state; the only positivity guard is bracketing inside the Newton solve. No cap was warranted.

### Notes on the reference
- The trust guard's widened solve was degenerate. `surrogate_ok.widened` is [-1.4375, 7.4875] and `truncation_shift` is NaN. This is exactly the domain artefact the spec's own CAUTION predicts, since the lower bound goes negative where the drift is undefined. The kernel still accepted the surrogate. The reference values (0.748873 / 0.049404) agree with the formulator's independent Crank-Nicolson solve on three domains (0.74887 / 0.04937 to 0.04940) and with its drift-implicit MC at dt = 5e-5 (0.74836 +/- 0.00035 / 0.04914), so I judge the reference sound. It is, however, a guard that did not actually evaluate, so the surrogate trust here rests on the formulator's recorded cross-checks and not on the kernel's own truncation test. This is a kernel and formulator observation. It is not a solver defect and it did not affect the score.
- No formulator defect found. The moment ODE genuinely does not close (the hierarchy is infinite in both directions), and the stationary density is not the scored quantity.

### Feedback for solver
- No accuracy defect. Mean and variance are 0.006% and 0.28% from the Kolmogorov reference against 5% and 10% tolerances. Strong order is 0.99 and the generator check is clean. Nothing needs to change in the scheme.
- **Frontmatter is unclosed, so the declaration was never read.** `SOLUTION.md` opens with `---` and has no closing `---`, so the kernel parsed nothing. It reports `plan.declared: false`, `scheme_family: null`, `scheme_family_source: unknown`, and so `order_band: one-sided (scheme family undeclared in the plan)`. Your `scheme_family: implicit` line never reached the kernel. Add a closing `---` after the `strategy:` line. This costs nothing in score here, because the one-sided band is satisfied (0.988 against a 0.5 floor, and an order above the floor is never charged). Closing the frontmatter would let the two-sided check apply. Be aware that `implicit` selects the spec's 0.5 floor, and with a two-sided band an observed 0.99 sits 0.49 above 0.5 and outside the 0.4 half-width. If you close it, declare the family that matches what the kernel measures: the Lamperti transform gives additive noise, so strong order 1.0 is the honest claim. Check which of the spec's two declared floors (`expected_strong_order` 0.5, `expected_strong_order_milstein` 1.0) your family key maps to before you rely on it. Leaving it undeclared is also safe.
- `exact_sampling_components` is correctly absent. The step consumes the supplied increments (order 0.99, not near zero), so no declaration is warranted.
- Weak order reads 0.24 on one surviving test function against the 1.0 you claim. The kernel treats it as noise (`weak_order_out_of_band`, never scored) and so do I. Do not quote 0.24 as a property of the scheme. Your Implementation Notes already, and rightly, decline to claim more than 1.0.
- Optional, for a more robust margin at the graded count: the Richardson shift is 2.2 se and `shift_within_noise` is false, though `richardson_stable` is true. At 200 000 paths this narrows by 2x on its own. No action required.

</review>
