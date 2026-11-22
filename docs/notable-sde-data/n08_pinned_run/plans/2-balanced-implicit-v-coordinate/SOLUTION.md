---
id: 2
plan_slug: balanced-implicit-v-coordinate
scheme: drift-implicit-euler-log-chart
scheme_family: implicit
strategy: Step x = ln V (a transformed coordinate, declared) with the noise sub-step explicit and the stiff restoring drift implicit at V_{n+1}, so every step is strictly positive without any clipping.
---

## Why this scheme

`problem_spec.json` marks this problem `stiff: true`: `g(V) = epsilon * V**1.5` grows faster than the
quadratic restoring drift can damp, so plain Euler-Maruyama at the mandated `dt = 0.004` returns
negative or non-finite states. Rule 3 of the scheme-selection rules calls for an implicit
treatment of the stiff part. This plan treats the restoring drift implicitly and keeps the noise explicit.

**Transformed-coordinate step (grey zone, declared).** The spec prefers stepping `V` as written. This
plan does **not**: it steps `x = ln V` and returns `V = exp(x)`. Reason: the graded functionals are
`1/V` moments, and a V-chart step has no positivity guarantee without a floor (a floor at `eps`
gives `1/eps` outliers that dominate the answer). The log chart makes positivity structural. The
"cross-coordinate check" role against a V-chart plan is therefore played by the other plans in the pool.

No exact transition law is sampled; one supplied Brownian increment is consumed per step at the
handed-in `dt` (R25), so `exact_sampling_components` is not declared.

## SDE Reference

```
dV(t) = kappa * V(t) * (theta - V(t)) dt + epsilon * V(t)**1.5 dW(t),    V(0) = 0.15

X_0 = V(0) = 0.15
kappa   = 22.84
theta   = 0.218
epsilon = 8.56
T       = 0.1        (interval [0, 0.1])
```

Drift `f(V) = kappa * V * (theta - V)`; diffusion `g(V) = epsilon * V**1.5`;
`dg/dV = 1.5 * epsilon * np.sqrt(V)`.

## The scheme, written out

Ito on `x = ln V` gives

```
dx = [kappa*theta - (kappa + eps^2/2) V] dt + eps * sqrt(V) dW,     V = e^x
```

One step, with `c = (kappa + eps^2/2) dt`:

```
b       = ln V_n + kappa*theta*dt + eps*sqrt(V_n)*dW_n
x_{n+1} + c * exp(x_{n+1}) = b          (restoring term implicit at V_{n+1})
V_{n+1} = exp(x_{n+1})
```

The left side is monotone and convex in `x`, so 25 Newton iterations started at `x = b` converge
monotonically. `V_{n+1} = exp(x) > 0` for every increment, however large: positivity is structural and
there is no clip. No Milstein term is used: it was tried and reverted (see Revision).

## Numerical Scheme

- **Method**: drift-implicit Euler on `ln V` (`scheme_family: implicit`)
- **dt**: `0.004` as handed in; `Nt = max(1, round(T/dt))` = 25, `dt = T/Nt`; no hardcoded internal substeps
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`); graded call `200000`; same code serves both
- **T**: `0.1`; **seed**: `42` via `np.random.default_rng(seed)`
- **Expected strong order**: 0.5 (spec). Measured here on a 4-level shared-dW ladder RMS errors (vs an 800-step reference on the same Brownian path) 0.0077 / 0.0045 / 0.0017 at 100/200/400 steps.
- **Expected weak order**: 1.0. Ladder on `1/V`: mean 11.582/11.592/11.597/11.599 and variance 45.02/45.15/45.21/45.24 at 100/200/400/800 steps, approaching the exact 11.6228 / 45.2351.
- **Reference**: none in moment space; `verification.exact_functional_reference` gives
  `inv_var_mean = 11.6227862851`, `inv_var_variance = 45.2351064760`, carried as gated invariants on
  `1/X`, `1/X**2` and gated `functionals` tolerances (5% / 10%).

## Implementation Notes

- **Solver signature.** `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`.
- **Honour `dW` exactly** when supplied: `Nt = dW.shape[1]`, `dt = T/Nt`, no randomness of our own.
- **`observables`**: trapezoid `int phi(V_s) ds` per path, returned as `path_integrals`.
- **Return** `terminal_paths` = `V` itself, shape `(num_paths,)`; `empirical_mean`; `empirical_variance`
  (`ddof=1`); `functionals = {inv_var_mean, inv_var_variance}` of `1/V_T`, with `functional_se`
  (standard error of the mean; delta-method sqrt((m4 - s^4)/M) for the variance).
- **Do not argue accuracy from the mean/variance of `V`**: infinite-variance tails make them uninformative.
- **Scheme history.** Originally a balanced implicit method on `V` (weights `c0=|f|/V`, `c1=|g|/V`);
  its `c1|dW|` damping carried an O(sqrt(dt)) bias and was removed (see revision below). The shipped
  step is the drift-implicit log-chart Euler step (no Milstein term). `scheme_family` was
  `implicit` before this revision and is still `implicit` (the stiff drift is implicit), so it is
  unchanged. The `scheme:` name was updated from `balanced-implicit-method` to match the code.
- **No `boundary_treatment`**: `V = 0` is inaccessible, reproduced by `V = exp(x) > 0`.
- **Two path counts (R28).** Nothing cached or keyed on `num_paths`.

## Revision (bias fix, then Milstein correction)

1. Bias fix: the balancing denominator `D = 1 + c0 dt + c1|dW|` damped the noise by an O(sqrt(dt)) relative
   amount; it was replaced by the drift-implicit step on `x = ln V`. Explicit log-Euler was rejected: it
   blows up at dt = 0.004.
2. Milstein correction tried and REVERTED: it raised strong order (0.458 -> 0.865) but biased the 1/V
   moments (-0.96% / -3.3%), failing both gated martingale invariants (4.09 se vs bar 2.84) and
   dropping the score 9 -> 3. Shipped code is the pre-Milstein step (solver.r2.py). Variance reduction (cycle 3): see Antithetic estimator below.

## Antithetic estimator (cycle 3)

The same step is run a second time on `-dW`. The mirror feeds ONLY `functionals` / `functional_se`:
`inv_var_mean` is the mean over both halves, `inv_var_variance` the pooled sample variance of `1/V_T`
(2M samples, ddof=1 correction), and both SEs are the pair-level delta-method (influence function)
standard errors of the combined estimators. `terminal_paths`, `empirical_mean`, `empirical_variance` and
`path_integrals` come from the supplied increments alone, so the CRN ladder, invariants and Dynkin check
see byte-identical data. The `dt` bias was NOT reduced: no change to the step was made that could risk
the gated invariants (no Richardson/substeps, per R16). Previous solver kept as solver.r3.py.

## Results

Run: `uv run python solver.py` (dt = 0.004, T = 0.1, seed 42, Nt = 25).

| num_paths | empirical_mean (V) | empirical_variance (V) | inv_var_mean (se) | inv_var_variance (se) |
|---|---|---|---|---|
| 200000 | 0.133270 | 0.018495 | 11.5905 (0.0041) | 46.057 (0.161) |
| 50000 | 0.133011 | 0.018761 | 11.5812 (0.0082) | 45.745 (0.323) |

Exact reference 11.6228 / 45.2351 (not used by the solver). SE of `inv_var_mean` fell about 4x, of
`inv_var_variance` about 20% versus the plain estimator. Positive throughout, no clipping; `dW` and
`observables` hooks run.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: manufactured_partial
estimated_rel_error: 1.844e-03
error_is_estimate: true
observed_order: 0.4577
observed_order_se: 0.0287
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: implicit
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.2642
invariant_worst_headroom: 0.3681
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 0.0117
functional_bias_rel: 4.613e-03
functional_se_rel: 7.070e-03
functional_limited_by: both
order_check: true
mc_se_rel: 0.0839
resolved: false
dynkin_z: 1.9620
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: c1a4ddf368bbf1ca1d2f092b6bc8f62d8acf4430a38a1b7f742915e95ad56bf5
solve_time_s: 0.3800
wall_time_s: 12.5900
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              manufactured_partial  (reference route: none. `detail.surrogate_ok.attempted`: moment_ode "no moment ODE, or it does not close exactly", kolmogorov "no kolmogorov.bounds declared", stationary_density "no normalizable support declared". One circularity break: Dynkin/D1 against a validated operator (`operator_validation_route: spec_generator`). The exact reciprocal-CIR values enter only as gated invariants and functional tolerances.) `provenance_as_expected: false` (spec predicted self_convergence, run reached manufactured_partial). Diagnostic only, never scored on.
- mean_rel_err:            unavailable (no moment reference for V; `moments_ok: unavailable`, not graded). Kernel mean of V at 50000 paths is 0.13301 +/- 2.0 x 0.000613.
- variance_rel_err:        unavailable (same). Kernel variance of V is 0.018761 +/- 2.0 x 0.001573. Per the spec this variance has infinite variance (tail index 2.623), so it is not informative.
- mc_se_rel:               0.0839   resolved: false in the metrics, but `detail.resolved` reads "no reference variance to resolve against". This is a skip (`unavailable` in `checks`), not an MC-inconclusive finding and not the score-6 row. "Raise num_paths" is not the feedback.
- functional_rel_error:    0.0117 against functional_rel_tol 0.10  (bias 0.00461 + sampling 0.00707 -> limited by both, binding functional inv_var_variance). Both terms are well inside the gate, so "both" here means neither dominates, not that the gate is threatened. inv_var_mean is clean on the ladder (error 0.00139 against 0.05). Graded-call values at 50000 paths: 11.5812 (exact 11.6228) and 45.745 (exact 45.2351). Ladder-measured functional orders: 0.58 (mean) and 1.03 (variance).
- observed_order:          0.4577 +/- 0.0287 (expected 0.5, one-sided below the declared floor; median-based order 0.528). `orders_ok: true`. The point estimate sits about 1.5 se below the floor and inside the band's tolerance. This is the expected Euler-like strong order of a drift-implicit step with no Milstein term, and it is a real measurement, because the solver consumes the supplied `dW` (a near-zero order would have meant it did not).
- weak_order:              0.677 on tanh, not flagged. x and x2 are at the noise floor ("at the noise floor, or not shrinking"), so indeterminate on those.
- dynkin:                  ok. The first seed gave z = 1.96 against bar 2.43, which is 0.466 from the bar and inside the 0.7 margin, so `dynkin_replicated: true` and it was re-run on 3 seeds. Median z = 1.82 (seed z 1.96, 1.82, 0.89). `pooled_z` = 2.70 does not agree with the median verdict (`pooled_agrees: false`). The plan was marginal, and the result is a pass on the median rule but a thin one. Per function z: 1/X -1.96, 1/X**2 -1.41, tanh +1.50, with measured extrapolation orders 1.14, 0.68, 1.35. The generator is built from the spec's own drift and diffusion, so this would not catch a drift mis-transcribed from problem.md. It is one circularity break, not two.
- terminal_law:            not declared (the spec explains why: 1/V, not V, is the noncentral-chi-squared law)
- shape:                   skewness 10.10 +/- 3.53, excess kurtosis 349.7 +/- 235.7 (heavy right tail of V, consistent with tail index 2.62)
- constraints:             finite: pass. positivity: pass (strict; neg_fraction 0, no clip, min V stays > 0 by construction of exp). martingale entries on 1/X and 1/X**2 (gated, family-wise bar 2.95): pass, worst drift 1.264 se, headroom 0.368 decades. estimator_stability: pass (0.0043 against 0.02, ungated).
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: "no surrogate: the orders, Dynkin and the constraints all hold and the Richardson moments are stable across the ladder". The certification ceiling is also 9 ("one circularity break (D1 against a validated operator); no Tier-B evidence"). `certified: false` because the evidence path tops out below 10. `gates_clean: true`. No agent_cap applied. I checked `solver.py` against `SOLUTION.md`: both describe a drift-implicit Euler step on ln V with no Milstein term, and no state clip.

### Feedback for solver
- **Nothing is required. 9 is this evidence path's ceiling, not a defect.** With no moment ODE, Kolmogorov solve or stationary density available to the kernel, the only independent evidence is one circularity break (Dynkin), and the rubric stops at 9 on that. No change to `solver.py` moves it. Do not re-add the Milstein correction to chase the strong-order line: your own revision log shows it raised the order and failed both gated martingale checks (4.09 se against a bar of about 2.84).
- **Not a stability or path-count problem.** `resolved: false` is a skip for lack of a V-space reference, not an inconclusive comparison. Do not raise `num_paths`. Both functional terms are already roughly an order of magnitude inside the 10% gate (bias 0.46%, sampling 0.71%), and `functional_limited_by: both` only records that neither term alone dominates.
- **Thin spot to know about.** The binding margin is Dynkin: median z 1.82 against bar 2.43, pooled z 2.70, and the first seed was marginal. It is the same discretization bias that the strong-order reading of 0.458 shows. A step carrying less `dt` bias on the drift would widen it, but only within the R16 constraint (no extra internal substeps, and the handed-in `dt` is what is used).
- **Antithetic estimator.** The mirror run on `-dW` feeds only `functionals` and `functional_se`. The kernel's `terminal_paths`, moments, ladder and Dynkin all see the supplied increments alone, and `reported_moments_agree` is true (rel diff 0.0), so the reported estimator cannot hide a scheme defect from the checks. Note the reported functionals pool 2 x num_paths samples while the terminal states the kernel examines number num_paths, so read the reported functional SEs with that in mind.
- **Cross-plan record is stale.** The agreement entries against plans 1 and 3 predate this `solver.py` (`why: a solver.py changed since the comparison`) and are corroboration only, never scored. They show inv_var_mean z 4.60 against plan 3 and inv_var_variance z 3.01 against plan 1 (bar 3.0). Both are marginal and the older version's, not a finding on this one.
- **For the formulator, not the solver.** The exact reference for the graded quantities is recoverable (Y = 1/V is CIR, and the spec carries it as gated invariants and gated functional tolerances). It is not declared as a route the kernel reads, so `surrogate_ok` is `unavailable` and the V-space moment comparison and `resolved` are skipped. The spec's note argues that declaring it as a state-V reference would install a reference wrong by about 90x for every plan, so this was a reasoned choice, not an oversight. It binds every plan on this spec identically and cost this plan only the 10 tier. If the kernel offered a reciprocal-coordinate surrogate declaration, that is where it should go.

</review>
