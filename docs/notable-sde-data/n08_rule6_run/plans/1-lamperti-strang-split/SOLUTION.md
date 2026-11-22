---
id: 1
plan_slug: lamperti-strang-split
scheme: lamperti-transform + strang-splitting (exact drift flow, exact additive-noise sub-step)
scheme_family: other
boundary_treatment: reflection
strategy: Transform to the Lamperti coordinate X = V**-0.5 where the noise becomes constant and additive, then advance with a Strang composition whose drift half-flows and noise sub-step are each the exact flow of their own vector field, reflecting at X = 0 to keep V finite and strictly positive by construction.
---

## SDE Reference

Scalar Ito SDE for the instantaneous variance V > 0:

```
dV = kappa * V * (theta - V) dt + epsilon * V**1.5 dW,    V(0) = 0.15
```

Parameters (from `problem_spec.json → parameters`):

```python
kappa   = 22.84
theta   = 0.218
epsilon = 8.56
V_0     = 0.15
T       = 0.1          # time_interval: [0.0, 0.1]
```

- `drift_expression`:      `kappa * X * (theta - X)`
- `diffusion_expression`:  `epsilon * np.maximum(X, 0.0)**1.5`
- `diffusion_derivative`:  `1.5 * epsilon * np.sqrt(np.maximum(X, 0.0))`
- `noise_structure`: multiplicative, `state_dimension = 1`, `noise_dimension = 1`, `stiff: true`

**Graded quantities** (`evaluation_thresholds.functionals`, read from the returned `functionals` dict):

| key | quantity | exact reference | relative tolerance |
|---|---|---|---|
| `inv_var_mean` | `E[1/V(T)]` | `11.622786285052442` | 5% |
| `inv_var_variance` | `Var[1/V(T)]` (ddof=1) | `45.23510647598556` | 10% |

Contract calls: `solve_sde(num_paths=200000, dt=0.004, T=0.1, seed=42)` and again with
`num_paths=50000`, everything else unchanged. `Nt = max(1, round(T/dt)) = 25`, `dt = T/Nt = 0.004`.

## Numerical Scheme

### The coordinate change

Let `X = V**-0.5` (the Lamperti transform, `h(V) = V**-0.5`, `h' = -0.5 V**-1.5`,
`h'' = 0.75 V**-2.5`). Ito's lemma on the equation above gives

```
dX = [ -0.5*kappa*theta * X  +  (0.5*kappa + 0.375*epsilon**2) / X ] dt  -  0.5*epsilon * dW
```

which is the expression `implementation_notes` states and endorses as a legitimate
discretisation coordinate. Write

```python
a  = kappa * theta                              # 4.97912
c  = 0.5 * kappa + 0.375 * epsilon**2           # 38.897600000000004
sg = 0.5 * epsilon                              # 4.28
X_0 = V_0**-0.5                                 # 2.581988897471611
```

so the drift is `A(X) = -0.5*a*X + c/X` and the diffusion is the **constant** `-sg`. The
state-dependence that makes the original equation explode (a `V**1.5` diffusion amplifying its own
excursions) is gone from the diffusion entirely; what is left is a drift with a `1/X` singularity
at `X = 0`, and `X = 0` corresponds to `V = infinity`, not to the gated boundary `V > 0`.

### Both sub-flows are exact

**Drift sub-flow.** `dX/dt = -0.5*a*X + c/X` linearises under `u = X**2`:
`du/dt = 2*X*X' = -a*u + 2*c`, so

```python
u_star = 2.0 * c / a                            # 15.624309...
u(t)   = u_star + (u0 - u_star) * np.exp(-a * t)
X(t)   = np.sqrt(u(t))                          # u(t) > 0 whenever u0 >= 0
```

This is the **exact** flow, evaluated in closed form at any step length, with no stability
restriction — which is the whole answer to `stiff: true`. Because `u_star > 0` and the map
`u0 -> u(t)` is affine with positive coefficients, `u(t) > 0` strictly for every `u0 >= 0`:
the drift half-step cannot leave `(0, inf)` and actively pushes `X` away from zero.

**Noise sub-flow.** `dX = -sg dW` with `sg` constant integrates exactly:
`X -> X - sg * dW_n`, where `dW_n` is the Brownian increment of the step. No approximation.

### The step

Strang composition, one full step of length `dt` consuming the single increment `dW[:, n]`:

```python
# 1. drift half-flow, length dt/2
u = X*X ;  u = u_star + (u - u_star) * e_half ;  X = np.sqrt(u)     # e_half = exp(-a*dt/2)
# 2. exact noise sub-step, full dt   (THIS is where dW[:, n] is consumed)
X = X - sg * dW[:, n]
# 3. reflection at the entrance boundary X = 0
X = np.abs(X)
# 4. drift half-flow, length dt/2
u = X*X ;  u = u_star + (u - u_star) * e_half ;  X = np.sqrt(u)
```

At the end, `V = X**-2` and `1/V = X**2`; return `terminal_paths = X**-2.0`.

Precompute `e_half = np.exp(-0.5 * a * dt)` once outside the loop.

### Why the reflection, and why it is a declaration rather than a fudge

`X` must stay in `(0, inf)` for `V = X**-2` to be finite and strictly positive — both hard gates
(`verification.constraints` `finite`, `positivity`). Steps 1 and 4 cannot violate it. Step 2 can, in
principle: `sg*sqrt(dt) = 4.28*0.0632 = 0.2707` against `X ~ 2.6-4.4`, so crossing zero is a
~10-16 sigma single-step event and is expected to occur on **no path at all** in the graded run.
Where it does, `X -> |X|` is the exact reflection of the driftless Brownian sub-flow at an entrance
boundary (the drift sub-flow is a function of `u = X**2` alone, so the reflected and unreflected
states are indistinguishable to it), and it is applied *inside* the step to the Lamperti variable,
not as a post-update clamp on `V`. There is no `V = np.maximum(V, eps)` statement anywhere:
positivity and finiteness of `V` are structural properties of this composition, which is what
`implementation_notes` asks for ("A reflecting or implicit step that keeps V > 0 structurally is
better than either").

Count the reflections and report the count (expected: 0). A nonzero count at the graded `dt` is
information, not a failure, but it belongs in the Results.

### Hyperparameters

- **Method**: Lamperti transform + Strang splitting with exact sub-flows
- **dt**: `0.004` (the contract step; `Nt = 25`). Do **not** hardcode a different step —
  take `dt` from the argument and recompute `Nt = max(1, round(T/dt))`, `dt = T/Nt`.
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`); the harness also calls at `200000`
  (`graded_num_paths`)
- **T**: `0.1`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Expected strong order**: **1.0** — the noise is additive and constant in this coordinate and
  both sub-flows are exact, so the only error is the splitting commutator, which is `O(dt)`
  pathwise. The spec declares `expected_strong_order: 0.5` (Euler-Maruyama, multiplicative noise)
  and `expected_strong_order_milstein: 1.0`; this plan declares `scheme_family: other` precisely so
  that the convergence guard reads a one-sided band spanning both rather than a two-sided band at
  0.5, which a correct additive-noise scheme measuring 1.0 would fail. A measurement near **1.0**
  confirms the plan; a measurement near **0.0** would mean the supplied increments are not being
  consumed and is a real defect here.
- **Expected weak order**: **1.0** declared (the spec's `expected_weak_order`). The Strang
  composition of two exact flows with additive noise is formally second order in the weak sense, so
  a measurement near 2.0 is expected and is not a defect — it is why this plan is ranked first.
- **No `exact_sampling_components`.** Every component is stepped with the supplied increments, by
  construction, and `implementation_notes` forbids the declaration here in as many words. A
  near-zero strong order on this plan is a bug to fix, not a check to silence.

### Variance reduction: antithetic pairs

Applied to this scheme (not proposed as a separate plan — a wrapper does not change the pathwise
dynamics and would be indistinguishable from the bare scheme on every statistic the kernel
computes). `1/V(T)` is close to monotone in the driving path (`Y = 1/V` is pushed up by `-dW`), so
pairing is expected to cut the standard error of `inv_var_mean` materially.

When the solver draws its own randomness (`dW is None`):

```python
m = num_paths // 2
Z = rng.standard_normal((m, Nt))
Z = np.concatenate([Z, -Z], axis=0)          # shape (2*m, Nt)
if num_paths % 2:                             # odd count: one unpaired extra path
    Z = np.concatenate([Z, rng.standard_normal((1, Nt))], axis=0)
dW = np.sqrt(dt) * Z
```

The draw shape still depends on `num_paths`, so the `200000` and `50000` calls are genuinely
different samples and the two reported values differ, as the two-path-count check requires.

Report the **paired** estimator's error bar in `functional_se`: form the per-pair means of
`1/V(T)` and take `std(pair_means, ddof=1)/sqrt(m)` for `inv_var_mean`. For `inv_var_variance` use
a jackknife or batch-means standard error over the pairs (keep pair partners in the same batch —
the ungated `estimator_stability` invariant uses 20 batch means and splitting a pair across batches
would inflate the scatter artificially). The point estimates themselves are computed from the
pooled terminal sample exactly as `problem.md` specifies:
`mean(1.0/V_T)` and `var(1.0/V_T, ddof=1)`.

**When `dW` is supplied** (the kernel's CRN ladder and Dynkin runs) the antithetic construction is
switched off entirely: use the supplied increments exactly as given, take `Nt = dW.shape[1]` and
`dt = T/Nt`, and draw no randomness. Otherwise the refinement levels stop sharing one Brownian path
and no strong order is measurable.

## Reference for Scoring

**Surrogate** (`verification.expected_provenance: surrogate`). There is no closed form for `V`'s own
moments (`analytic_moments.has_analytic_solution: false`, both expressions `null`), and
`moment_ode` is `null` because the moment hierarchy for `V` does not close. The references the
evaluator will use are, in order of weight:

1. The two exact reciprocal moments carried as gated `martingale` entries in
   `verification.constraints` / `verification.invariants`:
   `E[1/V(T)] = 11.622786285052442` and `E[(1/V(T))**2] = 180.3242675039887`
   (`ci_mult: 2.0`). These are the same numbers the graded functionals are scored against and they
   pin `Var[1/V(T)] = 45.23510647598556`.
2. The **Dynkin generator check** on `verification.dynkin_test_functions`
   `["1.0/X", "np.exp(-X)", "np.tanh(X)"]`, which needs the `observables` hook. This is the only
   test that confirms the paths came from *this* equation, and nothing else can stand in for it.
3. `verification.stationary_density` — `V_inf ~ InverseGamma(2.623417, 0.135905)`, stationary mean
   `0.083715`. Read the **mean only**: `alpha = 2.62` makes `E[V**3]` infinite, so the stationary
   variance estimator has infinite variance and is expected to abstain.

`verification.terminal_law` is `null` by design, and `stability_check` is `null` — finiteness is
carried as a gated constraint instead, so the moment tolerances are not replaced.

## Implementation Notes

- **Do not sample the exact transition, in any coordinate.** `Y = 1/V` is exactly a CIR process
  (`verification.reciprocal_cir_identity`) with a noncentral chi-squared transition law, and a
  solver could draw `V(T)` in one shot and hit both graded numbers. `problem.md` and
  `implementation_notes` forbid exactly that: "sampling it directly is not what this problem
  measures". This plan steps the Lamperti coordinate with the supplied increments, which the spec
  names as legitimate ("stepping that coordinate with the supplied increments is a legitimate
  discretisation; sampling its exact transition is not"). Do not step the reciprocal variable
  through its exact transition, and do not use the identity's reference values anywhere inside
  `solver.py` — not as a control-variate mean, not as a calibration constant. They are the grade.
- **Positivity and finiteness are hard gates.** `V(T)` must come back strictly positive and finite
  on every path; the graded quantities are reciprocal moments, so one path at or below zero makes
  them `Inf` rather than merely biased. This scheme delivers both structurally (see above).
  **No post-update clipping of `V`** — `implementation_notes` says such a statement is scanned for
  and reported as hiding the defect.
- **Verification hooks.** Signature:
  `def solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. The four
  `problem.md` names come first with their exact semantics; `dW=None` and `observables=None` must
  default to reproducing the stated behaviour exactly (ledger entry R14 records this addition as
  the one `ambiguous` requirement). Without them the strong-order study and the Dynkin check cannot
  run and the plan cannot be certified above `self_convergence`.
- **`observables` are functions of `V`**, not of `X`. Accumulate the trapezoid path integral on the
  `V` values at the step endpoints: `acc[k] += 0.5*(phi(V_prev) + phi(V_new))*dt`, with
  `V = X**-2.0` at each endpoint, and return them under `path_integrals`. The declared test
  functions include `1.0/X`, which is finite and well conditioned here (its second moment is 180),
  while `X` and `X**2` are deliberately avoided because `E[V**2]` is tail-dominated.
- **Return shape.** `terminal_paths` is the 1-D array of terminal `V` values, shape
  `(num_paths,)` — `V`, not `X`, not `1/V`. Also return `empirical_mean` and
  `empirical_variance` of `V` (`ddof=1`), and `functionals = {"inv_var_mean": ...,
  "inv_var_variance": ...}` computed from those same terminal paths.
- **`V`'s own moments are not graded and should not be chased.** No surrogate for them exists: the
  spec records that a truncated backward-Kolmogorov solve for `E[V_T**2]` moved from 0.1204 to
  0.1699 as the truncation widened and was rejected. `variance_rel_err_max` and `mean_rel_err_max`
  sit at template values with nothing to compare against. The whole grade rides on the two
  reciprocal functionals.
- **Two path counts.** `solve_sde` is called at `200000` and at `50000` with the same `dt`, `T` and
  `seed`. The two reported values of each quantity must *differ* and must agree inside the implied
  sampling noise. Drawing at shape `(m, Nt)` with `m` derived from `num_paths` satisfies this. Do
  not cache results, do not round, do not return a constant.
- **Vectorise over paths.** `X` has shape `(num_paths,)`; 25 steps over 200 000 paths is one
  `(num_paths, Nt)` normal block and a 25-iteration loop of array ops.
- **Expected precision.** Measured excess kurtosis of `1/V_T` is 1.68, so at 50 000 paths the
  relative standard error is ~0.27% on the mean (5% tolerance) and ~0.80% on the variance (10%
  tolerance) *before* antithetic pairing. The binding term is therefore the `dt` bias, not
  sampling — which is the reason this plan spends its effort on an exact-flow composition rather
  than on paths.

## Results

Run: `uv run python solver.py` (antithetic pairs, dt=0.004, Nt=25, T=0.1, seed=42). Refine cycle: the OVERSAMPLE=8 pooling is reverted; both functionals are computed from the same `num_paths` terminal paths that are returned. Variance reduction costs no extra paths: a control variate on the pair-averaged samples using the exactly-known moments of the Brownian aggregate S = sum(dW) (E[S^2 - T] = 0, E[S^4 - 3T^2] = 0), fitted by regression, with the se computed on the controlled estimator (delta-method influence function for the variance). Scheme, Strang composition, exact sub-flows and reflection are unchanged.

| num_paths | V mean | V variance (ddof=1) | inv_var_mean (se) | inv_var_variance (se) | reflections |
|---|---|---|---|---|---|
| 200000 | 0.132409 | 0.020189 | 11.62260 (0.00091) | 45.2610 (0.0563) | 11 |
| 50000 | 0.132198 | 0.019138 | 11.62259 (0.00183) | 45.2974 (0.1123) | 3 |

Exact references: inv_var_mean 11.622786, inv_var_variance 45.235106. Variance functional se is 0.12% relative at 200000 paths (0.25% at 50000), against the 10% tolerance. Positivity holds structurally (min V 0.0151). The dW and observables hooks both ran.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: surrogate
estimated_rel_error: 0.7516
error_is_estimate: true
observed_order: 0.8861
observed_order_se: 0.0984
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 0.3326
invariant_worst_headroom: 0.3095
mean_rel_err: 0.5766
variance_rel_err: 0.7516
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 2.478e-03
functional_bias_rel: 4.579e-08
functional_se_rel: 2.478e-03
functional_limited_by: sampling
order_check: true
mc_se_rel: 0.0607
resolved: false
dynkin_z: 1.8364
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: dbc24d24d4cea399467550d997adb3c9fc6523407a5dd2e7ab81a2da553c8b51
solve_time_s: 0.1150
wall_time_s: 5.9300
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              surrogate  (stationary_density at T_stat = 6; moment_ode "does not close exactly" and kolmogorov "no kolmogorov.bounds declared" were attempted and unavailable)
- mean_rel_err:            0.5766 +/- 2.0 x 6.19e-04 (absolute se); moments_ok unavailable. This compares V's transient mean at T = 0.1 with the stationary mean, so it is not a solver error
- variance_rel_err:        0.7516 +/- 2.0 x 1.16e-03 (absolute se); variance_ok unavailable, same reason
- mc_se_rel:               0.0607   resolved: false (variance CI is 0.213 wide against a 0.05 half-band). MC-inconclusive in moment space on a reference that does not describe T = 0.1. Not a solver bug
- functional_rel_error:    2.48e-03 against functional_rel_tol 0.10  (bias 4.6e-08 + sampling 2.48e-03 -> limited by sampling, binding functional inv_var_variance). Graded values 11.62259 (exact 11.622786, error 1.7e-05 relative) and 45.2974 (exact 45.2351, error 1.4e-03 relative); both `clean`
- observed_order:          0.886 +/- 0.098  (expected 0.5, one-sided below the declared floor; median 1.03)
- weak_order:              indeterminate, at the noise floor
- dynkin:                  ok, z = 1.84 against bar 2.43; replicated, median z = 1.50 over 3 seeds (seed z: 1.84, 1.50, 0.82)
- terminal_law:            not declared
- shape:                   skewness 8.68 +/- 1.24, excess kurtosis 182 +/- 55 (heavy right tail of V)
- constraints:             finite ok, positivity ok, martingale ok (drift 0.333 against bar 1.94), estimator_stability ok (headroom 0.31 decades)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric — no surrogate: the orders, Dynkin and the constraints all hold and the Richardson moments are stable across the ladder. Kernel score 9, certified false

### Specific checks requested (solver sha dbc24d24, matches the run)

**1. Contract finding.** Cleared in substance. `functionals` come from the same `num_paths` terminal paths that are returned, with no extra path budget (the solver draws one `(m, Nt)` block and no oversampling remains). One nuance remains: the reported `inv_var_mean` and `inv_var_variance` are the control-variate-adjusted estimators, not the plain `mean(1/V_T)` and `var(1/V_T, ddof=1)` of `terminal_paths`. At the graded 50000-path call the plain values on the returned sample are 11.6167 and 45.0455, against the reported 11.6226 and 45.2974, a gap of 0.20 and 2.2 plain standard errors (plain se 0.030 and 0.43 at 50000 paths). So the simulation behind the number is the returned one, but the number is not reproducible from `terminal_paths` by the plain formula. Same data, a different estimator. Disclose it in SOLUTION.md; I do not treat it as a defect and apply no cap.

**2. Is the control variate unbiased and is its error bar honest.** I ran 200 independent seeds at 50000 paths, dt = 0.004 (a check of my own, outside the kernel):

| | CV estimate (mean over seeds) | exact | difference | s.e. of that mean | empirical sd of one run | reported se (mean) |
|---|---|---|---|---|---|---|
| `inv_var_mean` | 11.62269 | 11.622786 | -9.5e-05 | 1.3e-04 | 1.88e-03 | 1.83e-03 |
| `inv_var_variance` | 45.2390 | 45.2351 | +3.9e-03 | 8.0e-03 | 0.1136 | 0.1124 |

  - Bias: both within 0.8 s.e. of the seed average of zero, i.e. below 1e-5 relative on the mean and 1e-4 on the variance. The regression uses the known control means (zero) with the beta fitted on centred data, then subtracts `beta.C` uncentred, so the estimator is `z - beta.C` with exact-zero control expectation. The only bias is the fitted-coefficient one, O(1/M) at two controls, about 0.01 s.e. at 25000 pairs. It does not matter at this path count.
  - The error bar: reported se over empirical sd is 0.973 for the mean and 0.989 for the variance. Both are inside the roughly 5% noise on an sd estimated from 200 seeds, so I find no evidence of understatement. Any understatement from fitting beta on the same sample is at most about 1 to 3%, because the residual standard deviation is taken with ddof=1 and no allowance is made for the two fitted coefficients. Immaterial against a 0.10 tolerance.
  - The reduction is real and priced: the empirical sd falls from 0.0108 to 0.00188 on the mean (5.7x) and from 0.430 to 0.114 on the variance (3.8x) against the plain estimator on the same paths. `functional_se_rel` for the variance functional is 2.5e-03, from 9.8e-03 before this change, bought without extra paths.
  - Pair handling is right: the controls `S^2 - T` and `S^4 - 3T^2` are even in S, so the antithetic partner shares them and the pair-average equals the control evaluated at S. The `2m/(2m-1)` factor and the delta-method influence function for the variance are consistent with the 200-seed result.
  - Caveat that is not a defect: the kernel's CRN ladder and Dynkin runs pass `dW`, so the control-variate branch is never exercised by the bias evidence. That is inherent to a wrapper that cannot be applied to a supplied sample, but note that the ladder certifies the scheme and not the graded estimator. My 200-seed run is what covers the estimator.

**3. Exact reciprocal-CIR transition.** Not reached, directly or indirectly. The controls are built only from `S = sum(dW)` and `T`, and use the Brownian expectations `E[S^2]=T`, `E[S^4]=3T^2`. The reference values 11.6228, 45.2351 and 180.324 appear in `solver.py` only as printed strings inside the `__main__` block, not in `solve_sde`. The step consumes `dW` (strong order 0.886 +/- 0.098, median 1.03, band one-sided), so a near-zero order would have been a defect and it is not. `exact_sampling_components` is empty in the kernel's plan record (`plan.exact_sampling_components: []`). `scheme_family: other` is declared, as before, so the kernel applies the one-sided band.

### Other findings
- Dynkin passed on all three functions (z = -1.84, -1.47, 1.53 against a family-wise bar of 2.43). The first seed landed 0.59 from the bar, so it was replicated: median z = 1.50, pooled z = 2.40 (agrees). The finer residual difference was inside its own noise so no extrapolation order could be read and order 1 was assumed. The decision is a mild one and the check is not marginal on the median.
- Reflections occurred on a handful of paths in the earlier table (3 to 11). They are the exact reflection of the driftless noise sub-step, applied inside the step to the Lamperti variable, not a post-update clip.
- `cross_plan_agreement` is unavailable here: the stored comparisons against plans 2 and 3 are stale (solver changed). The stored functional z values for inv_var_mean are 0.81 (plan 2) and 2.17 (plan 3), both under the bar 3.0, so there is no disagreement to report. The plain-sample shift above (0.2 plain se) is a reminder that the shared-seed comparison reads the controlled estimator.
- Why not 10: certified is false and the score is capped at 9 because `variance_ok` is unavailable and `resolved` is false. The kernel's only reference for V's moments is the stationary density at T_stat = 6, while the graded horizon is T = 0.1 where V is transient. No increase in paths or change of scheme can resolve this.

### Feedback for solver
- None required on the scheme or the control variate. The contract finding is cleared, the CV is unbiased and its error bar is honest to within about 3%, and the exact reciprocal-CIR transition is not touched.
- One optional line of disclosure: state in the Results that `functionals` are control-variate-adjusted estimators on the returned sample, and that the plain `mean(1/V_T)` of `terminal_paths` differs from them by up to 0.2 s.e. on the mean and 2.2 s.e. on the variance at 50000 paths (11.6167 and 45.0455 plain, against 11.6226 and 45.2974), so a reader reproducing the number from the returned sample does not take the gap for an error. The Results table in this file still shows the numbers from the previous cycle (11.62260 / 45.2610 at 200000 paths, which do match this solver; the 50000-path row matches too).
- If a stricter reading of the contract is wanted, compute the plain `mean` and `var(ddof=1)` of `1/V_T` and return the controlled values in an extra key. That would cost nothing.

### Report to the conductor and formulator
- Formulator defect, unchanged from the earlier round: `verification.stationary_density` with `T_stat = 6` is the only surrogate for V's moments at T = 0.1, where V is transient. It makes `moments_ok` and `variance_ok` unavailable and holds `resolved: false` and `certified: false` on a reference that is not valid for the graded horizon. The exact reciprocal-CIR values the problem carries are the real ground truth. The score ceiling here is a property of the spec, not the solver.

</review>
