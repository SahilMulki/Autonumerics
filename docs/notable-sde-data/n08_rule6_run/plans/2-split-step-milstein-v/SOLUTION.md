---
id: 2
plan_slug: split-step-milstein-v
scheme: strang-splitting in V (exact logistic drift half-flows) + exact Stratonovich noise flow in Z=V^-1/2 driven by the supplied dW
scheme_family: strang-splitting
boundary_treatment: none
strategy: Strang splitting of the Stratonovich form in u=1/V. Both vector fields are advanced by their exact flows (logistic drift linear in u; additive noise in Z=V^-1/2), so positivity is algebraic and no clip or cap is used.
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
- `diffusion_derivative`:  `dg/dV = 1.5 * epsilon * np.sqrt(np.maximum(V, 0.0))`
- `milstein_correction`:   `0.5 * g * dg/dV * (dW**2 - dt) = 0.75 * epsilon**2 * V**2 * (dW**2 - dt)`
- `noise_structure`: multiplicative, `state_dimension = 1`, `noise_dimension = 1`,
  `milstein_eligible: true`, `stiff: true`

**Graded quantities** (`evaluation_thresholds.functionals`, read from the returned `functionals` dict):

| key | quantity | exact reference | relative tolerance |
|---|---|---|---|
| `inv_var_mean` | `E[1/V(T)]` | `11.622786285052442` | 5% |
| `inv_var_variance` | `Var[1/V(T)]` (ddof=1) | `45.23510647598556` | 10% |

Contract calls: `solve_sde(num_paths=200000, dt=0.004, T=0.1, seed=42)` and again with
`num_paths=50000`, everything else unchanged. `Nt = max(1, round(T/dt)) = 25`, `dt = T/Nt = 0.004`.

## Numerical Scheme

The step is a Strang splitting of the **Stratonovich** form of the equation, advanced in the
reciprocal coordinate `u = 1/V`. Each of the two vector fields has a closed-form flow, so no
discretisation error is made inside either sub-step; the only error is the Strang commutator.

### Stratonovich conversion

For `g(V) = eps*V**1.5`, `0.5*g*g' = 0.75*eps**2*V**2`, so

```
dV = [ r*V - c*V**2 ] dt + eps*V**1.5 o dW,     r = kappa*theta = 4.97912,  c = kappa + 0.75*eps**2 = 77.7936
```

### Drift flow (exact, logistic, linear in u = 1/V)

`dV/dt = r*V - c*V**2` gives `du/dt = c - r*u`, so over a time `h`:
`u <- u_inf + (u - u_inf)*exp(-r*h)`, `u_inf = c/r`. The flow is applied for `h = dt/2` twice
(`a_h = exp(-0.5*r*dt)`). It maps `u > 0` to `u >= u_inf*(1 - a_h) > 0`, so V stays finite and the
drift can never produce a non-positive or unbounded state, whatever the step size.

### Noise flow (exact, additive in Z = V**-0.5)

The Stratonovich equation `dV = eps*V**1.5 o dW` becomes, by the ordinary chain rule,
`dZ = -(eps/2) o dW` for `Z = V**-0.5`: additive noise, so the exact flow over the full step is
`Z_new = Z - 0.5*eps*dW[:, n]`, then `u = Z_new**2`. The supplied increment is consumed exactly once
per step. No transition law (noncentral chi-squared or otherwise) is sampled and no reference
value appears in the code.

### The step (as implemented)

```python
u = u_inf + (u - u_inf)*a_h                 # drift, dt/2
z = np.sqrt(u) - 0.5*EPS*dW[:, n]           # noise, dt
u = z*z
u = u_inf + (u - u_inf)*a_h                 # drift, dt/2
```

`V = 1/u` at the end. Positivity is algebraic: `u = z*z >= 0` and the closing drift half-flow lifts
it to `>= u_inf*(1 - a_h) > 0`. (`z == 0` exactly has probability zero and is harmless anyway because
the drift flow follows.) There is no clip, no truncation and no cap.

### Hyperparameters

- **Method**: Strang splitting, exact logistic drift half-flows + exact Stratonovich noise flow in `Z = V**-0.5`
- **dt**: `0.004` (`Nt = 25`); `Nt = max(1, round(T/dt))`, `dt = T/Nt`, or `Nt = dW.shape[1]` when `dW` is supplied
- **num_paths**: `50000` (spec) and `200000` (graded); unchanged
- **T**: `0.1`; **seed**: `42` via `np.random.default_rng`
- **Expected strong order**: 1.0 (measured 0.89, median 1.03); the scheme has no Milstein correction term
- **Expected weak order**: 1.0 nominal (Strang is formally second order; measured weak order is at the noise floor)
- **No `exact_sampling_components`.** Every component is stepped with the supplied increments.
- **`boundary_treatment: none`** (positivity is algebraic).

## Reference for Scoring

Surrogate provenance. The graded references are the two exact reciprocal moments
`E[1/V(T)] = 11.622786285052442` and `Var[1/V(T)] = 45.23510647598556` (gated `martingale`
constraints and graded functionals), plus the Dynkin check on `1/X`, `exp(-X)`, `tanh(X)` through the
`observables` hook. V's own moments are not graded: the stationary-density reference does not apply at
`T = 0.1` (about half a relaxation time), and `E[V**3]` is infinite under the invariant law, so
`resolved: false` on V's variance is structural and not a path-count issue.

## Implementation Notes

- **Previous version.** Earlier cycles used a Milstein noise sub-step `V*(1 + s + 0.75 s^2 - 0.75 eps^2 V dt)`
  with a `V_cap = 4/(9 eps^2 dt)` truncation on the correction's drift part, between exact logistic
  drift half-flows in V. It was positive and finite but biased the reciprocal moments (about -0.9% and
  -2% to -3%) and failed the martingale constraints, because `1 + s + 0.75 s^2` approximates the
  multiplicative flow only to `O(s^3)` with `|s|` up to about 0.5. It was replaced by the exact noise
  flow above. That version's cap and its `cap_activations` counter no longer exist and are no longer returned.
- **Variance reduction considered and rejected.** Antithetic pairs `(dW, -dW)` at the same path count
  cut the standard error of `E[1/V]` by about 2.7x (pair correlation -0.86), but the binding functional
  is `Var[1/V]`, whose standard error got *worse* (0.46 vs 0.40 at 50000 paths), and the paired
  terminal sample is no longer i.i.d. for the kernel's own checks. Not used; the draw stays one
  `(num_paths, Nt)` block.
- **Do not sample the exact transition** (`Y = 1/V` is CIR) and do not use the reference values in `solver.py`.
- **Hooks.** `dW` is used exactly as supplied; `observables` are accumulated by the trapezoid rule
  `0.5*(phi(V_prev) + phi(V))*dt` and returned as `path_integrals`.
- **Return.** `terminal_paths` is V (shape `(num_paths,)`), `empirical_mean/variance` of V (`ddof=1`),
  `functionals = {inv_var_mean, inv_var_variance}` from the same terminal paths, and `functional_se`
  (`std/sqrt(M)` for the mean, a fourth-moment standard error for the variance).

## Results

(dt = 0.004, Nt = 25, T = 0.1, seed = 42; `uv run python solver.py`) -- refine cycle 2

| num_paths | mean(V) | var(V) | inv_var_mean (ref 11.6228) | inv_var_variance (ref 45.2351) |
|---|---|---|---|---|
| 200000 | 0.132530 | 0.018988 | 11.5908 (se 0.0150, -0.27%) | 45.0985 (se 0.193, -0.30%) |
| 50000 | 0.132329 | 0.018563 | 11.5875 (se 0.0299, -0.30%) | 44.6940 (se 0.387, -1.19%) |

- Terminal V: min 0.0156, max 6.4 (strictly positive, finite, no clipping, no cap). Refine cycle 3: cleanup only, `cap_activations` removed, numbers unchanged. Previous
  version had bias -0.9% / -2.1% to -3.0%; it is now -0.3% / -0.3% at 200000 paths.
- `dW` and `observables` hooks verified working; no `dW_sub` (spec declares none).

### Refine-cycle implementation notes

- Previously `scheme_family: milstein`; it is now `strang-splitting`, because the noise sub-step is no
  longer a Milstein update. The bias sat in that sub-step (`1 + s + 0.75 s^2` approximates the
  multiplicative flow only to O(s^3) with |s| up to ~0.5), so it was replaced by its exact flow.
- The SDE in Stratonovich form is `dV = [kappa*theta*V - (kappa + 0.75 eps^2) V^2] dt + eps V^1.5 o dW`.
  Drift flow: logistic, i.e. linear in u = 1/V, `u' = c - r u` (r = kappa*theta, c = kappa+0.75 eps^2),
  advanced exactly over dt/2 twice. Noise flow: `Z = V^-1/2` obeys `dZ = -(eps/2) o dW`, so
  `Z_new = Z - 0.5*eps*dW[:, n]`, `u = Z_new^2`. The step consumes the supplied increment once; no
  noncentral chi-square / CIR transition law is sampled and no reference value is used. The state is
  carried as u = 1/V, which stays >= (c/r)(1 - exp(-r dt/2)) > 0 by construction: positivity is
  algebraic and there is no clip, truncation or V cap.
- No `exact_sampling_components` declared.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: surrogate
estimated_rel_error: 0.6990
error_is_estimate: true
observed_order: 0.8861
observed_order_se: 0.0984
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: strang-splitting
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.3603
invariant_worst_headroom: 0.1516
mean_rel_err: 0.5782
variance_rel_err: 0.6990
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 8.660e-03
functional_bias_rel: 8.797e-07
functional_se_rel: 8.659e-03
functional_limited_by: sampling
order_check: true
mc_se_rel: 0.0439
resolved: false
dynkin_z: 1.8364
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 00b859aeaa3c484c4fe6992bd56bf8a2fe211287ed8fcfb16a03bfb5278a014a
solve_time_s: 0.1090
wall_time_s: 5.4400
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              surrogate  (reference route: stationary_density, T_stat = 6; moment_ode and kolmogorov attempted and unavailable; the exact reciprocal-CIR values enter only as the gated martingale constraints and graded functionals)
- mean_rel_err:            0.578 +/- 2.0 x 6.1e-04 (se_mean) -- not a solver error: the reference is the stationary mean 0.0838 and the graded time T = 0.1 is far from stationary
- variance_rel_err:        0.699 +/- 2.0 x 8.1e-04 (se_variance) -- same
- mc_se_rel:               0.0439   resolved: false (structural, see below; not a path-count defect)
- functional_rel_error:    8.66e-03 against functional_rel_tol 0.10  (bias 8.8e-07 + sampling 8.66e-03 -> limited by sampling, binding functional inv_var_variance)
- observed_order:          0.886 +/- 0.098 (median-based 1.031; expected floor 0.5, one-sided band)
- weak_order:              indeterminate -- at the noise floor (all three test functions)
- dynkin:                  ok; replicated over 3 seeds, median z = 1.50 (seed z = 1.84, 1.47, 1.53) against bar 2.43; first seed was 0.59 below the bar, so replication was warranted. Extrapolation order assumed 1, not read (finer residual differences inside their own MC noise)
- terminal_law:            not declared
- shape:                   skewness 6.99 +/- 0.40, excess kurtosis 94.4 +/- 10.5 (V itself, heavy tailed)
- constraints:             finite ok; positivity ok (no violations, neg_fraction 0); martingale ok, worst drift 1.36 against bar 1.93 (headroom 0.15 decades, thin); estimator_stability ok
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- no surrogate-based certification: orders, Dynkin and constraints hold and Richardson is stable; certified: false (headroom_fraction -14.9 driven by the V-moment comparison against the non-stationary reference)

**Body vs code (conductor request).** The body of SOLUTION.md now matches solver.py. Scheme, Stratonovich constants (r = kappa*theta, c = kappa + 0.75 eps^2), the logistic half-flows linear in u = 1/V, the full-dt additive flow z = sqrt(u) - 0.5*eps*dW[:, n], u = z*z, the positivity argument, the Nt = dW.shape[1] handling, the `scheme_family: strang-splitting` frontmatter and the empty `exact_sampling_components` all agree with the source. `cap_activations` is gone from both. No cap, no state_clip warning, no reference value in the code, and the strong order (0.886, median 1.03) confirms the supplied increments are consumed. The earlier cap-worthy mismatch is resolved; no agent_cap applied. Cosmetic only: the Results section's "refine cycle 2" header coexists with a "refine cycle 3" note, and the 50000-path table row shows inv_var_variance -1.19% (se 0.87%), consistent with the kernel's graded figure.

**Why 9 and not certified.** Unchanged from round 2 and not caused by the solver. V's own moments are compared to a stationary-density surrogate that does not apply at T = 0.1 (about half a relaxation time), so moments_ok and variance_ok read unavailable, resolved is false, and estimated_rel_error is 0.699. The graded reciprocal moments are accurate (ladder errors 6.1e-06 and 8.8e-07; graded-run values 11.588 and 44.69 against 11.6228 and 45.2351, both well inside 5% and 10%). This is a formulator/spec issue (stationary-density route attached to a non-stationary terminal time), not a plan defect.

**Antithetic decision.** The solver's call was right. The graded binding functional is Var[1/V] (sampling-limited, 8.66e-03 of a 0.10 tolerance), and pairing (dW, -dW) induces strong negative correlation within pairs that helps a mean statistic (E[1/V], se about 2.7x lower) but inflates the variance of a second-moment estimator, as the solver measured (0.46 vs 0.40). It also breaks the i.i.d. terminal sample that the kernel's own variance, KS and fourth-moment standard-error formulas assume, which would have misreported functional_se. E[1/V] has 5% slack and is not binding, so its gain buys nothing. Note this is a judgment from the solver's reported numbers and the kernel's functional split; the kernel did not run an antithetic variant.

### Feedback for solver
- None required for the score. No scheme change needed; the 9 is bound by the rubric and the V-moment reference, neither of which this solver controls.
- Optional robustness only: martingale headroom is thin (worst drift 1.36 against bar 1.93, 0.15 decades). The residual is the Strang commutator at dt = 0.004; Richardson shift is within sampling noise (0.8 to 1.1 se), so leave dt as is.
- Dynkin z (median 1.50, bar 2.43) is a property of the check at this path count and is not a defect.

</review>
