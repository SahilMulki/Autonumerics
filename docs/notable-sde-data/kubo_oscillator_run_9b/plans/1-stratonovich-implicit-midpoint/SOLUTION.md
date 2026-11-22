---
id: 1
plan_slug: stratonovich-implicit-midpoint
scheme: stratonovich-implicit-midpoint (Cayley rotation step)
scheme_family: milstein
strategy: Advance the state with the Stratonovich implicit midpoint rule, which for this skew-symmetric linear system closes in 2x2 form as the Cayley transform of the step angle — strong order 1 and exactly amplitude-preserving, without ever evaluating the closed-form rotation.
---

`scheme_family: milstein` is declared to select the spec's **order-1** strong-order
floor (`verification.expected_strong_order_milstein: 1.0`), which is the order this
scheme actually has. It is the only way the kernel can know the 0.5 default does not
apply: declaring a family the spec has no key for returns the 0.5 default with a
*two-sided* band, which would fail a correct order-1 scheme measuring 1.0 (the
`sde_ginzburg_landau_s6` precedent in `verifylib/kernel/sde/crn.py`). No
`exact_sampling_components` is declared, and must not be — `implementation_notes` (4)
and requirement R19 say both components must be stepped from the supplied `dW`, so a
near-zero measured strong order here would be a real defect rather than a waivable one.
There is no declared barrier or absorbing boundary, so no `boundary_treatment` line.

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
# Ito drift
f0 = -a*X[:, 1] - 0.5*sigma**2*X[:, 0]
f1 =  a*X[:, 0] - 0.5*sigma**2*X[:, 1]
# Stratonovich drift (the same, without the -0.5*sigma**2*X terms)
f0_strat = -a*X[:, 1]
f1_strat =  a*X[:, 0]
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

- **Method**: Stratonovich implicit midpoint rule, resolved in closed 2x2 form

  Write `J = [[0, -1], [1, 0]]`. Both the Stratonovich drift and the diffusion are
  `J` applied to the state: `f_strat(X) = a*J@X`, `g(X) = sigma*J@X`. The Stratonovich
  midpoint rule is therefore

  ```
  theta_n   = a*dt + sigma*dW[:, n]           # the step's total rotation increment
  X_{n+1}   = X_n + theta_n * J @ (X_n + X_{n+1}) / 2
  ```

  which for this 2x2 system solves **exactly, with no iteration**:

  ```
  X_{n+1} = (I - 0.5*theta_n*J)^{-1} (I + 0.5*theta_n*J) X_n
  ```

  and the 2x2 inverse is explicit. With `c = 1 - theta**2/4` and `d = 1 + theta**2/4`:

  ```
  X1_{n+1} = (c * X1_n - theta_n * X2_n) / d
  X2_{n+1} = (c * X2_n + theta_n * X1_n) / d
  ```

  No linear solve, no fixed-point loop, no matrix library call — just these two lines
  per step, vectorised over all paths.

- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 50000 for the kernel's moment checks
  (`evaluation_thresholds.num_paths`); the contract call is
  `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)` and the solver must honour
  whatever `num_paths` it is handed
- **T**: 1.0
- **seed**: 42 (`np.random.default_rng(seed)`; only used when `dW is None`)
- **Milstein correction**: not formed explicitly. The midpoint rule is a Stratonovich
  scheme and reaches strong order 1 without `Dg.g`. For reference, the spec's
  equivalent explicit correction is
  `0.5*(Dg.g)*(dW**2 - dt) = -0.5*sigma**2*X_n*(dW**2 - dt)` on the Ito step.
- **Expected strong order**: **1.0** (the spec's `expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: **analytic moments** (`has_analytic_solution: true`),
  cross-checked by the evaluator against `verification.moment_ode`, which
  `closes_exactly`. Provenance should come back `analytic`.

## Implementation Notes

1. **The Stratonovich-to-Ito correction is the whole problem.** This scheme avoids
   the convention error structurally rather than by remembering a sign: it uses the
   *Stratonovich* coefficients (no `-0.5*sigma**2*X` drift) inside a *Stratonovich*
   scheme. If you instead write an Ito scheme, you must add the
   `-0.5*sigma**2*X` drift. Reading the stated coefficients as Ito coefficients and
   using an Ito step integrates a different equation whose amplitude grows like
   `exp(0.5*sigma**2*t) = 1.65` over `[0, 1]`, which breaches the gated
   `amplitude_conservation` constraint.

2. **Why this scheme is first.** The Cayley transform of a real skew-symmetric matrix
   is exactly orthogonal, so `|X_{n+1}| = |X_n|` to **machine precision** on every
   path at every step. The gated constraint
   `abs(sqrt(X1**2 + X2**2) - 1) <= 0.25` is then satisfied with ~16 decades of
   headroom, not by being small-enough-at-this-dt. The remaining error is a pure
   *angle* error: the step rotates by `2*arctan(theta_n/2) = theta_n - theta_n**3/12 + ...`
   instead of `theta_n`, i.e. half the leading error constant of a second-order
   Taylor/Heun step (`theta**3/6`), and all of it tangential.

3. **Why this is a discretisation and not the closed form.** Required reading:
   `implementation_notes` (4) and (5). A per-step matrix exponential
   `exp(theta_n*J)` is exact for this flow and is at risk of **exclusion** from the
   comparison. Cayley is not `exp`: it differs by `theta_n**3/12` per step
   (~8e-4 at `dt = 0.04`, where `theta_n` has standard deviation 0.2), so the
   terminal states genuinely differ at `dt`, `dt/2` and `dt/4` by far more than
   rounding. Do **not** call `scipy.linalg.expm`, `np.cos`/`np.sin` of `theta_n`, or
   any trigonometric form of the step — those are the exact flow wearing a
   discretisation's clothes. Use the two rational lines above. (`numpy` and the
   standard library only, per R17.)

4. **Strong order 1 is reachable at `state_dimension = 2`** because
   `noise_dimension = 1`: with one channel no Levy area arises, which is why
   `milstein_eligible` is `true` here (see `diffusion_derivative`).

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

7. **No variance reduction is proposed.** The spec declares no payoff functional
   (`evaluation_thresholds` has no `functionals`) and the problem is graded path by
   path against the exact solution driven by the *given* increments, so an antithetic
   pair would halve nothing that is scored and would not change the pathwise dynamics
   the order ladder measures.

8. **Numerics are benign**: the state is bounded by 1, there is no stiffness, no
   positivity constraint, and `d = 1 + theta**2/4 >= 1` so the division is always safe.

9. The Dynkin check will be built from `verification.generator` (the **Ito** drift and
   the 2x1 diffusion matrix), with test functions `X1`, `X1**2 + X2**2`, `X1*X2`,
   `tanh(X1)`. The second of those has `L.phi = 0` identically — the amplitude
   invariant — so a scheme that preserves `|X|` exactly also makes that residual
   vanish exactly.

10. **Cycle 2: angle sub-step raised in order.** The measured mean bias came from the
   Cayley angle error `theta**3/12` per step. The step is now the symmetric composition
   `Cayley(g1*theta) o Cayley(g2*theta) o Cayley(g1*theta)` with
   `g1 = 1/(2 - 2**(1/3))`, `g2 = 1 - 2*g1` (Yoshida triple jump: `sum g = 1`,
   `sum g**3 = 0`), which cancels the cubic angle error (order 4 in the angle). Each factor
   is still the rational Cayley map, so no `exp`/`cos`/`sin`/`arctan` is evaluated and
   `|X|` stays preserved to rounding (4e-15). `scheme_family` stays `milstein`: the
   scheme is still an implicit-midpoint (Stratonovich) method, now composed; it was
   `milstein` before as well, and only its design order in the angle changed. The strong
   order measured against the exact flow may now exceed 1 (an overshoot is flagged, not failed).

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt = 25, composed-Cayley step).

- Empirical mean (X1, X2): [0.328328, 0.513936]  (analytic 0.327710, 0.510378)
- Empirical variance (X1, X2): [0.363270, 0.264813]  (analytic 0.364447, 0.267674)
- Max | |X| - 1 | over paths: 4.0e-15 (amplitude preserved to machine precision)
- `dW` and `observables` hooks confirmed working.
- Cycle 2 change: the angle sub-step was raised from one Cayley map (angle error theta^3/12) to a
  Yoshida triple-jump composition of three Cayley maps (previous mean (X1, X2): 0.335395, 0.515565).

## Evaluation

See the metrics and review blocks below.

<metrics>
kind: sde
score: 3
certified: false
provenance: analytic
estimated_rel_error: 0.0107
error_is_estimate: false
observed_order: 1.8328
observed_order_se: 0.0138
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 8.556e-03
invariant_worst_headroom: 0.0677
invariants_not_reported: ['second_moment_sum']
reference_outcome: validated
mean_rel_err: 6.972e-03
variance_rel_err: 0.0107
moment_bias_binding: mean
moment_bias_frac_tol: 2.842e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.953e-03
resolved: true
dynkin_z: 2.3124
dynkin_bar: 2.5310
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: e8cd2fe3687047c2739dc8fa683be0b6b1df8e67880e4e272a76c789d58525a2
solve_time_s: 0.1320
wall_time_s: 39.1000
cost_source: cpu
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated)
- mean_rel_err:            comp1 0.0019 +/- 2 x 0.0027 (abs se); comp2 0.0070 +/- 2 x 0.0023 (abs se); tol 0.05 -- pass
- variance_rel_err:        comp1 0.0032 +/- 2 x 0.0018; comp2 0.0107 +/- 2 x 0.0018; tol 0.10 -- pass
- mc_se_rel:               0.0070   resolved: true (CI widths 0.0100 / 0.0138 vs bar 0.05)
- functional_rel_error:    none declared
- observed_order:          1.833 +/- 0.014 (median 1.708; expected 1.0, one-sided below the declared floor) -- strong_order_above_band: true, flagged and not failed
- weak_order:              1.96, out of band (above the theoretical 1.0), reported only
- dynkin:                  FAIL by replication. First seed z = 2.31 (bar 2.53, 0.22 from it, so it was marginal and re-run on 3 seeds); per-seed worst z = 2.31 / 3.20 / 4.39, median z = 3.20 > 2.53, pooled z = 5.72. dynkin_replicated: true
- terminal_law:            not declared
- shape:                   comp1 skew -0.684 +/- 0.007, ex. kurtosis -0.753 +/- 0.014; comp2 skew -1.159 +/- 0.009, ex. kurtosis 0.417 +/- 0.028
- constraints:             estimator_stability true (drift 8.6e-3 vs 1e-2, headroom 0.07 decades); finite true; amplitude_conservation true (0); second_moment_sum not_reported
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- Dynkin fails (certification cap agrees, 3)

The Dynkin generator check reached a verdict this time (initial state resolved to x0 = [1, 0], deterministic). Per test function (extrapolated residual, s.e., z): X1: -4.0e-3, 2.7e-3, z -1.52; X1^2+X2^2: -8.1e-12, 3.5e-12, z -2.31 on seed 42 (up to 4.39 on another seed); X1*X2: 1.8e-3, 3.3e-3, z 0.55; tanh(X1): -3.5e-3, 2.1e-3, z -1.65. All four per_function_ok flags are true. The three functions that carry signal are comfortably inside noise, and the Stratonovich-vs-Ito drift discrimination this problem is about is not what is failing: an Ito-drift solver would put a residual of order 0.5*sigma^2 times the test function's curvature, i.e. O(1e-1), against these O(1e-3) values. The failing statistic is the one on X1^2+X2^2, whose residual is at 1e-11 (floating-point scale, 6e-12 to 8e-11 across the three levels) because the Cayley map conserves the amplitude exactly; its s.e. is 3.5e-12 and a z computed from a roundoff-scale residual over a roundoff-scale s.e. is not a statement about the SDE. I am transcribing the score the kernel computed and not capping upward, but I flag this to the conductor as a probable kernel artifact: a residual at roundoff relative to the terminal scale (cf. the order study's `roundoff_floor` of 6.6e-13 on the same terminal values) should abstain as `unresolved`, not fail, and if it did, this plan would score 10 rather than 3 on every other row (moments, order, constraints, resolved all clean).

### Feedback for solver
- The Yoshida triple-jump took effect. Strong order read 1.833 +/- 0.014 (median 1.708), above the 1.0 floor; the kernel's one-sided band flags the overshoot and does not charge it, and `orders_ok` is true. The overshoot is also consistent with the amplitude being exactly conserved so the remaining error is a pure phase error that the composition cancels to higher order. Not a defect, and no change to `scheme_family: milstein` is needed for scoring.
- Moments are well inside tolerance and MC-resolved: worst variance error 1.07% (comp 2) against 10%, worst mean error 0.70% against 5%. The measured dt bias is 0.28% of the mean tolerance, so there is nothing for a smaller dt to buy (`moment_bias_actionable: false`). Do not raise `num_paths` or reduce dt.
- The only blocker is `dynkin_ok: false` (score 3). Do not attribute it to the drift: the three functions with a measurable residual have |z| <= 1.65, and a wrong-drift integrator would give z in the tens to hundreds. The failing component is X1^2+X2^2, whose residual (1e-11) is floating-point noise from an exactly amplitude-preserving map. Nothing in `solver.py` needs to change for this. If you want to test that reading, perturb the update by one ulp-scale operation (e.g. a different but algebraically identical evaluation order of the Cayley rotation) and see whether that z moves; if it does, it is roundoff, not the equation.
- Report to the conductor/kernel maintainers: Dynkin z on a residual at roundoff scale should abstain. Not a solver action.
- `second_moment_sum` is `not_reported`: the solver does not expose the quantity that invariant reads, so it was neither passed nor failed. Report it if the invariant is meant to be checked.

</review>
