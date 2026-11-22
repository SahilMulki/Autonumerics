---
id: 3
plan_slug: euler-maruyama
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Euler-Maruyama on the Ito form of the system, including the Stratonovich-to-Ito drift correction — the strong-order-1/2 baseline against which the order-1 schemes are judged.
---

This plan is **last on purpose**: it is the baseline. It is a correct solve of the right
equation and it should clear the gated `amplitude_conservation` constraint and the moment
tolerances, but its strong order is 0.5, so it is the least accurate pathwise of the pool
and is expected to rank below either order-1 scheme. Its value is twofold: it is the
simplest scheme that can get the Ito/Stratonovich convention *right*, so agreement
between it and an order-1 scheme on the terminal moments is real evidence that the
convention was handled correctly in both; and it anchors the measured strong order at
0.5, which is what makes a measured 1.0 elsewhere in the pool meaningful rather than
unverifiable.

`scheme_family: euler-maruyama` selects the spec's `expected_strong_order: 0.5` with a
two-sided band, which is the right bar here. No `exact_sampling_components` is declared,
and must not be — `implementation_notes` (4) and requirement R19 say both components must
be stepped from the supplied `dW`. There is no declared barrier or absorbing boundary, so
no `boundary_treatment` line.

## SDE Reference

**Stratonovich form** (the form `problem.md` states, `a = 1`, `sigma = 1`):

```
dX1 = -a*X2 dt - sigma*X2 o dW
dX2 =  a*X1 dt + sigma*X1 o dW,     X(0) = (1, 0)
```

**Equivalent Ito form** — *this is the form this scheme integrates*
(Stratonovich-to-Ito correction `0.5*(Dg)g = -0.5*sigma**2*X`):

```
dX1 = (-a*X2 - 0.5*sigma**2*X1) dt - sigma*X2 dW
dX2 = ( a*X1 - 0.5*sigma**2*X2) dt + sigma*X1 dW
```

Both are driven by the **same single** scalar Brownian motion `W` (noise_dimension = 1).

Parameters: `a = 1.0`, `sigma = 1.0`, `X1_0 = 1.0`, `X2_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.

Coefficients, with state `X` of shape `(num_paths, 2)`:

```
# Ito drift -- this is what Euler-Maruyama must use
f0 = -a*X[:, 1] - 0.5*sigma**2*X[:, 0]
f1 =  a*X[:, 0] - 0.5*sigma**2*X[:, 1]
# diffusion, single channel (identical in both conventions)
g0 = -sigma*X[:, 1]
g1 =  sigma*X[:, 0]
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

- **Method**: Euler-Maruyama on the **Ito** form. Per step `n`, with `dWn = dW[:, n]`,
  computing both components from the *old* state:

  ```
  X1_new = X1 + (-a*X2 - 0.5*sigma**2*X1)*dt - sigma*X2*dWn
  X2_new = X2 + ( a*X1 - 0.5*sigma**2*X2)*dt + sigma*X1*dWn
  ```

  Both updates must read `X1`, `X2` from before the step — writing `X1` in place and
  then using it for `X2` turns this into a different (Gauss-Seidel-like) scheme.

- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 50000 for the kernel's moment checks
  (`evaluation_thresholds.num_paths`); the contract call is
  `solve_sde(num_paths=2000, dt=0.04, T=1, seed=42)` and the solver must honour
  whatever `num_paths` it is handed
- **T**: 1.0
- **seed**: 42 (`np.random.default_rng(seed)`; only used when `dW is None`)
- **Milstein correction**: deliberately **not** applied. The correction
  `-0.5*sigma**2*X_n*(dW**2 - dt)` is what would lift this to strong order 1; omitting
  it is what makes this plan the order-1/2 baseline. Do not add it here.
- **Expected strong order**: **0.5** (the spec's `expected_strong_order`)
- **Expected weak order**: 1.0
- **Reference for scoring**: **analytic moments** (`has_analytic_solution: true`),
  cross-checked by the evaluator against `verification.moment_ode`, which
  `closes_exactly`. Provenance should come back `analytic`.

## Implementation Notes

1. **The `-0.5*sigma**2*X` drift terms are not optional and are the entire trap.**
   `problem.md` states Stratonovich coefficients. Euler-Maruyama is an Ito scheme, so
   it must integrate the Ito form, which carries that correction. Dropping it
   integrates a different equation whose amplitude grows like
   `exp(0.5*sigma**2*t) = 1.65` over `[0, 1]` — a 65% breach of a gated constraint
   whose bound is 0.25, and the discriminating failure this problem is built around.

2. **Amplitude behaviour.** Euler-Maruyama on the correct Ito form conserves `|X|`
   only to its own local-error order. Per step,
   `|X_{n+1}|**2 = |X_n|**2 * (1 + O(dt**2) + O(dt*dWn) ...)`; at `dt = 0.04` the
   worst-path deviation should land in the low tens of a percent — inside the gated
   bound of 0.25, but with much less headroom than an order-1 scheme has. Report
   `max(abs(sqrt(X1**2 + X2**2) - 1))` over paths in Results. If that number is near
   0.65 or beyond, the drift correction is missing, and the fix is the correction, not
   a smaller `dt`.

3. **Expect to be the least accurate plan in the pool, and let that stand.** The
   problem's own gate is a 20% relative pathwise strong error at `dt = 0.04`, which
   Euler-Maruyama at strong order 0.5 is near rather than far inside. If the evaluator
   reports the pathwise error too close to the gate, the legitimate response for *this
   plan* is a smaller `dt` only if the harness permits one — the contract pins
   `dt = 0.04` (R10), so do not hardcode a different step. Do not "rescue" this plan by
   adding the Milstein correction: that erases the pool's only order-1/2 anchor and
   makes it a duplicate of another plan.

4. **Consume the supplied `dW` exactly.** `dW` is always supplied, shape
   `(num_paths, Nt)`, **already scaled by `sqrt(dt)`**. Take `Nt = dW.shape[1]` and
   `dt = T / Nt`; draw no randomness of your own for the Brownian path. When
   `dW is None` (the standalone `__main__` smoke test), build it as
   `np.sqrt(dt) * np.random.default_rng(seed).standard_normal((num_paths, Nt))` with
   `Nt = max(1, round(T / dt))` and `dt = T / Nt`. Ignoring `dW` makes the pathwise
   grading meaningless and makes the CRN order ladder read a strong order near zero —
   which on this plan would be a real defect, not a waivable one (R19).

5. **Return contract** (R12/R13): `terminal_paths` a `(num_paths, 2)` float array with
   columns `(X1, X2)` in that order; `empirical_mean` a list of 2 floats;
   `empirical_variance` a list of 2 floats with `ddof=1`. Keep `solve_sde` importable —
   no module-level work outside `if __name__ == "__main__"`. `numpy` and the standard
   library only (R17).

6. **No variance reduction is proposed.** The spec declares no payoff functional and
   the problem is graded path by path against the exact solution driven by the *given*
   increments, so an antithetic pair would reduce nothing that is scored and would not
   change the pathwise dynamics the order ladder measures.

7. **Numerics are benign**: the state is bounded near 1, no stiffness, no positivity
   constraint, nothing to guard.

8. The Dynkin check will be built from `verification.generator` (the Ito drift and the
   2x1 diffusion matrix), with test functions `X1`, `X1**2 + X2**2`, `X1*X2`,
   `tanh(X1)`. The second has `L.phi = 0` identically — the amplitude invariant — and
   is the test function most sensitive to the omitted drift correction.

9. **Refine cycle 2 decision: keep Euler-Maruyama unchanged; `scheme_family` stays `euler-maruyama`.**
   The evaluator scored 3/10 on the gated `amplitude_conservation` constraint (7.02% of
   paths with ||X|-1| > 0.25). Considered the Ito-Milstein correction. On the mathematics:
   with the correction, the per-step update is `X_{n+1} = (1 - 0.5*sigma^2*dW^2) X + J X (a dt + sigma dW)`,
   so `|X_{n+1}|^2 = |X_n|^2 (1 + theta^4/4)` with `theta = a dt + sigma dW` -- it would very
   likely clear the gate. But for this linear single-channel problem (`Dg` constant, `J^2 = -I`)
   Ito-Milstein is algebraically the second-order Taylor truncation `I + theta J + 0.5 theta^2 J^2`,
   which is what plan 2's Stratonovich Heun reduces to, differing only in O(dt^1.5) cross terms
   (plan 2's own header calls Heun "equivalently Stratonovich-Milstein"). It would therefore be a
   near-duplicate of plan 2 and would remove the pool's only order-0.5 anchor. I judge the
   anchor worth more than the score: the 3 is the measured, intended behaviour of plain EM at
   dt = 0.04, sigma = 1 (a real finding that the gate discriminates order-0.5 from order-1 schemes),
   not a defect. No clipping, renormalisation or projection was added; no cos/sin/expm; no
   `exact_sampling_components`. The earlier instruction in item 3 (do not add Milstein) is
   therefore upheld, now on stated reasoning. `solver.py` is unchanged.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt=25, Ito-form Euler-Maruyama, no Milstein correction).

- empirical_mean = [0.321722, 0.528647]  (exact [0.327710, 0.510378])
- empirical_variance = [0.386453, 0.281917]  (exact [0.364447, 0.267674])
- max |  |X| - 1 | over the 50000 paths = 1.074 (heavy tail of a 0.5-order scheme; exceeds the 0.25 gate on the worst path). Not measured at the 2000-path graded call: an ad-hoc `uv run python -c` measurement was denied by the permission system in cycle 2 as well. The evaluator's 50000-path figure is 7.02% of paths beyond 0.25.
- dW hook: with an explicit (1000, 25) dW, dt = 0.04 returned and pathwise RMS error vs cos/sin(a*T + sigma*W_T) = 0.103.
- observables hook: path_integrals shape (1000,).

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 3
certified: false
provenance: analytic
estimated_rel_error: 0.0604
error_is_estimate: false
order_floor: 0.5000
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: false
constraints_ok: false
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0702
invariant_worst_headroom: -6.8461
invariants_not_reported: ['second_moment_sum']
reference_outcome: validated
mean_rel_err: 0.0358
variance_rel_err: 0.0604
order_check: not_run
mc_se_rel: 6.977e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: af56c6c31aafe6d2d931d494097ddef9a4e34422f7a41829e55506708a4d4d06
solve_time_s: 0.1300
wall_time_s: 1.0700
cost_source: cpu
</metrics>

<review score=3>

**Score: 3/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the moment ODE)
- mean_rel_err:            X1 0.0183 (tol 0.05, se 0.00278 -> 2 x se = 0.0056); X2 0.0358 (tol 0.05, se 0.00237 -> 2 x se = 0.0047). Both pass.
- variance_rel_err:        X1 0.0604 (tol 0.10, se 0.00224 -> 2 x se = 0.0045); X2 0.0532 (tol 0.10, se 0.00197 -> 2 x se = 0.0039). Both pass.
- mc_se_rel:               0.00698   resolved: true (CI width rel 0.0123 and 0.0147 against bar 0.05)
- functional_rel_error:    none declared
- observed_order:          not_run (expected 0.5, two-sided band declared via scheme_family). Skipped for cost because a gate had already failed; this is not a pass and not a failure.
- weak_order:              not_run
- dynkin:                  not_run (same reason; not a failure)
- terminal_law:            not declared (null in spec)
- shape:                   X2 skewness -1.036 +/- 0.0089, excess kurtosis 0.434 +/- 0.0285; X1 skewness -0.786 +/- 0.0078, excess kurtosis -0.315 +/- 0.0185
- constraints:             finite: pass. amplitude_conservation (gate): FAIL, 7.02% of the 50000 terminal paths have | |X| - 1 | > 0.25 (the gate tolerates none, bar 1e-8). estimator_stability: pass (0.0089 < 0.01). second_moment_sum: not_reported (spec says declared but not evaluable).
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                certification - a gate:true structural violation (rubric row 4, certification ceiling 3)

### Feedback for solver
- The moments are fine and MC-resolved: this is NOT an MC-inconclusive case and raising `num_paths` will not help. The score is 3 solely because the gated `amplitude_conservation` constraint fails on about 7% of paths (`invariant_max_drift` 0.0702). Your own Results note says the worst path deviates by 1.074 and that the gate is exceeded; that is the same finding.
- The drift correction is present and correct in `solver.py` (`-half*X1`, `-half*X2`), so this is not the Ito/Stratonovich misreading. It is the scheme itself. Per step, Euler-Maruyama gives |X_{n+1}|^2 = |X_n|^2 * ((1 - 0.5 s^2 dt)^2 + (a dt + s dW)^2), which drifts at O(dW^2) per step. At dt = 0.04 and sigma = 1, tail paths with large |dW| compound this into the 25% breach. No amount of care in the EM update fixes it, and `dt` is pinned by the contract (R10).
- Do not mask it by clipping or renormalising |X|; that hides the violation without fixing the scheme. The fix is a different scheme, not a patch to this one.
- To lift the score, this plan must stop being plain EM. Adding the Ito-Milstein term `-0.5*sigma**2*X_n*(dW**2 - dt)` (with m = 1 channel there is no Levy area), or a Stratonovich Heun/midpoint predictor-corrector, makes the per-step norm error O(dW^4) rather than O(dW^2) and should clear the gate. That would make this plan a duplicate of the order-1 plans, which `SOLUTION.md` warns against, so the honest alternative is to keep this plan as the documented order-0.5 baseline and accept that it cannot pass the gate at dt = 0.04. Either is a plan-level decision; the kernel's verdict on the code as written is 3.
- Order and Dynkin were not run because the gate failure caps the score first (manual section 24 cost rule). They will run on the next pass once the gate is clean, and the declared `scheme_family: euler-maruyama` gives a two-sided band around 0.5. `dW` is consumed correctly (Nt from `dW.shape[1]`), so a near-zero order is not expected.
- `second_moment_sum` is `not_reported`; that is a formulator-side declaration with no evaluable expression and is not a solver defect.
- Plan claims match code (EM, no Milstein term), so no `agent_cap` was applied.
- Second pass (spec repaired: top-level `initial_condition` now present, `solver.py` hash unchanged `af56c6c3...`): the score reproduces at 3. Order ladder and Dynkin are STILL `not_run`, because the gated `amplitude_conservation` failure caps the score first and the kernel skips them for cost (manual section 24). The spec repair makes a Dynkin verdict reachable, but it only runs once the gate is clean; nothing previously skipped has newly run. This is not a pass and not a failure.
- Breach fraction at the graded 2,000-path count (`solve_sde(2000, 0.04, 1.0, seed)`, fraction with `| |X| - 1 | > 0.25`): 7.05% at seed 42 (141 of 2000), and 6.35%, 6.50%, 6.60% at seeds 1, 2, 3; binomial se about 0.55%. For comparison 7.016% at 50,000 paths (seed 42). Mean `| |X| - 1 |` is 0.110, max 0.95 at 2000 paths. The gate bar is 1e-8, so the count is irrelevant to the verdict: it fails at any path count, and the fraction is a property of the scheme at dt = 0.04, sigma = 1, not of sampling.
- Grading note: the plan's decision not to rescue the scheme (Implementation Note 9) is recorded and no clip/renormalise/projection is present; no `agent_cap` applied.

</review>
