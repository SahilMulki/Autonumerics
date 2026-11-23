---
id: 3
plan_slug: euler-maruyama-control
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Baseline Euler-Maruyama using the step increments dW only — the discretised cross-check that measures Clark and Cameron's c*sqrt(dt) lower bound and is expected to miss the 2% pathwise gate.
---

## SDE Reference

Two-dimensional Ito system driven by **two independent** Brownian motions W1, W2:

```
dS = mu dt + dW2,     S(0) = S_0 = 1
dU = S dW1,           U(0) = U_0 = 0
```

State is `(S, U)` in that order, so `terminal_paths[:, 0] = S(T)`, `terminal_paths[:, 1] = U(T)`.

Parameters: `S_0 = 1.0`, `U_0 = 0.0`, `mu = 1.0`. Time interval `[0, T]` with `T = 1.0`.

Drift / diffusion:

```
f(S, U) = [mu, 0.0]
G(S, U) = [[0.0, 1.0],
           [S,   0.0]]          against the channel pair (dW1, dW2)
```

**Channel convention (load-bearing for a pathwise grade):** the last axis of `dW` is
`channel 0 = W1` (drives U) and `channel 1 = W2` (drives S). This is the formulator's
requirement `R12`, recorded as `ambiguous` because `problem.md` fixes the numbering of W1/W2
in the equation display but never spells the axis mapping out. Do not swap them — against a
fixed reference path a swap is a different terminal state, not a distributionally equivalent one.

Exact moments at `T = 1` (from `analytic_moments`): `mean_S = 2`, `var_S = 1`, `mean_U = 0`,
`var_U = 17/6 = 2.8333333333333335`, `cov(S, U) = 0`. Closed forms:
`mean_X = S_0 + mu*t`, `variance_X = t`, `mean_Y = 0`,
`variance_Y = ((S_0 + mu*t)**3 - S_0**3)/(3*mu) + t**2/2`.

## Numerical Scheme

- **Method**: Euler-Maruyama, left-endpoint Ito, using the step increments `dW` only.
  `dW_sub` is accepted by the signature and **not used**.
- **dt**: 0.04 (the contract's step; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 16000 (`evaluation_thresholds.num_paths`; the contract call itself is
  `solve_sde(num_paths=1000, dt=0.04, T=1, seed=42)` and the kernel uses
  `graded_num_paths = 1000`, `conv_num_paths = 2000` on the CRN ladder)
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: none. `milstein_eligible` is `false` with
  `milstein_ineligible_reason: levy_area` — the multi-dimensional correction needs the Levy area
  of `(W1, W2)`, which the step increments do not carry. This plan takes no correction and so is
  graded against the `dW`-only floor.
- **Expected strong order**: 0.5 (`verification.expected_strong_order`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`has_analytic_solution: true`), backed by the
  exactly-closing moment ODE on component 0 and the four `dynkin_test_functions`. The pathwise
  2% gate is scored by the harness against its own fine-grid reference, not by this plan.

### The step

```
S_{n+1} = S_n + mu*dt + dW[:, n, 1]            # exact, no discretization error
U_{n+1} = U_n + S_n * dW[:, n, 0]              # left-endpoint Ito, order 0.5 strong
```

## Implementation Notes

- **This plan is expected to fail the 2% pathwise gate, and that is its job.** The omitted term
  is the Levy-area integral `I_{1,2} = int (W2(s) - W2(t_n)) dW1(s)`, with per-step variance
  `dt^2/2`. Over `Nt = T/dt` steps the strong error is `sqrt(T*dt/2) ~ sqrt(dt)` — exactly
  Clark & Cameron's (1980) `c*sqrt(dt)` lower bound for any scheme seeing only `dW`. At
  `dt = 0.04` the RMS residual is `sqrt(25 * dt^2/2) = 0.141`, i.e. **8.4%** of
  `sqrt(E[U_T^2]) = 1.683` and **5.1%** of the full-state `sqrt(E[|X_T|^2]) = 2.799`. Both are
  above the 2% gate, and no refinement of `dt` at the contract step rescues it: reaching 2%
  pathwise would need `dt` smaller by roughly a factor of 7-18, which `R11` forbids.
  **It is numbered last for that reason** — it is the discretised baseline against which the
  Levy-area plans' order-1 claim is measured, not a candidate to ship.
- **What it is good for.** The strong-order measurement is clean and cheap here: reading 0.49 on
  this plan and ~1.0 on a Levy-area plan, on the same Brownian path, is the sharpest available
  confirmation that the sub-increment assembly in those plans is correct rather than accidentally
  equivalent to Euler. Its **moments are also correct**: the left-endpoint Ito step gives
  `E[U] = 0` exactly at every `dt`, and `Var[U]` converges to `17/6` as `dt -> 0`, so this plan
  should pass the moment gates, the `finite` constraint, the martingale invariant and Dynkin
  while missing only the pathwise accuracy bar.
- **One update per step of `dW`.** `Nt = dW.shape[1]`, `dt = T / Nt`. `R22`'s exclusion rule
  applies to this plan too — do not loop at the sub-step.
- **`dW_sub` is accepted and ignored.** Keep it in the signature so the call contract is
  satisfied; do not read it. The 1.6 GB array at 16000 paths costs nothing extra if untouched.
- **Never draw the Brownian path when `dW` is given** (`R13`). Keep the `dW is None` branch for
  standalone runs only, seeded with `np.random.default_rng(seed)` — not `np.random.seed` (`R17`).
- **Positivity / clipping:** none. The system is linear with affine coefficients; `S` and `U`
  are both signed and unbounded. The only gated constraint is `finite` (`np.isfinite(X)`).
- **Left endpoint, not midpoint or right.** `S_n`, not `S_{n+1}`. A right-endpoint or midpoint
  integrand makes the step Stratonovich, picks up a non-zero `E[U(T)]`, and is caught by the
  `martingale` invariant and the `X[:, 0]*X[:, 1]` Dynkin probe even when the variance looks right.
- **`exact_sampling_components` is deliberately absent.** `S`'s step is exact but consumes the
  supplied `dW[:, n, 1]`, so the kernel's automatic excuse for an exactly-integrating component
  applies; nothing here samples a transition law.
- **No variance reduction, on purpose** (manual rule 4). The problem declares no payoff or
  functional and is graded **path by path** against a fixed supplied Brownian path, so an
  antithetic pair or control variate would require drawing or negating that path (`R13` forbids
  it) and would break the pathwise comparison. There is no `functional_se` to report.
- **Mean check on component 1 is skipped**: `|mean_U| = 0 < near_zero_mean_threshold = 0.01`.
- **`U(T)` is not Gaussian** (measured excess kurtosis 1.32); `verification.terminal_law` is
  `null`, so expect no distributional test and assume no normality.
- **Return contract**: `terminal_paths` shape `(num_paths, 2)`, `empirical_mean` and
  `empirical_variance` as lists of floats, one per component, variance with `ddof=1` (`R15`,
  `R16`). `solve_sde` must stay importable — no module-load work outside
  `if __name__ == "__main__"` (`R19`). numpy and the standard library only (`R20`).

## Results

Ran `solver.py` (16000 paths, dt=0.04, T=1, seed=42, Nt=25):

- Empirical mean (S, U):     [2.004149, -0.021424]   (exact: [2, 0])
- Empirical variance (S, U): [0.993789, 2.794445]    (exact: [1, 17/6 = 2.8333])

Variance relative errors: S 0.6%, U 1.4% (both within the 10% gate). Mean of S 0.2% off; mean of U skipped as |exact| < 0.01 (observed -0.021 is within sampling noise, SE about 0.013). The `dW` hook (dt taken from `dW.shape[1]`) and the `observables` hook (`path_integrals`) both run. `dW_sub` is accepted and unused, as planned. The pathwise 2% strong-error gate is expected to be missed (Clark-Cameron sqrt(dt) bound).

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 0.0162
error_is_estimate: false
observed_order: 0.4965
observed_order_se: 0.0348
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0178
invariant_worst_headroom: 0.1714
invariants_not_reported: ['zero_channel_cross_variation']
reference_outcome: validated
mean_rel_err: 3.675e-03
variance_rel_err: 0.0162
moment_bias_binding: variance
moment_bias_frac_tol: 0.3874
moment_bias_actionable: true
order_check: true
mc_se_rel: 0.0149
resolved: true
dynkin_z: 2.2261
dynkin_bar: 2.5310
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 9ecba45853196392a018bcbdb6021c80156916382a4c8ee5ca603813e902a2e0
solve_time_s: 1.3790
wall_time_s: 31.3700
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated; moment ODE/Kolmogorov/stationary density not needed)
- mean_rel_err:            S 0.0037 +/- 2.0 x 0.0079 (abs SE); U absolute mode, |err| 0.0178 +/- 2.53 x 0.0132 (family-wise); exact mean 0 so the near-zero skip applies
- variance_rel_err:        S 0.0039 +/- 2.0 x 0.0111 (abs SE); U 0.0162 +/- 2.0 x 0.0416 (abs SE) against tol 0.10
- mc_se_rel:               0.0149   resolved: true
- functional_rel_error:    not declared (no functional in the spec)
- observed_order:          0.4965 +/- 0.0348 (expected 0.5, one-sided below the floor; median-based 0.438). Component 0 (S) abstained as unresolved because the scheme integrates it exactly (ladder differences at roundoff); a property of the estimator, not of the solver.
- weak_order:              0.83 on x^2 (other probes at the noise floor); reported, not scored on
- dynkin:                  ok. Replicated: first seed z = 2.226 against family-wise bar 2.531 (within the 0.7 margin), so re-run on 3 seeds (42, 3444837047, 2669555309), worst-function z = 2.226 / 1.942 / 1.516, median 1.942, below the bar. Per function (seed 42): X0^2 z = -0.48, X1^2 z = -0.79, X0*X1 z = 2.23, tanh(X1) z = 1.79; all four pass. Initial state read as deterministic [1.0, 0.0]. Extrapolation order measured for X1^2 (1.36), assumed 1 for the others because the finer residual difference was inside its own noise.
- terminal_law:            not declared (U is non-Gaussian by design)
- shape:                   skewness U 0.028 +/- 0.045, excess kurtosis U 1.567 +/- 0.190 (spec 1.32; about 1.3 SE high, consistent)
- constraints:             finite: true; martingale: true (drift 0.0178 vs bar 0.0264, headroom 0.17 decades, the tightest); estimator_stability: true; zero_channel_cross_variation: not_reported (declared unchecked in the spec)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments, the estimates are MC-resolved, and Dynkin now reaches a verdict (pass). Tier A, certified.

Other readings. `reported_moments_agree: true` (worst rel diff 3e-15, so `ddof=1` is used). `provenance_as_expected: true`. `richardson_stable: false` is reported only. The moment-bias study finds a real, resolved dt bias on Var[U] of -3.9% at dt=0.04 (0.39 of tolerance, extrapolated 2.913 against graded 2.800); this is the expected O(dt) Euler bias and is inside the gate. The sub-increment ladder was reduced from 256 to 167 sub-steps to fit the 1 GiB cap; this plan ignores `dW_sub`, so it is unaffected.

Caveat on what Dynkin establishes here. It builds the generator from the spec's own drift and diffusion, so it confirms the paths satisfy that SDE as transcribed, not that the transcription matches `problem.md`. The X0*X1 residual (z = 2.23) is a marginal excursion on one test function and the 3-seed median settles it below the bar. The plan was marginal, not clear-cut.

### Feedback for solver
- None. The previous 9 was caused solely by the spec's initial condition being unreadable by the Dynkin check; with that fixed and `solver.py` unchanged (same sha256), the check passes and the plan is certified at 10. As a control plan its strong order (0.4965 +/- 0.035) matches the Clark-Cameron sqrt(dt) bound it was written to expose, and the -3.9% Var[U] bias is the baseline the Levy-area plans should improve on.

</review>
