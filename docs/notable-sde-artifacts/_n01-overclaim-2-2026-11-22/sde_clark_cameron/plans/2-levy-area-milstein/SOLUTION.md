---
id: 2
plan_slug: levy-area-milstein
scheme: milstein-levy-area
scheme_family: milstein
levy_area_treatment: sub-increments
strategy: The canonical order-1 Levy-area Milstein scheme for the Clark-Cameron system — exact step for S, and U advanced with S_n*dW1_n plus the iterated integral I_{1,2} assembled from the supplied sub-increments, with no other correction term.
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

- **Method**: Levy-area Milstein exactly as the spec's `milstein_correction` states it — one
  update per step of `dW`, iterated integral assembled from `dW_sub`.
- **dt**: 0.04 (the contract's step; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 16000 (`evaluation_thresholds.num_paths`; the contract call itself is
  `solve_sde(num_paths=1000, dt=0.04, T=1, seed=42)` and the kernel uses
  `graded_num_paths = 1000`, `conv_num_paths = 2000` on the CRN ladder)
- **T**: 1.0
- **seed**: 42
- **Milstein correction** (verbatim from `problem_spec.json`): for the Clark-Cameron system the
  only non-trivial iterated integral is
  `I_{1,2}(t_n, t_{n+1}) = int_{t_n}^{t_{n+1}} (W2(s) - W2(t_n)) dW1(s)`. The order-1 step is
  `S_{n+1} = S_n + mu*dt + dW2_n` (exact) and
  `U_{n+1} = U_n + S_n*dW1_n + I_{1,2}_n`, with `I_{1,2}_n` assembled from the sub-increments
  as `sum_j (partial sum of dW2 up to j) * dW1_j`.
- **Expected strong order**: 1.0 (`verification.expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`has_analytic_solution: true`), backed by the
  exactly-closing moment ODE on component 0 and the four `dynkin_test_functions`. The pathwise
  2% gate is scored by the harness against its own fine-grid reference, not by this plan.

### The step

`S` is exactly integrable and consumes the supplied increment:

```
S_{n+1} = S_n + mu*dt + dW[:, n, 1]            # exact, no discretization error
U_{n+1} = U_n + S_n * dW[:, n, 0] + I12[:, n]
```

Vectorised assembly of `I_{1,2}` (let `M = dW_sub.shape[2]`, do **not** hardcode 256):

```python
d1 = dW_sub[:, :, :, 0]                                 # W1 sub-increments
d2 = dW_sub[:, :, :, 1]                                 # W2 sub-increments
W2_partial = np.cumsum(d2, axis=2) - d2                 # exclusive prefix sum: W2(s_j) - W2(t_n)
I12 = np.einsum("pnj,pnj->pn", W2_partial, d1)          # one value per path per step
```

### Where the remaining error lives

Expanding `S_s = S_n + mu*(s - t_n) + (W2(s) - W2(t_n))` inside `int S_s dW1(s)` gives three
terms: `S_n*dW1_n`, `mu*I_{0,1}` with `I_{0,1} = int (s - t_n) dW1(s)`, and `I_{1,2}`. This
scheme carries the first and third exactly, so the per-step residual is `mu*I_{0,1}` with
variance `mu^2 * dt^3 / 3`. Over `Nt = T/dt` independent steps the strong error is
`sqrt(T*dt^2/3) ~ dt` — order 1, which is the floor the plan is held to. At `dt = 0.04`:
RMS residual `sqrt(25 * dt^3/3) = 0.0231`, i.e. **1.37%** of `sqrt(E[U_T^2]) = 1.683` and
**0.82%** of the full-state `sqrt(E[|X_T|^2]) = 2.799`. Clears the 2% pathwise gate; Euler-Maruyama,
whose residual is the omitted `I_{1,2}` with per-step variance `dt^2/2`, lands at 0.141 (8.4% / 5.1%)
and cannot — that is Clark & Cameron's `c*sqrt(dt)` lower bound, and it is the reason this scheme
reads `dW_sub` at all.

**Do not also assemble `I_{0,1}` exactly from `dW_sub`.** Adding `mu * sum_j (s_j - t_n) * dW1_j`
makes the update algebraically identical to a plain Euler sum over the whole sub-grid, because
`sum_j S_{s_j} dW1_j = S_n*dW1_n + mu*I_{0,1}^{sub} + I_{1,2}^{sub}`. The kernel holds the
sub-step fixed at the graded `dt/256`, so such a solver returns terminal states identical to
rounding at `dt`, `dt/2` and `dt/4` and is **excluded** from the comparison as having bypassed
the step (`problem.md`, requirement `R22`).

## Implementation Notes

- **One update per step of `dW`.** `Nt = dW.shape[1]`, `dt = T / Nt`. Never loop at the sub-step
  to advance the state: reduce `dW_sub` to the per-step scalar `I12` first (the `einsum` above
  contracts the sub-axis away), then take a single step. This is `R9`/`R22`/`R23`.
- **Never draw the Brownian path when `dW` is given** (`R13`). Keep the
  `dW is None` branch of the template for standalone runs only, seeded with
  `np.random.default_rng(seed)` — not `np.random.seed` (`R17`).
- **`dW_sub` may be withheld.** The contract says it is always supplied, but
  `implementation_notes` records that the kernel may omit it at the 16000-path moment call
  (1.6 GB at that path count). Handle `dW_sub is None` by dropping `I12` to zero — i.e. falling
  back to the left-endpoint Euler step for `U`. The moment evidence is unaffected: the Ito
  left-endpoint step already gives `E[U] = 0` and the exact `Var[U]` in the `dt -> 0` limit, and
  it is only the pathwise strong error that needs the Levy area. Do **not** silently substitute
  a self-drawn sub-path.
- **Memory.** `dW_sub` is `num_paths * Nt * M * 2` float64 = `num_paths * 102400` bytes,
  independent of ladder level. Prefer chunking over paths when forming `W2_partial`, so the
  cumulative-sum temporary does not double the 1.6 GB input at 16000 paths.
- **Sign and endpoint of the correction.** The prefix sum must be **exclusive** —
  `cumsum(d2) - d2`, the W2 displacement *before* sub-step `j` — and it must multiply the W1
  sub-increment of that same sub-step. An inclusive prefix sum computes the Stratonovich-corrected
  integral and adds a spurious `O(dt)` bias; this is the multi-dimensional analogue of the
  classic Milstein error of applying the correction at `X_{n+1}` instead of `X_n`. If the measured
  strong order comes back near 0.5, the prefix sum or the channel mapping is the first place to look.
- **Positivity / clipping:** none. The system is linear with affine coefficients; `S` and `U`
  are both signed and unbounded. The only gated constraint is `finite` (`np.isfinite(X)`).
- **Martingale invariant (non-gate).** `E[U(T)] = 0` exactly, and every term in the `U` update is
  a left-endpoint-adapted Ito term with zero mean, so it holds by construction. Evaluating `S` at
  the right endpoint or the midpoint makes the step Stratonovich and picks up a non-zero mean; the
  `martingale` invariant and the `X[:, 0]*X[:, 1]` Dynkin probe are what catch it.
- **`exact_sampling_components` is deliberately absent.** `S`'s step is *exact*, but it consumes
  the supplied `dW[:, n, 1]` and draws no auxiliary randomness, so the CRN ladder sees its
  pathwise differences collapse to rounding and the kernel already excuses an exactly-integrating
  component automatically. Nothing in this plan samples a transition law, so declaring the field
  would forfeit the order check for nothing.
- **No variance reduction, on purpose** (manual rule 4). The problem declares no payoff or
  functional and is graded **path by path** against a fixed supplied Brownian path. An antithetic
  pair or a control variate would require drawing or negating that path, which `R13` forbids and
  which would break the pathwise comparison. There is no `functional_se` to report.
- **Mean check on component 1 is skipped**: `|mean_U| = 0 < near_zero_mean_threshold = 0.01`, so
  `U` is graded on variance (`17/6`) and on the martingale invariant, not on a relative mean error.
- **`U(T)` is not Gaussian** (measured excess kurtosis 1.32), which is why
  `verification.terminal_law` is `null` — expect no KS-style distributional test, and do not
  assume normality anywhere in the solver.
- **Path-count resolution.** `variance_rel_err_max = 0.1` with `ci_mult = 2.0` and kurtosis 1.32
  puts the moment-resolution floor near 10600 paths, which is why `num_paths = 16000` and not the
  contract's 1000.
- **Return contract**: `terminal_paths` shape `(num_paths, 2)`, `empirical_mean` and
  `empirical_variance` as lists of floats, one per component, variance with `ddof=1` (`R15`,
  `R16`). `solve_sde` must stay importable — no module-load work outside
  `if __name__ == "__main__"` (`R19`). numpy and the standard library only (`R20`).

## Results

Run: `solve_sde(num_paths=16000, dt=0.04, T=1.0, seed=42)` (standalone, no `dW_sub`, so the
`dW_sub is None` branch with conditional-mean area 0.5*dW1*dW2 was used for the moment run).

- Empirical mean: [2.004149, -0.020963]   (exact: [2, 0])
- Empirical variance: [0.993789, 2.805298]   (exact: [1, 17/6 = 2.833333])

Both hooks (`dW` + `dW_sub`, `observables`) confirmed working. With `dW_sub` supplied, `I12` is
assembled with an exclusive prefix sum, chunked over paths. Note: the fallback when `dW_sub` is
withheld uses the conditional-mean area 0.5*dW1*dW2 rather than zero.

## Evaluation

See the metrics and review blocks below.

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 9.803e-03
error_is_estimate: false
observed_order: 0.9827
observed_order_se: 0.0307
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0183
invariant_worst_headroom: 0.1597
invariants_not_reported: ['zero_channel_cross_variation']
reference_outcome: validated
mean_rel_err: 3.675e-03
variance_rel_err: 9.803e-03
moment_bias_binding: variance
moment_bias_frac_tol: 0.2131
moment_bias_actionable: true
order_check: true
mc_se_rel: 0.0150
resolved: true
dynkin_z: 2.2030
dynkin_bar: 2.5310
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 1bf8136df5597817d1c9c3cb3208c596580f47ef5760c2cbc1d5b20739bb7191
solve_time_s: 1.7620
wall_time_s: 35.1200
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**  (certified: true)

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the spec's own generator; Path A)
- mean_rel_err:            S: 3.68e-03 +/- 2.0 x 7.89e-03 (se); U: absolute 1.83e-02 +/- 2.53 x 1.32e-02 (family-wise), |exact_mean| < 0.01 rule applies, within bar
- variance_rel_err:        S: 3.86e-03 +/- 2.0 x 1.11e-02 (se); U: 9.80e-03 +/- 2.0 x 4.21e-02 (se), tolerance 0.10
- mc_se_rel:               0.0150   resolved: true (16000 paths)
- functional_rel_error:    not applicable, the spec declares no functional
- observed_order:          0.983 +/- 0.031 (expected 1.0, one-sided below the declared floor); measured on U, the only discretised component. S integrates exactly and abstained as "unresolved" by design, at no cost
- weak_order:              0.990 (from E[U^2], within band)
- dynkin:                  ok. Four test functions, all within the family-wise bar 2.53. First seed worst z = 2.20, 0.33 from the bar, so the check was replicated: seeds z = 2.20 / 1.91 / 1.13, median z = 1.91, decision on the median. Pooled z = 3.03 and does not agree with the median; this is reported, not decisive under rule D9. Per-function z: X0^2 -0.48, X1^2 -0.42, X0*X1 2.20, tanh(X1) 1.69. The X0*X1 residual is the largest and sits near the bar, so it is the one to watch at a higher path count
- terminal_law:            not declared (U is not Gaussian)
- shape:                   U skewness 0.033 +/- 0.046, excess kurtosis 1.611 +/- 0.186
- constraints:             finite: pass (gate); martingale: pass (E[U] drift 0.0183 against bar 0.0265, headroom 0.160); estimator_stability: pass; zero_channel_cross_variation: not_reported (declared but not evaluable, non-gate)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments, the estimates are MC-resolved, and the generator check reached a verdict

### What changed since the 9
The previous pass returned `dynkin_ok: unavailable` because the Dynkin reader could not read the initial condition. That was a formulator defect. With `initial_condition` declared at the top level as `[1.0, 0.0]`, the check ran (initial_state deterministic, x0 = [1, 0]) and passed. Nothing in `solver.py` changed (sha256 1bf8136d...). The 10 is earned by that check and not by anything the solver did differently.

What the Dynkin pass does and does not establish: the generator is built from the spec's own drift and diffusion, so it confirms the paths follow the SDE as the spec states it. A drift mis-transcribed from `problem.md` would not be caught by it.

### Other skip-vocabulary items
`surrogate_ok: unavailable` (Path A, analytic route used), `cross_plan_agreement: unavailable`, `functionals_ok: unavailable` (none declared), `terminal_law_ok` and `cross_moment_ok: unavailable` (neither declared). None is a failure.

### Observations
- Strong order 0.983 +/- 0.031 on U sits on the claimed 1.0, so the exclusive prefix sum, the channel mapping (0 = W1, 1 = W2) and use of the supplied `dW` are all correct. `scheme_family: milstein` with `levy_area_treatment: sub-increments` is declared, so no `milstein_claim` cap applies. The scheme is genuinely implemented, so no `agent_cap`.
- The U variance carries a real, resolved dt bias: graded 2.843 against Richardson limit 2.904, 2.13 percent of the value and 0.213 of the 10 percent tolerance. It comes from the omitted `mu * I_{0,1}` term, which the plan documents on purpose.

### Feedback for solver
- None.

</review>
