---
id: 1
plan_slug: levy-area-milstein-trapezoid
scheme: milstein-levy-area-with-drift-noise-trapezoid
scheme_family: milstein
levy_area_treatment: sub-increments
strategy: Order-1 Levy-area Milstein step for U with the iterated integral I_{1,2} assembled exactly from the supplied sub-increments, plus a trapezoidal (dt/2)*dW1 surrogate for the drift-noise integral I_{0,1} that cuts the remaining residual variance by 4x at the same order.
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

- **Method**: Levy-area Milstein, one update per step of `dW`, with the iterated integral
  assembled from `dW_sub` and a trapezoidal surrogate for the drift-noise integral.
- **dt**: 0.04 (the contract's step; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 16000 (`evaluation_thresholds.num_paths`; the contract call itself is
  `solve_sde(num_paths=1000, dt=0.04, T=1, seed=42)` and the kernel uses
  `graded_num_paths = 1000`, `conv_num_paths = 2000` on the CRN ladder)
- **T**: 1.0
- **seed**: 42
- **Expected strong order**: 1.0 (`verification.expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`has_analytic_solution: true`), backed by the
  exactly-closing moment ODE on component 0 and the four `dynkin_test_functions`. The pathwise
  2% gate is scored by the harness against its own fine-grid reference, not by this plan.

### The step

`S` is exactly integrable and consumes the supplied increment:

```
S_{n+1} = S_n + mu*dt + dW[:, n, 1]            # exact, no discretization error
```

All discretization error lives in `U`. Write the exact increment over `[t_n, t_{n+1}]`,
substituting `S_s = S_n + mu*(s - t_n) + (W2(s) - W2(t_n))`:

```
U_{n+1} - U_n = int S_s dW1(s)
              = S_n * dW1_n  +  mu * I_{0,1}  +  I_{1,2}
I_{0,1} = int_{t_n}^{t_{n+1}} (s - t_n) dW1(s)
I_{1,2} = int_{t_n}^{t_{n+1}} (W2(s) - W2(t_n)) dW1(s)        # the Levy-area term
```

`I_{1,2}` is the one Clark & Cameron (1980) proved is not recoverable from `dW` alone; it is
assembled **exactly on the supplied sub-grid** from `dW_sub`, and `I_{0,1}` is replaced by the
trapezoidal surrogate `(dt/2) * dW1_n`:

```
U_{n+1} = U_n + S_n * dW1_n + mu * (dt/2) * dW1_n + I_{1,2}_n
```

Vectorised assembly of `I_{1,2}` (let `M = dW_sub.shape[2]`, do **not** hardcode 256):

```python
d1 = dW_sub[:, :, :, 0]                                 # W1 sub-increments
d2 = dW_sub[:, :, :, 1]                                 # W2 sub-increments
W2_partial = np.cumsum(d2, axis=2) - d2                 # exclusive prefix sum: W2(s_j) - W2(t_n)
I12 = np.einsum("pnj,pnj->pn", W2_partial, d1)          # one value per path per step
```

### Why the trapezoid term is in, and why `I_{0,1}` is not assembled exactly

Per step the residual of the `I_{1,2}`-only Milstein step is `mu * I_{0,1}`, with variance
`mu^2 * dt^3 / 3`. Replacing it by `(dt/2) * dW1_n` leaves residual variance
`Var[I_{0,1}] - 2*Cov + Var[(dt/2)dW1] = dt^3/3 - 2*(dt^3/4) + dt^3/4 = dt^3/12`
(using `E[I_{0,1} dW1_n] = int (s - t_n) ds = dt^2/2`). That is a **4x** variance reduction,
i.e. 2x on the strong error, at the identical order. Summing over `Nt = T/dt` independent steps:

| scheme | residual RMS at dt = 0.04 | vs `sqrt(E[U_T^2]) = 1.683` | vs the full-state `sqrt(E[|X_T|^2]) = 2.799` |
|---|---|---|---|
| Euler-Maruyama (`dW` only) | `sqrt(Nt*dt^2/2)` = 0.141 | 8.4% | 5.1% |
| Levy-area Milstein, `I_{1,2}` only | `sqrt(Nt*dt^3/3)` = 0.0231 | 1.37% | 0.82% |
| **this plan** | `sqrt(Nt*dt^3/12)` = 0.0115 | **0.69%** | **0.41%** |

Both Milstein rows clear the 2% pathwise gate; this one clears it with roughly 3x the headroom,
which is the whole reason it is plan 1.

**Do not assemble `I_{0,1}` exactly from `dW_sub`.** Adding `mu * sum_j (s_j - t_n) * dW1_j`
alongside the exact `I_{1,2}` makes the update algebraically identical to a plain Euler sum over
the entire sub-grid, because
`sum_j S_{s_j} dW1_j = S_n*dW1_n + mu*I_{0,1}^{sub} + I_{1,2}^{sub}`. The kernel holds the
sub-step fixed at the graded `dt/256`, so such a solver returns terminal states identical to
rounding at `dt`, `dt/2` and `dt/4` and is **excluded** from the comparison as having bypassed
the step (`problem.md`, requirement `R22`). The `dt/2` surrogate keeps a genuine, measurable
`dt`-dependence in the update while still beating the `I_{1,2}`-only step.

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
- **Positivity / clipping:** none. The system is linear with affine coefficients; `S` and `U`
  are both signed and unbounded. The only gated constraint is `finite` (`np.isfinite(X)`).
- **Martingale invariant (non-gate).** `E[U(T)] = 0` exactly. Every noise term in the `U`
  update — `S_n*dW1_n`, `(mu*dt/2)*dW1_n`, `I_{1,2}` — is a left-endpoint-adapted Ito term with
  zero mean, so the invariant holds by construction. Evaluating `S` at the right endpoint or at
  the midpoint of the step turns this into a Stratonovich step and picks up a non-zero mean; the
  `martingale` invariant and the `X[:, 0]*X[:, 1]` Dynkin probe are what catch it.
- **`exact_sampling_components` is deliberately absent.** `S`'s step is *exact*, but it consumes
  the supplied `dW[:, n, 1]` and draws no auxiliary randomness, so the CRN ladder sees its
  pathwise differences collapse to rounding and the kernel already excuses an exactly-integrating
  component automatically. Nothing in this plan samples a transition law, so declaring the field
  would forfeit the order check for nothing.
- **No variance reduction, on purpose** (manual rule 4 / step 1c). The problem declares no payoff
  or functional — `evaluation_thresholds` carries no `functionals` entry — and it is graded
  **path by path** against a fixed supplied Brownian path. An antithetic pair or a control variate
  would require drawing or negating the path the harness handed us, which `R13` forbids and which
  would break the pathwise comparison outright. There is no `functional_se` to report.
- **Mean check on component 1 is skipped**: `|mean_U| = 0 < near_zero_mean_threshold = 0.01`, so
  `U` is graded on variance (`17/6`) and on the martingale invariant, not on a relative mean error.
- **`U(T)` is not Gaussian** (measured excess kurtosis 1.32), which is why
  `verification.terminal_law` is `null` — expect no KS-style distributional test, and do not
  assume normality anywhere in the solver or in any self-reported diagnostic.
- **Path-count resolution.** `variance_rel_err_max = 0.1` with `ci_mult = 2.0` and kurtosis 1.32
  puts the moment-resolution floor near 10600 paths, which is why `num_paths = 16000` and not the
  contract's 1000.
- **Return contract**: `terminal_paths` shape `(num_paths, 2)`, `empirical_mean` and
  `empirical_variance` as lists of floats, one per component, variance with `ddof=1` (`R15`,
  `R16`). `solve_sde` must stay importable — no module-load work outside
  `if __name__ == "__main__"` (`R19`). numpy and the standard library only (`R20`).

## Results

Standalone run (16000 paths, dt = 0.04, T = 1, supplied dW and dW_sub with 256 sub-steps):

- Empirical mean: S = 1.995927, U = 0.033488 (exact 2, 0)
- Empirical variance: S = 0.981998, U = 2.809457 (exact 1, 17/6 = 2.833333)
- Without dW_sub (conditional-mean area 0.5*dW1*dW2): variance U = 2.796410
- Self-drawn path (seed 42, no dW): mean (2.004149, -0.021149), variance (0.993789, 2.865534)
- dW hook and observables hook both confirmed working.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 0.0114
error_is_estimate: false
observed_order: 0.9343
observed_order_se: 0.0351
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0186
invariant_worst_headroom: 0.1573
invariants_not_reported: ['zero_channel_cross_variation']
reference_outcome: validated
mean_rel_err: 3.675e-03
variance_rel_err: 0.0114
order_check: true
mc_se_rel: 0.0150
resolved: true
dynkin_z: 2.2030
dynkin_bar: 2.5310
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 78612944dd9c1d53a3af432cfeca13373450e54f66af179ddfa9be64b805bf96
solve_time_s: 1.6630
wall_time_s: 35.0700
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the spec's generator; surrogate_ok unavailable because the analytic route took precedence)
- mean_rel_err:            S: 3.68e-03 +/- 2.0 x 7.89e-03 (abs). U: |mean| = 0.0186 +/- 2.53 x 0.0134 (family-wise multiplier), absolute mode since the exact mean is 0.
- variance_rel_err:        S: 3.86e-03 +/- 2.0 x 1.11e-02. U: 1.14e-02 +/- 2.0 x 4.29e-02 (se in absolute variance units). Tolerance 0.10.
- mc_se_rel:               0.0150   resolved: true
- functional_rel_error:    not applicable, the spec declares no gated functional
- observed_order:          0.934 +/- 0.035 on U (median 0.927), expected 1.0 (one-sided below the declared floor). S is integrated exactly, so the estimator abstained on it (unresolved, a property of the estimator, not a defect); the U component binds and orders_ok is true.
- weak_order:              indeterminate, at the noise floor
- dynkin:                  ok. z = 2.20 against bar 2.53 on the first seed. The first seed was within the margin (0.7) of the bar, so it was replicated on 3 seeds (dynkin_replicated: true); the median z is 1.91. The plan was marginal on a single seed, and the replication settled it. The earlier unavailable verdict was a spec defect (initial_condition not read at top level), now fixed. Caveat: the generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from problem.md would fool it.
- terminal_law:            not declared (U(T) is not Gaussian, deliberately null in the spec)
- shape:                   S: skewness 0.010 +/- 0.020, excess kurtosis -0.015 +/- 0.039. U: skewness 0.032 +/- 0.045, excess kurtosis 1.579 +/- 0.183.
- constraints:             finite: pass. martingale: pass (drift 0.0186 against bar 0.0268, headroom 0.157, thin and sampling-limited). estimator_stability: pass. zero_channel_cross_variation: not_reported (spec-declared as not checked).
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments and the estimates are MC-resolved; Tier A, certified.

Other checks: reported_moments_agree true (6.9e-15, ddof=1 correct). richardson_stable true. Moment dt bias is small (variance 0.57% of its tolerance, mean 3.0%). The kernel reduced the declared 256 sub-steps to 167 at 16000 paths to fit the 1 GiB cap; this is a kernel memory decision, not a solver defect. cross_plan_agreement unavailable (single plan).

### Feedback for solver
- None. The score is 10 and the plan is certified. For the record: the Levy-area step consumes dW and dW_sub as intended, U's strong order 0.934 +/- 0.035 is consistent with the claimed order 1, and the milstein plus sub-increments declaration was accepted with no milstein_claim cap.
- Observation only: the martingale invariant has the thinnest margin (0.157). The sample mean of U is 0.0186 +/- 0.0134, about 1.4 se from zero, so this is sampling noise and not a scheme bias.

</review>
