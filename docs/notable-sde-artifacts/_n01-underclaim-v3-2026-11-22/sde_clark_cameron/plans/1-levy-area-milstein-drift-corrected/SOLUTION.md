---
id: 1
plan_slug: levy-area-milstein-drift-corrected
scheme: milstein-levy-area-with-drift-noise-correction
scheme_family: milstein
levy_area_treatment: sub-increments
strategy: Step at dt with the exact S update, and build U's step from the left-point Ito sub-increment sum for the noise-noise iterated integral I_21 plus the increment-measurable conditional mean of the drift-noise iterated integral I_01, halving the residual strong error of the plain Levy-area scheme.
---

## SDE Reference

Two-dimensional Ito system driven by **two independent** Brownian motions W1, W2:

```
dS = mu dt + dW2,     S(0) = S_0 = 1.0
dU = S dW1,           U(0) = U_0 = 0.0
```

State vector is `(S, U)` in that column order. Diffusion matrix against the
independent channels `(dW1, dW2)`:

```
G(S, U) = [[0, 1],
           [S, 0]]
```

so column 0 multiplies `dW[:, :, 0] = dW1` and column 1 multiplies `dW[:, :, 1] = dW2`.

Parameters: `mu = 1.0`, `S_0 = 1.0`, `U_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.

Exact moments at `t` (component X = S, component Y = U):

```python
mean_S = S_0 + mu*t
var_S  = t
mean_U = 0.0
var_U  = ((S_0 + mu*t)**3 - S_0**3)/(3*mu) + t**2/2     # = 17/6 = 2.8333333333 at t = 1
cov(S_T, U_T) = 0.0 exactly
```

`U_T` is **not** Gaussian: conditionally on the W2 path it is `N(0, int_0^T S^2 ds)`,
a Gaussian variance mixture with excess kurtosis 1.4444.

## Numerical Scheme

- **Method**: order-1.0 Milstein-type Taylor step with the Levy-area term assembled from
  the supplied sub-increments, plus the increment-measurable part of the drift-noise
  iterated integral
- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 8000 (`evaluation_thresholds.num_paths`); `problem.md` grades the single
  contract call at `num_paths = 1000`, `dt = 0.04`, `T = 1`, `seed = 42`
- **T**: 1.0
- **seed**: 42
- **Expected strong order**: 1.0 (`verification.expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (exact, above), plus the harness's own
  pathwise reference built on the 256x sub-grid from the same Brownian path

### The step

Over step `k` on `[t_k, t_k + dt]` the exact increment of U, with
`S(s) = S_k + mu*(s - t_k) + (W2(s) - W2(t_k))`, is

```
U_{k+1} - U_k = S_k * dW1_k  +  mu * I_01  +  I_21
I_01 = int (s - t_k) dW1(s)
I_21 = int (W2(s) - W2(t_k)) dW1(s)        <- the Levy-area term
```

`I_21` is not a function of the step increments at all (Clark and Cameron, 1980) and is
assembled from `dW_sub` by the **left-point (Ito) Riemann sum**:

```python
sub       = dW_sub[:, k, :, :]                      # (num_paths, 256, 2)
W2_before = np.concatenate(
    [np.zeros((num_paths, 1)), np.cumsum(sub[:, :, 1], axis=1)[:, :-1]], axis=1)
I21 = np.sum(W2_before * sub[:, :, 0], axis=1)      # left point = Ito convention
```

`I_01` is treated by its conditional expectation given the step increment, which is exact
and increment-measurable: `E[I_01 | dW1_k] = 0.5 * dt * dW1_k`. So the updates are

```python
S_next = S + mu*dt + dW[:, k, 1]                                   # exact
U_next = U + S * dW[:, k, 0] + I21 + 0.5 * mu * dt * dW[:, k, 0]
```

S's update is **exact** (its coefficients are constant), so the whole discretization error
lives in U.

### Why `I_01` is not assembled in full from `dW_sub`

It could be — `I_01 = sum_j (j*h) * dW1_sub_j` — and for this particular linear system that
would make the step reproduce the harness's sub-grid reference to rounding. That is exactly
the state `problem.md` **excludes** rather than scores (requirement R23): terminal states
identical at `dt`, `dt/2` and `dt/4` are recorded as having bypassed the step rather than
integrated at it. The conditional-mean term above is the sharpest treatment that keeps a
genuinely `dt`-dependent residual, so do not replace it with the full sub-grid sum.

### Expected accuracy (derived, not measured here)

The only term omitted relative to the sub-grid reference is the component of `mu*I_01`
orthogonal to `dW1_k`. Per step `Var[I_01] = dt^3/3` and `Var[E[I_01|dW1]] = dt^3/4`, so the
residual per-step variance is `dt^3/12` instead of `dt^3/3` — a factor-2 reduction in
strong error. Accumulated over `Nt = 25` steps at `dt = 0.04` and normalised by
`sqrt(E[U_T^2]) = sqrt(17/6) = 1.683`:

| scheme | predicted relative strong error |
|---|---|
| plain Euler-Maruyama (no `I_21`) | ~8.9e-02 (measured in the spec) |
| `I_21` only | ~1.4e-02 (measured in the spec; derivation agrees: 1.37e-02) |
| `I_21` + `0.5*mu*dt*dW1` (this plan) | ~6.9e-03 |

against the 2% gate — roughly 2.9x margin.

## Implementation Notes

- **Channel convention is the single highest-risk detail.** `dW[:, :, 0] = dW1` (drives U),
  `dW[:, :, 1] = dW2` (drives S). A swap is invisible to every moment, invariant and Dynkin
  check because W1 and W2 are i.i.d., but gives ~100% pathwise strong error. Self-check in
  `__main__`: `S_T` must equal `S_0 + mu*T + dW[:, :, 1].sum(axis=1)` to rounding (~1e-13).
- **Left-point convention matters.** A midpoint or trapezoid sub-sum yields the
  *Stratonovich* iterated integral and leaves a non-vanishing `0.5*dt` bias in U.
- `dW_sub` may be `None` when the solver is run standalone; in that case draw the sub-grid
  first with `np.random.default_rng(seed)` and aggregate it to form `dW`, so the two are
  consistent. When both are supplied, use them exactly and draw no randomness.
- Contract: `Nt = max(1, round(T/dt))` then `dt = T/Nt`; take `Nt = dW.shape[1]`; advance the
  state exactly once per step of `dW`; `terminal_paths` is `(num_paths, 2)` in column order
  `(S, U)`; `empirical_variance` uses `ddof=1`; numpy and the standard library only; no work
  at module load outside `if __name__ == "__main__"`.
- **Do not declare `exact_sampling_components`.** Both components are stepped with the
  supplied increments: S's update consumes `dW2` and U's consumes `dW1` plus the
  sub-increments that sum to it. Nothing here draws auxiliary randomness. S is exactly
  integrated, which the kernel already excuses automatically; and because this problem is
  graded *pathwise*, drawing U from its conditional Gaussian law (which is available) would
  decorrelate from the reference path and produce ~100% strong error — exact sampling and
  conditional Monte Carlo are both wrong answers here, not options.
- **No variance reduction.** The problem declares no payoff or functional; it is graded on a
  pathwise strong error, where antithetics and control variates change nothing.
- **No declared barrier or absorbing boundary**, so no `boundary_treatment` applies.
- Memory: `dW_sub` is `(num_paths, Nt, 256, 2)` float64 — 820 MB at 8000 paths and `Nt = 25`,
  3.3 GB at the `dt/4` ladder level. `conv_num_paths = 2000` is set for the CRN strong-order
  and Dynkin ladders. Build `I_21` step by step (slice `dW_sub[:, k]`) rather than
  materialising a cumulative sum over the whole array. If the Dynkin ladder refuses on the
  memory budget it reports `unavailable` and caps the score at 9; that is a correct refusal,
  not a solver defect.
- Both coefficients are globally Lipschitz here, so no correct scheme produces Inf/NaN
  (gated constraint `finite`). U is a true martingale: `E[U_T] = 0` exactly at every `dt`
  (ungated invariant `martingale`).
- **Refine cycle 2: no change to `solver.py`.** Checked against the derivation: every term of the
  U update (`S_k dW1`, the left-point `I_21`, `0.5*mu*dt*dW1`) has zero conditional mean given
  the past, so `E[U_T] = 0` at every `dt`; the 0.0434 reading (z ~ 2.29, 8000 paths) is a seed
  excursion, not a scheme bias, and `0.5*mu*dt*dW1` is left as is. The strong order on U
  (0.934 +/- 0.035, one-sided floor 1.0) is within about 1.9 se of 1.0 and is consistent with
  the derived order 1; the `dt^3/12` residual of the conditional-mean `I_01` keeps genuine
  `dt`-dependence, and assembling `I_01` in full would trigger R23, so it stays as the
  conditional mean. `scheme_family` remains `milstein` (unchanged).

## Results

Standalone run (`num_paths=8000`, `dt=0.04`, `T=1`, `seed=42`, sub-grid drawn internally, 256 sub-steps):

- Empirical mean (S, U): [2.003095, 0.043400]   (exact: [2.0, 0.0])
- Empirical variance (S, U): [0.987931, 2.865558]   (exact: [1.0, 2.8333])

Hooks confirmed: explicit `dW` + `dW_sub` (dt = 0.04, S matches `S_0 + mu*T + sum dW2` to 2e-15),
`dW` without `dW_sub` (conditional-mean area branch), and a one-entry `observables` dict
(`path_integrals['x']` shape `(num_paths,)`).

## Evaluation

(Filled by evaluator; see the review block below.)

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 0.0121
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
invariant_max_drift: 3.192e-03
invariant_worst_headroom: 0.4959
reference_outcome: validated
mean_rel_err: 1.548e-03
variance_rel_err: 0.0121
order_check: true
dynkin_z: 2.0342
dynkin_bar: 2.4284
dynkin_seeds: 3
dynkin_replicated: true
dynkin_median_z: 1.8957
mc_se_rel: 0.0208
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 7a10d2583d0bd5c10c6d312731f7640f98b34496325a3e6609e5159d368d34d0
solve_time_s: 1.3830
wall_time_s: 23.6500
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Headline: the Dynkin check now runs and passes
`dynkin_ok: true`. With `initial_condition` at the top level of the spec, the reader picked up `law: deterministic, x0 = [1.0, 0.0]` (`detail.dynkin_ok.initial_state`). The first seed gave `dynkin_z = 2.034` against `dynkin_bar = 2.428`, which is 0.394 under the bar and inside the 0.7 margin. The check was therefore replicated on 3 seeds (`dynkin_replicated: true`) and the median `z` decided: **median z = 1.896** (seeds 42 / 3444837047 / 2669555309 gave worst-z 2.034 / 1.252 / 1.896). This plan was marginal on the first seed, which is why it was replicated. The bar is the family-wise multiplier over 3 test functions.

Per test function on seed 42: `X[:,0]*X[:,1]` z = 2.034, `X[:,1]**2` z = 0.632, `tanh(X0+X1)` z = 0.691. All three pass. The first is the U-drift-sensitive function, and it is the one nearest the bar. It is the only function in the suite that covers a spurious additive drift in U, the mode that both moment checks cannot see.

Three things this does not establish:
- **Pooled z disagrees.** `pooled_z = 2.99` is above the 2.43 bar and `pooled_agrees: false`. The decision rule is the median, which passes, but this is a soft pass and not a clean zero.
- **Direction.** The `phi` residual is about 0.0946 and is flat across dt (0.0941, 0.0947, 0.0946). It does not shrink with the step. Its SE is 0.046, so it is about 2 se from zero, with the same sign on all three seeds. The U mean also reads +0.043 (z about 2.3 in absolute mode). Both point the same way. A seed-42 excursion explains this without a scheme bias, and the derivation (every term of the U update has zero conditional mean) agrees. But two independent-looking readings leaning the same way is worth a higher-path re-check if anyone wants more than the median rule gives.
- **Extrapolation order.** No residual rate could be read (`extrapolation_order_source: assumed_1`). The finer residual differences are inside their own MC noise, and on `tanh` the differences change sign. This is a property of the check at 8000 paths and not a solver defect. The residual at the finest level is also below the coarse one on all three functions, so nothing contradicts the pass.

The generator is built from the spec's own drift and diffusion. A drift mis-transcribed from `problem.md` would fool it, so it breaks circularity only on the solver side.

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, reference_outcome validated)
- mean_rel_err:            S: 1.55e-03 +/- 2.0 x 0.0111 (abs se); U: |mean| = 0.0434 +/- 2.53 x 0.0189 (family-wise multiplier, absolute mode, tol 0.05), passes
- variance_rel_err:        S: 1.21e-02 +/- 2.0 x 0.0157 (abs se); U: 1.14e-02 +/- 2.0 x 0.0595 (abs se); both inside the 10% tolerance
- mc_se_rel:               0.0208   resolved: true (CI width 3.1% for S, 4.2% for U, against the 5% bar, 8000 paths)
- functional_rel_error:    not declared by this spec (no gated functional)
- observed_order:          0.934 +/- 0.035 on U, read against this plan's own design rate 1.0 (`expected_strong_order_milstein`), not the pool-wide 0.5. One-sided band: only a shortfall counts, and 0.066 below design is about 1.9 se, inside the 0.4 band. Median-based order 0.927. S is integrated exactly, so its order abstained as "unresolved" by design and costs nothing
- weak_order:              indeterminate (at the noise floor)
- dynkin:                  ok. z = 2.034 on seed 42 against bar 2.428; median z = 1.896 over 3 seeds (replicated; first seed 0.394 from the bar); pooled z = 2.99, which does not agree with the median decision
- terminal_law:            not declared
- shape:                   U skewness 0.111 +/- 0.055, excess kurtosis 1.454 +/- 0.169 (exact 1.444, so the variance mixture is reproduced); S skewness 0.042 +/- 0.028, excess kurtosis 0.032 +/- 0.055
- constraints:             finite: pass (gate); estimator_stability: pass (0.0032 vs bar 0.01, headroom 0.50 decades)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                strong_gate_not_measured (cap 9): the problem declares a pathwise strong-error gate (`strong_rel_err_max` 0.02), and `verifylib` has no pathwise reference, so nothing measured a strong error and moment evidence does not stand in for it. This is the designed ceiling on this problem and not a defect in this plan. Dynkin was the other open item (rubric 10 needs a verdict); it is now closed, so the rubric no longer binds below the cap. The ceiling stays 9 and the plan stays uncertified

### Claim audit (no cap applied)
SOLUTION.md describes the Levy-area Milstein step and `solver.py` implements it (unchanged hash). `levy_area_treatment: sub-increments` is declared, so there is no `milstein_claim` cap. The solver consumes the supplied `dW`, takes `Nt = dW.shape[1]`, and `reported_moments_agree` holds to 4e-15. No `exact_sampling_components` is needed. The ladder shows real dt-dependence on U (s0 7.6e-3 to s1 4.0e-3), so the conditional-mean `I_01` treatment is not an exact bypass of the step.

### Feedback for solver
- None. Moments, order against its own design 1.0, constraints, shape and the generator check all pass. Richardson moment shifts are within noise (mean 0.11 se, variance 1.08 se). Nothing in `solver.py` bears on the remaining gap, which is the strong-gate ceiling.
- Watch item for the conductor only, not a solver change: the `X[:,0]*X[:,1]` residual sits about 2 se above zero on every seed and does not shrink with dt, and the U mean is also +0.043. If a stronger statement than the median rule is wanted, raise `num_paths` on the Dynkin run. A true additive U drift would grow with `num_paths`, and a seed excursion would not.

</review>
