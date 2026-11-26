---
id: 1
plan_slug: milstein-bridge-cv
scheme: milstein
scheme_family: milstein
strategy: Scalar Milstein on S at the contract dt, the path integral of ln S by the trapezoid rule on the nodes plus the conditional Brownian-bridge term (unbiased), and a control variate on the discounted terminal asset for the call price.
---

## SDE Reference

    dS = r S dt + sigma S dW,     S(0) = X_0

| parameter | value |
|---|---|
| X_0 | 100.0 |
| r | 0.05 |
| sigma | 0.2 |
| K | 100.0 |
| T | 1.0 (T0 = 0) |

- drift f(S) = r S; diffusion g(S) = sigma S; dg/dS = sigma
- Graded functionals, from the same paths:
  - `call_price` = e^{-rT} E[(A - K)^+], A = exp((1/T) int_0^T ln S(t) dt)
  - `geo_average_variance` = Var[A], ddof=1
- Closed-form references (`problem_spec.json -> functional_reference`): ln A ~ N(mu_A, s2_A),
  mu_A = ln X_0 + (r - sigma^2/2) T/2, s2_A = sigma^2 T/3. call = 5.5468, Var[A] = 140.17.
- Terminal law of S(T): lognormal, mean X_0 e^{rT}, variance X_0^2 e^{2rT}(e^{sigma^2 T} - 1).

## Numerical Scheme

- **Method**: Milstein (scalar, multiplicative noise)
  `S_{n+1} = S_n + r S_n dt + sigma S_n dW_n + 0.5 sigma^2 S_n (dW_n^2 - dt)`
  — the correction uses S_n (the pre-step value), never S_{n+1}.
- **dt**: 0.04 (contract value; `Nt = max(1, round(T/dt))`, `dt = T/Nt`, so Nt = 25)
- **num_paths**: 400000 (graded), 100000 (second count)
- **seed**: 42, via `np.random.default_rng(seed)`
- **T**: 1.0
- **Milstein correction**: `0.5 * sigma**2 * X_n * (dW**2 - dt)`
- **Expected strong order**: 1.0
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic — closed-form functional values in
  `functional_reference` (call 5.5468, Var[A] 140.17), plus analytic terminal moments of S(T)
  reproduced by `verification.moment_ode` and the lognormal `terminal_law`.

### Path integral of ln S (the binding choice)

Accumulate `I = int_0^T ln S dt` per path as

    I = sum_n [ 0.5 * (ln S_n + ln S_{n+1}) * dt  +  sigma * sqrt(dt**3 / 12) * xi_n ]

The first term is the trapezoid rule on the nodes. The second is the conditional Brownian-bridge
contribution: given the endpoints, `int_{t_n}^{t_{n+1}} W dt` deviates from the trapezoid value
by an independent N(0, dt^3/12), so adding it makes the quadrature unbiased in law for the exact
log-process. Without it the trapezoid rule's Var[ln A] shortfall is sigma^2 dt^2/(12 T), about
0.04% of s2_A — inside the gate either way, but the term removes the last quadrature bias.

- `xi_n ~ N(0,1)` is **fresh randomness**, independent of dW. Draw it from the same
  `default_rng(seed)` **after** the dW block (one `(num_paths, Nt)` draw), so the dW sequence is
  untouched.
- **When the evaluator supplies `dW`**, draw no randomness of your own: omit the bridge term and use
  the trapezoid rule alone, so the functional is a deterministic function of the supplied
  increments. The bridge term has zero mean and only touches A, not the state, so this changes
  neither terminal_paths nor the strong-order measurement.
- Then `A = exp(I / T)`. Never a left-Riemann sum — it biases the log-average variance at O(dt/T)
  and fails the 1% price gate at 25 steps.

### Variance reduction (call price)

Control variate `Y = e^{-rT} S_N`, whose expectation **under this scheme** is exactly
`e^{-rT} X_0 (1 + r dt)^Nt` (the Milstein correction and dW terms have zero mean). Use the scheme's
own mean, not X_0, so the control adds no bias.

    P = e^{-rT} max(A - K, 0)
    beta = Cov(P, Y) / Var(Y)                (estimated from the same paths)
    call_price = mean(P) - beta * (mean(Y) - E[Y])
    functional_se["call_price"] = std(P - beta*Y, ddof=1) / sqrt(num_paths)

`geo_average_variance = np.var(A, ddof=1)`; report its SE too (e.g. via the delta-method or a
20-batch estimate) in `functional_se["geo_average_variance"]`.

## Implementation Notes

- Return `terminal_paths = S_N` shape `(num_paths,)`, `empirical_mean`, `empirical_variance`
  (ddof=1), `functionals = {call_price, geo_average_variance}`, `functional_se`, `num_paths`, `dt`,
  `T`.
- Support the kernel hooks from the manual template: optional `dW` (shape `(num_paths, Nt)`, already
  scaled; take Nt from its shape and use it exactly) and optional `observables` (trapezoid path
  integrals of phi(S), returned as `path_integrals`). `verification.dynkin_test_functions` includes
  `np.log(X)`, so the state must stay positive — it does (sigma sqrt(dt) = 0.04, a negative step is
  ~25 sigma out); do not clip.
- Positivity and finiteness are gated constraints on S. Discounted S(T) is a (non-gated) martingale
  invariant with value 100.
- Memory: at 400000 x 25 the dW block is 80 MB float64, the xi block another 80 MB. Acceptable; or
  generate per-step slices from the same generator in a fixed order (dW block first, then xi) — but
  keep the order fixed and deterministic.
- Do not cache or hardcode: the 100000-path call must give a different estimate that agrees within
  sampling noise. The control-variate beta is re-estimated at each count.
- Kink at A = K (at the money) affects only the payoff, not the dynamics; no smoothing needed.
- Weak bias of the Milstein S-step on ln S is O(sigma^4 dt) in the log mean — orders of magnitude
  below 1%.
- numpy + stdlib only; no work at import; seed with `np.random.default_rng(seed)`.

## Results

Run: `uv run python workspace/sde_geometric_asian/plans/1-milstein-bridge-cv/solver.py`
(seed 42, dt = 0.04 -> Nt = 25, T = 1). Terminal S(T) moments (exact: mean 105.127, variance 451.05):

| num_paths | empirical_mean | empirical_variance | call_price (+/- SE) | geo_average_variance (+/- SE) |
|---|---|---|---|---|
| 400000 | 105.124795 | 448.626880 | 5.53092 +/- 0.00737 (rel err 2.9e-3 vs 5.5468) | 139.679 +/- 0.329 (rel err 3.5e-3 vs 140.17) |
| 100000 | 105.175765 | 448.282649 | 5.50790 +/- 0.01475 (rel err 7.0e-3) | 139.168 +/- 0.658 (rel err 7.2e-3) |

- Both functionals sit inside their gates (1% price, 3% variance) at both path counts; the two
  counts differ and agree within sampling noise. The control variate cuts the price SE to ~0.13%.
- Hooks confirmed: explicit `dW` (dt taken as T / dW.shape[1], bridge term omitted, no own
  randomness) and a one-entry `observables` dict (`path_integrals` shape `(num_paths,)`).
- Both price estimates read about 2.2-2.6 SE below the closed form (same seed, so the two counts
  are correlated, not independent evidence). No multi-seed bias study was run in this pass.

## Evaluation

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 5.325e-03
error_is_estimate: false
observed_order: 0.9835
observed_order_se: 3.470e-03
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 3.645e-04
strong_rel_err_se: 1.043e-06
strong_rel_err_component: 0
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 2.202e-03
invariant_worst_headroom: 1.5457
reference_outcome: validated
mean_rel_err: 2.202e-05
variance_rel_err: 5.325e-03
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 1.664e-03
functional_bias_rel: 3.320e-04
functional_se_rel: 1.332e-03
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 0.0416
moment_bias_actionable: false
order_check: true
mc_se_rel: 2.581e-03
resolved: true
dynkin_z: 0.3346
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 6b09dcbb4e0f2061a0549470616cbcf5ad3a8a02e77c6018003859f4099b3018
solve_time_s: 0.4840
wall_time_s: 40.2100
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments; reference_outcome validated, Tier A; `surrogate_ok` unavailable because the closed form made the surrogate route unnecessary)
- mean_rel_err:            2.20e-05 +/- 2.0 x 3.35e-02 absolute (SE 3.2e-04 relative; family-wise multiplier 2.43). Empirical 105.1248 against exact 105.1271
- variance_rel_err:        5.33e-03 +/- 2.0 x 1.158 absolute (SE 2.6e-03 relative; tolerance 0.10). Empirical 448.627 against exact 451.029
- mc_se_rel:               2.58e-03   resolved: true (CI width 5.13e-03 relative against a 0.05 bar, at 400000 paths)
- functional_rel_error:    1.664e-03 against functional_rel_tol 1.0e-02  (bias 3.32e-04 + sampling 1.332e-03 -> limited by sampling, binding functional call_price). The second gated functional, geo_average_variance, reads a ladder dt-bias of 9.3e-04 against its 3e-02 gate. Both ladder outcomes are "clean".
- observed_order:          0.9835 +/- 0.0035  (expected 1.0, one-sided below the declared floor; median-based 0.958). Strong rel. error implied at the graded dt: 3.6e-04 +/- 1.0e-06.
- weak_order:              0.998 (x: 0.9986, x^2: 0.9975; the tanh test function was at the noise floor and not used). Within band.
- dynkin:                  ok. z = 0.26 / 0.33 / 0.18 on X, X^2 and log X against a family-wise bar of 2.43, with measured extrapolation orders 0.98 / 0.98 / 1.00. The plan was not marginal (|z - bar| = 2.09 is outside the 0.7 margin), so one seed settled it (dynkin_replicated: false).
- terminal_law:            KS statistic 8.78e-04 against ks_critical 1.45e-03 (standardized, parametric bootstrap). Passes.
- shape:                   skewness 0.610 +/- 0.0053 (reference 0.614), excess kurtosis 0.664 +/- 0.0228 (reference 0.678). Both reported, neither graded.
- constraints:             positivity true, finite true (both gated); martingale true (drift 2.2e-03 against bar 7.7e-02, headroom 1.55 decades); estimator_stability true.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments, and the estimates are MC-resolved.

Other measurements:
- Moment dt-bias at the graded step is 0.0416 of the variance tolerance (`moment_bias_actionable: false`), so no refinement instruction applies.
- `reported_moments_agree` is true (relative difference 0.0), so the `ddof=1` arithmetic is correct.
- Cost: `solve_time_s` 0.484 s (cpu), peak memory 268 MB at the graded call, against a 4096 MB limit.
- `cross_plan_agreement` and `cross_check_ok` are unavailable: no sibling plan was evaluated against this one and the spec declares no ordering between functionals. This is a skip, not a pass.
- Scope of `functional_rel_error`: the kernel reports it as the ladder's dt bias plus the solver's own SE, and does not read the closed-form `functional_reference` values (spec ledger R10). It is therefore not an error against 5.5468 or 140.17. As a separate observation, the solver's own `call_price` of 5.5309 +/- 0.0074 sits 2.2 SE below the closed form, and the kernel's ladder value (5.5334, trapezoid only, 100000 paths) sits below it too. Both are well inside the 1% gate, and nothing in the kernel's measurements shows a defect.

### Feedback for solver
- None

</review>
