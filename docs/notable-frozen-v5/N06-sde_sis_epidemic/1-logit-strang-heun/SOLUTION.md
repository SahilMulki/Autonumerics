---
id: 1
plan_slug: logit-strang-heun
scheme: strang-splitting-logit-heun
scheme_family: strang-splitting
strategy: Transform to the logit coordinate Y = log(I/(N-I)) where the noise is additive (sigma*N dW), then Strang-split the stiff logistic part (exact flow, two half steps) around a Heun (predictor-corrector) step of the bounded Ito-correction drift plus the additive noise.
---

## SDE Reference

Ito SDE (state I, written X in the spec):

    dI = I [ (beta N - mu - gamma) - beta I ] dt + sigma I (N - I) dW,     I(0) = 90,   t in [0, 0.5]

| parameter | value |
|---|---|
| X_0 = I(0) | 90.0 |
| N | 100.0 |
| beta | 0.5 |
| mu | 20.0 |
| gamma | 25.0 |
| sigma | 0.03 |
| T | 0.5 |

Derived: `a = beta*N - mu - gamma = 5`, endemic level `a/beta = 10`, `mu + gamma = 45`.

- drift `f(I) = I*(a - beta*I)`
- diffusion `g(I) = sigma*I*(N - I)`, `g'(I) = sigma*(N - 2I)`
- the exact process stays in the open interval (0, N); the harness gates every terminal state finite and in [0, N].

## Numerical Scheme

- **Method**: Lamperti/logit transform, then Strang splitting, with a two-stage (Heun) residual stage. Composition per step of length `h = dt`:
  1. **Half logistic flow, exact** (h/2): advance `dI = I(a - beta I) dt` exactly,
     `I(t) = a I0 / (beta I0 + (a - beta I0) e^{-a t})`, and re-express as Y.
  2. **Residual + noise, Heun** (h): in Y, `dY = R(Y) dt + sigma*N dW` with
     `R(Y) = 0.5 sigma^2 N (2I - N) = 0.5 sigma^2 N^2 tanh(Y/2)` (bounded by 4.5).
     `Y* = Y + R(Y) h + sigma N dW_n`,  `Y <- Y + 0.5 (R(Y) + R(Y*)) h + sigma N dW_n`.
  3. **Half logistic flow, exact** (h/2) again.
  Then `I = N / (1 + exp(-Y))` at the end (or carry I and Y consistently).
- Derivation: `Y = log(I/(N-I))` has `dY/dI = N/(I(N-I))`, so by Ito
  `dY = [ N(a - beta I)/(N - I) + 0.5 sigma^2 N (2I - N) ] dt + sigma N dW`.
  The first bracket term is exactly the logistic ODE written in Y (handled by step 1/3), the second is R(Y).
- **dt**: 0.02 (the contract step; `Nt = max(1, round(T/dt))`, `dt = T/Nt` -> 25 steps)
- **num_paths**: 100000 for development (spec `evaluation_thresholds.num_paths`); the graded call is `solve_sde(num_paths=200000, dt=0.02, T=0.5, seed=42)`
- **seed**: 42, via `np.random.default_rng(seed)`
- **T**: 0.5
- **Milstein correction**: not applicable — noise is additive in Y, so the Milstein term vanishes identically (`g_Y' = 0`).
- **Expected strong order**: 1.0 (additive noise in Y; the map back to I is smooth on the reachable set)
- **Expected weak order**: 1.0 declared (measured on this problem: the mean bias is already at its floor by dt = 0.02 — +0.054% at 0.02, +0.017% at 0.01 — so it behaves effectively second order)
- **Reference for scoring**: Kolmogorov solve (`verification.kolmogorov`: backward Kolmogorov CN on [-5, 105], Nx = 8001, Nt = 4000), giving E[I(0.5)] = 6.60172, Var[I(0.5)] = 39.7989. Dynkin check on `X`, `X**2`, `tanh(X/10)` is also available.

## Implementation Notes

- **Hooks (required by the evaluator)**: accept `dW=None` (pre-scaled Brownian increments, shape `(num_paths, Nt)`; when given, take `Nt = dW.shape[1]`, `dt = T/Nt`, and draw no randomness of your own) and `observables=None` (accumulate the trapezoidal path integral of each `phi(I)` over each full step, evaluated in the I coordinate). Every step must consume `dW[:, n]` exactly once, in the noise stage — this plan steps every component with the supplied increments, so do **not** declare `exact_sampling_components`.
- **Numerically stable maps** — never form `N - I` by subtraction when I is near N:
  `I = N / (1 + exp(-Y))`, `N - I = N / (1 + exp(Y))`; `Y = log(I) - log(N - I)`.
  For the exact logistic flow written cancellation-free:
  `D = beta*I0 + (a - beta*I0)*E` with `E = exp(-a*h/2)`; `I1 = a*I0/D`;
  `N - I1 = [ N*(a - beta*I0)*E + (mu+gamma)*I0 ] / D` (uses `N*beta - a = mu + gamma`).
  Then `Y1 = log(I1) - log(N - I1)`. D > 0 for all I0 in (0, N) because a > 0, so the flow is well-defined; alternatively write the flow directly in Y (it is the same ODE).
  Since paths can become very small (the law is right-skewed with mass near 0), compute `log(I1)` as `log(a) + log(I0) - log(D)` when I0 is tiny, or carry `Y` as the primary state and compute `I0 = N/(1+exp(-Y))`, `log(I0) = -log1p(exp(-Y))` + log N via `np.logaddexp`.
- **Domain preservation without clipping**: the logit maps 0 and N to -inf and +inf, so every finite Y maps into (0, N). Do not clip; check at the end that `np.all(np.isfinite(I)) and I.min() >= 0 and I.max() <= N`.
- **Stiffness**: f'(90) = -85 and in Y the logistic part has Jacobian `-(mu+gamma) I/(N-I)` = -405 at I = 90. Both are handled by the exact flow; the only explicit stage is R(Y), whose derivative is bounded by `0.25 sigma^2 N^2 = 2.25`, so `|R'| dt = 0.045` — no stability issue.
- **Why this composition**: measured on this problem (sde_manual rule 6), holding the Strang skeleton and the exact half-flows fixed and changing only the noise sub-step, Milstein in I gave a -0.98% mean bias at dt = 0.02 while logit + Heun gave +0.054%, an 18x gain on the binding moment.
- **Scheme family**: `strang-splitting` carries no kernel order floor (its rate depends on its sub-steps), so the order study's band is one-sided; the expected strong order 1.0 above is the plan's own claim.
- Vectorise over paths; loop only over the 25 steps. Return `terminal_paths` (shape `(num_paths,)`), `empirical_mean`, `empirical_variance` (ddof=1), plus `num_paths`, `dt`, `T`, and `path_integrals` when observables are passed. No work at import time.

## Results

Run: `solve_sde(num_paths=200000, dt=0.02, T=0.5, seed=42)` (25 steps), about 1 s.

- Empirical mean: 6.625505 (Kolmogorov reference 6.60172, rel. err 0.36%, inside MC noise)
- Empirical variance: 39.509713 (reference 39.7989, rel. err 0.73%)
- Terminal range [0.00330, 53.07], all finite and inside [0, N] with no clipping.
- `dW` hook (dt taken from `dW.shape[1]`) and `observables` hook (path integrals of shape `(num_paths,)`) both confirmed.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: surrogate
estimated_rel_error: 7.517e-03
error_is_estimate: true
observed_order: 0.9993
observed_order_se: 0.0157
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 0.0398
strong_rel_err_se: 4.124e-04
strong_rel_err_component: 0
design_order: 0.5000
scheme_family: strang-splitting
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 2.708e-03
invariant_worst_headroom: 0.5673
mean_rel_err: 2.989e-03
variance_rel_err: 7.517e-03
moment_bias_binding: variance
moment_bias_frac_tol: 0.0862
moment_bias_actionable: false
order_check: true
mc_se_rel: 7.162e-03
resolved: true
dynkin_z: 1.3763
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: bb7c98bb59d0f6fc1087f8a33c6e15ff500883b62873d26582aa5bee63769a68
solve_time_s: 0.2540
wall_time_s: 12.7000
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              surrogate  (reference route: kolmogorov; moment_ode attempted and did not close exactly, as the spec predicted; the Kolmogorov truncation guard passed, shift ~3e-7 and ~2e-7 relative on the mean and variance)
- mean_rel_err:            2.99e-03 +/- 2.0 x 0.0199 (se_mean absolute; empirical 6.6215 vs reference 6.6017, tol 5%)
- variance_rel_err:        7.52e-03 +/- 2.0 x 0.2829 (se_variance absolute; empirical 39.4997 vs reference 39.7989, tol 10%)
- mc_se_rel:               7.16e-03   resolved: true (CI width 1.42e-02 against a 0.05 bar, ci_mult 2.0, 100000 paths)
- functional_rel_error:    not applicable, the spec declares no gated functional (`functionals_ok: unavailable`)
- moment bias (report-only): the dt = 0.02 bias in the variance is 8.6e-03 relative, 0.086 of its tolerance, against a Richardson value of 39.569. The mean bias is 9.0e-04 relative. `moment_bias_actionable: false`.
- observed_order:          0.9993 +/- 0.0157 (median-based 1.015), against expected strong order 0.5 declared in the spec. Candidates 0.5 and 1.0 both exist, and the plan claimed 1.0. Band: one-sided; the plan's `strang-splitting` family carries no kernel floor. `strong_order_above_band: true` is flagged and not scored on. Strong relative error at dt0 is 3.98e-02 +/- 4.1e-04, and the spec sets no `strong_rel_err_max`, so it is not gated.
- weak_order:              indeterminate, at the noise floor. All three weak differences (x, x^2, tanh) are inside their own Monte Carlo noise. Not scored on, and consistent with the plan's own note that the mean bias is already at its floor at dt = 0.02.
- dynkin:                  ok. z = (-0.88, -1.38, +0.44) on X, X^2 and tanh(X/10), against a family-wise bar of 2.43. Measured extrapolation orders are 1.95 and 1.93 for the two polynomial functions, and the check ran on 4 levels. Not replicated: |z - bar| = 1.05 is outside the 0.7 margin, so one seed settled it and the plan was not marginal.
- terminal_law:            not declared
- shape:                   skewness 1.636 +/- 0.012, excess kurtosis 3.130 +/- 0.077 (right-skewed, mass near 0, as the spec states)
- constraints:             finite: pass. support [0, 100]: pass, drift 0, so no state was clipped and none left the domain. invariant `estimator_stability` (non-gate): pass, drift 2.7e-3 against a bar of 1e-2, headroom 0.57 decades. `neg_fraction` is 0.
- reported_moments_agree:  true. The solver's own mean and variance (ddof=1) match the kernel's recomputation exactly.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. A surrogate reference passed its trust guard and the moments match it (Tier A').

### Skips and vocabulary
- `cross_plan_agreement`, `functionals_ok` and `cross_check_ok` are `unavailable`. Their inputs do not exist here: there is only one plan, and the spec declares no functionals or cross-check. None of them counts against the plan.
- The accuracy seed study did not run (`accuracy_seeds` = 0). That is a skip, not a pass.
- Dynkin evidence has two limits. The generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from problem.md would fool it. The same holds for the Kolmogorov reference, which was built from the same spec expressions. Both are circularity breaks against the scheme, not against the spec. This solver's coefficients do match the spec: A = 5, sigma = 0.03, N = 100.

### Diagnosis
- The strong order reads 0.9993 +/- 0.016 with clean factor-of-four Dynkin residual ratios. That is a genuine first-order result and not an artefact of ignoring `dW`. The step takes `Nt = dW.shape[1]` and consumes `dW[:, n]` exactly once, in the noise stage. Rule 0 and rule 0b do not apply, and the plan correctly declares no `exact_sampling_components`.
- The one measurable discretization effect is the variance bias at the graded step: 8.6e-03 relative, 3.4 standard errors from the Richardson value. It is real but sits at 8.6% of the 10% tolerance, so it is not actionable.
- The source scan found no `state_clip` warning. The domain is preserved by the logit map, as the plan states, and not by a post-update clip.
- The plan describes logit transform, Strang splitting, exact logistic half-flows and Heun on the bounded residual, and `solver.py` implements exactly that. No cap applies.

### Feedback for solver
None. Score 10, certified.

Optional and not scored: the graded variance carries a real 8.6e-03 dt bias (0.086 of its tolerance). The `moment_bias` block reports it as not actionable, so no change is needed at the contract step.

</review>
