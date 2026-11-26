---
id: 1
plan_slug: beta-moment-matched
scheme: moment-matched-beta-step
scheme_family: other
exact_sampling_components: 0
boundary_treatment: none
strategy: Step X at the given dt by drawing each X_{n+1} | X_n from the Beta law on [0,1] that carries the exact conditional mean and variance of the Wright-Fisher transition over one step, so the graded mean and variance carry no dt bias and the support is preserved by construction.
---

## SDE Reference

Scalar Itô SDE on the state space [0, 1]:

```
dX = ( a (1 - X) - b X ) dt + sigma sqrt( X (1 - X) ) dW,     X(0) = X_0
```

| Parameter | Value |
|---|---|
| X_0 | 0.05 |
| a | 0.25 |
| b | 0.25 |
| sigma | 1.0 |
| T0, T | 0.0, 1.0 |

- drift `f(X) = a*(1.0 - X) - b*X`
- diffusion `g(X) = sigma*np.sqrt(np.maximum(X*(1.0 - X), 0.0))`
- Both boundaries are attainable (2a/sigma^2 = 2b/sigma^2 = 1/2 < 1) and reflecting via the mutation drift.

Exact reference at T = 1: `E[X(1)] = 0.22706120312931494`, `Var[X(1)] = 0.0609928976136204`.

## Numerical Scheme

- **Method**: per-step conditional-moment-matched Beta draw (sde_manual.md rule 0c). This is a moment-matched discretisation, like Andersen QE. It is **not** a draw from the transition law, which is an infinite series here, and it is not a terminal-law draw: the state is advanced step by step at the given `dt`, Nt = 25 steps at the contract step.
- **dt**: 0.04 (the contract step; `Nt = max(1, round(T/dt))`, `dt = T/Nt`)
- **num_paths**: 50000 for the kernel evaluation; the contract call uses 200000
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable
- **Expected strong order**: not measurable. The step draws its own randomness, so the CRN order study abstains on component 0 (declared in `exact_sampling_components`). That abstention is expected and is not a pass.
- **Expected weak order**: the graded mean and variance are **exact at every dt** (bias-free by the tower property). For other smooth test functions, such as the Dynkin probe `sin(pi X)`, it is weak order 1.0.
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`, with the closed-form `mean_expression` / `variance_expression`). The exact moment ODE in `verification.moment_ode` reproduces them to 2e-14.

### The step

The generator maps polynomials to polynomials of the same degree, so the conditional first two moments over a step of length `h` from `X_n = x` are explicit. With

```
lam1  = a + b
lam2  = 2*(a + b) + sigma**2
m_inf = a / (a + b)
m2_inf = (2*a + sigma**2) * m_inf / lam2
```

they are

```
mean(x, h)   = m_inf + (x - m_inf) * exp(-lam1*h)
B(x)         = (2*a + sigma**2) * (x - m_inf) / (lam1 + sigma**2)
C(x)         = x*x - m2_inf - B(x)
second(x, h) = m2_inf + B(x)*exp(-lam1*h) + C(x)*exp(-lam2*h)
var(x, h)    = second(x, h) - mean(x, h)**2
```

At a = b = 1/4 and sigma = 1 this reduces to `mean = 0.5 + (x-0.5) e^{-h/2}` and `second = 0.375 + (x-0.5) e^{-h/2} + (x^2 - 0.375 - (x-0.5)) e^{-2h}`. Use the general-parameter form, computed from `a, b, sigma`, not a hardcoded specialisation.

Then draw `X_{n+1} ~ Beta(alpha, beta)` with

```
nu    = mean*(1 - mean)/var - 1
alpha = mean * nu
beta  = (1 - mean) * nu
X_{n+1} = rng.beta(alpha, beta)          # vectorised over paths
```

The Beta has exactly that mean and variance. By the tower property, `E[X_T]` and `E[X_T^2]` are then exact, and the dt bias on the graded moments is identically zero.

## Implementation Notes

- **Contract** (problem.md): `solve_sde(num_paths, dt, T, seed=42)`, imported and called once as `solve_sde(num_paths=200000, dt=0.04, T=1, seed=42)`. It must return `terminal_paths` (shape `(num_paths,)`), `empirical_mean`, and `empirical_variance` (`ddof=1`). Seed all randomness with `np.random.default_rng(seed)`. Use numpy and the standard library only (no scipy). Do no work at import time. Use the `dt` you are given.
- **Kernel hooks**: also accept `dW: np.ndarray | None = None` and `observables: dict | None = None`.
  - When `dW` is supplied, take `num_paths, Nt = dW.shape` and `dt = T/Nt`. This scheme **cannot** consume `dW`: the Beta draw needs its own randomness, and numpy has no Beta inverse CDF to push `dW` through. So keep drawing from `np.random.default_rng(seed)` and ignore the values in `dW`. This is exactly why the frontmatter declares `exact_sampling_components: 0`. Do not try to fake consumption of `dW`.
  - When `observables` is supplied, accumulate `0.5*(phi(X_prev) + phi(X))*dt` per step and return it as `out["path_integrals"]`. Dynkin's test functions are `X`, `X**2` and `np.sin(np.pi*X)`.
- **Numerical guards**:
  - Clamp the variance to `max(var, tiny)` and to strictly below `mean*(1-mean)`, e.g. `var = np.minimum(var, (1 - 1e-12)*mean*(1-mean))`. Both hold analytically for every x in [0, 1]; the guards only absorb roundoff. A positive `nu` is required.
  - Near the boundary, at x = 0 and h = 0.04, the variance is about 1.9e-4 and comes out of a subtraction of O(0.4) terms. Double precision leaves about 1e-12 relative error there, which is fine. If you see trouble at small `h` in the Dynkin ladder, rewrite the `exp(-lam*h)` differences with `np.expm1`.
  - Near the boundary the Beta's shape parameter tends to about 2a/sigma^2 = 0.5, so `alpha, beta` stay O(0.5) or larger. Do not floor them artificially.
- **Boundary treatment (`none`)**: the Beta law is supported on [0, 1], so every state is in the support by construction. No projection, reflection or clipping is applied or needed. Both gated constraints (`finite`, `support` on [0,1]) hold automatically. Do not add a post-hoc `np.clip` on the output: nothing needs it, and the leakage scan would flag it.
- **What the bias-free claim covers**: the abstention on the dt-bias term is exactly right for the two graded moments (mean, variance). It is stronger than this scheme earns for anything else: a third moment, a quantile, or a path functional would still carry O(dt) error. problem.md grades only the mean and variance, so this is not exercised. It is stated here so that nobody reads the plan as exact in law.
- **Expected accuracy**: pure Monte Carlo error. At 200000 paths the mean's standard error is about sqrt(0.061/2e5) ≈ 5.5e-4 (about 0.24% relative). The variance's relative standard error is of similar order. Both are far inside the 5% / 10% gates.
- **Legitimacy under problem.md**: "A discretised time-stepping scheme is required. Advance the state step by step at the `dt` you are given." This plan does that, with 25 Beta steps of size 0.04. It does not draw X_T from a terminal law, and it uses no transformation that makes one available. The spec's `implementation_notes` name it as a legitimate step-wise option.

## Results

Run: `solve_sde(num_paths=200000, dt=0.04, T=1.0, seed=42)` (25 Beta steps).

| Quantity | Empirical | Exact | Rel. error |
|---|---|---|---|
| Mean | 0.227025 | 0.227061 | 1.6e-4 |
| Variance | 0.061096 | 0.060993 | 1.7e-3 |

All terminal states lie in [0, 1] (min 4.2e-11, max 1.0). The `dW` hook (Nt taken from `dW.shape[1]`, values not consumed) and the `observables` hook (path integrals of shape `(num_paths,)`) both run.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 2.922e-03
error_is_estimate: false
observed_order_se: 7.148e-03
order_floor: 0.4000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.4000
scheme_family: other
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 3.794e-03
invariant_worst_headroom: 0.4209
reference_outcome: validated
mean_rel_err: 2.922e-03
variance_rel_err: 6.052e-05
order_check: unresolved
mc_se_rel: 7.407e-03
resolved: true
dynkin_z: 1.1931
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 7279c30f9ec7927ed1b7c5815047f19347dbaaed625311748d0c0447c8a4ea34
solve_time_s: 0.1580
wall_time_s: 2.8400
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated against the spec's operator; Tier A)
- mean_rel_err:            2.92e-03 +/- 2.0 x 1.10e-03 absolute se (tolerance 5e-2); mean 0.227725 against exact 0.227061
- variance_rel_err:        6.05e-05 +/- 2.0 x 4.52e-04 absolute se (tolerance 1e-1); variance 0.060997 against exact 0.060993
- mc_se_rel:               7.41e-03   resolved: true (CI width rel 1.48e-02 against bar 5e-2, 50000 paths)
- functional_rel_error:    not applicable, the spec declares no gated functional
- observed_order:          not measured. The order study abstained (`orders_ok: "unresolved"`) because the plan declares component 0 exactly sampled. The raw fitted order was 0.0006 +/- 0.0071, which is the expected reading for a step that does not consume dW. Declared floor 0.4, one-sided band.
- weak_order:              indeterminate, at the noise floor
- dynkin:                  ok. z = 1.19 against bar 2.43 (worst of X, X**2, sin(pi X)). Not marginal (|z - bar| = 1.235, outside the 0.7 margin), so one seed settled it. The first-order extrapolation was assumed because the finer residual difference is inside its own noise.
- terminal_law:            not declared
- shape:                   skewness 1.263 +/- 0.009, excess kurtosis 0.743 +/- 0.032 (context only)
- constraints:             finite pass, support [0,1] pass (both gated, drift 0), estimator_stability pass (drift 3.8e-3 against 1e-2, headroom 0.42 decades)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments and the estimates are MC-resolved. The order study abstained and is not counted against the plan either way.

### Feedback for solver
- None for the score. Notes for the record, none of them defects:
  - The order check abstained. Strong-order evidence is therefore absent for this plan. An abstention is not a pass, and the 10 rests on the moments, the Dynkin check and the constraints.
  - The `exact_sampling_components: 0` declaration is correct. The Beta draw does not consume `dW`, and the plan did not reason itself out of declaring it.
  - The bias-free claim covers the mean and variance only. The Dynkin residuals on `X**2` and `sin(pi X)` show a larger scatter (z = 1.19 on `sin(pi X)`) because those functionals do carry O(dt) error under moment matching. This is inside the bar here and is not exercised by the grading.
  - Two source-level observations. `X_0`, `A`, `B` and `SIGMA` are module constants that match the spec, so the ledger scan found no mismatch. `solve_sde` ignores `dW` values but takes `Nt` from `dW.shape[1]`, as the contract requires.
</review>
