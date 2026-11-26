---
id: 3
plan_slug: log-implicit-euler
scheme: drift-implicit Euler on L = log V (Newton solve per step)
scheme_family: implicit
strategy: Step log V with its quadratic-in-V drift treated implicitly (a monotone scalar equation solved by Newton) and its multiplicative noise explicitly, giving a strictly positive, overflow-free discretisation in the original variable as the cross-check on the reciprocal-coordinate plans.
---

## SDE Reference

Ito SDE for the instantaneous variance V > 0 (the 3/2 model):

    dV = kappa * V * (theta - V) dt + epsilon * V**1.5 dW,     V(0) = V_0

| parameter | value |
|---|---|
| V_0 | 0.15 |
| kappa | 22.84 |
| theta | 0.218 |
| epsilon | 8.56 |
| T | 0.1 (T0 = 0) |

Drift `f(V) = kappa*V*(theta - V)`, diffusion `g(V) = epsilon*V**1.5`, `g'(V) = 1.5*epsilon*sqrt(V)`.

**Coordinate used by this plan.** `L = log V` satisfies (Ito)

    dL = [ kappa*theta - (kappa + epsilon**2/2) * e^L ] dt + epsilon * e^(L/2) dW,     L(0) = log(0.15)

with `kappa*theta = 4.97912` and `c = kappa + epsilon**2/2 = 59.4768`. Same sign of `dW` as the V equation.

Graded quantities (returned in `functionals`, computed from terminal V):

- `inv_var_mean = mean(1/V_T)` — exact reference 11.622786285052442, tolerance 5%
- `inv_var_variance = var(1/V_T, ddof=1)` — exact reference 45.23510647598556, tolerance 10%

## Numerical Scheme

- **Method**: drift-implicit Euler in L with explicit noise. With `V_n = e^{L_n}` and

      r = L_n + kappa*theta*dt + epsilon*sqrt(V_n)*dW_n

  solve `F(x) = x + c*dt*e^x - r = 0` for `x = L_{n+1}`. `F` is strictly increasing and convex, so
  the root is unique and lies below `r`. Since `F' >= 1` and F is convex, Newton started at or to
  the right of the root decreases monotonically onto it; `x0 = r` is always such a start
  (`F(r) = c*dt*e^r > 0`). Where `r > 1`, `x0 = min(r, log(r/(c*dt)))` is closer and still safe
  (Newton from the left of a convex increasing F overshoots once to the right, then is monotone).
  Iterate to `|dx| < 1e-12`, vectorised over paths, with a cap of ~50 iterations. Closed form alternative: `x = r - W0(c*dt*e^r)` with Lambert W, but
  numpy has no Lambert W, so use Newton.
- **Why implicit**: the explicit log-Euler step is unstable here. At V ~ 10 with a 4-sigma increment
  the explicit drift `-c*V*dt = -2.4` loses to the noise `epsilon*sqrt(V)*dW = +6.8`, V jumps to
  ~900, and the next explicit drift `-c*900*dt = -214` sends V to ~1e-90, so 1/V explodes and
  ruins both graded moments. The implicit drift bounds the update (`e^{L_{n+1}} < (r - L_{n+1})/(c*dt)`),
  so a large positive increment no longer produces a huge V and the collapse never starts.
- **dt**: 0.004 (contract value; `Nt = max(1, round(T/dt)) = 25`, `dt = T/Nt`)
- **num_paths**: 200000 graded, 50000 second contract call (spec default 50000)
- **seed**: 42 via `np.random.default_rng(seed)`
- **T**: 0.1
- **Expected strong order**: 0.5 (the noise `epsilon*e^{L/2}` is multiplicative in L and no Milstein
  term is applied)
- **Expected weak order**: 1.0
- **Reference for scoring**: no closed-form moments of V; the evaluator scores against the two gated
  `martingale` invariants (`1/X` -> 11.622786285052442, `1/X**2` -> 180.3242675039887), the exact
  reciprocal moments, plus Dynkin (`1/X`, `1/X**2`, `tanh(X)`) and the CRN order study.
  `expected_provenance: self_convergence`.

## Implementation Notes

- Contract: `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`. When `dW` is given,
  take `Nt` from its shape, set `dt = T/Nt`, use it exactly, and draw no randomness of your own.
- `terminal_paths = exp(L_T)`, shape `(num_paths,)`. Must be finite and strictly positive on every
  path; do not clip. Guard against `exp` underflow by checking `min(L_T) > -700` and report it.
  `empirical_mean` / `empirical_variance` (ddof=1) are of V.
- `observables` are functions of V: evaluate `phi(exp(L))` at both ends of each step, trapezoid rule,
  return as `path_integrals`.
- `functionals = {"inv_var_mean": mean(exp(-L_T)), "inv_var_variance": var(exp(-L_T), ddof=1)}`, plus
  `functional_se` for both (`std/sqrt(M)`; delta-method `sqrt((m4 - s2**2)/M)` for the variance).
- **Bias risk, stated up front**: this is a strong-order-0.5 scheme and its weak bias on the
  reciprocal moments at dt = 0.004 is not known in advance. The gates (5% / 10%) are
  bias-dominated (MC relative SE ~0.13% / ~0.43% at 200k paths). The solver must report the graded
  functionals at dt, dt/2 and dt/4 (self-convergence) so the bias can be read off; if the dt-value
  misses a gate, report it rather than changing the contract step.
- An alternative, if the Newton solve proves troublesome: the spec also lists taming or
  balanced-implicit steps in V. Do not use plain explicit Euler in V — it can return V <= 0, which
  fails the gated positivity constraint.
- No exact transition (forbidden by problem.md); every supplied increment is consumed, so
  `exact_sampling_components` is not declared.
- **Variance reduction (control variates from the generator)**: along each path accumulate the trapezoid
  integrals of `L(Y)` and `L(Y^2)` for `Y = 1/V` (`dY = (kappa+eps^2 - kappa*theta*Y) dt - eps*sqrt(Y) dW`,
  `L(Y) = kappa+eps^2 - kappa*theta*Y`, `L(Y^2) = 2Y L(Y) + eps^2 Y`). The Dynkin martingales
  `c1 = Y_T - Y_0 - int L(Y)`, `c2 = Y_T^2 - Y_0^2 - int L(Y^2)` have mean zero for the SDE; `inv_var_mean` and
  `inv_var_variance` are the intercepts of the regressions of `Y` and `Y^2` on `(1, c1, c2)`, combined as
  `(q - m^2) M/(M-1)`. `functional_se` is the residual / influence-function SE of that combined estimator.
  No exact value is used; the stepping scheme is unchanged. Plain sample moments are returned as
  `plain_functionals`. Antithetic pairs were tried and rejected (cut the mean SE but raised the variance SE).
- numpy only; no work at import time.

## Results

Run: `uv run python solver.py` (dt = 0.004, Nt = 25, seed 42, T = 0.1). Newton solve per step, no clipping; all terminal V finite and > 0 (min V = 1.53e-2, min L = -4.18, far above the -700 underflow guard).

| num_paths | empirical_mean (V) | empirical_variance (V) | inv_var_mean | inv_var_variance |
|---|---|---|---|---|
| 200000 | 0.133270 | 0.018495 | 11.61551 (ref 11.62279, -0.06%), se 0.00192 | 45.24750 (ref 45.23511, +0.03%), se 0.0534 |
| 50000 | 0.133011 | 0.018761 | 11.62074 (-0.02%), se 0.00383 | 45.20875 (-0.06%), se 0.1073 |

Functional estimators now use the generator control variates (SE of inv_var_variance 0.12% vs 0.44% plain; inv_var_mean 0.017% vs 0.13%). Terminal V finite and > 0 (min V = 1.53e-2).

CRN self-convergence (50000 paths, shared Brownian path), inv_var_mean / inv_var_variance:
dt = 0.004: 11.6182 / 45.134; dt/2: 11.6205 / 45.191; dt/4: 11.6217 / 45.215. Bias at the graded dt is about 0.03% / 0.2% (was 0.1% / 0.45%).

Hooks verified: explicit `dW` (dt = T/Nt used) and `observables` (path_integrals shape (num_paths,)).

## Evaluation

<metrics>
kind: sde
score: 9
certified: false
provenance: manufactured_partial
estimated_rel_error: 1.844e-03
error_is_estimate: true
observed_order: 0.4577
observed_order_se: 0.0287
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
strong_rel_err: 0.1453
strong_rel_err_se: 3.646e-03
strong_rel_err_component: 0
design_order: 0.5000
scheme_family: implicit
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.2642
invariant_worst_headroom: 0.3622
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 2.750e-03
functional_bias_rel: 3.771e-04
functional_se_rel: 2.373e-03
functional_limited_by: sampling
moment_bias_actionable: false
order_check: true
mc_se_rel: 0.0839
resolved: unavailable
dynkin_z: 1.9620
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 6e434fed76b1b4668d9e452287349b65b100d8e5f8ef219e71aeca9dfc6f08c9
solve_time_s: 0.1690
wall_time_s: 6.8700
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              manufactured_partial  (reference route: none. `surrogate_ok: unavailable`. Attempted: moment_ode "no moment ODE, or it does not close exactly", kolmogorov "no kolmogorov.bounds declared", stationary_density "no normalizable support declared". The spec's `expected_provenance` was `self_convergence`, so `provenance_as_expected: false` means the run reached a better tier than predicted, one circularity break from Dynkin. It is not a defect and is not scored on.)
- mean_rel_err:            not_reported (`moments_ok: unavailable`, no reference for the moments of V). Context only: Richardson-extrapolated mean shift at the graded dt is 0.41% (1.75 se, within noise) against the 5% tolerance.
- variance_rel_err:        not_reported (same reason). Context only: Richardson-extrapolated variance shift at the graded dt is 12.3% of the value, but only 1.65 se, so it is not a resolved bias (`moment_bias.real: false`, `actionable: false`). The estimator of Var[V] has infinite variance because E[V^4] is infinite.
- mc_se_rel:               0.0839 at 50000 paths, resolved: unavailable. Kernel reason: "no reference variance to resolve against". This is a missing reference, not a width failure, and it is not a solver bug.
- functional_rel_error:    2.75e-03 against functional_rel_tol 0.10 (bias 3.77e-04 + sampling 2.37e-03 -> limited by sampling, binding functional inv_var_variance). Both terms are far inside the gate. The other functional, inv_var_mean, shows a ladder error of 5.2e-05 against 0.05. Both functionals are "clean". Graded values at 50000 paths: inv_var_mean 11.6207 and inv_var_variance 45.2088, against exact 11.62279 and 45.23511.
- observed_order:          0.458 +/- 0.029 (median-based 0.528), expected 0.5, one-sided band below the declared floor. orders_ok true, strong_order_shrinking true. The one-sided band applies because the floor came from the spec and the shortfall only is judged.
- weak_order:              0.68 from tanh only; x and x2 are at the noise floor and not used. Reported, never scored on.
- dynkin:                  ok. Replicated: first seed z = 1.96 against bar 2.43, which is 0.47 from the bar, so it was marginal and the check was re-run. Median z over 3 seeds = 1.82, per-seed z 1.96 / 1.82 / 0.89. The pooled z of 2.70 sits above the bar, but the rule decides on the median. Per test function (1/X, 1/X**2, tanh(X)) all three pass, z = -1.96 / -1.41 / +1.50.
- terminal_law:            not declared
- shape:                   skewness 10.1 +/- 3.5, excess kurtosis 350 +/- 236 (heavy tail of V, as the spec predicts)
- constraints:             finite true, positivity true (gate), martingale E[1/X] and E[1/X**2] true (max drift 1.264 against bar 6.49), estimator_stability true. No gate_violation, no negative paths, and no state clip in the source.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. "no surrogate: the orders, Dynkin and the constraints all hold and the Richardson moments are stable across the ladder". The certification ceiling also lands at 9 (one circularity break, D1 against a validated operator; no Tier-B evidence). `certified: false` reflects the absence of Tier-A/B evidence, not a failed gate (`gates_clean: true`).

Skip vocabulary: `moments_ok`, `variance_ok`, `resolved`, `cross_plan_agreement`, `cross_check_ok` and `surrogate_ok` are `unavailable` (the inputs do not exist). `accuracy_seeds` was `not_run` (0 requested). None of these is a pass or a failure.

The strong error is large in absolute terms (`strong_rel_err` 0.145 +/- 0.004 at dt = 0.004), as expected for an order-0.5 scheme with eps = 8.56 noise. The spec sets no `strong_rel_err_max`, so it is not gated.

### Feedback for solver
- The code matches SOLUTION.md. The log-coordinate drift-implicit Euler is implemented as described: Newton on a convex increasing F, no state clipping, every supplied `dW` consumed, `Nt = dW.shape[1]` and `dt = T/Nt`. Nothing needs fixing and no cap was applied.
- The 9 is the evidence-tier ceiling, not a defect. There is no surrogate for V, so the run earned one circularity break from Dynkin. Dynkin does not establish that the drift was transcribed correctly from `problem.md`, because the generator is built from the spec's own coefficients. The exact reciprocal moments, enforced through the two martingale invariants, are the independent evidence, and they pass.
- The accuracy margin is large and the binding term is sampling, not bias: `functional_bias_rel` 3.8e-04 against `functional_se_rel` 2.4e-03, against a 10% gate. A smaller dt would buy nothing. More paths or the existing control variates are the only lever, and neither is needed to pass.
- The 9 comes from the evidence tier. The plan only fails to certify because no independent reference for V is available, not because of anything in the solver.
- Dynkin was marginal on the first seed (z = 1.96 against 2.43). The 1/X test function has the largest residual and its extrapolation order is 1.14, consistent with a first-order weak bias. The median over 3 seeds is 1.82, so the verdict is a pass. This is a property of the check at the path count used, not a sign of a wrong equation.
- Spec side, for the formulator: the exact reciprocal moments E[1/V_T] and Var[1/V_T] exist but cannot be carried as a `moment_ode` or `terminal_law` reference, because those fields compare moments of `terminal_paths` (= V). The spec's own notes already record this as a vocabulary limit rather than a missed surrogate, so I do not count it as a formulator defect.

</review>
