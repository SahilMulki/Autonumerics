---
id: 1
plan_slug: jump-adapted-milstein-control-variate
scheme: jump-adapted-milstein
scheme_family: milstein
strategy: Milstein on the GBM-like continuous leg with the Poisson jump times inserted as extra sub-step boundaries (strong order 1.0), with the jump-free GBM call as an analytically priced control variate on the graded price.
---

**Declared scheme.** This plan steps the state with the supplied `dt` and consumes the supplied
Brownian increments on every sub-step, so `scheme_family: milstein` is the family the order study
should hold it to, at the spec's `verification.expected_strong_order_milstein = 1.0` floor. Order 1.0
is reachable here **only because the stepping is jump-adapted** — the jump times are inserted as
sub-step boundaries so the diffusion is advanced exactly up to each jump and the state is multiplied
by `Y` at the jump time itself. A Milstein scheme that instead applies accumulated jumps at grid
points carries an O(√dt) jump-placement error and measures ≈0.5 (`verification.strong_order_note`);
this plan does not do that.

**`exact_sampling_components` is deliberately absent**, per `implementation_notes` item 1. The
diffusion leg must consume the evaluator's `dW`, the terminal Poisson-mixture-of-lognormals law is
explicitly *not* what this problem measures, and a near-zero measured strong order here would be a
real defect rather than something to declare away.

No barrier or absorbing boundary is declared in `verification.structural_facts`, so no
`boundary_treatment` field applies. Positivity is carried as a gated constraint and is handled by the
step form (see Implementation Notes).

## SDE Reference

Scalar Merton jump diffusion:

```
dX = (r - lam*kappa) * X dt + sigma * X dW + X dJ,     X(0) = 100
```

`J` is a compound Poisson process with intensity `lam`; each jump multiplies `X` by
`Y = exp(Z)`, `Z ~ N(m_J, delta_J**2)`. The compensator uses
`kappa = E[Y] - 1 = exp(m_J + delta_J**2/2) - 1`, so `E[X(t)] = X_0 * exp(r*t)`.

Parameters (from `problem_spec.json → parameters`):

```python
X_0      = 100.0
r        = 0.05
sigma    = 0.15
lam      = 0.1
m_J      = -0.90
delta_J  = 0.45
K        = 100.0          # strike; used only by the reported functional
kappa    = -0.550109023493067      # = E[Y] - 1   (NEGATIVE here)
eta2     = -0.7521669636327712     # = E[Y**2] - 1
```

Drift coefficient: `a = r - lam*kappa = 0.05 - 0.1*(-0.550109...) = +0.10501090...`
(**larger** than `r`, not smaller).

Time interval `[0, T]` with `T = 0.25`. Continuous part: `f(X) = a*X`, `g(X) = sigma*X`,
`dg/dX = sigma`. Milstein correction: `0.5 * sigma**2 * X_n * (dW**2 - dt)`.

Reference values at `T = 0.25` (`exact_values_at_T`): mean `101.25784515406345`,
variance `147.94671758296317`.

## Numerical Scheme

- **Method**: jump-adapted Milstein (Milstein on the continuous leg; jumps applied at their exact
  times as extra sub-step boundaries)
- **dt**: 0.01 (contract-fixed; `Nt = max(1, round(T/dt)) = 25`, then `dt = T/Nt`)
- **num_paths**: 400000 (graded); the same function is called again at 100000
- **T**: 0.25
- **seed**: 42
- **Milstein correction**: `0.5 * sigma**2 * X_n * (dW_sub**2 - dt_sub)`, evaluated at `X_n`
  (the value at the *start* of the sub-step), with `dt_sub` the length of that sub-step
- **Expected strong order**: 1.0
- **Expected weak order**: 1.0
- **Variance reduction**: control variate — the jump-free GBM call, priced in closed form
- **Reference for scoring**: analytic moments (`analytic_moments`, cross-checked by the exactly
  closing `verification.moment_ode`) for mean/variance, and `verification.analytic_functional`
  (Merton's Poisson-weighted series) for the gated `call_price`

## Implementation Notes

**Function signature.** The contract is
`solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict`. Add `dW=None` and
`observables=None` as optional keyword arguments so the evaluator's CRN order study and Dynkin hook
can call it; defaults must reproduce the contract behaviour exactly. Where `dW` is supplied, take
`Nt = dW.shape[1]`, set `dt = T/Nt`, use the increments exactly and draw no Brownian randomness of
your own.

**Jump stream must be dt-independent** (`implementation_notes` item 2). From the single
`np.random.default_rng(seed)`, in a fixed order: draw per-path counts `N ~ Poisson(lam*T)`, then
`sum(N)` uniform jump times on `[0, T]`, then `sum(N)` sizes `Z ~ N(m_J, delta_J**2)`. Do **not**
draw per-step Poisson counts — that makes the jump realisation a function of `dt` and the CRN ladder
would then be measuring jump resampling instead of the scheme. With `lam*T = 0.025`, about 2.5% of
paths carry at least one jump; sorting each path's times is cheap.

**Splitting the supplied increment at a jump.** A grid step `[t_n, t_n + dt]` containing a jump at
interior fraction `u` must be advanced in two Milstein sub-steps of lengths `u*dt` and `(1-u)*dt`,
with the step's Brownian increment split by the Brownian bridge:

```
dW1 = u * dW_step + sqrt(u*(1-u)*dt) * xi,      dW2 = dW_step - dW1
```

Draw one `xi ~ N(0,1)` **per jump**, alongside the jump times and sizes, from the same seeded
generator — keyed by jump index, not by step index. That keeps the auxiliary draw dt-independent, so
the ladder's `dt`, `dt/2`, `dt/4` levels share it exactly as they share `dW`, and the split remains
consistent with the supplied total increment at every level. Multiply by `Y = exp(Z)` between the two
sub-steps. Two jumps inside one grid step is possible and must be handled by the same recursion
(sort the times, walk them in order) — at `lam*T = 0.025` it is rare but not impossible.

**Compensation sign is the single most likely bug** (`implementation_notes` item 4). Use
`a = r - lam*kappa = +0.10501090...` as the drift coefficient. Using `r` directly, or flipping
`kappa`'s sign, breaks the gated discounted-martingale invariant
(`E[exp(-r*T) * X_T] = X_0 = 100`, `ci_mult = 3.0`) by about 1.4%, which is many standard errors.
Self-check before reporting: `exp(-r*T) * mean(terminal_paths)` should sit at 100 within ~3 s.e.

**Control variate on the graded price.** Carry a second state `S` stepped with the *same* Milstein
scheme, the *same* `dW`, and **no jumps** — i.e. plain GBM with drift coefficient `a`. Its discounted
call has a closed form (this is exactly the `n = 0` term of Merton's series, *not* the answer):

```
M  = X_0 * exp(a*T);   v = sigma**2 * T
d2 = (log(M/K) - 0.5*v) / sqrt(v);   d1 = d2 + sqrt(v)
P_cv = exp(-r*T) * (M*Phi(d1) - K*Phi(d2))
```

with `Phi(x) = 0.5*(1 + math.erf(x/sqrt(2)))` (stdlib only — scipy is not permitted here). Report

```
price = mean(payoff_X) - beta * (mean(payoff_S) - P_cv)
```

with `payoff_* = exp(-r*T) * maximum(* - K, 0)`, and `beta` the least-squares coefficient
`cov(payoff_X, payoff_S)/var(payoff_S)` estimated from the same sample (or simply `beta = 1.0`, which
is near-optimal because ~97.5% of paths have `X == S` pathwise). Report
`functional_se["call_price"]` as the standard error of the **combined** estimator
`payoff_X - beta*payoff_S`, not of `payoff_X` — that number is the only thing that tells the kernel
the control variate was worth writing. Two further notes: (i) because `S` is stepped with the same
scheme and increments, its own discretization bias largely cancels `X`'s, so this is a mild bias
reduction as well as a variance reduction; (ii) an in-sample `beta` introduces an O(1/num_paths)
bias, negligible at 400000 paths but a reason to prefer `beta = 1.0` if anything looks off.

**`functionals` and the two path counts.** `functionals["call_price"]` must be a float computed from
the returned `terminal_paths` (plus the control term). It is gated at 1% relative. At 400000 paths the
raw payoff's own 2-s.e. band is ≈0.4% of a value near 4.39 — inside the gate but not comfortably, which
is why the control variate is here and why `dt` must stay at the contract value rather than being
coarsened. The estimate must *move* between the 400000-path and 100000-path calls (both use seed 42):
drawing one block of randomness sized by `num_paths` does this naturally; anything that returns a
constant at both counts fails outright.

**Positivity and finiteness** (both gated constraints). The Milstein step is additive in form, so a
large negative increment could in principle push `X` below zero; at `sigma*sqrt(dt) = 0.015` that
needs a ~66-sigma draw, so a violation means a coding error. Do not clip silently — if any terminal
path is `<= 0` or non-finite, that is a bug to find, not to mask.

**Verification ceiling.** `verification.dynkin_note` makes the Dynkin check a declared skip: the
schema cannot express the non-local jump term `lam*E[phi(xY) - phi(x)]`, and the diffusion-only
generator would false-fail every correct solver. Per the project manual's 9 row this caps
certification at **9**. That is a property of the schema, not of this plan; do not chase it.

## Results

Run at num_paths=400000, dt=0.01, T=0.25, seed=42:

- empirical_mean = 101.252761 (exact 101.257845)
- empirical_variance = 147.608120 (exact 147.946718)
- call_price = 4.389490, functional_se = 1.97e-03 (raw payoff se 8.69e-03, so the control variate cuts the se about 4.4x); 100000-path call_price = 4.395180
- discounted-martingale check exp(-rT)*mean = 99.995 (expect 100); min terminal path 3.26 (positive, finite)
- dW and observables hooks verified; CRN strong error (vs Nt=100 reference) 6.3e-3 at Nt=25, 2.1e-3 at Nt=50

## Evaluation

(See the metrics and review blocks below.)

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 2.289e-03
error_is_estimate: false
observed_order: 0.9969
observed_order_se: 1.155e-03
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 5.021e-03
invariant_worst_headroom: 1.0544
reference_outcome: validated
mean_rel_err: 5.021e-05
variance_rel_err: 2.289e-03
functional_name: call_price
functional_rel_tol: 0.0100
functional_rel_error: 4.530e-04
functional_bias_rel: 5.289e-06
functional_se_rel: 4.477e-04
functional_limited_by: sampling
moment_bias_binding: variance
moment_bias_frac_tol: 8.565e-03
moment_bias_actionable: false
order_check: true
mc_se_rel: 6.766e-03
resolved: true
dynkin_z: 72.0709
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: abc0a68d83869708f34967ab1f0a4f0f461aa121a1ef67bb903d0fd82408a768
solve_time_s: 0.4750
wall_time_s: 38.5500
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the moment ODE; Path A)
- mean_rel_err:            5.0e-05 +/- 2.0 x 0.0192 absolute (se on 101.25; 101.2528 vs exact 101.2578), tol 5%
- variance_rel_err:        2.29e-03 +/- 2.0 x 0.999 absolute (se on 147.6; 147.608 vs exact 147.947), tol 10%
- mc_se_rel:               6.77e-03   resolved: true (variance CI width 1.35% against a 5% bar)
- functional_rel_error:    4.53e-04 against functional_rel_tol 1.0e-02  (bias 5.3e-06 + sampling 4.48e-04 -> limited by sampling, binding functional call_price). Reported call_price 4.38949 with the control-variate se 1.97e-03.
- observed_order:          0.9969 +/- 0.0012 (expected 1.0, one-sided below the declared floor); weak order 0.9999
- weak_order:              0.9999, in band
- dynkin:                  unresolved (score 9, not 10). The check ran on X and X**2; z = 72.1 against bar 2.43 is NOT a verdict, since the extrapolated limit moved between the ladder's two windows by 4.0 and 3.2 paired s.e. (X, X**2), so the levels are not in one asymptotic regime. Single seed, not replicated.
- terminal_law:            not declared (Poisson mixture of lognormals, by design)
- shape:                   skewness -3.051 +/- 0.012, excess kurtosis 16.31 +/- 0.09
- constraints:             martingale ok (drift 5.0e-03 against bar 5.7e-02), estimator_stability ok (1.6e-04 against 1e-2), positivity ok, finite ok. No gate violation.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. The generator check reached no verdict, so the 10 row is closed. Tier A would otherwise allow 10.

### Feedback for solver
- Nothing to fix. The scheme is correct: strong order 0.997 +/- 0.001 against 1.0 confirms that the jump-adapted split with the Brownian bridge and the dt-independent jump stream works. Compensation sign (a = r - lam*kappa) is right, as the martingale invariant shows.
- The ceiling is the check, not the solver. `dynkin_ok: unresolved` is a property of this problem and not of the plan. `verification.dynkin_note` says the schema can only express the diffusion generator, not the non-local jump term, so the residual does not behave as a clean extrapolation. Do not chase it.
- Sampling, not bias, limits the call price (bias 5.3e-06 against sampling 4.5e-04, a 1% gate). No change to dt or paths is needed. The control variate is already doing its job.
- The 100000-path rerun is unaffected: all comparisons here are resolved at 400000 paths.
- Formulator note: the spec's own Dynkin declaration is `dynkin_test_functions: null`, yet the kernel still evaluated X and X**2 using the diffusion-only generator. That the result was `unresolved` and not a hard 3 is lucky, since a jump-free generator would false-fail a correct solver. This is worth the formulator checking.

</review>
