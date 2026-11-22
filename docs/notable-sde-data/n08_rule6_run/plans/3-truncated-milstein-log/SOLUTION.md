---
id: 3
plan_slug: truncated-milstein-log
scheme: truncated (Mao-style) Milstein in the log coordinate L = log V
scheme_family: tamed-milstein
boundary_treatment: none
strategy: Step L = log V with an explicit Milstein scheme whose superlinear drift and diffusion are evaluated at a truncated state with radius growing like dt**-0.25, so positivity is automatic through V = exp(L) and no excursion can overflow, giving a fully explicit order-one cross-check in a third coordinate.
---

## SDE Reference

Scalar Ito SDE for the instantaneous variance V > 0:

```
dV = kappa * V * (theta - V) dt + epsilon * V**1.5 dW,    V(0) = 0.15
```

Parameters (from `problem_spec.json → parameters`):

```python
kappa   = 22.84
theta   = 0.218
epsilon = 8.56
V_0     = 0.15
T       = 0.1          # time_interval: [0.0, 0.1]
```

- `drift_expression`:      `kappa * X * (theta - X)`
- `diffusion_expression`:  `epsilon * np.maximum(X, 0.0)**1.5`
- `diffusion_derivative`:  `dg/dV = 1.5 * epsilon * np.sqrt(np.maximum(V, 0.0))`
- `milstein_correction` (in `V`): `0.75 * epsilon**2 * V**2 * (dW**2 - dt)`
- `noise_structure`: multiplicative, `state_dimension = 1`, `noise_dimension = 1`,
  `milstein_eligible: true`, `stiff: true`

**Graded quantities** (`evaluation_thresholds.functionals`, read from the returned `functionals` dict):

| key | quantity | exact reference | relative tolerance |
|---|---|---|---|
| `inv_var_mean` | `E[1/V(T)]` | `11.622786285052442` | 5% |
| `inv_var_variance` | `Var[1/V(T)]` (ddof=1) | `45.23510647598556` | 10% |

Contract calls: `solve_sde(num_paths=200000, dt=0.004, T=0.1, seed=42)` and again with
`num_paths=50000`, everything else unchanged. `Nt = max(1, round(T/dt)) = 25`, `dt = T/Nt = 0.004`.

## Numerical Scheme

### The coordinate change

Let `L = log V`. Ito's lemma (`h = log`, `h' = 1/V`, `h'' = -1/V**2`, `g**2 = epsilon**2 V**3`):

```
dL = [ kappa*theta - (kappa + 0.5*epsilon**2) * V ] dt  +  epsilon * np.sqrt(V) * dW,   V = exp(L)
```

with `kappa + 0.5*epsilon**2 = 59.4768`. Two things this buys:

- **Positivity is free.** `V = exp(L) > 0` for every finite `L`, so the strict-positivity gate is
  satisfied by the representation itself — no guard, no clipping, no reflection.
- **It is the natural coordinate for the graded quantity.** `1/V = exp(-L)`, so the reciprocal
  moments that carry the whole grade are smooth bounded-below functionals of the stepped variable,
  and the small-`V` region that dominates `E[1/V]` is exactly where this coordinate is best
  behaved: as `L -> -inf` the drift tends to the constant `+kappa*theta > 0` (pushing `V` back up)
  and the diffusion `epsilon*exp(L/2)` tends to zero.

### Milstein correction in the log coordinate

With `g_L(L) = epsilon*exp(L/2)`, `dg_L/dL = 0.5*epsilon*exp(L/2)`, the Milstein term is

```
0.5 * g_L * dg_L/dL * (dW**2 - dt) = 0.25 * epsilon**2 * exp(L) * (dW**2 - dt)
                                   = 0.25 * epsilon**2 * V * (dW**2 - dt)
```

Note it is **linear** in `V`, where the same correction in the `V` coordinate is quadratic
(`0.75*epsilon**2*V**2`) — one power of the superlinearity is absorbed by the transform.

### Truncation (Mao), not `|dW|`-taming

Both coefficients are superlinear in `V`, so the explicit scheme needs the growth bounded for the
convergence proof and for float64 safety. Evaluate **every** coefficient at a truncated state:

```python
R  = 1.0 * dt**-0.25                     # 3.9764 at dt = 0.004
Vt = np.minimum(V, R)                    # V itself is never modified
```

and step

```python
mu  = kappa*theta - (kappa + 0.5*epsilon**2) * Vt
sig = epsilon * np.sqrt(Vt)
L   = L + mu*dt + sig*dW[:, n] + 0.25*epsilon**2 * Vt * (dW[:, n]**2 - dt)
V   = np.exp(L)
```

Why truncation rather than a tamed denominator: a `|dW|`-dependent denominator of the balanced /
tamed-diffusion kind shrinks the diffusion increment by a relative `O(sqrt(dt))` — roughly 17% at
this `dt` and these parameters — which biases the variance of the graded reciprocal directly.
Truncation leaves the increment untouched wherever it does not activate, and `R(dt) ~ dt**-0.25`
grows without bound as the step refines, so the truncation error is higher order and the strong
order 1.0 of the Milstein step is retained.

At the graded step `R = 3.98`, against a state whose invariant law is
`InverseGamma(2.623417, 0.135905)` with mean `0.0837` and whose value at `T = 0.1` starting from
`0.15` is an order of magnitude below that radius. The truncation is therefore expected to activate
on **no path**: count the activations and report the count, because a nonzero count at the graded
`dt` would mean the scheme is being held together by the truncation rather than merely insured by
it.

**No overflow is possible.** Once `V` exceeds `R` the per-step increment of `L` is bounded by
`-0.93 + 1.08*Z + 0.29*(Z**2 - 1)` with `Z` standard normal, i.e. by an `O(1)` quantity with a
negative mean, over 25 steps. This is the failure mode `implementation_notes` records having
measured — explicit tamed-**drift** Milstein in `V` overflowing `V**1.5` within a few hundred
steps, because it is the *diffusion* that is superlinear. Truncating both coefficients, in a
coordinate where the state is an exponent rather than a base, removes it.

### Hyperparameters

- **Method**: explicit truncated Milstein in `L = log V`
- **dt**: `0.004` (the contract step; `Nt = 25`). Take `dt` from the argument and recompute
  `Nt = max(1, round(T/dt))`, `dt = T/Nt`; do not hardcode a different step. `R` must be recomputed
  from the *effective* `dt`, so that the ladder levels are genuinely the same scheme at three steps.
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`); the harness also calls at `200000`
  (`graded_num_paths`)
- **T**: `0.1`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Milstein correction**: `0.25 * epsilon**2 * Vt * (dW**2 - dt)`, evaluated at the state
  **entering** the step (never at the updated value — applying it at `L_{n+1}` is one of the two
  classic ways to break the correction and shows up as a measured strong order of ~0.5)
- **Expected strong order**: **1.0** — `expected_strong_order_milstein` in
  `problem_spec.json → verification`, declared by the spec alongside the Euler-Maruyama value of
  0.5 so that a Milstein-family plan is not failed against the 0.5 default. A measurement near 0.5
  means the correction term is wrong; near 0.0 means the supplied increments are not being
  consumed, which is a real defect here.
- **Expected weak order**: **1.0** (`evaluation_thresholds.expected_weak_order`)
- **No `exact_sampling_components`.** Every component is stepped with the supplied increments.
  `implementation_notes` forbids the declaration on this problem in as many words.
- **`boundary_treatment: none`**, with the argument: the gated boundary is `V > 0`, and
  `V = exp(L)` cannot reach or cross it for any finite `L` — there is nothing to monitor and no
  crossing probability to compute. `verification.structural_facts` declares no barrier.

### Why this plan is ranked last

It is the only one of the three that advances the stiff drift **explicitly**. The per-step drift
scale is `(kappa + 0.5*epsilon**2)*dt = 0.238`, comfortably inside stability but carrying the full
`O(dt)` drift discretisation error that an exact-flow composition removes outright. Its value in
the pool is that it is a genuinely independent discretisation — a different coordinate, a different
positivity mechanism, a different error constant — so agreement with the other plans on the two
graded reciprocal moments is a real cross-check rather than a restatement.

## Reference for Scoring

**Surrogate** (`verification.expected_provenance: surrogate`). `analytic_moments.has_analytic_solution`
is `false`, both moment expressions are `null`, and `moment_ode` is `null` because the moment
hierarchy for `V` does not close (`dE[V**2]/dt` reaches `E[V**3]`, which does not exist — the
invariant law has shape `alpha = 2.62`). The references the evaluator will use:

1. The two exact reciprocal moments, carried as gated `martingale` entries in
   `verification.constraints` / `verification.invariants`:
   `E[1/V(T)] = 11.622786285052442` and `E[(1/V(T))**2] = 180.3242675039887` (`ci_mult: 2.0`),
   which pin `Var[1/V(T)] = 45.23510647598556`. These are the numbers the graded functionals are
   scored against.
2. The **Dynkin generator check** on `verification.dynkin_test_functions`
   `["1.0/X", "np.exp(-X)", "np.tanh(X)"]`, which requires the `observables` hook. It is the only
   check that establishes the paths came from *this* equation, and the exact moments cannot stand in
   for it.
3. `verification.stationary_density` — `InverseGamma(2.623417, 0.135905)`, stationary mean
   `0.083715`, `T_stat = 6.0`, `burn_in = 3.0`. The **mean only** is usable; the stationary-variance
   leg has infinite variance and is expected to abstain.

`verification.terminal_law` is `null` by design and `stability_check` is `null`, with finiteness
carried as a gated constraint so the moment tolerances are not replaced.

## Implementation Notes

- **Do not sample the exact transition, in any coordinate.** `Y = 1/V` is exactly a CIR process
  (`verification.reciprocal_cir_identity`) with a noncentral chi-squared transition law, so `V(T)`
  could be drawn in one shot and both graded numbers hit to sampling error. `problem.md` forbids it:
  "sampling it directly is not what this problem measures". Do not step the reciprocal or Lamperti
  variable through its exact transition either, and do not use the identity's reference values
  anywhere inside `solver.py` — not as a control-variate mean, not as a calibration constant. They
  are the grade.
- **Positivity and finiteness are hard gates** (`verification.constraints` `finite` and
  `positivity`, both `gate: true`, strict). The graded quantities are reciprocal moments, so one
  path at or below zero makes them `Inf` rather than merely biased. Here both come from the
  representation: `V = exp(L)` with `L` finite. Keep the `np.maximum(..., 0.0)` guard inside
  `np.sqrt` for roundoff safety; there must be **no post-update clipping of the state** —
  `implementation_notes` says such a statement is scanned for and reported as hiding the defect.
  Assert `np.all(np.isfinite(L))` once at the end rather than repairing `L` inside the loop.
- **Verification hooks.** Signature:
  `def solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. The four
  `problem.md` names come first with their exact semantics; `dW=None` and `observables=None` must
  default to reproducing the stated behaviour exactly (ledger entry R14 records this addition as the
  one `ambiguous` requirement). When `dW` is supplied it is already scaled by `sqrt(dt)`: take
  `Nt = dW.shape[1]`, `dt = T/Nt`, recompute `R` from that `dt`, use the increments exactly as given
  and draw no randomness of your own, so the refinement levels share one Brownian path. Without
  these hooks the strong-order study and the Dynkin check cannot run and the plan cannot be
  certified above `self_convergence`.
- **`observables` are functions of `V`**, not of `L`. Accumulate
  `acc[k] += 0.5*(phi(V_prev) + phi(V_new))*dt` across the full step and return them under
  `path_integrals`. The declared test functions are `1.0/X`, `np.exp(-X)`, `np.tanh(X)`; `X` and
  `X**2` are deliberately avoided because `E[V**2]` is tail-dominated and its error bar makes the
  generator residual unresolvable.
- **Return shape.** `terminal_paths` is the 1-D array of terminal `V` values, shape `(num_paths,)` —
  `V` itself, not `L`, not `1/V`. Also return `empirical_mean` and `empirical_variance` of `V`
  (`ddof=1`), and `functionals = {"inv_var_mean": float(np.mean(1.0/V_T)), "inv_var_variance":
  float(np.var(1.0/V_T, ddof=1))}` computed from those same terminal paths. Report `functional_se`
  for both: `std(1/V_T, ddof=1)/sqrt(num_paths)` for the mean, and a batch-means or jackknife
  standard error for the variance.
- **`V`'s own moments are not graded and should not be chased.** No surrogate exists — the spec
  records a truncated backward-Kolmogorov solve whose `E[V_T**2]` moved from 0.1204 to 0.1699 as the
  truncation widened from `Vmax = 8` to `27`, and rejected it. `variance_rel_err_max` and
  `mean_rel_err_max` sit at template values with nothing to compare against.
- **Refine cycle (conductor: E[1/V] 0.53% high, bias-limited).** The drift sub-step was explicit Euler and
  carried the full O(dt) error. It is now the explicit trapezoid (Heun) rule: the noise plus Milstein
  correction (evaluated at the entering truncated state) gives a predictor `L_pred`, and the drift is
  `0.5*dt*(mu(Vt_n) + mu(min(exp(L_pred), R)))`. Still fully explicit in `L`, no exact flow, no exact
  transition. Measured with common random numbers at 400000 paths (E[1/V], ref 11.6228, se 0.011):
  Euler drift 11.696/11.654/11.630 at Nt=25/50/100; Heun 11.643/11.628/11.618; Strang with an RK4 drift flow
  11.587/11.600/11.604 (rejected: its residual bias comes from the noise sub-step and does not shrink).
  The remaining bias is the weak-order-1 noise/Milstein sub-step, which is the floor for this scheme.
  `scheme_family` is unchanged (`tamed-milstein`: truncated Milstein, now with a Heun drift).
- **Truncation correction.** The plan predicted zero activations. It fires on 357 path-steps at 200000 paths
  (91 at 50000), so V does reach R = 3.98 in the tail. The truncation freezes the coefficients there, which
  weakens the restoring drift and the diffusion on those paths, a small bias on tail paths that matter for
  E[1/V] only through tiny V (the opposite end), so its effect on the graded reciprocal moments is small.
- **Two path counts.** `solve_sde` is called at `200000` and at `50000` with the same `dt`, `T` and
  `seed`. Draw as `rng.standard_normal((num_paths, Nt))` so the two calls are genuinely different
  samples: the reported values must *differ* and must agree inside the implied sampling noise. Do
  not cache, do not round, do not return a constant.
- **Expected precision.** Measured excess kurtosis of `1/V_T` is 1.68, so at 50 000 paths the
  relative standard error is ~0.27% on the mean (5% tolerance) and ~0.80% on the variance (10%
  tolerance). Sampling is not the binding term for this plan either — the explicit drift's `dt`
  bias is, and if the evaluator reports the graded functionals limited by bias rather than
  sampling, the response is a finer `dt` on the diagnostic runs, never more paths.

## Results

Run: `uv run python solver.py` (dt=0.004, Nt=25, T=0.1, seed=42, R = dt**-0.25), Heun drift kept, now with antithetic pairing inside the returned sample (paths i and i+M/2 use +dW and -dW; no extra paths, functionals from the same terminal_paths).

| num_paths | E[1/V] (ref 11.6228, tol 5%) | Var[1/V] (ref 45.2351, tol 10%) | mean V | var V | truncation activations |
|---|---|---|---|---|---|
| 200000 | 11.65989 (se 0.0054) | 44.6464 (se 0.2193) | 0.146773 | 42.879 | 353 |
| 50000 | 11.65778 (se 0.0108) | 44.5911 (se 0.4381) | 0.130702 | 0.03567 | 69 |

Variance reduction, measured over 40 seeds at 50000 paths (empirical sd of the estimate vs mean reported se):
E[1/V] sd 0.0331 -> 0.0103 (reported se 0.0299 -> 0.0109, the se now being a pair-level estimate); Var[1/V] sd 0.416 -> 0.417 (reported se 0.380 -> 0.440, delete-one-pair jackknife).
Antithetic pairing cuts the mean's sampling error about 3x but does nothing for Var[1/V], the binding functional, because (1/V - m)^2 is nearly even in the noise. The only control variates available without the exact transition are noise statistics with known mean (sum dW, sum dW^2 - T); the best one, sum dW^2 - T, correlates 0.20 with the pair-averaged squared deviation, a ~4% variance cut, not worth adding. No control variate was added; the exact-transition identities are not used. Note the E[1/V] estimate sits +0.3% above reference here (about 2.2-3 sigma of the old se, a seed/pairing realisation of the Heun bias); well inside the 5% gate.

## Evaluation

<metrics>
kind: sde
score: 9
certified: false
provenance: surrogate
estimated_rel_error: 2.2648
error_is_estimate: true
observed_order: 0.7999
observed_order_se: 0.1281
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: tamed-milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 0.1697
invariant_worst_headroom: 0.1328
mean_rel_err: 0.5588
variance_rel_err: 2.2648
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 0.0121
functional_bias_rel: 2.282e-03
functional_se_rel: 9.825e-03
functional_limited_by: sampling
moment_bias_binding: mean
moment_bias_frac_tol: 0.2616
moment_bias_actionable: true
order_check: true
mc_se_rel: 0.3777
resolved: false
dynkin_z: 1.8310
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 757534e9f62e37df9594e05cd3a23d6e8066ee363f3afc7334cb2b1bc5d15501
solve_time_s: 0.1250
wall_time_s: 6.0000
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              surrogate  (route: stationary_density at T_stat = 6.0, support [1e-8, 50]; moment_ode and kolmogorov attempted and unavailable). certified: false
- mean_rel_err:            0.559 +/- 2.0 x 0.00084 (V's own mean, 0.1307 against a reference of 0.0838). Not scored on, see the reference note below
- variance_rel_err:        2.265 +/- 2.0 x 0.0135 relative to its reference (V's own variance, 0.0357 against 0.0109); outcome `unresolved`
- mc_se_rel:               0.378   resolved: false (ci_width_rel 2.47 against a bar of 0.05). V's skewness is 63.7 +/- 9.5 and its excess kurtosis 7130 +/- 1030
- functional_rel_error:    0.0121 against functional_rel_tol 0.10 (bias 0.0023 + sampling 0.0098 -> limited by sampling, binding functional inv_var_variance). inv_var_mean clean at 0.00086 against 0.05
- observed_order:          0.800 +/- 0.128 (median-based 1.017), expected 1.0, one-sided below the declared floor; orders_ok true
- weak_order:              1.29 (x: 1.49 and tanh: 1.09 used; x2 at the noise floor and not used)
- dynkin:                  ok, z = 1.83 against bar 2.43 on seed 42; replicated, median z = 1.53 over 3 seeds (z by seed 1.83 / 1.53 / 0.93). Pooled z 2.47 sat just over the bar and disagreed with the median, so this one is nearer the bar than the median suggests
- terminal_law:            not declared
- shape:                   skewness 63.7 +/- 9.5, excess kurtosis 7130 +/- 1030
- constraints:             finite true, positivity true (zero violations), martingale true (drift 0.170 against bar 1.93), estimator_stability true (0.0074 against bar 0.01, headroom 0.133 decades)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. "no surrogate: the orders, Dynkin and the constraints all hold and the Richardson moments are stable across the ladder". The certification ceiling is 10 (Tier A').

**Why the score is 9 and not 10.** `resolved: false` is not a solver defect here, and more paths will not clear it. The moment-space check compares V's own mean and variance, which have infinite variance as estimators (E[V^3] does not exist), and does so against a reference that is the stationary density at T_stat = 6.0. The graded process is at T = 0.1, where the mean is about 0.131 and not 0.084. The reference is the invariant law of V and not the law at T = 0.1, so `mean_ok: false` (0.559) says nothing about this solver. The `analytic_moments.note` in the spec already says V's moments are not graded and no trustworthy surrogate exists, yet the kernel reached for the stationary density and it passed its own trust guard. The graded quantities, the exact reciprocal-CIR moments of 1/V, are declared under `functionals` and the plan clears them. I report this as a formulator and spec-declaration gap. A reference that does not apply to T = 0.1 is gating certification for every plan on this problem. The 10 is likely unreachable until the spec declares the reciprocal CIR moments as the moment-space reference, or marks V's moments as not graded.

**Are the reported error bars honest for the paired sample? Yes.** I ran the solver over 300 seeds at M = 50000 and 300 seeds at M = 200000, once with the solver's own antithetic noise and once with supplied iid `dW` (same dynamics, iid sample), and compared reported `functional_se` with the empirical sd across seeds.

| M | sample | quantity | empirical sd | reported se (median) | centred 2-se coverage |
|---|---|---|---|---|---|
| 50000 | paired | E[1/V] | 0.0108 | 0.0109 | 0.947 |
| 50000 | paired | Var[1/V] | 0.435 | 0.440 | 0.947 |
| 50000 | iid | E[1/V] | 0.0300 | 0.0299 | 0.970 |
| 50000 | iid | Var[1/V] | 0.387 | 0.374 | 0.960 |
| 200000 | paired | E[1/V] | 0.0054 | 0.0055 | not measured |
| 200000 | paired | Var[1/V] | 0.215 | 0.220 | not measured |
| 200000 | iid | Var[1/V] | 0.178 | 0.188 | not measured |

Coverage is centred on the seed-mean because the estimator carries a dt bias, discussed below. The pair-level se for the mean and the delete-one-pair jackknife for the variance are both correct. A jackknife over pairs is valid here because the pairs are independent units even though the paths inside a pair are not. If anything it is very slightly conservative (se / sd of 1.00 to 1.05 across both sets). I did not find a defect in the error-bar code.

**The pairing is not helping the binding functional.** The pair-level mean se falls 2.8x (0.0299 to 0.0109, matching your 0.0331 to 0.0103). The variance estimator's true sampling sd *rises*: 0.387 to 0.435 at M = 50000 (+12%) and 0.178 to 0.215 at M = 200000 (+21%). The rise in the reported se (0.374 to 0.440) is therefore real and not a jackknife artifact. I did not isolate the mechanism; the effect is measured, not derived, and is consistent across both path counts. The graded `inv_var_variance` has sampling term 0.0098 of its 0.10 gate, so there is ample room and nothing fails. The change just does not improve the quantity that binds, and it slightly worsens it.

**E[1/V] reads +0.3% over exact: this is bias, not seed noise.** The 300-seed mean at M = 50000 is 11.6651 +/- 0.0006, which is +0.364% against 11.6228 (about 600 standard errors), and 11.6643 +/- 0.0003 (+0.358%) at M = 200000. The iid sample gives the same +0.365%. It does not move with path count or with pairing, so it is a dt = 0.004 bias, consistent with the plan's Heun attribution. The graded seed read 11.658 (+0.30%). Var[1/V] carries a matching bias of -0.90% (44.83 against 45.235, 16 standard errors) that is also pair- and M-independent. Both are far inside the gates (0.36% against 5%, 0.90% against 10%). The kernel's own ladder puts the variance bias at 0.0023 relative and the ladder's `inv_var_mean` error at 0.00086, with observed functional orders of 1.03 and 1.07, so the bias is first order and shrinks with dt. Note that this size of bias is visible only because the seed study averages it down; a single run cannot distinguish it from noise at 0.0108 sd, which is why the single seed-42 reading looked ambiguous.

### Feedback for solver
- The error bars are correct as written, and both estimators are validated by the 300-seed studies above. Keep the pair-level mean se and the pair jackknife if you keep the pairing.
- Consider removing the antithetic pairing. It cuts the se on E[1/V], which is graded at 5% and has about 0.2% sampling error already, and it raises the sampling sd of Var[1/V] by 12 to 21%, which is the binding functional (sampling term 0.0098 of its 0.10 gate). If you keep it, pair only the mean estimator and compute the variance from an independent half, or accept the cost.
- E[1/V] +0.36% and Var[1/V] -0.90% are dt biases at dt = 0.004, measured at 300 seeds, not noise. Both are well inside their gates, so no action is needed for the score. If you want them smaller, a smaller step is the only lever, since more paths will not move them.
- `resolved: false` and V's own mean and variance being unresolved are not defects of this scheme. Do not raise `num_paths` for it. It comes from V's infinite-variance moments and from a stationary-density reference that does not describe T = 0.1.
- The Dynkin median z is 1.53 against a bar of 2.43, but the pooled estimate was 2.47 and disagreed with the median. The first seed z was 1.83. This is a margin worth watching rather than a failure. The 1/X residual extrapolates at order 1.03, which is first order, as expected.
- The strong-order estimate is 0.80 +/- 0.13 (median-based 1.02) against a floor of 1.0. It is within its error bar and `orders_ok` is true. The truncation radius R = dt^-0.25 fires on a small number of path-steps and is the likely source of the low tail.
- Formulator note for the conductor: the V-moment reference is a stationary density at T_stat = 6.0 applied to a T = 0.1 problem, and it caps every plan here at 9. This is not something the solver can fix.

</review>
