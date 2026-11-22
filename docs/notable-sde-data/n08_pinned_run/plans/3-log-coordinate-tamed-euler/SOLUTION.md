---
id: 3
plan_slug: log-coordinate-tamed-euler
scheme: tamed-milstein
scheme_family: tamed-milstein
strategy: Step U = log V with a tamed-drift Euler-Maruyama scheme — the simplest positivity-preserving discretisation of this equation, carried as the order-0.5 baseline the higher-order plans are measured against.
---

## Why this scheme

The plan steps `U = log V` with a log-chart Milstein correction and second-order taming of both the
drift and the noise. It began as the uncorrected Euler baseline of the pool and was revised twice for
measured bias (see the revision sections). The body below describes the code in `solver.py`.

Positivity is structural, not clipped: `V = exp(U) in (0, inf)` identically. A floored `V`-chart
scheme would be the wrong choice, because `1/eps` at a small floor dominates `E[1/V]`.

No transition law is sampled (`problem.md` R25). The step consumes one supplied Brownian increment
per step at the handed-in `dt`, and `exact_sampling_components` is not declared.

## SDE Reference

```
dV(t) = kappa * V(t) * (theta - V(t)) dt + epsilon * V(t)**1.5 dW(t),    V(0) = 0.15

X_0 = V(0) = 0.15
kappa   = 22.84
theta   = 0.218
epsilon = 8.56
T       = 0.1        (interval [0, 0.1])
```

Drift `f(V) = kappa * V * (theta - V)`; diffusion `g(V) = epsilon * V**1.5`;
`dg/dV = 1.5 * epsilon * np.sqrt(V)`.

## The scheme, written out

With `U = log V`, Itô gives `dU = a(V) dt + b(V) dW`, with

```
a(V) = kappa * (theta - V) - 0.5 * epsilon**2 * V
b(V) = epsilon * np.sqrt(V)
```

Per-step update, `V_n = np.exp(U_n)` (pre-step value used throughout):

```
a_tamed = a_n / np.sqrt(1 + (dt * a_n)**2)             # drift taming, error O(dt^2)
s       = dt * b_n**2
damp    = 1 / (1 + s**2)                               # noise taming, 1 + O(dt^2)
U_{n+1} = U_n + a_tamed*dt + b_n*np.sqrt(damp)*dW_n
              + 0.25*epsilon**2*V_n*(dW_n**2 - dt)*damp   # log-chart Milstein term
```

Taming is needed because the log drift carries `-0.5*eps**2*V`, an explicit step of `0.1465*V` in
`U` for a large excursion. Both tamings bound the increment while agreeing with the untamed terms to
`O(dt**2)`, so they do not cost order.

## Numerical Scheme

- **Method**: Milstein in the `U = log V` chart with second-order drift and noise taming
  (`scheme_family: tamed-milstein`)
- **dt**: `0.004` — use the value handed in; `Nt = max(1, round(T/dt))` = 25, then `dt = T/Nt`
- **num_paths**: `50000` (`evaluation_thresholds.num_paths`); the problem's graded call is
  `200000` (`graded_num_paths`) and the same code must serve both
- **T**: `0.1`
- **seed**: `42`, via `np.random.default_rng(seed)` only
- **Milstein correction**: applied, `0.25 * epsilon**2 * V_n * (dW**2 - dt)` (log chart, pre-step `V`)
- **Expected strong order**: 1.0 (measured 1.27 +/- 0.16)
- **Expected weak order**: 1.0
- **Reference for scoring**: **none in moment space** — `analytic_moments.has_analytic_solution`
  is `false` and `moment_ode`, `kolmogorov` and `terminal_law` are all explicitly declined by the
  formulator with structural reasons. The reference is
  `verification.exact_functional_reference`: `inv_var_mean = 11.6227862851`,
  `inv_var_variance = 45.2351064760`, `inv_var_second_moment = 180.3242675040`, carried as two
  **gated** `verification.invariants` checks on `1.0/X` and `1.0/X**2` evaluated on the returned
  `terminal_paths`, plus the two gated `functionals` tolerances (5% / 10%).
  `verification.expected_provenance` is `self_convergence`.

## Implementation Notes

- **Solver signature.** `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None)`. The
  first four are mandated by `problem.md`, which calls the signature exact; the two extras default
  to `None` so every mandated call is valid and unchanged (ledger entry **R29**, status
  `ambiguous`). Without `dW` this plan's entire purpose — supplying the measured 0.5 against which
  a 1.0 is read — cannot be served.
- **Honour `dW` exactly** when supplied: `Nt = dW.shape[1]`, `dt = T/Nt`, no randomness of your own.
  Otherwise draw one path-major block `np.sqrt(dt) * rng.standard_normal((num_paths, Nt))`.
- **`observables`**: accumulate `∫ phi(V_s) ds` per path by trapezoid, returned as
  `path_integrals`. The spec's Dynkin functions are `1.0/X`, `1.0/X**2`, `np.tanh(X)` — `X` and
  `X**2` are excluded because `V` has tail index 2.623, so `E[V**3]` and `E[V**4]` are infinite and
  their estimators would be unresolvable for reasons that say nothing about the solver.
- **Return** `terminal_paths` = terminal values of **`V` itself**, shape `(num_paths,)` (R12);
  `empirical_mean`; `empirical_variance` (`ddof=1`); and
  `functionals = {"inv_var_mean": float((1.0/Vt).mean()), "inv_var_variance": float((1.0/Vt).var(ddof=1))}`,
  plus `functional_se` for both.
- **Do not argue accuracy from `empirical_mean` / `empirical_variance` of `V`** — the sample
  variance of `V` has infinite variance at this tail index (`analytic_moments.note`). The
  reciprocal functionals are the grade.
- **Overflow guard, not a scheme element.** Clip `U` above at `50.0` (`V <= 5.2e21`, contributing
  `~2e-22` to `E[1/V]`) so the gated `finite` constraint cannot be lost to an `np.exp` overflow on
  a rare path, and report how many paths the clip touched. The large-`V` tail is real and
  irrelevant to the graded functionals, since `1/V -> 0` there; the small-`V` region is what must
  be right.
- **No `boundary_treatment` is declared**: the spec declares no `barrier` and no
  `absorbing_boundary`. `V = 0` is inaccessible, not absorbing — the Feller index of `Y = 1/V` is
  5.2468 > 2 — and the log chart reproduces that exactly rather than approximating a crossing.
  Positivity is a gated `constraints` entry and holds structurally.
- **Two path counts (R28).** The same code must run at 200 000 and 50 000, the two reported values
  must differ and must agree within the implied sampling noise. Nothing cached or keyed on
  `num_paths`.

## Results

Run: `uv run python solver.py` (dt=0.004, T=0.1, seed=42, Nt=25).

| num_paths | E[V] (empirical_mean) | Var[V] (empirical_variance) | inv_var_mean (exact 11.6228) | inv_var_variance (exact 45.2351) |
|---|---|---|---|---|
| 200000 | 0.134545 | 0.069593 | 11.7041 (se 0.0039) | 45.826 (se 0.158) |
| 50000 | 0.133911 | 0.031879 | 11.6937 (se 0.0077) | 45.494 (se 0.317) |

No clipped paths. `dW` and `observables` hooks work. Var[V] is heavy-tailed (tail index 2.6) and not graded.

## Implementation Notes (revision 1, noise sub-step)

`scheme_family` was `tamed-euler-maruyama`; it became `tamed-milstein`. The noise taming `b/(1+dt*b**2)` was only 1+O(dt) accurate
and shrank the diffusion by about 4% at the graded step. It is now `b/sqrt(1+(dt*b**2)**2)` and the log-chart Milstein term is added.

## Implementation Notes (revision 2, drift sub-step, measured mean bias 1.21 of tolerance)

The kernel's binding bias was the mean of V (+6% against the extrapolated limit). E[V] is dominated by large-V excursions, where the
drift `a ~ -0.5*eps**2*V` is large and the first-order taming `a/(1+dt|a|)` shrinks it by `dt|a|` (about 15% at V=0.5). That sub-step
carried the bias. It is now `a/sqrt(1+(dt*a)**2)`, error O(dt**2). The `scheme_family` stays `tamed-milstein`. Same-noise ladder
(400k paths, Nt=25..400), E[V]: before 0.1407, 0.1369, 0.1348, 0.1338, 0.1334; after 0.1346, 0.1337, 0.1332, 0.1331, 0.1330. The bias at
Nt=25 falls from +0.0077 to +0.0016. Cost: E[1/V] at Nt=25 moves from +0.2% to +0.6% above the limit (inside the 5% gate), while the
variance functional bias falls from +2.1% to +1.1%. Not tuned toward sibling plans; other power-law tamings (`p=3`, `1/(1+(dt a)**2)`)
were tried and do not do better.

## Implementation Notes (revision 3, bias and variance)

Bias: a weak-order-2 drift term `0.5*L a*dt**2` (generator applied to the log drift) was tried on the drift sub-step. It made E[V] blow
up on rare large-V paths and raised the gated reciprocal moments at every dt (E[1/V] +0.7% -> +0.7..0.8%, variance functional +1.1% -> +1.5%),
so it was rejected. The drift sub-step stays at second-order taming; the residual graded mean bias (+1.2% vs the limit, 0.31 of tolerance) is
the weak-order-1 error of the exp-of-Euler chart, and scheme_family stays `tamed-milstein`.
Variance: the graded functionals are now an antithetic pair estimator (supplied `dW` and `-dW`, same scheme); `functional_se` is the
standard error of the pair-averaged estimator (inv_var_mean se falls ~4x, variance functional ~20%). `terminal_paths` and `path_integrals`
are those of the supplied increments only. No internal substeps, no change in num_paths.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: manufactured_partial
estimated_rel_error: 1.658e-03
error_is_estimate: true
observed_order_se: 0.2008
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: tamed-milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.1441
invariant_worst_headroom: 0.4074
d1_outcome: clean
operator_validated: true
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 0.0101
functional_bias_rel: 3.179e-03
functional_se_rel: 6.968e-03
functional_limited_by: both
moment_bias_binding: mean
moment_bias_frac_tol: 0.3146
moment_bias_actionable: true
order_check: unresolved
mc_se_rel: 0.2325
resolved: unavailable
dynkin_z: 1.9413
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 7b3ad0f5fcb1bbff4544a55cbac2e4244d312fa36e8a063020cb466b0521b9f2
solve_time_s: 0.1560
wall_time_s: 6.8400
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              manufactured_partial  (reference route: none. moment_ode "no moment ODE, or it does not close exactly", kolmogorov "no kolmogorov.bounds declared", stationary_density "no normalizable support declared"). One circularity break (D1 against a validated operator), no Tier-B; certified: false
- mean_rel_err:            unavailable (moments_ok "no reference to compare against"); sample mean of V 0.13391 +/- 2 x 0.00080 at 50000 paths
- variance_rel_err:        unavailable; sample variance of V 0.03188 +/- 2 x 0.00741 (V has tail index 2.62, not a graded quantity)
- mc_se_rel:               0.2325   resolved: unavailable ("no reference variance to resolve against"). Not an MC-inconclusive verdict in the "raise num_paths" sense; there is no reference to resolve against
- functional_rel_error:    0.0101 against functional_rel_tol 0.10  (bias 3.18e-03 + sampling 6.97e-03 -> limited by both, binding functional inv_var_variance). Both terms are well inside the gate. Graded-dt values: inv_var_mean 11.6937, inv_var_variance 45.4939 (exact 11.6228 / 45.2351); ladder-extrapolated 11.6364 / 45.3863
- observed_order:          1.13 +/- 0.20 (2-sigma interval 0.40 wide; expected 1.0, one-sided below the declared floor). Median-based order 0.972. orders_ok "unresolved": the estimator abstained because its own interval is wider than the 0.4 band. A property of the estimator, not of the solver; costs the plan nothing
- weak_order:              1.62 (x only; x2 and tanh at the noise floor or not shrinking). Not scored on
- dynkin:                  ok, replicated (first seed landed 0.49 from the bar); median z = 1.72 over 3 seeds (1.94, 1.72, 0.95), bar 2.43; pooled z 2.66 does not agree with the median rule, so a marginal pass. Extrapolation orders 0.88 / 0.94 / 1.47
- terminal_law:            not declared
- shape:                   skewness 31.2 +/- 13.6, excess kurtosis 2701 +/- 1372 (heavy tail, as the spec warns)
- constraints:             finite pass; positivity pass; estimator_stability pass (0.0058 vs bar 0.02); martingale (gate, E[1/X] and E[1/X^2] against the exact CIR values) pass, max drift 1.144 against bar 2.92
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. No surrogate: Dynkin and the constraints hold and Richardson moments are stable; the order study abstained and is not counted against the plan. Certification also caps at 9 (one circularity break, no Tier-B)

### Feedback for solver
- No defect to fix. The 9 rather than 10 is structural: with no surrogate route a 10 needs two circularity breaks and this path supplies one. The solver cannot close it.
- Source and claim agree. The log-chart Milstein term (0.25 * eps**2 * V_prev * (dW**2 - dt) * damp) is present and uses the pre-step V; dW is consumed exactly (Nt = dW.shape[1]). No exact_sampling_components declaration is needed and none should be added. The U_MAX = 50 clip is an overflow guard on log V, not a state clip, and touched no paths.
- Strong order reads 1.13 +/- 0.20 (median 0.97) against design 1.0. The ladder abstained as unresolved; it is consistent with a working Milstein correction and is not a finding.
- Remaining bias: graded-step mean of V is 0.1349 against a Richardson limit of 0.1328 (+1.6% against a 5% tolerance, 2.7 standard errors), report-only since V's mean has no reference. The graded functionals carry bias 0.32% and sampling 0.70% against a 10% gate, so neither more paths nor a smaller dt is needed.
- Dynkin z of 1.94 on 1/X (bar 2.43, median 1.72) is a marginal pass. It cannot detect a drift mis-transcribed from problem.md, since the generator is built from the spec's own coefficients.
- SOLUTION.md hygiene (not scored): the "Why this scheme", "The scheme, written out" and "Results" sections still describe the uncorrected first-order-taming baseline; only the revision sections describe the code. Make the body match solver.py.

### Notes for the conductor and formulator (not for the solver)
- provenance_as_expected: false. The spec predicted self_convergence and the run reached manufactured_partial via D1. Not scored on; the better outcome.
- Possible formulator gap: the exact reciprocal-CIR moments (11.6228, 45.2351) are recoverable and carried only as gated `invariants` entries, with no declared surrogate route bearing them. moments_ok, variance_ok and resolved are therefore all unavailable and the path is capped at 9. The reasons for declining the V-space routes are sound; the gap is that no mechanism lets a reference be declared for a derived functional of the state.
- Cross-plan: the stored comparison against plan 1 is stale (solver sha changed since); it corroborates nothing. Re-run `compare`.

</review>
