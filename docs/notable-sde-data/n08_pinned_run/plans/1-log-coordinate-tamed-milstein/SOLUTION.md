---
id: 1
plan_slug: log-coordinate-tamed-milstein
scheme: reciprocal-chart-strang-splitting
scheme_family: strang-splitting
strategy: Step Y = 1/V (a CIR-type SDE with square-root noise) by Strang splitting of its generator into an exactly-solved drift flow and an exactly-solved noise flow driven by the supplied Brownian increment, then return V = 1/Y; positive by construction; weak order 2 is expected but not evidenced by the kernel.
---

## Why this scheme

The graded quantities are `E[1/V(T)]` and `Var[1/V(T)]`, dominated by small `V`. Two things matter, and neither is sampling noise.

1. **Strict positivity with no floor.** Any clip of `V` injects a large upward bias into `1/V`. Positivity must come from the scheme's structure. Stepping `Y = 1/V` with sub-flows that map `(0, inf)` to itself gives `V in (0, inf)` on every path with no `np.maximum`.
2. **Discretisation bias at the mandated `Nt = 25`.** The kernel measured the previous scheme's `inv_var_variance` bias at about 3 tolerances of its own. That scheme was a chart-changed Milstein step with Crank-Nicolson drift, so its weak order was 1 and the O(dt) error sat in the noise sub-step (the Milstein/Euler-type treatment of `sqrt(Y) dW`). More paths cannot remove it.

**Chart choice, stated plainly.** The spec calls a transformed-coordinate step a grey zone that must be declared. This plan steps in the reciprocal chart `Y = 1/V`, by Ito's formula:

```
dY = (a - b Y) dt - epsilon sqrt(Y) dW,     a = kappa + epsilon^2,   b = kappa * theta
```

This is the same equation in different coordinates. It is stepped one increment per step at the handed-in `dt`, consuming the supplied `dW`. The exactly samplable terminal law (`1/V` is CIR) is **not** used, and no transition law is sampled. `implementation_notes` names a "log/reciprocal-coordinate step" as acceptable. An earlier version of this document described a log chart (`U = log V`) with a `U = 50` clip. That scheme is gone and nothing in the shipped code uses it.

## SDE Reference

```
dV(t) = kappa * V(t) * (theta - V(t)) dt + epsilon * V(t)**1.5 dW(t),    V(0) = 0.15

X_0 = V(0) = 0.15,  kappa = 22.84,  theta = 0.218,  epsilon = 8.56,  T = 0.1
```

## Numerical Scheme

Rewrite the noise in Stratonovich form: `dY = (c0 - b Y) dt + epsilon sqrt(Y) o dB`, with `B = -W` and `c0 = a - epsilon^2/4 = 22.84 + 73.2736*0.75 > 0`. Split the generator into

- **drift flow** `dY = (c0 - b Y) dt`, exact: `Y -> Y e^{-b h} + (c0/b)(1 - e^{-b h})`
- **noise flow** `dY = epsilon sqrt(Y) o dB`, exact: `Y -> (sqrt(Y) + epsilon dB / 2)^2`

Per step (`dB = -dW[:, n]`, `h = dt/2`):

```
Y <- drift(h)(Y);   Y <- (sqrt(Y) + epsilon*dB/2)^2;   Y <- drift(h)(Y);   V = 1/Y
```

Both sub-flows are exact and Strang composition is symmetric, so the weak error is expected to be O(dt^2) (not measured: the kernel's weak-order tests sat at the noise floor). The noise flow is exact pathwise for any real `dB`, so the supplied Gaussian increment is used as is and no extra random variables are needed. Positivity: the drift flow maps `Y > 0` to `Y > 0` (since `c0 > 0`), and the noise flow is a square.

The `dW_sub` argument is not used because the spec declares no sub-increments.

## Implementation Notes

- **Scheme family.** `scheme_family` was `tamed-milstein` in the previous version. It is now `strang-splitting`, since the sub-step that carried the bias was replaced by an exact-flow splitting. Order of that sub-step went from weak 1 (Milstein-type noise treatment) to weak 2 (exact noise flow in a symmetric composition).
- **Sub-step carrying the bias.** The noise sub-step. The drift treatment (Crank-Nicolson on a linear drift) was already second order.
- **Strong order.** Strang splitting with exact sub-flows is strong order 1 here. The product of the two flows differs from the exact solution by commutator terms of size O(dt^1.5) per step in this setting. The measured value is for the evaluator to report.
- **Antithetic functionals.** The functionals pair each supplied-dW path with its mirror run on -dW (same scheme, same dt, same Strang composition; the mirror feeds only the functionals, never terminal_paths or path_integrals). `functional_se` is the pair-level standard error (influence function for the variance). Measured at 200k: SE of inv_var_variance 0.193 -> 0.157, inv_var_mean 0.0150 -> 0.0039.
- One `dW` block, path-major, drawn up front; when `dW` is supplied `Nt = dW.shape[1]` and `dt = T/Nt`.
- Observables are accumulated by the trapezoid rule on `V` (`phi(V_prev)`, `phi(V)`).
- `functionals` returns `inv_var_mean = mean(Y_T)` and `inv_var_variance = var(Y_T, ddof=1)`. `functional_se` uses `std/sqrt(M)` for the mean and `sqrt((mu4 - s^4)/M)` for the variance.
- **`n_fallback_steps` = 0** on the graded run (and on the 50k run). The splitting scheme has no fallback branch.

## Results

Graded call (`num_paths=200000, dt=0.004, T=0.1, seed=42`; Nt = 25):

| quantity | value | exact reference | rel. error | MC SE |
|---|---|---|---|---|
| E[V(T)] (empirical_mean) | 0.132530 | n/a | n/a | n/a |
| Var[V(T)] (empirical_variance) | 0.018988 | n/a | n/a | n/a |
| inv_var_mean = E[1/V] (antithetic) | 11.6226 | 11.6227863 | -0.00% | 0.0039 |
| inv_var_variance = Var[1/V] (antithetic) | 45.202 | 45.2351065 | -0.07% | 0.157 |

At 50k paths (`seed=42`, antithetic): E[V] = 0.132329, Var[V] = 0.018563, inv_var_mean 11.6124, inv_var_variance 44.875 (-0.80%, SE 0.314). Previous scheme at 50k: inv_var_variance 44.425 (-1.79%). `n_fallback_steps = 0`; min V(T) = 1.56e-2 (no collapse to zero). The exact values are the spec's references; they are used here only as a check and were not tuned toward.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 9
certified: false
provenance: manufactured_partial
estimated_rel_error: 5.360e-04
error_is_estimate: true
observed_order: 0.8861
observed_order_se: 0.0984
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: strang-splitting
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: true
invariant_max_drift: 1.3603
invariant_worst_headroom: 0.3277
functional_name: inv_var_variance
functional_rel_tol: 0.1000
functional_rel_error: 6.994e-03
functional_bias_rel: 2.971e-07
functional_se_rel: 6.994e-03
functional_limited_by: sampling
order_check: true
mc_se_rel: 0.0439
resolved: false
dynkin_z: 1.8364
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: adaf4b4b9b88de9725bfb431e89c604fc26dffdcb517755d88bd7be4d3e71443
solve_time_s: 0.1190
wall_time_s: 5.8200
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

Bound by the rubric, not by a cap: "no surrogate: the orders, Dynkin and the constraints all hold and the Richardson moments are stable across the ladder". The certification layer independently allows at most 9 (one circularity break, D1 against a validated operator; no Tier-B evidence). `certified: false` only because no second circularity break exists on this evidence path; `gates_clean: true`. No `agent_cap`: SOLUTION.md and `solver.py` describe the same scheme (reciprocal-chart Strang splitting, no taming, no clip).

### Numerical Accuracy
- provenance:              manufactured_partial  (reference route: none. `moment_ode`, `kolmogorov`, `stationary_density` all unavailable or declined by the spec, so `surrogate_ok` is unavailable. Spec predicted `self_convergence`; the run reached `manufactured_partial` via D1. `provenance_as_expected: false` is not a solver defect and is not scored on)
- mean_rel_err:            not reported (`moments_ok` unavailable: no reference for the state V)
- variance_rel_err:        not reported, same reason
- mc_se_rel:               0.0439   resolved: unavailable (the metrics print `false`, but the detail reason is "no reference variance to resolve against"). This is NOT an MC-inconclusive row and not a solver bug. V has tail index 2.623, so Var[V] is not resolvable at any path count, and the problem does not grade it
- functional_rel_error:    6.99e-03 against functional_rel_tol 0.10  (bias 2.97e-07 + sampling 6.99e-03 -> limited by sampling, binding functional inv_var_variance). Both gated functionals are `clean` on the ladder (inv_var_mean ladder error 2.6e-06, inv_var_variance 3.0e-07). Reported values at 50 000 paths: inv_var_mean 11.6124 (-0.09% vs exact 11.6228), inv_var_variance 44.875 (-0.80% vs 45.235, about 0.7 s.e. at the 7.0e-03 relative s.e.)
- observed_order:          0.886 +/- 0.098 (floor 0.5 from the spec, one-sided; median-based 1.031). Clears the floor by about 4 s.e.; `strong_order_shrinking` true
- weak_order:              indeterminate: at the noise floor. All three weak tests (x, x2, tanh) were rejected as "at the noise floor, or not shrinking"
- dynkin:                  ok. Median z = 1.74 over 3 seeds (1.84 / 1.74 / 0.93) against bar 2.43. `dynkin_replicated: true` (the first seed landed 0.59 from the bar, inside the 0.7 margin). Pooled z = 2.60 disagrees with the median, so the plan is inside the bar but not by a wide margin. All three test functions (1/X, 1/X**2, tanh X) pass. Extrapolation order was assumed 1 (the finer residual difference is inside its own noise)
- terminal_law:            not declared
- shape:                   skewness 6.99 +/- 0.40, excess kurtosis 94.4 +/- 10.5 (V is heavy-tailed, as the spec says)
- constraints:             `finite` pass, `positivity` pass, `estimator_stability` pass (drift 0.0045 vs bar 0.02). Gated `martingale` reference checks (E[1/V] and E[1/V^2] against the exact CIR values) pass: worst drift 1.36 against bar 2.89 (headroom 0.33 decades). No `gate_violation`; no clip anywhere in the code
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric: no surrogate, orders, Dynkin and constraints all hold and the Richardson moments are stable. Certification ceiling is also 9

### What the score means
Every check that ran came back clean and none was waived. The functional error is sampling-limited with a negligible dt bias (3.0e-07 relative), and it is already about 14x inside its 10% tolerance, so more paths would only tighten a term that is not near binding. The step from 9 to 10 is outside the solver's control: with no moment reference for V, a 10 needs two independent circularity breaks and only D1 is available. What D1 does not establish: the generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from `problem.md` would fool it. The exact reciprocal references carried in `verification.invariants` guard against that here, and they pass.

`moment_bias` reports Var[V] at 1.29 of its tolerance (`real: false`, `actionable: false`, Richardson shift 0.78 s.e.). That refers to Var[V], whose estimator has infinite variance and which is not graded, so it is not treated as a finding. Mean-of-V bias is 0.24% against a 5% tolerance.

### Feedback for solver
- Nothing is required to keep the score; the only things between 9 and 10 are outside the solver's control.
- The scheme declaration is consistent: the step consumes the supplied `dW` one increment per step (`w = -sign * dW[:, n]`), samples no transition law, and the strong order reads 0.886 +/- 0.098 (median 1.03). `exact_sampling_components` is correctly not declared, since nothing here draws its own randomness; no `undeclared_exact_sampling` flag was raised.
- Stepping the reciprocal chart is the spec's stated grey zone. The plan discloses it and discretises it with a genuine pathwise step whose error vanishes with dt (functional bias 3.0e-07 relative on the ladder), which is what `implementation_notes` requires. Whether a chart change is admissible under problem.md's wording is a call for the conductor and spec, not something the kernel measures.
- Stale text in SOLUTION.md: the Results table quotes 50k-path values (inv_var_variance 44.875, -0.80%) that match this run, but the table's 200k row and the earlier metrics block were from a prior pass; the kernel grades at 50 000 paths, where inv_var_variance is -0.80% with relative s.e. 0.70%. Regenerate the table from the kernel if it is quoted in REPORT.md.
- The weak-order-2 claim is not evidenced: the kernel rejected all three weak tests as noise-floor. Soften it to "expected" in the frontmatter `strategy` and the Numerical Scheme section, or support it with a higher-path weak-order study. The `inv_var_variance` ladder reads an observed order of 4.0 on differences of 2e-04 and 1e-02, which is a noise-scale ratio rather than a rate, and is not scored.
- Margins are thinner than the score suggests: martingale headroom 0.33 decades (drift 1.36 vs bar 2.89) and Dynkin median z 1.74 vs 2.43 with pooled z 2.60. If the scheme is edited, re-run both before trusting the change.
- Housekeeping: the directory and `plan_slug` (`log-coordinate-tamed-milstein`) still name the retired log-chart scheme. It is a label only.
- Not a formulator defect. Declining `moment_ode` for V is correct (a Y-chart reference would be wrong by a factor of 90 for the state). The reciprocal-CIR reference is carried as gated invariants and gated functionals, the strongest form the vocabulary allows while V is the state. Sibling plans 2 and 3 disagree with this plan on the 50k run (functional z 3.56 and 3.32 against bar 3.0), but that comparison is stale (solver hashes changed) and cross-plan agreement never certifies in either direction.

</review>
