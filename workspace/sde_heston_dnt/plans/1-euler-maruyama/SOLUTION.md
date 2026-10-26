---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Full-truncation Euler-Maruyama on V, log-Euler on S with frozen per-step variance, and a per-step Brownian-bridge survival weight in log space for continuous monitoring of the (50, 150) corridor; one (V, log-return) path set serves all three spots.
---

## SDE Reference

Heston system (Ito), zero rate, state (X, Y) = (S, V):

    dS = sqrt(V) * S dW1,                              S(0) = S0
    dV = kappa * (eta - V) dt + sigma * sqrt(V) dW2,   V(0) = V0
    d<W1, W2>_t = rho dt

| Parameter | Value |
|---|---|
| kappa | 2.58 |
| eta | 0.043 |
| sigma (vol-of-vol) | 1.0 |
| rho | -0.36 |
| r | 0.0 |
| V0 (`Y_0`) | 0.114 |
| S0 for terminal_paths (`X_0`) | 100.0 |
| Spots for the functional | 75, 100, 125 |
| Barriers | lower 50, upper 150 (knock-out on touch: survival requires 50 < S_t < 150 strictly, for all t) |
| T | 1.0 |

Feller condition FAILS: 2*kappa*eta = 0.22188 < sigma^2 = 1, so V reaches 0.

Noise against independent channels (W1, W_perp):
`dW1 = dW[:, n, 0]`, `dW2 = rho*dW[:, n, 0] + sqrt(1 - rho**2)*dW[:, n, 1]`.

Spec drift / diffusion (full-truncation form, coincides with the true generator on V >= 0):
- drift: `[0, kappa*(eta - max(V, 0))]`
- G: row S = `[sqrt(max(V,0))*S, 0]`; row V = `[rho*sigma*sqrt(max(V,0)), sigma*sqrt(1-rho**2)*sqrt(max(V,0))]`

## Numerical Scheme

- **Method**: Euler-Maruyama family (system scheme). Full truncation (Lord, Koekkoek & van Dijk 2010) on V; log-Euler on S.
- **dt**: 0.002 (mandated by problem.md; use the `dt` passed in, `Nt = max(1, round(T/dt))`, `dt = T/Nt`)
- **num_paths**: 400000 (graded run); second run at 100000 with identical dt, T, seed
- **seed**: 42 (`np.random.default_rng(seed)` only)
- **T**: 1.0
- **Milstein correction**: none (`milstein_eligible: false`, reason `levy_area`)
- **Expected strong order**: 0.3 (the spec's declared `expected_strong_order`; the Lipschitz 0.5 of EM is lost because the Feller condition fails — measured 0.29-0.32 for exactly this scheme under the kernel's CRN ladder dt0 = 0.008 -> 0.004 -> 0.002)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`, formulator-derived and validated: E[S_T] = 100, Var[S_T] = 606.655, E[V_T] = 0.048380, Var[V_T] = 0.0102127) plus the Dynkin check against `verification.generator`; the graded quantity of interest is the no-touch functional, against the formulator's 2-D backward-Kolmogorov references 0.83454 / 0.89983 / 0.66840 (S0 = 75 / 100 / 125), rel_tol 0.006 each, gate true.

### Per-step update (n = 0 .. Nt-1)

    Vp      = max(V_n, 0)
    V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*sqrt(Vp)*dW2        # do NOT clip V_{n+1}
    y_{n+1} = y_n - 0.5*Vp*dt + sqrt(Vp)*dW1                       # y = ln(S/S0), y_0 = 0

For each spot k: `a = ln S0_k + y_n`, `b = ln S0_k + y_{n+1}`, `lU = ln 150`, `lL = ln 50`, `s2 = Vp*dt`:

    if b <= lL or b >= lU:          step survival = 0   (endpoint touched/crossed)
    elif s2 == 0:                   step survival = 1   (deterministic step, endpoint inside)
    else:
        p_up = exp(-2*(lU - a)*(lU - b)/s2)
        p_dn = exp(-2*(a - lL)*(b - lL)/s2)
        step survival = clip(1 - p_up - p_dn, 0, 1)
    W_k *= step survival            # W_k starts at 1; once 0 stays 0

`no_touch_Sk = mean(W_k)` — a conditional-expectation estimator (no Bernoulli draw), fully driven by the Brownian increments.

## Implementation Notes

1. **Solver contract** — `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`. The two hooks are needed for internal evaluation (CRN order study and Dynkin); problem.md's external call passes only the first four.
2. **dW hook** — when supplied, shape is `(num_paths, Nt, 2)` of independent increments already scaled by sqrt(dt); take `Nt = dW.shape[1]`, `dt = T/Nt`, draw no randomness. When `dW is None`, draw `rng.standard_normal((num_paths, 2)) * sqrt(dt)` **inside the time loop** (a pre-drawn 400000 x 500 x 2 block is 3.2 GB). Do not store path histories.
3. **Feller violation** — full truncation as above. Use `Vp` everywhere V enters the S step, the drift, the diffusion and the bridge variance. Never write `V = np.maximum(V, 0)` as a statement (that is reflection-by-clipping, a different scheme, flagged by the leakage scan). Report `np.maximum(V_T, 0)` as the V column of `terminal_paths`.
4. **Positivity of S** — log-Euler keeps S > 0 exactly (gated constraint `S_positive`) and is an exact martingale step given frozen V (`martingale` invariant).
5. **One path set, three spots** — the log-return path y and V are identical for every S0; only `a, b` shift by `ln S0_k`. Keep three survival-weight arrays of shape `(num_paths,)`.
6. **Continuous monitoring is the whole problem** — grid-point-only monitoring at dt = 0.002 overstates survival by +1.0% / +0.5% / +2.0% (S0 = 75 / 100 / 125), failing the 0.6% bar at two spots. The bridge weight is required. Guard `s2 -> 0` (Vp = 0) to avoid 0/0; the double-crossing term is negligible at this dt. Measured with exactly this recipe at 400k paths, seed 42: 0.83510 / 0.89962 / 0.66715 — all within 0.2% of reference, within 1.7 s.e.
7. **MC resolution** — relative s.e. at 400k paths: 0.07% / 0.05% / 0.11%. The 0.6% tolerance is almost entirely bias budget.
8. **Two path counts** — the 400k and 100k runs must give different values that agree within MC noise. With per-step draws from `default_rng(seed)` of shape `(num_paths, 2)` the streams differ automatically. Never return a constant.
9. **Outputs** — `terminal_paths = column_stack([S_T for S0 = 100, max(V_T, 0)])` for every path, knocked out or not (shape `(num_paths, 2)`); `empirical_mean` / `empirical_variance` per-column Python lists (ddof = 1); `functionals = {"no_touch_S75", "no_touch_S100", "no_touch_S125"}` as Python floats; also `num_paths`, `dt`, `T`.
10. **Observables hook** — `observables` is `{name: phi}` with phi taking the `(num_paths, 2)` state `[S (S0 = 100), max(V, 0)]`; accumulate trapezoid integrals `0.5*(phi(X_prev) + phi(X))*dt` and return them under `path_integrals`. With the Feller violation, the V-dependent Dynkin residual may converge slowly; `dynkin_ok: "unresolved"` is a legitimate outcome, not a solver bug.
11. **Constraints** — numpy + stdlib only; no work at import time outside `if __name__ == "__main__"`; do not use Andersen QE or exact noncentral-chi-square sampling (they do not consume the supplied increments and the CRN order study would read ~0).

## Results

Run: `uv run python workspace/sde_heston_dnt/plans/1-euler-maruyama/solver.py` (dt = 0.002, T = 1, seed 42; about 25 s per 400k run).

| num_paths | mean [S, V] | variance [S, V] | no_touch S75 / S100 / S125 |
|---|---|---|---|
| 400000 | [100.0667, 0.048499] | [607.22, 0.010193] | 0.83510 / 0.89962 / 0.66715 |
| 100000 | [100.1084, 0.048049] | [605.22, 0.009993] | 0.83457 / 0.89917 / 0.66796 |

Reference: mean [100, 0.048380], variance [606.655, 0.010213]; no-touch 0.83454 / 0.89983 / 0.66840. The 400k functionals are within 0.07%, 0.02% and 0.19% of the references (tolerance 0.6%). The two path counts give different values that agree within MC noise. The `dW` and `observables` hooks both ran.

## Evaluation

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 2.451e-03
error_is_estimate: false
observed_order: 0.2817
order_floor: 0.3000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0667
invariant_worst_headroom: 0.2435
invariants_not_reported: ['variance_nonnegative']
reference_outcome: validated
mean_rel_err: 2.451e-03
variance_rel_err: 1.962e-03
order_check: true
mc_se_rel: 8.139e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 9838edce7629b609eb624eb6f57e8849c5c0c78b7f3e088b88cb5548c8128c25
solve_time_s: 40.7850
wall_time_s: 49.7700
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated; moment ODE agrees with the closed form per the spec)
- mean_rel_err:            S 6.67e-04 (100.0667 vs 100, se 0.039, bar 2 x 0.039 = 0.078); V 2.45e-03 (0.048499 vs 0.048380, se 1.6e-4, bar 3.2e-4). Tolerance 5%
- variance_rel_err:        S 9.33e-04 (607.22 vs 606.66, se 2.50); V 1.96e-03 (0.010193 vs 0.010213, se 8.3e-5). Tolerance 10%
- mc_se_rel:               8.1e-03 (worst component, CI width 1.6% of value against a 5% bar)   resolved: true
- observed_order:          0.282 +/- 0.019 (se; worst component V). S component 0.315 +/- 0.013. Expected 0.3, two-sided band +/- 0.4. Median-based order 0.325
- weak_order:              indeterminate - at the noise floor (all three phi discarded)
- dynkin:                  unavailable - not run. The finest ladder level needs a 12.8 GB increment array (400000 paths x 2000 steps x 2 channels) against a 12 GB budget. Not a pass and not a failure
- terminal_law:            not declared (deliberately, per spec). Shape for the record: S skewness 0.419 +/- 0.049, excess kurtosis 4.76 +/- 1.20; V skewness 4.05 +/- 0.044, excess kurtosis 24.5 +/- 0.98
- constraints:             finite pass, shape pass, S_positive pass (all gates); martingale E[S_T]=100 pass (drift 0.067 against bar 0.117, headroom 0.24 decades); variance_nonnegative not_reported (free-form note, nothing to evaluate, by design)
- reported_moments_agree:  true (worst relative difference 2e-13)
- no-touch functionals (re-run by me this pass, outside the kernel, which does not score SDE functionals; identical to the earlier pass): 400k run 0.83510 / 0.89962 / 0.66715 against reference 0.83454 / 0.89983 / 0.66840, relative errors 0.067% / 0.023% / 0.187% (bar 0.6%, MC s.e. about 0.06% / 0.05% / 0.11%). 100k run 0.83457 / 0.89917 / 0.66796. The two runs differ (0.00053, 0.00045, -0.00081) and agree well within their combined sampling noise (about 0.0013 / 0.0011 / 0.0017 at 2 s.e. even before doubling). All three gates met
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric - every checked moment is within tolerance and the MC is resolved, but the generator (Dynkin) check could not run, so the 10 row is not available. Uncertified, not suspect

### Feedback for solver
- The score is 9 and not 10 for one reason that is not a defect in the code: `dynkin_ok` came back `unavailable` because the kernel's Dynkin ladder at 400000 paths and dt/4 = 0.0005 would need a 12.8 GB `(num_paths, Nt, 2)` increment block, over the 12 GB budget. The kernel will not shrink the path count, because a smaller run could turn a real generator inconsistency into a pass. This is a property of the spec's `num_paths`/`dt`/`noise_dimension` and not of `solver.py`. Nothing in the solver can fix it. Do not change the scheme on account of it.
- Everything that was measurable passes with margin. Moments are within 0.25% of the exact values against 5% and 10% tolerances and MC-resolved. The CRN strong order is 0.282 +/- 0.019 (0.315 on S) against the spec's measured 0.3, so the solver consumes the supplied `dW` as required (shape `(M, Nt, 2)`, `dW2 = rho*dW1 + sqrt(1-rho^2)*dW_perp`, `Nt = dW.shape[1]`, `dt = T/Nt`). The no-touch functionals sit inside 0.2% against a 0.6% bar, so the Brownian-bridge survival weight is doing its job.
- Source review: `V` is not clipped as a statement. The only truncation is `Vp = np.maximum(V, 0.0)` inside the coefficients, which is full truncation and endorsed. The `state()` helper clips only the reported V column. `SOLUTION.md` (Euler-family, full truncation on V, log-Euler on S, no Milstein) matches `solver.py`, so no cap applies.
- Two small notes, neither scored. (1) `weak_order` could not be read: all three test functions were at the noise floor at 20000 paths, which is the estimator and not the solver. (2) `invariant_worst_headroom` on the martingale check is 0.24 decades, because the bar is 3 s.e. of a 400k-path mean. That is the Monte Carlo ceiling and not something a scheme change would improve.
- Spec-side finding (formulator/conductor, not solver): the spec's own `num_paths = 400000` with `dt = 0.002` and `noise_dimension = 2` makes the Dynkin ladder's finest level (dt/4, 2000 steps) 12.8 GB against a 12 GB budget, so every correct plan on this problem is capped at 9 by the same mechanism. The spec's `dynkin_note` anticipated a slow V residual (`unresolved`), but the check never got that far.
- Nothing to change in `solver.py`. A 10 here depends on the Dynkin check being given a budget it fits in, which is a decision for the formulator/conductor (for example a lower `dynkin` path count declared in the spec, or a larger memory budget), not for the solver.

</review>
