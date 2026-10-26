---
id: 2
plan_slug: euler-maruyama-cir-milstein-v
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Euler-family system scheme with a V-only scalar CIR Milstein correction (truncated), log-Euler on S with frozen per-step variance, and a per-step Brownian-bridge survival weight in log space for continuous monitoring; one (V, log-return) path set serves all three spots.
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

For the autonomous V equation alone: g(V) = sigma*sqrt(V), dg/dV = sigma/(2*sqrt(V)), g*g' = sigma^2/2 (state-independent).

## Numerical Scheme

- **Method**: Euler-Maruyama family (system scheme). V: truncated scalar CIR Milstein step; S: log-Euler. This is NOT a system Milstein scheme — the noise is non-commutative (`milstein_eligible: false`, reason `levy_area`), so S stays at Euler order and the plan is declared `scheme: euler-maruyama`, `scheme_family: euler-maruyama` as `implementation_notes` item 8 requires.
- **dt**: 0.002 (mandated by problem.md; use the `dt` passed in, `Nt = max(1, round(T/dt))`, `dt = T/Nt`)
- **num_paths**: 400000 (graded run); second run at 100000 with identical dt, T, seed
- **seed**: 42 (`np.random.default_rng(seed)` only)
- **T**: 1.0
- **Milstein correction** (V component only): `(sigma**2/4) * (dW2**2 - dt)` — from the sde_manual.md CIR row, with dW2 the correlated increment
- **Expected strong order**: 0.3 (the spec's declared `expected_strong_order`, measured on the worst component; the S component is at Euler order and the Feller violation removes the Lipschitz rate. The spec's two-sided band of +-0.4 admits up to 0.7 if the V correction lifts it)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`, formulator-derived and validated: E[S_T] = 100, Var[S_T] = 606.655, E[V_T] = 0.048380, Var[V_T] = 0.0102127) plus the Dynkin check against `verification.generator`; the graded quantity of interest is the no-touch functional, against the formulator's 2-D backward-Kolmogorov references 0.83454 / 0.89983 / 0.66840 (S0 = 75 / 100 / 125), rel_tol 0.006 each, gate true.

### Per-step update (n = 0 .. Nt-1)

    Vp      = max(V_n, 0)
    V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*sqrt(Vp)*dW2 + 0.25*sigma**2*(dW2**2 - dt)   # do NOT clip V_{n+1}
    y_{n+1} = y_n - 0.5*Vp*dt + sqrt(Vp)*dW1                                                  # y = ln(S/S0), y_0 = 0

All right-hand sides use the state at the start of the step (`V_n`, `Vp`), never `V_{n+1}`.

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
3. **Feller violation / what happens at V = 0** — truncation as above: `Vp = max(V_n, 0)` everywhere V enters the drift, the diffusion, the S step and the bridge variance; the scheme state V itself may go negative. The Milstein term `0.25*sigma**2*(dW2**2 - dt)` is state-independent and has zero conditional mean, so it is applied on every path including those with `Vp = 0` (it does not change the generator, so the Dynkin check is unaffected). Never write `V = np.maximum(V, 0)` as a statement (reflection-by-clipping, a different scheme, flagged by the leakage scan). Report `np.maximum(V_T, 0)` as the V column of `terminal_paths`.
4. **Correction sign and argument** — `+0.25*sigma**2*(dW2**2 - dt)` with the **correlated** dW2, not `dW[:, n, 1]`. A sign error or using the wrong channel shows up as a V-marginal bias (E[V_T] off 0.048380) and a Dynkin failure on phi = V.
5. **Positivity of S** — log-Euler keeps S > 0 exactly (gated constraint `S_positive`) and is an exact martingale step given frozen V (`martingale` invariant).
6. **One path set, three spots** — the log-return path y and V are identical for every S0; only `a, b` shift by `ln S0_k`. Keep three survival-weight arrays of shape `(num_paths,)`.
7. **Continuous monitoring is the whole problem** — grid-point-only monitoring at dt = 0.002 overstates survival by +1.0% / +0.5% / +2.0% (S0 = 75 / 100 / 125), failing the 0.6% bar at two spots. The bridge weight is required. Guard `s2 -> 0` (Vp = 0) to avoid 0/0; the double-crossing term is negligible at this dt.
8. **MC resolution** — relative s.e. at 400k paths: 0.07% / 0.05% / 0.11%. The 0.6% tolerance is almost entirely bias budget; the V correction is intended to reduce the near-zero distortion of the variance path, which feeds the bridge variance `s2`.
9. **Two path counts** — the 400k and 100k runs must give different values that agree within MC noise. With per-step draws from `default_rng(seed)` of shape `(num_paths, 2)` the streams differ automatically. Never return a constant.
10. **Outputs** — `terminal_paths = column_stack([S_T for S0 = 100, max(V_T, 0)])` for every path, knocked out or not (shape `(num_paths, 2)`); `empirical_mean` / `empirical_variance` per-column Python lists (ddof = 1); `functionals = {"no_touch_S75", "no_touch_S100", "no_touch_S125"}` as Python floats; also `num_paths`, `dt`, `T`.
11. **Observables hook** — `observables` is `{name: phi}` with phi taking the `(num_paths, 2)` state `[S (S0 = 100), max(V, 0)]`; accumulate trapezoid integrals `0.5*(phi(X_prev) + phi(X))*dt` and return them under `path_integrals`. With the Feller violation the V-dependent Dynkin residual may converge slowly; `dynkin_ok: "unresolved"` is a legitimate outcome.
12. **Constraints** — numpy + stdlib only; no work at import time outside `if __name__ == "__main__"`; do not use Andersen QE or exact noncentral-chi-square sampling (they do not consume the supplied increments and the CRN order study would read ~0).

## Results

Run: `solve_sde(400000, 0.002, 1.0, seed=42)` (about 25 s), columns (S, max(V,0)):

- empirical_mean: [100.0745, 0.048449] (analytic 100, 0.048380)
- empirical_variance: [607.385, 0.010219] (analytic 606.655, 0.0102127)
- functionals (400k): no_touch_S75 = 0.83513, no_touch_S100 = 0.89998, no_touch_S125 = 0.66688
  (references 0.83454 / 0.89983 / 0.66840; relative errors 0.07% / 0.02% / 0.23%)
- functionals (100k): 0.83462 / 0.89932 / 0.66754 (differ from the 400k run, within MC noise)
- `dW` hook (shape (M, Nt, 2)) and `observables` hook both exercised and working.

## Evaluation

(Filled by evaluator)

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 1.422e-03
error_is_estimate: false
observed_order: 0.2425
order_floor: 0.3000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0745
invariant_worst_headroom: 0.1954
invariants_not_reported: ['variance_nonnegative']
reference_outcome: validated
mean_rel_err: 1.422e-03
variance_rel_err: 1.204e-03
order_check: true
mc_se_rel: 8.172e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: a6aaa664a93f06a6ada3ed7eefa566de3580866ce4eea51bf57d188e43bd5339
solve_time_s: 41.1380
wall_time_s: 49.8700
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, reference_outcome validated)
- mean_rel_err:            S 7.45e-4 (100.0745 vs 100, se 0.039); V 1.42e-3 (0.048449 vs 0.048380, se 1.6e-4) -- both inside 5% tol, +/- 2 x se
- variance_rel_err:        S 1.20e-3 (607.39 vs 606.66, se 2.46); V 6.0e-4 (0.010219 vs 0.010213, se 8.4e-5) -- both inside 10% tol, +/- 2 x se
- mc_se_rel:               8.17e-3   resolved: true (CI width rel 0.0081 for S, 0.0164 for V, against bar 0.05)
- observed_order:          0.242 +/- 0.020 (expected 0.3, two-sided band +/-0.4); S component 0.283 +/- 0.013, V component 0.242 +/- 0.020 (median-based 0.374)
- weak_order:              indeterminate -- at the noise floor for all three test functions
- dynkin:                  not run (unavailable): finest ladder level needs a 12.8 GB increment array (400000 paths x 2000 steps x 2 channels), above the 12 GB budget. Single seed; not replicated (replication cannot change an "unavailable")
- terminal_law:            not declared
- shape:                   S skewness 0.416 +/- 0.044, excess kurtosis 4.57 +/- 0.99; V skewness 4.07 +/- 0.044, excess kurtosis 24.7 +/- 0.99
- constraints:             martingale ok (drift 0.0745 vs bar 0.1169, headroom 0.195); finite, shape, S_positive ok; variance_nonnegative not_reported
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- all measurable checks pass, but the generator (Dynkin) check never reached a verdict because of the memory budget; uncertified, not suspect

### Feedback for solver
- Nothing is wrong with the scheme. Moments, resolution, constraints, reported-vs-recomputed moments (rel diff 1.6e-12) and the CRN order (0.24 vs 0.3, inside the band) all pass, and the solver consumes `dW` correctly (Nt from `dW.shape[1]`, correlated dW2 built from the supplied channels).
- The only thing between this plan and a 10 is the Dynkin check, which was refused for memory (12.8 GB > 12 GB), not failed. This is a property of the spec's num_paths x dt and of the check's budget, not a defect in `solver.py`. Nothing in the solver can fix it; the cap lifts only if the spec/kernel reduces `dynkin_levels` or the finest-level footprint. This is not a request to change the scheme.
- Observed strong order of 0.24 is the lowest-order reading across components (V). It is within the band and not scored against, but sits below the 0.3 target; the Feller violation (2*kappa*eta = 0.22 < sigma^2 = 1) is the expected cause, not a bug.
- Cosmetic: `variance_nonnegative` is declared but not reported by the solver. It is non-gating and not scored, but returning it would remove the not_reported entry.
- Not assessed by this kernel run: the no-touch functionals (0.83513 / 0.89998 / 0.66688 against references 0.83454 / 0.89983 / 0.66840 in SOLUTION.md) are not graded by the verifylib output above; they are the solver's own reported numbers. The spec's `second_run` (100k vs 400k, values must differ and agree within sampling noise) was likewise not run by the kernel, and the evaluator's attempt to re-run the solver for it was denied by the permission system, so neither the functionals nor the second run are independently confirmed here. This is a gap in what was checked, not a finding against the solver.
- Re-run on 2026-10-26 reproduced the stored result exactly (same solver sha256, same score 9, same `dynkin_ok: unavailable` for the 12.8 GB vs 12 GB budget). Not a solver defect and not fixable from `solver.py`.

</review>
