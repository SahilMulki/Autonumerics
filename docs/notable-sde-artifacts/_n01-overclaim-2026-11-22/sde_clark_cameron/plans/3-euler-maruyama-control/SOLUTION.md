---
id: 3
plan_slug: euler-maruyama-control
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Plain multi-dimensional Euler-Maruyama on the step increments alone — the deliberate control that demonstrates the Clark-Cameron sqrt(dt) floor and is expected to miss the 2% pathwise gate.
---

**This plan is the control, and it is numbered last on purpose.** It is expected to
**fail** the problem's 2% relative pathwise strong-error gate. It exists so the pool
contains a discretised scheme built from the step increments alone, against which the
Lévy-area schemes' strong order can be read as a genuine difference rather than an
assertion. It is not the candidate to ship, and it must not win a tie.

## SDE Reference

Two-dimensional Itô system driven by **two independent** Brownian motions `W1`, `W2`:

```
dS = mu*dt + dW2,     S(0) = S_0
dU = S*dW1,           U(0) = U_0
```

State vector is `(S, U)` in that order; returned columns `(X, Y) = (S, U)`.

Diffusion matrix (rows = state components `(S, U)`, columns = noise channels `(W1, W2)`):

```
G(S, U) = [[0, 1],
           [S, 0]]
```

So `dW[..., 0] = dW1` drives `U` and `dW[..., 1] = dW2` drives `S`.

Parameters: `S_0 = 1.0`, `U_0 = 0.0`, `mu = 1.0`. Time interval `[0, 1]`, `T = 1.0`.

Exact moments (`analytic_moments.has_analytic_solution == true`):

```python
exact_mean_S     = S_0 + mu*t                                      # = 2.0 at t = 1
exact_variance_S = t                                               # = 1.0 at t = 1
exact_mean_U     = U_0                                             # = 0.0 for all t
exact_variance_U = ((S_0 + mu*t)**3 - S_0**3)/(3*mu) + t**2/2      # = 17/6 = 2.8333333333 at t = 1
# also exactly Cov(S_T, U_T) = 0
```

## Numerical Scheme

- **Method**: multi-dimensional Euler–Maruyama, `X_{n+1} = X_n + f(X_n)*dt + G(X_n) @ dW_n`. `milstein_eligible` is `false` (`milstein_ineligible_reason: levy_area`), so a scheme restricted to `dW` has no correction term available to it at all.
- **dt**: 0.04 (`evaluation_thresholds.dt`; `Nt = max(1, round(T/dt))` = 25, then `dt = T/Nt`)
- **num_paths**: 50000 (`evaluation_thresholds.num_paths`); the external contract calls it once at `graded_num_paths = 1000`
- **T**: 1.0
- **seed**: 42 (`np.random.default_rng(seed)` only; `dW` is always supplied, so no Brownian randomness is drawn here)

Per-step update:

```
dW1 = dW[:, n, 0]            # drives U
dW2 = dW[:, n, 1]            # drives S

S_next = S + mu*dt + dW2
U_next = U + S*dW1
S, U = S_next, U_next
```

- **Milstein correction**: none, and none is available — the required term is the Lévy area `I_21`, which `dW` does not carry. `dW_sub` is accepted in the signature and deliberately unused.
- **Expected strong order**: **0.5** (`verification.expected_strong_order`). `scheme_family: euler-maruyama` selects that declared floor; the spec carries 1.0 separately under `expected_strong_order_milstein`, which does not apply here.
- **Expected weak order**: 1.0
- **Reference for scoring**: **analytic moments** — exact mean and variance for both components as quoted above, plus the exactly-closing four-state moment ODE in `verification.moment_ode`.

## Implementation Notes

- **Expected to miss the pathwise gate, by design.** The omitted per-step Lévy area has variance `dt^2/2`, and the omissions are uncorrelated across steps, so the terminal error standard deviation in `U` is about `sqrt(Nt * dt^2/2) = sqrt(T*dt/2) ≈ 0.141` at `dt = 0.04`. Against `sqrt(E[U_T^2]) ≈ 1.68` that is roughly **8.4% relative** — about four times the 2% bar. It shrinks only as `sqrt(dt)`, so reaching 2% would need `dt` about 17× smaller; that is the Clark–Cameron (1980) lower bound, not a defect in the implementation, and it is why refining `dt` is **not** the right response to a failing evaluation here.
- Note what this scheme gets *right*, which is why the control is informative rather than merely bad: the `S` leg is exact, and `U`'s terminal **variance** is unbiased in the moment-space reading because the dropped `I_21` terms are mean-zero and orthogonal to the retained ones. So the plan can pass the moment thresholds while failing the pathwise gate — the strong-order ladder, reading ≈ 0.5 against the Lévy-area plans' ≈ 1.0, is what distinguishes it.
- Use the increments exactly as handed over: `Nt = dW.shape[1]`, `dt = T/Nt`. Never regenerate the path. `dW` is already scaled by `sqrt(dt)`; do not rescale.
- Do **not** declare `exact_sampling_components`: both components consume the supplied increments. `S`'s step is exact, but the kernel already excuses an exactly-integrating step automatically, and a declaration here would make the order study abstain — which would destroy the only thing this control is for, namely a measured 0.5 to contrast against.
- **Channel order is load-bearing.** `dW[..., 0] = dW1` drives `U`; `dW[..., 1] = dW2` drives `S`. Swapping them keeps both marginal variances correct while breaking `Cov(S_T, U_T) = 0` (the spec's ungated `zero_cross_moment` invariant). Assert `|corr(S_T, U_T)|` is within a couple of standard errors of zero.
- `exact_mean_U = 0`, below `near_zero_mean_threshold = 0.01`, so the mean check on component 1 is skipped; the ungated `martingale` constraint (bar `ci_mult * SE`) covers it. Component 0's mean (2.0) is checked normally.
- `verification.terminal_law` is `null` on purpose, so the distributional tiers abstain. Expected.
- Return `terminal_paths` of shape `(num_paths, 2)` with columns `(S, U)`, plus `empirical_mean` and `empirical_variance` as lists of two floats computed with `ddof=1`. Keep `dW_sub=None` in the signature's default and ignore the array when it is passed. Nothing may execute at import outside `if __name__ == "__main__"`. Standard library and numpy only.

## Results

Run: `solve_sde(num_paths=50000, dt=0.04, T=1.0, seed=42)` (Nt = 25, no Levy area; `dW_sub` unused).

- Empirical mean (S, U): [2.005045, 0.000897]  (exact: 2.0, 0.0)
- Empirical variance (S, U): [0.996042, 2.756433]  (exact: 1.0, 2.8333; U within 2.7% of exact)
- corr(S, U) = -0.0068 (se about 0.0045), consistent with zero
- `dW` hook (explicit 50-step increments -> dt = 0.02) and `observables` hook (path_integrals shape (1000,)) both confirmed working.
- Expected: strong order about 0.5 (control); pathwise error about 8% relative at dt = 0.04.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 0.0337
error_is_estimate: false
observed_order: 0.4868
observed_order_se: 0.0110
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
invariants_not_reported: ['symmetry_of_U', 'zero_cross_moment', 'martingale']
reference_outcome: validated
mean_rel_err: 1.993e-03
variance_rel_err: 0.0337
moment_bias_binding: variance
moment_bias_frac_tol: 0.2880
moment_bias_actionable: true
order_check: true
mc_se_rel: 8.279e-03
resolved: true
dynkin_z: 0.9392
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: e79300b8d09b57f2d42914de34135a2fe4521ade71fff1d5e6ad16c2ab462a0c
solve_time_s: 1.4160
wall_time_s: 22.6800
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated)
- mean_rel_err:            S: 1.99e-03 +/- 2.0 x 4.48e-03 (abs); U: abs 8.2e-03 +/- 2.53 x 7.40e-03 (near-zero mean, absolute mode); both pass
- variance_rel_err:        S: 5.7e-03 +/- 2.0 x 6.3e-03; U: 3.37e-02 +/- 2.0 x 2.27e-02 (tol 0.10); pass
- mc_se_rel:               8.28e-03   resolved: true
- functional_rel_error:    not declared by the spec (functionals_ok unavailable)
- observed_order:          0.487 +/- 0.011 (expected 0.5, one-sided below the declared floor)
- weak_order:              indeterminate for the mean and tanh tests (noise floor); not scored
- dynkin:                  ok. max z = 0.939 against family-wise bar 2.428 (3 test functions, all pass; initial state [1.0, 0.0] read from the top-level initial_condition). Not replicated: |z - bar| = 1.49 is outside the 0.7 margin, so one seed (42) settled it and the plan was not marginal. The residual on X[:,1]**2 shrinks at measured order 0.82; the other two residuals are at roundoff or inside MC noise, so no rate is read from them.
- terminal_law:            not declared
- shape:                   U skewness -0.025 +/- 0.022, excess kurtosis 1.43 +/- 0.067 (expected: U is a Gaussian scale mixture); S skewness -0.005 +/- 0.011, excess kurtosis -0.008 +/- 0.022
- constraints:             finite pass; symmetry_of_U, zero_cross_moment, martingale not_reported (ungated)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments, the estimates are MC-resolved, and the generator (Dynkin) check now reaches a verdict and passes.

Notes on the evidence:
- The Dynkin check now runs. The earlier 9 came only from its being unavailable, because the initial condition was not where `dynkin.py` reads it. With the top-level `initial_condition` in place it passes with z = 0.94 against 2.43. `solver.py` is unchanged (sha256 e79300b8...), so the score change is entirely the spec fix.
- What Dynkin does not establish: the generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from problem.md would fool it. It is one circularity break, not an independent reference.
- Strong order 0.487 +/- 0.011 matches the claimed 0.5 and the Clark-Cameron limit. This is the contrast the control was built to show. The solver consumes dW and no `undeclared_exact_sampling` flag was raised.
- There is a real discretization bias in U's variance: 2.778 against an extrapolated 2.861, about 0.288 of the 10% tolerance. It is actionable but well inside the gate, and more paths will not change it.
- The CRN ladder used a reduced sub-increment count to fit the memory cap. This is irrelevant to this plan, which ignores `dW_sub`.

### Feedback for solver
- None. The score is 10 and nothing in `solver.py` needs to change.
- Context only: this is the control. Its order of 0.49 against a Levy-area scheme's expected ~1.0 is the intended contrast. The problem's pathwise 2% strong gate is not scored by this kernel pass.

</review>
