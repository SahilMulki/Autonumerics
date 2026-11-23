---
id: 1
plan_slug: levy-area-milstein-midpoint-drift
scheme: milstein
scheme_family: milstein
levy_area_treatment: sub-increments
strategy: Lévy-area Milstein built from the 256 per-step sub-increments, with the S leg integrated exactly and the integrand's within-step drift carried at its midpoint value, halving the pathwise error constant at the same strong order 1.
---

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

- **Method**: Milstein for non-commutative noise — the iterated integral is assembled from the supplied sub-increments, with a midpoint drift correction on the `U` integrand
- **dt**: 0.04 (`evaluation_thresholds.dt`; `Nt = max(1, round(T/dt))` = 25, then `dt = T/Nt`)
- **num_paths**: 50000 (`evaluation_thresholds.num_paths`); the external contract calls it once at `graded_num_paths = 1000`
- **T**: 1.0
- **seed**: 42 (`np.random.default_rng(seed)` only; `dW` and `dW_sub` are always supplied, so no Brownian randomness is drawn here)
- **sub_increments_per_step**: 256

Per-step update, exactly once per step `n` of `dW`:

```
dW1 = dW[:, n, 0]            # drives U
dW2 = dW[:, n, 1]            # drives S

# S is exact in one step: its equation has constant drift and unit additive noise
S_next = S + mu*dt + dW2

# Lévy area of (W1, W2) over the step, from that step's 256 sub-increments:
#   I_21 = int_0^dt (W2(s) - W2(t_n)) dW1(s)
W2tilde = cumsum(dW_sub[:, n, :, 1]) - dW_sub[:, n, :, 1]     # running partial sum, left-point
I_21    = sum_j W2tilde[:, j] * dW_sub[:, n, j, 0]

# U: frozen-integrand Milstein, plus the midpoint value of the within-step drift
U_next = U + (S + 0.5*mu*dt)*dW1 + I_21

S, U = S_next, U_next
```

- **Milstein correction**: `I_21` as above — the spec's `milstein_correction` field verbatim, `I_{21} = sum_j (W2(t_{n,j}) - W2(t_n)) * dW1_sub[j]`. The partial sum must be **left-point** (exclude the `j`-th sub-increment from `W2tilde[j]`), which is what makes it an Itô iterated integral rather than a Stratonovich one.
- **Midpoint drift term**: the exact increment of `U` over the step is `int S_s dW1 = S_n*dW1 + I_21 + mu*int (s - t_n) dW1`. The last term is mean zero with variance `dt^3/3`; replacing it with its conditional-mean-matched surrogate `mu*(dt/2)*dW1` leaves a residual of variance `dt^3/12` — a factor-2 reduction in the pathwise error standard deviation at identical cost and identical strong order. It uses only `dW1`, so it does **not** make the step sub-grid-exact (see Implementation Notes).
- **Expected strong order**: **1.0** (`verification.expected_strong_order_milstein`). `scheme_family: milstein` selects this declared floor rather than the 0.5 Euler–Maruyama default.
- **Expected weak order**: 1.0
- **Reference for scoring**: **analytic moments** — exact mean and variance for both components as quoted above, plus the exactly-closing four-state moment ODE in `verification.moment_ode` (`rhs = [mu, 2*mu*m1 + 1, 0, m2]`). The kernel also runs the CRN strong-order ladder, which is the statistic that separates this plan from a plan using only `dW`.

## Implementation Notes

- **This is the plan to ship.** The whole point of the Clark–Cameron system is that the step increments `dW` do not determine the strong error: any scheme built from `dW` alone is bounded below by `c*sqrt(dt)` (Clark and Cameron, 1980), because `U`'s error is driven by the per-step Lévy area `I_21`, which is conditionally independent of `(dW1, dW2)` given them. Dropping `I_21` costs roughly `sqrt(T*dt/2) / sqrt(E[U_T^2]) ≈ 8%` relative pathwise error at `dt = 0.04`, four times the 2% gate. Including it is not an optimisation; it is the difference between passing and failing.
- **Do not add the sub-grid drift integral `J_n = sum_j (t_j - t_n)*dW1_sub[j]`.** Adding it makes the `U` update algebraically identical to Euler integration on the full 256× sub-grid, which returns terminal states agreeing across `dt`, `dt/2` and `dt/4` to roundoff — and `problem.md` (R19) **excludes** such a run from the comparison rather than failing it. The midpoint term `mu*(dt/2)*dW1` is the accuracy this plan takes without crossing that line: it is a function of the step increment only, so the scheme remains a genuine `dt`-stepped discretisation whose error shrinks as `O(dt)`.
- `S` is integrated **exactly** (`S_n + mu*dt + dW2_n` is the exact solution increment) while still consuming the supplied `dW2`. The kernel already excuses an exactly-integrating step automatically, since its CRN level differences collapse to roundoff. Do **not** declare `exact_sampling_components` here: both components consume the supplied increments, so the field does not apply and declaring it would forfeit the order check that is this plan's main evidence.
- Use the increments exactly as handed over: `Nt = dW.shape[1]`, `dt = T/Nt`. Never regenerate the path. `dW` is already scaled by `sqrt(dt)` and `dW_sub` by `sqrt(dt/256)`; do not rescale either.
- Sanity check to run once: `np.abs(dW_sub.sum(axis=2) - dW).max()` should be at roundoff. If it is not, the sub-increment axis has been misread.
- **Channel order is load-bearing.** `dW[..., 0] = dW1` drives `U`; `dW[..., 1] = dW2` drives `S`. Swapping them keeps both marginal variances correct while breaking `Cov(S_T, U_T) = 0` — the spec declares that as the ungated `zero_cross_moment` invariant precisely because it is the signature of this mistake. Assert `|corr(S_T, U_T)|` is within a couple of standard errors of zero before reporting.
- `exact_mean_U = 0`, which is below `near_zero_mean_threshold = 0.01`, so the mean check on component 1 is skipped; the `martingale` constraint (ungated, bar `ci_mult * SE`) covers it instead. Component 0's mean (2.0) is checked normally.
- `verification.terminal_law` is `null` on purpose — `U(T)` is a variance mixture of centred normals and matches none of the implemented laws — so the distributional tiers abstain. That is expected and costs nothing.
- Vectorise over paths: state arrays of shape `(num_paths,)` each, stacked into `(num_paths, 2)` at the end. The `I_21` accumulation is a `cumsum` plus a contraction over the 256 axis; at 50000 paths that is a `(50000, 256)` temporary per step, which is fine, but prefer `np.einsum` or a chunked loop if memory pressure appears.
- Return `terminal_paths` of shape `(num_paths, 2)` with columns `(S, U)`, plus `empirical_mean` and `empirical_variance` as lists of two floats computed with `ddof=1`. Nothing may execute at import outside `if __name__ == "__main__"`. Standard library and numpy only.

## Results

Run (50000 paths, dt=0.04, T=1, 256 sub-increments per step; locally generated Brownian path, dW_sub.sum(axis=2) == dW exactly):

- Empirical mean (S, U): [1.99869, 0.00470] (exact [2.0, 0.0])
- Empirical variance (S, U): [0.99882, 2.82634] (exact [1.0, 2.83333]; rel. err 1.2e-3, 2.5e-3)
- corr(S, U) = -6.4e-4 (SE about 4.5e-3), consistent with zero cross moment
- dW-only hook and observables hook both confirmed working; dW_sub=None falls back to I_21 = 0.5*dW1*dW2.

## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 10
certified: true
provenance: analytic
estimated_rel_error: 5.667e-03
error_is_estimate: false
observed_order: 0.9932
observed_order_se: 0.0110
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 0.0000
invariant_worst_headroom: 6.0000
invariants_not_reported: ['symmetry_of_U', 'zero_cross_moment', 'martingale']
reference_outcome: validated
mean_rel_err: 1.993e-03
variance_rel_err: 5.667e-03
order_check: true
mc_se_rel: 8.323e-03
resolved: true
dynkin_z: 1.3946
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 1c110e6ad60b0cde07418245fec0ab3e600a5c7c68d0e035d4c44eafc6bd3412
solve_time_s: 2.1380
wall_time_s: 28.3400
cost_source: cpu
</metrics>

<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated)
- mean_rel_err:            1.99e-03 (S mean 1.99601 vs 2.0, se 4.48e-03, 2.0x = 8.97e-03); U is near-zero mode, absolute 7.93e-03 +/- 2.53 x 7.51e-03 (family-wise), within bar
- variance_rel_err:        S: 5.67e-03 +/- 2 x 6.35e-03; U: 5.19e-03 +/- 2 x 2.35e-02 (variance 2.81862 vs 17/6 = 2.83333); tolerance 10%
- mc_se_rel:               8.32e-03   resolved: true
- functional_rel_error:    not applicable, the spec declares no functionals (functionals_ok: unavailable)
- observed_order:          0.993 +/- 0.011 on U (expected 1.0, one-sided below the declared floor). S is integrated exactly, its ladder differences sit at roundoff, so the estimator abstained on S and the verdict binds to U. That abstention is a property of the estimator, not a defect.
- weak_order:              indeterminate, at the noise floor
- dynkin:                  ok. z = 1.3946 against family-wise bar 2.4284 (50000 paths, 3 levels, deterministic start x0 = [1, 0] now read from the top-level initial_condition). dynkin_replicated: false, the plan was not marginal (|z - bar| about 1.0, outside the replication margin), so one seed settled it. Order extrapolation was assumed 1 on all three test functions; on the first the residual is at roundoff and its rate is unreadable. Caveat: the generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from problem.md would fool it.
- terminal_law:            not declared (spec leaves it null on purpose; U(T) is a variance mixture)
- shape:                   skewness S -0.005 +/- 0.011, U -0.026 +/- 0.022; excess kurtosis S -0.008 +/- 0.022, U 1.464 +/- 0.069 (leptokurtic, as expected for a variance mixture)
- constraints:             finite: pass (gate). symmetry_of_U, zero_cross_moment, martingale: not_reported (no registry evaluator, ungated). invariants_ok true
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. Every checked moment is within tolerance of the exact moments, the estimates are MC-resolved, and the Dynkin check now reaches a verdict.

Change from the previous pass: the score moves 9 to 10 solely because dynkin_ok went from unavailable to true. The cause was the formulator-side placement of the initial condition, now fixed at the top level. solver.py is unchanged (sha256 1c110e6a...).

### Feedback for solver
- None.
- Optional hygiene, not scored: the fallback `I21 = 0.5*dW1*dW2` when `dW_sub is None` is not an Ito Levy area. If it were reached with `dW` supplied it would add a spurious bias, since the conditional mean of I_21 given the step increments is 0. The correct fallback is `I21 = 0`. It never ran in this evaluation.

</review>
