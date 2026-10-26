---
id: 1
plan_slug: full-truncation-euler
scheme: euler-maruyama
scheme_family: euler-maruyama
expected_strong_order: 0.22
expected_weak_order: 1.0
strategy: Full-truncation Euler-Maruyama on V (drift and diffusion at V^+, auxiliary V_hat kept unclipped) with log-Euler on S, one simulation of R = S/S0 priced at all three spots.
---

## SDE Reference

Heston, Ito, zero interest rate, on t in [0, T], T = 1:

    dS = sqrt(V) * S dW1,                          S(0) = S0
    dV = kappa * (eta - V) dt + sigma * sqrt(V) dW2,  V(0) = V0
    corr(dW1, dW2) = rho

| Parameter | Value | Note |
|---|---|---|
| kappa | 2.58 | mean reversion |
| eta | 0.043 | long-run variance |
| sigma | 1.0 | vol-of-vol (problem.md writes the V diffusion as sqrt(V) dW2) |
| rho | -0.36 | corr(dW1, dW2) |
| r | 0.0 | no drift in S, no discounting anywhere |
| V0 (`Y_0`) | 0.164 | well above eta: V decays toward eta over [0, 1] |
| S0 for `terminal_paths` (`X_0`) | 100.0 | |
| K | 100.0 | strike |
| spots | 75, 100, 125 | one price per spot |

Feller is violated: 2*kappa*eta = 0.22188 < sigma^2 = 1 (CIR dimension delta = 0.44376 < 1), so V
reaches 0 and spends real time near it.

Noise in two independent channels Z1, Z2:

    dW1 = dZ1
    dW2 = rho * dZ1 + sqrt(1 - rho**2) * dZ2

Diffusion matrix against (Z1, Z2):
`G = [[sqrt(V^+) S, 0], [sigma rho sqrt(V^+), sigma sqrt(1-rho^2) sqrt(V^+)]]`.

Quantity of interest (the whole external grade): `price_S{75,100,125} = mean((S_T - 100)^+)` for each
spot, undiscounted. Tolerances: 1.5% / 0.4% / 0.2% relative (0.0199 / 0.0412 / 0.0589 in price
units). `price_S125` is the binding one.

## Numerical Scheme

- **Method**: Euler-Maruyama, full-truncation variant on V (Lord, Koekkoek and van Dijk 2010), log-Euler on S
- **dt**: 0.002 (the contract value; use the `dt` passed in, `Nt = max(1, round(T/dt))`, `dt = T/Nt` -> Nt = 500)
- **num_paths**: 2000000 (graded call), then 500000 (second call, same seed)
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable (multi-D system; spec `milstein_eligible: false`, reason `levy_area`)
- **Expected strong order**: 0.22 (spec-declared `verification.expected_strong_order`; Feller-violated CIR with delta = 0.444 caps any equidistant-increment scheme at delta/2 for V. The nominal Euler-Maruyama 0.5 is not achievable here; the kernel band admits 0.0-0.62)
- **Expected weak order**: 1.0 (diagnostic only; may read below 1 for the same reason)
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`: E[S_T] = 100, Var[S_T] = 768.581477379, E[V_T] = 0.0521686544868, Var[V_T] = 0.0115699466208), plus `verification.functional_reference` for the three prices and the cross moment E[S_T V_T] (covariance -0.775885) as a correlation probe

Per step n (state: `logR` = log(S/S0), length num_paths, starting at 0; `Vh` = auxiliary variance, starting at V0):

    Vp      = max(Vh, 0)
    sq      = sqrt(Vp)
    dW1     = dZ1
    dW2     = rho*dZ1 + sqrt(1 - rho**2)*dZ2
    logR   += -0.5*Vp*dt + sq*dW1          # uses V at the LEFT point (Ito)
    Vh     += kappa*(eta - Vp)*dt + sigma*sq*dW2

Both updates use `Vp` from the start of the step. Do not clip `Vh` itself.

At T: `R = exp(logR)`; `price_S{s0} = mean(maximum(s0*R - 100, 0))` for s0 in (75, 100, 125);
`terminal_paths = column_stack([100*R, max(Vh, 0)])`.

## Implementation Notes

- **The V = 0 boundary is the point of the problem.** Full truncation evaluates drift *and*
  diffusion at V^+ while keeping the auxiliary `Vh` unclipped. Do **not** use absorption
  (`V = max(V, 0)` on the evolving state) or reflection (`V = |V|`): both change the law and bias the
  price (problem.md, and Lord et al. 2010 rank them worst). Never take `sqrt` of an unguarded `Vh`:
  NaN fails the `finite` gate.
- **Reported V is `max(Vh, 0)` at output only** (gate `variance_nonnegative: X[:,1] >= 0`). That is the
  scheme's variance output by definition, not a clip of the evolving state.
- **S via log-Euler** keeps S > 0 (gate `asset_positive`) and makes each step an exact conditional
  martingale, so E[S_T] = S0 up to MC noise (invariant `martingale`, reference 100.0).
- **One simulation serves all three spots.** S is linear in S0 and V does not depend on S: simulate
  R = S/S0 once and price `(s0*R_T - 100)^+` for each spot. The three prices then share their noise,
  as problem.md intends. Simulating three times costs 3x for nothing.
- **Memory**: at 2e6 paths a full increment block `(num_paths, 500, 2)` is 16 GB. Draw increments one
  step at a time: `Z = rng.standard_normal((num_paths, 2))` inside the loop, from
  `rng = np.random.default_rng(seed)`. Keep state as float64 arrays of length num_paths. Work in
  place where possible (`np.maximum(Vh, 0, out=Vp)`, `np.sqrt(Vp, out=sq)`).
- **Verification hooks** (keep the contract signature `solve_sde(num_paths, dt, T, seed=42)` and add
  keyword-only optionals with defaults):
  - `dW`: shape `(num_paths, Nt, 2)`, the **independent** channel increments (dZ1, dZ2), already
    scaled by sqrt(dt). When given, take `Nt = dW.shape[1]`, `dt = T/Nt`, build the correlated dW2
    from them exactly as above, and draw no randomness of your own.
  - `observables`: `{name: phi}` with `phi` taking an `(num_paths, 2)` array of (S, V) for S0 = 100
    and V the reported `V^+`. Accumulate the trapezoid path integral `0.5*(phi(X_prev)+phi(X))*dt`
    per path and return it under `path_integrals`. Only build the (num_paths, 2) state array on steps
    where observables are requested, so the graded run does not pay for it.
- **Return dict**: `terminal_paths` (num_paths, 2) = (S, V) for S0 = 100; `empirical_mean` and
  `empirical_variance` (ddof=1) as Python lists of floats, one per component; `functionals` =
  `{"price_S75", "price_S100", "price_S125"}` as Python floats; also `num_paths`, `dt`, `T`.
- **Two-count check**: the 2e6 and 5e5 calls with the same seed must give different prices that agree
  within MC noise. Per-step `(num_paths, 2)` draws do that naturally. Do not cache, round or hard-code
  a price; never read `verification.functional_reference`.
- **No work at import**; numpy and the standard library only.
- **Accuracy risk (main concern for this plan)**: payoff sampling error at 2e6 paths is about 0.3% /
  0.1% / 0.07% (1 s.e.) for S75 / S100 / S125, so the time-discretization bias at dt = 0.002 must stay
  well under ~0.1% for S125. Full truncation's weak bias at sigma = 1 and delta = 0.44 is O(dt) in
  principle but with a large constant from paths sitting at V = 0. `dt` is fixed by the contract and
  may not be changed, so if the evaluator reports a price bias outside tolerance, the remedy is a
  better boundary treatment within this scheme family (e.g. evaluating the S increment with the
  trapezoid of V^+ at both ends of the step), not a smaller step. Report the measured bias against
  the S125 tolerance in Results.
- **Expected ceiling**: the Dynkin ladder at 2e6 paths and dt = 0.002 needs a 64 GB increment array,
  above the kernel's 12 GB budget, so `dynkin_ok` will be `"unavailable"` and the score caps at 9
  (uncertified, not suspect). This is a resource limit stated in the spec, not a solver defect.

## Results

Run: `solve_sde(2000000, 0.002, 1.0, seed=42)` (Nt = 500, about 95 s for both calls).

| Quantity | Empirical | Analytic |
|---|---|---|
| mean S_T | 100.0276 | 100.0 |
| mean V_T | 0.052037 | 0.052169 |
| variance S_T | 769.430 | 768.581 |
| variance V_T | 0.011562 | 0.011570 |

Prices (undiscounted), 2e6 paths / 5e5 paths (same seed):

| Price | 2e6 | 5e5 |
|---|---|---|
| price_S75 | 1.33163 | 1.33214 |
| price_S100 | 10.31597 | 10.32350 |
| price_S125 | 29.44750 | 29.46740 |

Against the spec's stated price-unit tolerances (0.0199 / 0.0412 / 0.0589), the measured offsets from the spec reference are about +0.007 / +0.016 / +0.022, i.e. inside tolerance, with S125 at roughly 37% of its budget (payoff s.e. at 2e6 paths is about 0.02). The solver itself never reads the reference. Both hooks (`dW` of shape (M, Nt, 2) and `observables`) ran cleanly.

## Evaluation

<metrics>
score: 9
certified: false
provenance: analytic
estimated_rel_error: 2.528e-03
error_is_estimate: false
observed_order: 0.3361
order_floor: 0.2200
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
invariant_max_drift: 0.0276
invariant_worst_headroom: 0.1529
reference_outcome: validated
mean_rel_err: 2.528e-03
variance_rel_err: 1.104e-03
order_check: true
mc_se_rel: 3.563e-03
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 8cce19e192420749e23808cf513d7f8a0601868e00c664dacb52f460c0e89452
solve_time_s: 80.4740
wall_time_s: 93.7300
</metrics>

<review score=9>

**Score: 9/10**

### Numerical Accuracy
- provenance:              analytic  (reference route: analytic_moments, validated against the spec's moment ODE; no surrogate needed)
- mean_rel_err:            S 2.76e-04 (100.0276 vs 100, se 0.0196); V 2.53e-03 (0.052037 vs 0.052169, se 7.6e-05). Worst 2.53e-03 against tol 0.05, 2 x se well inside
- variance_rel_err:        S 1.10e-03 (769.43 vs 768.58, se 1.36); V 6.66e-04 (0.011562 vs 0.011570, se 4.1e-05). Against tol 0.10
- mc_se_rel:               3.56e-03 (S), 7.1e-03 half-width for V at 2 x se, both under the 0.05 bar.  resolved: true
- observed_order:          0.336 +/- 0.013 (expected 0.22, two-sided band +/-0.4; V component 0.287 +/- 0.019). Median-based 0.362. Inside the band
- weak_order:              indeterminate -- all three test functions at the noise floor, so not scored
- dynkin:                  unavailable -- not a verdict. The finest ladder level needs 64.0 GB of increments (2e6 paths x 2000 steps x 2 channels) against the 12 GB budget. The spec predicted this in `dynkin_note`. Resource limit, not a solver defect
- terminal_law:            not declared in the spec (see `terminal_law_note`)
- shape:                   S skewness 0.549 +/- 0.016, excess kurtosis 4.22 +/- 0.31; V skewness 3.99 +/- 0.017, excess kurtosis 23.4 +/- 0.32 (heavy-tailed, as expected of a Feller-violated CIR; this is why the general se_v was used)
- cross moment:           E[S V] = 4.4315 +/- 0.0062 vs 4.4410 (rel 2.1e-03); covariance -0.7736 +/- 0.0034 vs -0.7759; terminal correlation -0.259 +/- 0.0009. The sign and size of rho are carried correctly
- constraints:             finite pass, asset_positive pass, variance_nonnegative pass (neg_fraction 0, min V 0.0 since V^+ is taken at output only); martingale E[S_T] = S0 drift 0.0276 against a bar of 0.0392 (2 x se), pass with 0.15 decades headroom
- reported moments:        solver's own empirical_mean / empirical_variance agree with the kernel's recomputation to 5.4e-12 (ddof=1 used)
- prices (graded by the evaluator against `verification.functional_reference`, since the kernel does not grade SDE functionals; recomputed by calling `solve_sde` at the contract settings, seed 42, dt 0.002):

  | price | 2e6 paths | payoff s.e. | rel err vs ref | rel_tol | 5e5 paths | s.e. | rel err | 2e6 minus 5e5 |
  |---|---|---|---|---|---|---|---|---|
  | price_S75 | 1.33163 | 0.0048 | +5.1e-03 | 1.5e-02 | 1.33214 | 0.0099 | +5.5e-03 | -0.0005 |
  | price_S100 | 10.31597 | 0.0123 | +1.5e-03 | 4.0e-03 | 10.32350 | 0.0249 | +2.3e-03 | -0.0075 |
  | price_S125 | 29.44750 | 0.0200 | +7.4e-04 | 2.0e-03 | 29.46740 | 0.0403 | +1.4e-03 | -0.0199 |

  All three pass at 2e6 paths, with S125 (the binding one) using 37% of its 0.0589 budget. Each offset from the reference is 1.4, 1.3 and 1.1 payoff s.e., so the data cannot separate any discretization bias from sampling noise: the time bias is below the resolution of this run, not shown to be zero. The two counts differ (not a constant) and the difference is inside 2 x sqrt(se_a^2 + se_b^2) for each price (0.022 / 0.056 / 0.090; the runs share a seed, so this is conservative), so the two-count check holds.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric -- the generator (Dynkin) check could not run because of the memory budget. Everything measurable passed. Uncertified, not suspect

### Feedback for solver
- None that would change the score. The 9 is the ceiling for this problem at the contract's `num_paths = 2000000` and `dt = 0.002`. The 10 row needs `dynkin_ok is True`, and the kernel declines to run the ladder at 64 GB. Do not reduce `num_paths` to try to unlock it: the contract value is fixed and the kernel refuses to run Dynkin at fewer paths for the reason it states (a wider interval could hide a real generator inconsistency).
- The scheme matches `SOLUTION.md`: full truncation inside the coefficients (`Vp = max(Vh, 0)` feeds both the drift and the diffusion), `Vh` itself never clipped, `max(Vh, 0)` only at output. The state_clip scan has nothing to flag. The `dW` hook (shape `(M, Nt, 2)`, independent channels) is consumed exactly with `Nt = dW.shape[1]`, which is why the CRN strong order of 0.336 +/- 0.013 is a real measurement and not near zero.
- Optional, not scored: the price offsets sit about +1.1 to +1.4 payoff s.e. above the reference on all three spots, with the same sign at both path counts. Because the S125 offset is only 37% of its budget, no change is needed. If a later cycle wants more margin, the improvement is in the S increment (the trapezoid of V^+ across the step, as `SOLUTION.md` already suggests), not a smaller `dt`.

### Notes for the formulator and conductor (not solver defects)
- The spec's own `dynkin_note` predicted `dynkin_ok: "unavailable"` and a cap of 9 at the contract path count. That is what happened. Any Heston plan at 2e6 paths and 2000 Dynkin steps will have the same ceiling unless the kernel's Dynkin budget or ladder configuration changes for this problem. No other plan can reach 10 by moments alone.
- No surrogate was missing: the spec declares `analytic_moments` (validated), a closing `moment_ode` and an exact Heston semi-closed-form price reference, so the no-closed-form rubric did not apply. `cross_plan_agreement` is `unavailable` because there is no sibling plan to compare against yet.

</review>
