---
id: 2
plan_slug: euler-maruyama-drift-implicit
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Full-truncation Euler-Maruyama for V; drift-implicit Euler step in R = 1/(c_lv*S) (closed-form positive quadratic root, driven only by dW1) with a per-step Brownian-bridge survival weight for both barriers, run with common random numbers for the three spots.
---

## SDE Reference

Ito, zero rate, t in [0, T], T = 1:

    dS = sqrt(V) * f(S) dW1,                       f(s) = c_lv * s^2,  S(0) = S0
    dV = kappa*(eta - V) dt + sigma*sqrt(V) dW2,   V(0) = V0
    d<W1, W2> = rho dt

| Parameter | Value |
|---|---|
| c_lv | 0.01 |
| kappa | 2.58 |
| eta | 0.043 |
| sigma | 1.0 |
| rho | -0.36 |
| V0 | 0.114 |
| L (lower barrier) | 50.0 |
| U (upper barrier) | 150.0 |
| T | 1.0 |
| Spots | S0 = 75, 100, 125 |

Diffusion matrix against INDEPENDENT channels (Z1, Z2):

    G = [[c_lv*S^2*sqrt(V^+),  0                         ],
         [rho*sigma*sqrt(V^+), sqrt(1-rho^2)*sigma*sqrt(V^+)]]

i.e. dW1 = dZ1, dW2 = rho*dZ1 + sqrt(1 - rho^2)*dZ2.

Feller fails: 2*kappa*eta = 0.22188 < sigma^2 = 1, so V hits 0.

**Graded quantity.** `functionals["no_touch_S75" | "no_touch_S100" | "no_touch_S125"]` =
P(L < S_t < U for all t in [0, T]), continuously monitored, undiscounted. Each must be within
0.8% relative of an independent reference. The run is repeated at num_paths = 100000 (same dt,
T, seed). The two values must differ, and they must agree within
2*sqrt(se_400k^2 + se_100k^2).

**Return contract.** `solve_sde(num_paths, dt, T, seed=42, dW=None, observables=None) -> dict`
with:
- `terminal_paths` of shape (num_paths, 2), columns (S(T), V(T)) for S0 = 100. Every path is
  included, including knocked-out ones, with its simulated S(T).
- `empirical_mean` and `empirical_variance` (ddof=1) as 2-element lists.
- `functionals` holding the three floats.
- `num_paths`, `dt`, `T`, and `path_integrals` when `observables` is given.

Use `Nt = max(1, round(T/dt)); dt = T/Nt`, `np.random.default_rng(seed)` only, numpy and the
standard library only, and no work at import time. The extra `dW`/`observables` keyword
arguments must default to `None` so that the graded 4-argument call still works.

## Numerical Scheme

- **Method**: Euler-Maruyama family. Full-truncation EM for V, and a drift-implicit
  (backward-Euler-in-drift) EM step for R = 1/(c_lv*S).
- **dt**: 0.002 (given by the contract; do not change it). Nt = 500.
- **num_paths**: 400000 (graded), plus 100000 (path-count check).
- **seed**: 42
- **T**: 1.0
- **Milstein correction**: not applicable. Milstein is ineligible because the 2-D, two-channel
  diffusion is non-commutative and needs the Levy area.
- **Expected strong order**: 0.5 (spec `verification.expected_strong_order`). The V component
  may read about 0.2-0.3 because Feller fails, with delta = 0.444. That is inside the
  kernel's band.
- **Expected weak order**: 1.0
- **Reference for scoring**: no closed-form moments. The kernel's terminal-moment rows use
  self-convergence only. The graded functionals have a deterministic reference
  (`verification.functional_reference`): a 2-D backward Kolmogorov PDE with Dirichlet barriers,
  validated at rho = 0 against an exact series (0.946627 / 0.885323 / 0.577294). The V column
  also has exact CIR references (`verification.component_references.V`, E[V_T] = 0.048380), and
  there is the martingale invariant E[exp(kappa*T)(V_T - eta)] = 0.071.

### Per-step update (vectorised over paths; all three spots share the same increments)

By Ito, R = 1/(c_lv*S) satisfies dR = -sqrt(V) dW1 + (V/R) dt. This is a time-changed Bessel-3
process with additive noise in R. Barriers in R: a = 1/(c_lv*U) = 2/3 and b = 1/(c_lv*L) = 2.
Initial R0 = 1/(c_lv*S0) = 4/3, 1, 0.8 for S0 = 75, 100, 125.

Given already-scaled independent increments dZ1, dZ2:

1. `Vp = max(V_n, 0)`, `s = sqrt(Vp)`, `tau = Vp*dt`.
2. **V (full truncation, Lord et al.)**:
   `V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*s*(rho*dZ1 + sqrt(1-rho^2)*dZ2)`.
   Do NOT clip the stored state. Clip only inside the coefficients.
3. **R (drift-implicit Euler)**: solve `R_{n+1} = R_n - s*dZ1 + tau / R_{n+1}` for the positive
   root:
   `Y = R_n - s*dZ1;  R_{n+1} = 0.5*(Y + sqrt(Y^2 + 4*tau))`.
   This is always > 0, so S = 1/(c_lv*R) > 0 always and the positivity_S hard gate holds. The
   explicit step `R_n - s*dZ1 + tau/R_n` can go negative, which is exactly the failure that
   problem.md warns about. Implement the root in a cancellation-safe form when Y < 0:
   `R_{n+1} = 2*tau / (sqrt(Y^2 + 4*tau) - Y)`. If tau = 0 and Y <= 0, which cannot happen from
   R_n > 0 with s = 0, guard anyway with a tiny floor such as `np.maximum(R, 1e-300)` applied
   only inside 1/R.
   The scheme consumes ONLY the supplied dW and draws no auxiliary randomness, so the kernel's
   CRN order study sees a fully coupled path.
4. **Continuous monitoring (Brownian-bridge weight)**, per spot. If R_{n+1} <= a or
   R_{n+1} >= b, the weight becomes 0. Otherwise multiply by
   `(1 - exp(-2*(R_n - a)*(R_{n+1} - a)/tau)) * (1 - exp(-2*(b - R_n)*(b - R_{n+1})/tau))`.
   When `tau == 0`, the factor is 1. Guard the division. For frozen coefficients the bridge
   law given the endpoints does not depend on the drift, so the formula applies to this
   drifted step unchanged.
5. The functional is `mean(weight)` per spot. Its MC s.e. is `std(weight, ddof=1)/sqrt(M)`.
   Report the s.e. in Results (optionally also in the dict, e.g. `functionals_se`). Keep
   simulating S for the S0 = 100 spot after knock-out, because its terminal S is returned
   regardless.

## Implementation Notes

- **Continuous monitoring is mandatory.** Grid-only monitoring overstates survival by several
  percent at dt = 0.002, which is outside the 0.8% band. The product of single-barrier factors
  ignores the double-crossing series term, which is negligible at this dt.
- **Bias budget**: the drift-implicit step has O(dt) weak bias in the Bessel drift. Inside the
  corridor R is in [2/3, 2], so V/R is bounded and well-behaved there, and the bias should be
  far below 0.8%. If the evaluator reports a functional just outside the band, the
  likely culprit is the drift discretisation near the upper barrier (small R). One option is
  to apply the drift-implicit step on the trapezoid clock (Vp_n + Vp_{n+1})/2 * dt. That is a
  legitimate refinement but must keep using the given dt.
- **Feller fails**: use full truncation (V^+ inside sqrt and drift, V stored unclipped).
  `variance_nonnegative` is report-only. The drift-implicit sqrt-V scheme is NOT available for
  V (it needs 4*kappa*eta >= sigma^2). Do not sample V from its exact noncentral chi-square
  law, because that would not consume dW[..., 1].
- **Kernel dW hook**: when `dW` is supplied it has shape (num_paths, Nt, 2) and holds INDEPENDENT,
  already-scaled increments. Take Nt from its shape and set dZ1 = dW[..., 0],
  dZ2 = dW[..., 1]. When `dW` is None, draw it from `default_rng(seed)` path-major.
- **Observables hook**: accumulate a trapezoid `int_0^T phi(X_s) ds` per path for the S0 = 100
  state X = (S, V), stacked as (num_paths, 2). The Dynkin test functions are tanh(S/100), 1/S, V
  and V*exp(-S/100).
- **Memory**: a full (400000, 500, 2) float64 block is 3.2 GB. Process paths in chunks (e.g.
  25k to 50k), drawing each chunk's (chunk, Nt, 2) block sequentially from one generator. This
  sequential draw reproduces the single path-major block exactly, so the 100k run is a prefix of
  the 400k run. Alternatively, draw per time step. Either satisfies the symmetric path-count
  check.
- **Tails**: S_T has an infinite third moment, so the kernel's S-variance s.e. is unreliable.
  This is a property of the law and not a solver bug.
- **Expected magnitudes**: at rho = 0 the exact survivals are 0.9466 / 0.8853 / 0.5773. With
  rho = -0.36 the values will differ somewhat.
- **Sanity checks the solver should print**: E[V_T] against 0.048380, the three functionals with
  their s.e., min S_T > 0, and the 100k vs 400k difference against
  2*sqrt(se_400k^2 + se_100k^2).

## Results

Run: num_paths=400000, dt=0.002 (Nt=500), T=1, seed=42, 67 s (chunks of 25k paths).

- Empirical mean (S, V): [100.011529, 0.048073]
- Empirical variance (S, V), ddof=1: [752.0437, 0.0101385]
- E[V_T] = 0.048073 vs exact 0.048380 (full-truncation bias, ~2 s.e.)
- min S_T = 23.57 > 0
- Functionals (value, MC s.e.):
  - no_touch_S75  = 0.933832 (0.000390)
  - no_touch_S100 = 0.914332 (0.000439)
  - no_touch_S125 = 0.592065 (0.000769)
- 100k run: 0.934206 / 0.913724 / 0.591068; |diff| = 3.7e-4 / 6.1e-4 / 1.0e-3, all within
  2*sqrt(se_400k^2 + se_100k^2) = 1.7e-3 / 2.0e-3 / 3.4e-3; values differ.
- dW hook (shape (M, Nt, 2)) and observables hook both exercised and working.

Refinement pass (after evaluator feedback; solver.py unchanged):
- rho = -0.36 bias check by dt refinement (200k paths, seed 7, full truncation). The three
  functionals at dt = 0.002 are 0.93419 / 0.91603 / 0.59563. At dt = 0.0005 they are
  0.93330 / 0.91552 / 0.59559. The differences (<= 0.1% relative) are inside the MC s.e., far
  below the 0.8% band, so the discretisation bias at rho = -0.36 is small.
- Tried a reflecting V step (|V_{n+1}|), which is nonnegative by construction, to clear the
  report-only `variance_nonnegative` invariant. Rejected: E[V_T] = 0.0613 at dt = 0.002 and
  0.0570 at dt = 0.0005 against exact 0.0484, and the functionals shift by 1-4%. Full
  truncation is kept; the negative stored V is intrinsic to it and report-only.

## Evaluation

(Filled by evaluator)

<metrics>
score: 7
certified: false
provenance: self_convergence
estimated_rel_error: 0.0170
error_is_estimate: true
observed_order: 0.3167
order_floor: 0.5000
scheme_family: euler-maruyama
invariants_ok: false
constraints_ok: true
invariant_max_drift: 0.1846
invariant_worst_headroom: -7.2663
order_check: true
mc_se_rel: 0.0882
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: a0565256f96e62ff7ef6c52679ac9d024379ab829bba73dc61afb5c60459e530
solve_time_s: 61.4410
wall_time_s: 69.1200
</metrics>

<review score=7>

**Score: 7/10**

### Numerical Accuracy
- provenance:              self_convergence  (reference route: none; moment_ode, kolmogorov and stationary_density all unavailable, as the spec predicted)
- mean_rel_err:            not computable (no moment reference; `moments_ok` unavailable). Cross-check on V only: E[V_T] = 0.048073 +/- 2 x 0.000159 vs exact 0.048380 (rel 0.63%, 1.9 s.e.). The martingale invariant passes: drift 0.00405 against a bar of 0.0092
- variance_rel_err:        not computable (no moment reference). S variance 752.0 +/- 2 x 66.4 (heavy tailed, the s.e. is unreliable as the spec warned). V variance 0.010138 +/- 2 x 0.0000824
- mc_se_rel:               0.0882   resolved: false. This is the S-variance s.e. against a reference that does not exist, so it is a property of the heavy tail and not a solver defect
- observed_order:          0.317 +/- 2 x 0.018 (expected 0.5, two-sided band +/-0.4, |diff| = 0.18, pass; declared `scheme_family` so the band is two-sided). S component 0.320 +/- 0.016, V component 0.317 +/- 0.018 (median-based 0.497). Consistent with the Feller-violated CIR rate the spec anticipates, and far from the ~0 that ignoring `dW` would give. The solver consumes `dW` correctly
- weak_order:              indeterminate, at the noise floor for x, x^2 and tanh
- dynkin:                  not run (unavailable). The finest level needs 12.8 GB (400000 paths x 2000 steps x 2 channels), above the 12 GB budget. This is a property of the spec's path count and not of the solver. It is a skip and not a pass. It means no generator evidence exists, which is why this plan has no circularity break. `dynkin_replicated: false` because the outcome is a skip that more seeds cannot change, not because the plan was non-marginal
- terminal_law:            not declared
- shape:                   S skewness 19.8 +/- 13.7, excess kurtosis 3112 +/- 2489 (strict local martingale, infinite third moment, unreliable). V skewness 4.08 +/- 0.035, excess kurtosis 24.5 +/- 0.63
- constraints:             finite pass; shape pass; positivity_S pass (gate, 0 negative S, min S_T = 23.57); martingale pass; variance_nonnegative FAIL (non-gate, report-only): neg_fraction drift 0.1846, i.e. 18.5% of V_T are negative under full truncation. The most negative value was not reported by the kernel
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. The kernel states "Richardson-extrapolated moments unstable, or Dynkin failed", but neither happened. `richardson_stable` is true and `dynkin_ok` is unavailable. The row fired because `invariants_ok` is false, and the only entry that drifted is the spec's own report-only `variance_nonnegative`. Without it the plan would sit at the 8 ceiling for self-convergence. See the note to the conductor below

### Functionals (THE grade; the kernel does not score these)
Reference at rho = -0.36 is the 2-D backward Kolmogorov PDE (`functional_reference`). I did not build it, so the values below are **not graded against a reference**. Only the rho = 0 closed-form series was available.
- rho = -0.36 (graded run, 400k, dt 0.002): no_touch_S75 = 0.933832 +/- 2 x 0.000390; no_touch_S100 = 0.914332 +/- 2 x 0.000439; no_touch_S125 = 0.592065 +/- 2 x 0.000769. No reference comparison
- rho = 0 bias check (carried from the previous evaluation pass; `solver_sha256` is unchanged at a0565256..., so it still applies; not re-run this pass), done by setting the module constant `RHO = 0` and calling the solver unchanged at 400k paths, dt 0.002, seed 42, against the exact series:

  | functional | solver | 2 x s.e. | exact | rel err | z |
  |---|---|---|---|---|---|
  | no_touch_S75 | 0.947180 | 0.000702 | 0.946627 | +0.058% | 1.57 |
  | no_touch_S100 | 0.884786 | 0.001000 | 0.885323 | -0.061% | -1.07 |
  | no_touch_S125 | 0.576267 | 0.001543 | 0.577294 | -0.178% | -1.33 |

  All three are inside the 0.8% band and inside the MC interval. The implicit Bessel step and the bridge weight are therefore unbiased to within about 0.2% when there is no correlation. The cross term, which is the only place rho enters, is not covered by this check.
- path-count check (from the solver's own run, not re-run by me): 100k values differ from 400k and every |diff| is below 2 x sqrt(se^2 + se^2).

### Feedback for solver
- Nothing in `solver.py` is defective that I could find. `dW` is consumed exactly (`Nt = dW.shape[1]`, `dt = T/Nt`), no randomness is drawn on that path, S stays positive (0 negatives), and V is clipped only inside the coefficients (full truncation), not as a state statement. `SOLUTION.md` and `solver.py` describe the same scheme, so no `agent_cap` applies.
- The score of 7 is mostly the kernel's reading of the ungated `variance_nonnegative` entry (18.5% of V_T below 0, which is intrinsic to full truncation with Feller delta = 0.444). Do not "fix" it by clipping the stored V, because the leakage scan would flag that. If you want that invariant to pass legitimately, you would need a scheme whose stored V is nonnegative by construction. That means a reflecting or absorbing rule in V, an exact CIR transition (which does not consume `dW[..., 1]` and would read strong order about 0 on V), or a Alfonsi-type second-order scheme. All of these change the V statistics and carry their own risk, so it is a judgement call and not a required change.
- Remaining accuracy risk is at rho = -0.36 only. The bias check at rho = 0 is clean, so if the 2-D Kolmogorov reference later disagrees by more than 0.8%, look at the cross-correlation of the V increment (`RHO*dz1 + rho_c*dz2` is used) and at the bridge weight using `tau = V_n*dt` while V moves within the step. The trapezoid-clock refinement your notes mention is the first thing to try.
- A hard-to-see edge in `inside`: the weight requires `R` and `Rn` both strictly inside (a, b), which is correct. Keep it.

### Notes for the conductor (not the solver)
- Dynkin is unavailable because 400000 x 2000 x 2 doubles is 12.8 GB. Fewer paths for Dynkin would widen the interval and is disallowed by the kernel. A lower `num_paths` or `dynkin_levels` in the spec, or a larger memory budget, is the only way to get the 9.
- Possible formulator/spec defect: `variance_nonnegative` is declared `gate: false` and "REPORTED ONLY", yet its failure sets `invariants_ok: false` and appears to bind the rubric at 7 with the misleading reason "Richardson unstable or Dynkin failed". A plan following the spec's own endorsed scheme (full truncation) is penalised for it. The spec might declare the entry as a diagnostic without a tolerance instead.
- The three graded functionals are not evaluated by the kernel. A rho = -0.36 Kolmogorov reference run (Rannacher + ADI, truncation guard v_max 4 vs 6) is still needed before this plan can be called passing on the actual grade.

</review>
