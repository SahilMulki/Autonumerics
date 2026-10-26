---
id: 1
plan_slug: euler-maruyama-bessel
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Full-truncation Euler-Maruyama for V; exact Bessel-3 step in R = 1/(c_lv*S) on the frozen clock V_n^+*dt (driven by -dW1 plus two auxiliary Gaussians), with a per-step Brownian-bridge survival weight for both barriers, run with common random numbers for the three spots.
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

- **Method**: Euler-Maruyama family. Full-truncation EM for V, and an exact-given-clock Bessel-3
  step for the transformed asset R = 1/(c_lv*S).
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

Barriers in R: a = 1/(c_lv*U) = 2/3 and b = 1/(c_lv*L) = 2. Initial R0 = 1/(c_lv*S0) = 4/3, 1,
0.8 for S0 = 75, 100, 125.

Given Z1_n, Z2_n with sqrt(dt) scaling already applied (dZ1, dZ2):

1. `Vp = max(V_n, 0)`, `s = sqrt(Vp)`, `tau = Vp*dt` (frozen clock increment).
2. **V (full truncation, Lord et al.)**:
   `V_{n+1} = V_n + kappa*(eta - Vp)*dt + sigma*s*(rho*dZ1 + sqrt(1-rho^2)*dZ2)`.
   Do NOT clip the stored state. Clip only inside the coefficients.
3. **R (Bessel-3, exact given tau)**: dR = -sqrt(V) dW1 + (V/R) dt is BES(3) on the clock
   int V dt. Then
   `R_{n+1} = sqrt( (R_n - s*dZ1)^2 + tau*(G2^2 + G3^2) )`,
   where G2 and G3 are auxiliary iid N(0,1) draws from the same `default_rng(seed)` (drawn after
   the dW block for each path chunk). This value is always > 0, so S = 1/(c_lv*R) > 0 always,
   which satisfies the positivity_S hard gate. Use the same G2 and G3 for all three spots.
4. **Continuous monitoring (Brownian-bridge weight)**, per spot. If R_{n+1} <= a or
   R_{n+1} >= b, the weight becomes 0. Otherwise multiply the survival weight by
   `(1 - exp(-2*(R_n - a)*(R_{n+1} - a)/tau)) * (1 - exp(-2*(b - R_n)*(b - R_{n+1})/tau))`.
   When `tau == 0` (V clipped to 0), the factor is 1 because there is no motion. Guard the
   division with `np.where` or `errstate`. Once a weight hits 0 it stays at 0. Do not stop
   simulating S for the S0 = 100 spot, because its terminal S is returned regardless of
   knock-out.
5. The functional is `mean(weight)` per spot. Its MC s.e. is `std(weight, ddof=1)/sqrt(M)`. Report
   the s.e. in the Results section (it is optional in the dict, e.g. under `functionals_se`).

The weight-product estimator is smoother than sampling a uniform per step: it has lower
variance and is a deterministic function of the increments.

## Implementation Notes

- **Continuous monitoring is mandatory.** Grid-only monitoring overstates survival by several
  percent at dt = 0.002, which is outside the 0.8% band. The bridge formula is exact for frozen
  coefficients because R has additive diffusion -sqrt(V) dW1. The Bessel drift does not change
  the bridge law given the endpoints. The product of single-barrier factors ignores the
  double-crossing series term, which is negligible at this dt.
- **Feller fails**: use full truncation (V^+ inside sqrt and drift, V stored unclipped).
  `variance_nonnegative` is report-only, so slightly negative stored V is acceptable. The
  drift-implicit sqrt-V scheme is not available (it needs 4*kappa*eta >= sigma^2).
  Do not sample V from the exact noncentral chi-square law, because that would not consume
  dW[..., 1] and would zero the CRN order study.
- **Kernel dW hook**: when `dW` is supplied it has shape (num_paths, Nt, 2) and holds INDEPENDENT,
  already-scaled increments. Take Nt from its shape and set dZ1 = dW[..., 0], dZ2 = dW[..., 1].
  The auxiliary Bessel Gaussians G2 and G3 still come from `default_rng(seed)`. Note the
  consequence for the CRN order study: the G2/G3 contribution to R is an O(dt)-per-step
  fluctuation, uncoupled across levels, so the S column's strong order should still read about
  0.5 and not 0.
- **Observables hook**: accumulate a trapezoid `int_0^T phi(X_s) ds` per path for the S0 = 100
  state X = (S, V), stacked as (num_paths, 2). The Dynkin test functions are tanh(S/100), 1/S, V
  and V*exp(-S/100). Do not include phi = S: S is a strict local martingale.
- **Memory**: a full (400000, 500, 2) float64 block is 3.2 GB. Loop over path chunks (e.g. 25k
  to 50k paths). Draw each chunk's (chunk, Nt, 2) dW block sequentially from one generator, so
  that the dW stream stays path-major. Then draw that chunk's (chunk, Nt, 2) auxiliary Gaussians.
  Alternatively, loop over time with per-step draws. Either is fine, and the path-count bar does
  not require the 100k run to be a prefix of the 400k run. The time loop is 500 steps with 3
  spots on vectorised arrays.
- **Tails**: S_T has an infinite third moment, so the kernel's S-variance s.e. is unreliable.
  This is a property of the law and not a solver bug.
- **Expected magnitudes**: at rho = 0 the exact survivals are 0.9466 / 0.8853 / 0.5773. With
  rho = -0.36 the values will differ somewhat. A value far from these, e.g. survival > 0.95 at
  S0 = 125, usually means grid-only monitoring or a sign error in the correlation.
- **Sanity checks the solver should print**: E[V_T] against 0.048380, the three functionals with
  their s.e., min S_T > 0, and the 100k vs 400k difference against
  2*sqrt(se_400k^2 + se_100k^2).

## Results

Run: num_paths = 400000, dt = 0.002 (Nt = 500), T = 1, seed = 42 (about 45 s per 400k run).

- empirical_mean (S, V): [100.0150, 0.047916]
- empirical_variance (S, V): [734.82, 0.010122]
- E[V_T] check: exact 0.048380, simulated 0.047916 (about 1% low, about 2.9 s.e.; full-truncation bias)
- min S_T = 24.22 > 0

| Functional | 400k | s.e. | 100k | s.e. | abs diff | 2*sqrt(se^2+se^2) |
|---|---|---|---|---|---|---|
| no_touch_S75 | 0.933916 | 3.90e-4 | 0.933544 | 7.82e-4 | 3.72e-4 | 1.75e-3 |
| no_touch_S100 | 0.914639 | 4.38e-4 | 0.913120 | 8.84e-4 | 1.52e-3 | 1.97e-3 |
| no_touch_S125 | 0.592359 | 7.69e-4 | 0.592061 | 1.54e-3 | 2.98e-4 | 3.44e-3 |

The 100k and 400k values differ and agree within the bar. The `dW` hook (shape (N, Nt, 2)) and the `observables` hook both run. The functionals have not been compared to the PDE reference; that is the evaluator's job.

Solver-side validation added in the refinement pass (200k paths, seed 7, solver.py unchanged):

| Check | no_touch_S75 | no_touch_S100 | no_touch_S125 |
|---|---|---|---|
| rho = 0, dt = 0.002 | 0.94662 | 0.88541 | 0.57757 |
| rho = 0, dt = 0.0005 | 0.94645 | 0.88479 | 0.57851 |
| rho = 0 exact series | 0.946627 | 0.885323 | 0.577294 |
| rho = -0.36, dt = 0.002 | 0.93372 | 0.91481 | 0.59299 |
| rho = -0.36, dt = 0.0005 | 0.93385 | 0.91482 | 0.59377 |

At rho = 0 the solver reproduces the exact series to within about 0.1% at dt = 0.002 (within MC s.e.).
The dt = 0.002 vs dt = 0.0005 shift at rho = -0.36 is at most 0.13% (S125), well inside the 0.8% band, so
discretization bias is not the limiting factor. The V mean bias (about 1% low at dt = 0.002) shrinks
with dt as expected for full truncation.

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
invariant_max_drift: 0.1857
invariant_worst_headroom: -7.2688
order_check: true
mc_se_rel: 0.0848
resolved: false
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 7806bfab683e7f222965119663b1aa8d77539e8367e41c0dd44bbe4bbb5a7440
solve_time_s: 77.3790
wall_time_s: 86.1000
</metrics>

<review score=7>

**Score: 7/10**

### Numerical Accuracy
- provenance:              self_convergence  (reference route: none; moment_ode, kolmogorov and stationary_density were all attempted for the S column and unavailable)
- mean_rel_err:            not measured (moments_ok unavailable, no S reference). Observed S mean 100.015 +/- 2 x 0.0429 (E[S_T] < S0 is expected, S is a strict local martingale). V mean 0.047916 +/- 2 x 0.000159 against the exact CIR 0.048380: 0.96% low, 2.9 s.e., the full-truncation O(dt) bias (inside the spec's martingale tolerance)
- variance_rel_err:        not measured. S variance 734.8 +/- 2 x 62.3; V variance 0.010122 +/- 2 x 8.2e-5
- mc_se_rel:               0.0848 (S variance, heavy tail)   resolved: unavailable (no reference variance to resolve against; this is a skip, not an MC-inconclusive verdict)
- observed_order:          0.317 +/- 0.018 on the last component (V); S column 0.337 +/- 0.015; expected 0.5, two-sided band +/-0.4, passes. Medians 0.497 (V) and 0.352 (S), so the mean-based figure is pulled down by tails; Feller fails here (2 kappa eta = 0.222 < sigma^2 = 1), which is the expected cause. The dW hook is consumed
- weak_order:              indeterminate, at the noise floor
- dynkin:                  unavailable (not a failure; dynkin_replicated: false, because no verdict exists for more seeds to replicate). The finest ladder level needs a 12.8 GB increment array (400000 paths x 2000 steps x 2) against the 12 GB budget; the kernel will not run it at fewer paths. A property of the spec's path count, not of this solver
- terminal_law:            not declared
- shape:                   S skewness 18.4 +/- 13.0, excess kurtosis 2872 +/- 2330 (S has an infinite third moment, so these are not usable); V skewness 4.08 +/- 0.037, excess kurtosis 24.5 +/- 0.67
- constraints:             finite, shape, positivity_S (all gated) pass; martingale (non-gate) passes, drift 0.0061 vs bar 0.0092; variance_nonnegative (non-gate, reported only) FAILS, drift 0.1857 vs bar 1e-8. This is full truncation storing slightly negative V by design
- functionals:             NOT GRADED BY THE KERNEL. The three gated no-touch probabilities are the actual grade and the kernel does not read them. The solver reports 0.933916 (+/- 2 x 3.9e-4), 0.914639 (+/- 2 x 4.4e-4), 0.592359 (+/- 2 x 7.7e-4) for S0 = 75 / 100 / 125. The 100k vs 400k path-count check passes for all three (diffs 3.7e-4, 1.5e-3, 3.0e-4 against bars 1.75e-3, 1.97e-3, 3.44e-3) and the values differ. I did not run the 2-D backward Kolmogorov PDE reference, because that needs numerical code outside this role. Accuracy against the 0.8% band is therefore unverified. The only independent anchor on hand is the rho = 0 exact series (0.9466 / 0.8853 / 0.5773), which the plan has not been run against
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric. The printed reason, "Richardson-extrapolated moments are unstable, or Dynkin failed", is generic and does not match this run: richardson_stable is true and dynkin_ok is unavailable, not failed. The binding facts are invariants_ok false (the non-gate variance_nonnegative) with no reference and no Dynkin verdict. Certification caps at 8 in any case (self-convergence, no circularity break)

### Feedback for solver
- Nothing in the code needs fixing. The source matches SOLUTION.md (full-truncation EM for V, Bessel-3 step for R, bridge weights, CRN across spots). It consumes `dW` of shape (N, Nt, 2) and takes `Nt` from it. S is positive (min S_T = 24.2). V is not clipped after the update; the clip sits inside the coefficients.
- The order of 0.32 on V is not a scheme defect. Feller fails and the strong order of full-truncation EM is degraded there. The median pathwise order is 0.497.
- `variance_nonnegative` failing is the intended behaviour of full truncation and is non-gate. It is the only thing holding `invariants_ok` false and costing the plan the row above 7. Do not clip V after the update to remove it. The spec asks you to leave it alone.
- One real accuracy risk: the Bessel step draws G2/G3 from `default_rng(seed)` even when `dW` is supplied. In the CRN ladder those auxiliary draws are the same across levels while `Nt` changes, so the levels are not coupled on that part of the noise. The ladder still read about 0.3-0.5, so it did not break here.
- The product of single-barrier bridge factors drops the double-crossing term. With the barriers at R = 2/3 and 2 and tau = V dt of order 1e-4 this should be negligible. Check it against the PDE reference rather than assuming it.
- The score cannot rise by changing the solver alone. To lift it, the three functionals need grading against the PDE reference (the conductor or formulator step), and Dynkin needs a spec `num_paths` or dynkin budget that fits in 12 GB.

### Spec and formulator notes (not solver defects)
- The kernel never grades the three gated functionals, which are the stated grade. The spec's own `functionals_note` says so. Until something computes the PDE reference, a 7 here says nothing about the 0.8% criterion.
- A non-gate, "REPORTED ONLY" constraint still sets `invariants_ok: false` and holds the score at 7. The spec flags the intent, but the kernel counts it.
- The reference route that matters (Kolmogorov PDE) was declared as `functional_reference`, not as `verification.kolmogorov`, so the kernel could not use it. That is why `surrogate_ok` is unavailable.

</review>
