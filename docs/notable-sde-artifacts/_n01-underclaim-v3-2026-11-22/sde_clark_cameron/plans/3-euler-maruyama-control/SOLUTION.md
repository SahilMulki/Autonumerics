---
id: 3
plan_slug: euler-maruyama-control
scheme: euler-maruyama
scheme_family: euler-maruyama
strategy: Increment-only Euler-Maruyama at the contract step, as the Clark-Cameron control that exhibits the sqrt(dt) strong-order lower bound and is expected to MISS the 2% pathwise gate.
---

**This plan is the control and is expected to fail the problem's accuracy gate.** It is in
the pool to measure the Clark-Cameron (1980) lower bound, not to be shipped: an
increment-only scheme cannot clear the 2% pathwise strong-error gate at `dt = 0.04`
(measured in the spec: 8.9e-02 relative strong error in U). Rank it last.

## SDE Reference

Two-dimensional Ito system driven by **two independent** Brownian motions W1, W2:

```
dS = mu dt + dW2,     S(0) = S_0 = 1.0
dU = S dW1,           U(0) = U_0 = 0.0
```

State vector is `(S, U)` in that column order. Diffusion matrix against the
independent channels `(dW1, dW2)`:

```
G(S, U) = [[0, 1],
           [S, 0]]
```

so column 0 multiplies `dW[:, :, 0] = dW1` and column 1 multiplies `dW[:, :, 1] = dW2`.

Parameters: `mu = 1.0`, `S_0 = 1.0`, `U_0 = 0.0`, `T0 = 0.0`, `T = 1.0`.

Exact moments at `t` (component X = S, component Y = U):

```python
mean_S = S_0 + mu*t
var_S  = t
mean_U = 0.0
var_U  = ((S_0 + mu*t)**3 - S_0**3)/(3*mu) + t**2/2     # = 17/6 = 2.8333333333 at t = 1
cov(S_T, U_T) = 0.0 exactly
```

## Numerical Scheme

- **Method**: Euler-Maruyama, vector form `X_{k+1} = X_k + f(X_k) dt + G(X_k) dW_k`, using
  the step increments only
- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 8000 (`evaluation_thresholds.num_paths`); `problem.md` grades the single
  contract call at `num_paths = 1000`, `dt = 0.04`, `T = 1`, `seed = 42`
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: not applicable — `milstein_eligible: false`,
  `milstein_ineligible_reason: levy_area`. The only non-zero state derivative of G is
  `dG[1][0]/dS = 1`, so the second-order correction is purely the mixed iterated integral
  `I_21`, i.e. the Levy area of the channel pair, which the step increments do not carry.
  This plan deliberately omits it.
- **Expected strong order**: 0.5 (`verification.expected_strong_order`) — and the
  Clark-Cameron bound is *attained* here, not merely a bound
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (exact, above), plus the harness's own
  pathwise reference built on the 256x sub-grid from the same Brownian path

### The step

```python
S_next = S + mu*dt + dW[:, k, 1]        # exact: constant coefficients
U_next = U + S * dW[:, k, 0]            # omits I_21 -> O(dt) per step, O(sqrt(dt)) in L2
```

## Implementation Notes

- **Expected outcome.** S is reproduced to ~8e-13 (its update is exact), so the whole gate
  is a statement about U; the omitted `I_21` leaves a measured 8.9e-02 relative strong error,
  about 4.5x over the 2% gate. The *moment* checks should still pass: `I_21` has mean zero
  and the omission costs U's variance about 4% (the kernel measured graded 2.800 against a
  Richardson-extrapolated 2.913, i.e. 3.9% or 0.387 of the 10% tolerance; an earlier
  version of this note said 0.7% from `Nt * dt^2/2 = 0.02`, which was optimistic),
  still inside the 10% variance tolerance. That contrast — moments clean, pathwise error
  4.5x over — is the evidence this plan exists to produce.
- **Bias instruction (refine cycle 2): the increment-only U sub-step is the floor.** The
  measured variance bias is the omitted Levy-area term `I_21` in U's step. Removing or
  reducing it requires `dW_sub`, which is exactly the Levy-area correction carried by
  plans 1 and 2; adding it here would make this plan a duplicate and destroy its value as
  the pool's measured Clark-Cameron lower bound. No bias reduction exists that uses only
  the step increments and stays at strong order 0.5. `scheme_family` stays
  `euler-maruyama` (unchanged); `solver.py` is unchanged.
- **Channel convention is the single highest-risk detail.** `dW[:, :, 0] = dW1` (drives U),
  `dW[:, :, 1] = dW2` (drives S). A swap is invisible to every moment, invariant and Dynkin
  check because W1 and W2 are i.i.d., but gives ~100% pathwise strong error. Self-check in
  `__main__`: `S_T` must equal `S_0 + mu*T + dW[:, :, 1].sum(axis=1)` to rounding (~1e-13).
  Without this check a swapped control is indistinguishable from an honest one.
- `dW_sub` must be accepted in the signature (`problem.md` always supplies it) and is simply
  **not used** by this scheme — that is the point of the control. `dW_sub` may be `None` when
  the solver is run standalone; `dW` may be too, in which case draw it with
  `np.random.default_rng(seed)`. When `dW` is supplied, use it exactly and draw no
  randomness.
- Contract: `Nt = max(1, round(T/dt))` then `dt = T/Nt`; take `Nt = dW.shape[1]`; advance the
  state exactly once per step of `dW`; `terminal_paths` is `(num_paths, 2)` in column order
  `(S, U)`; `empirical_variance` uses `ddof=1`; numpy and the standard library only; no work
  at module load outside `if __name__ == "__main__"`.
- **Do not declare `exact_sampling_components`.** Both components are stepped with the
  supplied increments; nothing draws auxiliary randomness. S is exactly integrated, which the
  kernel excuses automatically.
- **No variance reduction.** The problem declares no payoff or functional; it is graded on a
  pathwise strong error, where antithetics and control variates change nothing.
- **No declared barrier or absorbing boundary**, so no `boundary_treatment` applies.
- Both coefficients are globally Lipschitz here, so no correct scheme produces Inf/NaN
  (gated constraint `finite`). U is a true martingale: `E[U_T] = 0` exactly at every `dt`
  (ungated invariant `martingale`), and Euler-Maruyama preserves that exactly.

## Results

Run: `num_paths=8000, dt=0.04, T=1, seed=42` (own draw, shape (8000, 25, 2)).

- Empirical mean (S, U): [2.005133, -0.008320]  (exact [2, 0])
- Empirical variance (S, U): [0.999024, 2.786294]  (exact [1, 2.8333])

Hooks confirmed: explicit `dW` (S equals S_0 + mu*T + sum dW2 to 2.7e-15, so channel mapping is correct), `observables` path integrals, `dW_sub` accepted and unused. Expected to miss the 2% pathwise strong gate (control plan).


## Evaluation

(Filled by evaluator)

<metrics>
kind: sde
score: 5
certified: false
provenance: analytic
estimated_rel_error: 0.0200
error_is_estimate: false
observed_order: 0.4965
observed_order_se: 0.0348
order_floor: 0.5000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 0.5000
scheme_family: euler-maruyama
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 3.192e-03
invariant_worst_headroom: 0.4959
reference_outcome: validated
mean_rel_err: 1.548e-03
variance_rel_err: 0.0200
moment_bias_binding: variance
moment_bias_frac_tol: 0.3874
moment_bias_actionable: true
order_check: true
mc_se_rel: 0.0207
resolved: true
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
agent_cap: {"reason": "Pool control, declared EM at strong order 0.5. The problem states a 2% pathwise strong-error gate; the kernel does not measure it, but its own CRN ladder shows the dt=0.04 vs dt=0.02 pathwise difference in U is 0.0788 against terminal scale 1.246 (6.3%, 3x the gate) and the formulator measured 8.9e-2 (4.5x). Order 0.4965 clearing its own floor of 0.5 is conformance to a low design rate, not accuracy. Moment-only 9 would overclaim.", "score": 5}
solver_sha256: 78fc58a63d9546682ef70d4fbc9af71b8ca09ab29ba70decac36c230cd4960d1
solve_time_s: 1.0480
wall_time_s: 2.3300
cost_source: cpu
</metrics>

<review score=5>

**Score: 5/10**

Kernel score before the agent cap: **9** (bound by the rubric: Dynkin unavailable; the `strong_gate_not_measured` cap is also 9). Final **5** is a recorded downward `agent_cap`, explained under "Why capped".

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, reference validated)
- mean_rel_err:            S 1.5e-03 (2.0031 vs 2.0, se 0.0111); U absolute 0.0436 vs 0 (se 0.0186, family-wise x2.53, tol 0.05) -- passes
- variance_rel_err:        S 0.0121 (se 0.0157); U 0.0200 (2.7767 vs 17/6, se 0.0574, x2.53) -- passes the 10% tolerance
- mc_se_rel:               0.0207   resolved: true (CI widths 0.031 / 0.041 against bar 0.05)
- functional_rel_error:    not applicable, no functional declared
- observed_order:          0.4965 +/- 0.0348 on U (median-based 0.438), against this plan's own declared expected_strong_order 0.5, one-sided band. S is integrated exactly (abstained, roundoff). This is conformance to a deliberately low design rate; the plan reads at its rate, the two Levy-area plans read 0.93 and 0.98 against a design rate of 1.0
- weak_order:              0.83 on one usable observable (x2), indicative only
- dynkin:                  unavailable (still not run; see below)
- terminal_law:            not declared
- shape:                   U skewness 0.113 +/- 0.055, excess kurtosis 1.422 +/- 0.165 (spec derives 1.444, consistent)
- constraints:             finite: pass (gate); estimator_stability: pass (drift 0.0032 vs bar 0.01). martingale invariant no longer declared
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                agent_cap (5) -- kernel score 9 from rubric (Dynkin unavailable) and strong_gate_not_measured

### Dynkin: still unavailable, and the cause is now a spec-placement defect (formulator)
The numeric `initial_condition.value: [1.0, 0.0]` did not change the outcome. The kernel reads `spec["initial_condition"]` at the **top level** (`verifylib/kernel/sde/dynkin.py`, `initial_moments`), and the spec carries it under `verification.initial_condition`. The reason string is unchanged: it still lists `S_0, U_0` as unmatched. Fix: move the `initial_condition` object to the top level of `problem_spec.json` (or add `X_0`/`Y_0` to `parameters`). This is not a solver defect and not a Dynkin failure. Until it runs, every plan in this pool is capped at 9 by the rubric.

### What the kernel did not measure, and why the 5
- **The pathwise strong-error gate (`strong_rel_err_max` 0.02) is unmeasured by the kernel** (`strong_rel_err: null`, `strong_gate_not_measured` cap 9). It is the problem's only stated accuracy criterion, and this plan is known to fail it.
- Kernel-visible evidence that it fails: the CRN ladder's pathwise difference in U between dt = 0.04 and dt = 0.02 is 0.0788 (se 0.0014) against a terminal scale of 1.246, i.e. 6.3%, about 3x the 2% gate from one level difference alone. The formulator's independent measurement is 8.9e-2 relative strong error, 4.5x over.
- **A clean order reading is not evidence of accuracy here.** Order 0.4965 clearing the floor 0.5 says the scheme converges at the rate Euler-Maruyama is designed to, which is the Clark-Cameron sqrt(dt) bound. It inverts the true ranking against the Levy-area plans, which read slightly under their design rate but sit inside the gate.
- Moments passing (and the 9) are weak-error evidence. Euler-Maruyama and the Levy-area schemes share weak order, so moment space cannot separate them; that is exactly the axis this problem's gate exists to test.

### Why capped
The 9 would tie or exceed a plan that is inside the gate while this one is known to be about 3x to 4.5x outside it. I judged that writing it up at 9 would overclaim, so the score is capped at 5. If the kernel score for the Levy-area plans is also 9, the ranking by score alone cannot separate them from this control; only a pathwise strong-error measurement -- which this pipeline cannot make, since it has no pathwise reference -- can.

### Other findings
- The invariant removal lifted the previous bind of 5 as stated; it is not what produced this 5.
- `richardson_stable: false` comes from the mean statistic (target 0, shift within noise, 1.07 se). The variance statistic is stable and shows a real 3.9% variance bias (0.387 of tolerance, 7.4 se), consistent with the corrected Implementation Notes.

### Feedback for solver
- No code change is requested for the kernel-visible checks: `dW` is consumed exactly, `Nt = dW.shape[1]`, `dt = T/Nt`, `ddof=1`, channels mapped correctly. The decision to keep the increment-only step as the control is accepted.
- State plainly in REPORT.md that this plan is expected to miss the 2% pathwise gate (about 6-9% relative strong error in U at dt = 0.04) and that moment agreement does not contradict that.
- Do not cite `orders_ok: true` or the variance 2.0% error as evidence of accuracy on the graded criterion.

</review>
