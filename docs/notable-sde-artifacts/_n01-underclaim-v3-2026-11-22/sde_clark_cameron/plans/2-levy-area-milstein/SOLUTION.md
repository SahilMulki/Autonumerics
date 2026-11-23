---
id: 2
plan_slug: levy-area-milstein
scheme: milstein-levy-area
scheme_family: milstein
levy_area_treatment: sub-increments
strategy: Step at dt with the exact S update and the canonical Clark-Cameron order-1 U update, S_k dW1_k plus the noise-noise iterated integral I_21 assembled as a left-point Ito sum over the supplied sub-increments.
---

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

`U_T` is **not** Gaussian: conditionally on the W2 path it is `N(0, int_0^T S^2 ds)`,
a Gaussian variance mixture with excess kurtosis 1.4444.

## Numerical Scheme

- **Method**: order-1.0 Milstein-type step whose correction is the mixed iterated integral
  `I_21` (the Levy area of the channel pair), assembled from the supplied sub-increments
- **dt**: 0.04 (the contract step; `Nt = dW.shape[1] = 25`, then `dt = T / Nt`)
- **num_paths**: 8000 (`evaluation_thresholds.num_paths`); `problem.md` grades the single
  contract call at `num_paths = 1000`, `dt = 0.04`, `T = 1`, `seed = 42`
- **T**: 1.0
- **seed**: 42
- **Expected strong order**: 1.0 (`verification.expected_strong_order_milstein`)
- **Expected weak order**: 1.0
- **Reference for scoring**: analytic moments (exact, above), plus the harness's own
  pathwise reference built on the 256x sub-grid from the same Brownian path

### The step

`milstein_correction` from the spec, verbatim in effect:

```
S_{k+1} = S_k + mu*dt + dW2_k                      (exact; constant coefficients)
U_{k+1} = U_k + S_k * dW1_k + I_21,k
I_21,k  = int_{t_k}^{t_k+dt} (W2(s) - W2(t_k)) dW1(s)
```

`I_21` has mean zero and variance `dt^2/2`, and is **not** a function of the step
increments — which is the Clark-Cameron (1980) obstruction: any scheme driven by `dW`
alone carries an `O(dt)` per-step error in U that accumulates to `O(sqrt(dt))` in strong
L2, whatever its design. It is recoverable from `dW_sub` by the **left-point (Ito)**
Riemann sum:

```python
sub       = dW_sub[:, k, :, :]                      # (num_paths, 256, 2)
W2_before = np.concatenate(
    [np.zeros((num_paths, 1)), np.cumsum(sub[:, :, 1], axis=1)[:, :-1]], axis=1)
I21 = np.sum(W2_before * sub[:, :, 0], axis=1)
U   = U + S * dW[:, k, 0] + I21
```

Measured in the spec, independently of any plan (4000 paths, `dt = 0.04`, reference on the
256x sub-grid): this scheme leaves a **1.4e-02** relative strong error, inside the 2% gate;
plain Euler-Maruyama leaves 8.9e-02 and cannot clear it. Both reproduce S to 8e-13.

## Implementation Notes

- **Channel convention is the single highest-risk detail.** `dW[:, :, 0] = dW1` (drives U),
  `dW[:, :, 1] = dW2` (drives S). A swap is invisible to every moment, invariant and Dynkin
  check because W1 and W2 are i.i.d., but gives ~100% pathwise strong error. Self-check in
  `__main__`: `S_T` must equal `S_0 + mu*T + dW[:, :, 1].sum(axis=1)` to rounding (~1e-13).
- **Left-point convention matters.** A midpoint or trapezoid sub-sum yields the
  *Stratonovich* iterated integral and leaves a non-vanishing `0.5*dt` bias in U.
- The residual error against the harness's sub-grid reference is exactly the omitted
  drift-noise iterated integral `mu * int (s - t_k) dW1(s)`, whose per-step variance is
  `dt^3/3`; over `Nt = 25` steps, normalised by `sqrt(17/6) = 1.683`, that predicts
  1.37e-02 and matches the spec's measured 1.4e-02. Do **not** assemble that term in full
  from `dW_sub`: for this linear system it would reproduce the sub-grid reference exactly
  and make the terminal states `dt`-invariant, which `problem.md` requirement R23
  **excludes** from the comparison rather than scoring.
- `dW_sub` may be `None` when the solver is run standalone; in that case draw the sub-grid
  first with `np.random.default_rng(seed)` and aggregate it to form `dW`, so the two are
  consistent. When both are supplied, use them exactly and draw no randomness.
- Contract: `Nt = max(1, round(T/dt))` then `dt = T/Nt`; take `Nt = dW.shape[1]`; advance the
  state exactly once per step of `dW`; `terminal_paths` is `(num_paths, 2)` in column order
  `(S, U)`; `empirical_variance` uses `ddof=1`; numpy and the standard library only; no work
  at module load outside `if __name__ == "__main__"`.
- **Do not declare `exact_sampling_components`.** Both components are stepped with the
  supplied increments: S consumes `dW2`, U consumes `dW1` and the sub-increments that sum to
  it. Nothing draws auxiliary randomness. S is exactly integrated, which the kernel excuses
  automatically; and because this problem is graded *pathwise*, drawing U from its
  conditional Gaussian law would decorrelate from the reference path and give ~100% strong
  error, so exact sampling and conditional Monte Carlo are wrong answers here, not options.
- **No variance reduction.** The problem declares no payoff or functional; it is graded on a
  pathwise strong error, where antithetics and control variates change nothing.
- **No declared barrier or absorbing boundary**, so no `boundary_treatment` applies.
- Memory: `dW_sub` is `(num_paths, Nt, 256, 2)` float64 — 820 MB at 8000 paths and `Nt = 25`,
  3.3 GB at the `dt/4` ladder level. `conv_num_paths = 2000` is set for the CRN strong-order
  and Dynkin ladders. Build `I_21` step by step (slice `dW_sub[:, k]`) rather than
  materialising a cumulative sum over the whole array. If the Dynkin ladder refuses on the
  memory budget it reports `unavailable` and caps the score at 9; that is a correct refusal,
  not a solver defect.
- Both coefficients are globally Lipschitz here, so no correct scheme produces Inf/NaN
  (gated constraint `finite`). U is a true martingale: `E[U_T] = 0` exactly at every `dt`
  (ungated invariant `martingale`).

## Results

Standalone run (`solver.py`, num_paths=8000, dt=0.04, T=1, seed=42, self-drawn 256x sub-grid):

- empirical_mean = [2.00310, 0.04291]  (exact [2, 0])
- empirical_variance = [0.98793, 2.80541]  (exact [1, 2.8333])
- Self-check: S_T equals S_0 + mu*T + sum(dW2) to 2e-15 (channel convention correct).
- Explicit-dW + dW_sub, dW-only fallback (I21 = 0.5*dW1*dW2) and observables hooks all run.

### Refine cycle 2: answer to the bias instruction (no code change; `solver.py` and `scheme_family: milstein` unchanged)

Verdict: **the current U sub-step is the floor for this plan.** Which sub-step carries the variance bias: the term `S_k dW1_k`. Its left-point weight is the source, and it is not the dt^3/3 strong-error residual that the Implementation Notes cite.

- Exact `Var(U_T) = int E[S^2] ds = 17/6`. The scheme gives `sum_k E[S_k^2] dt + Nt*dt^2/2`. With `f(t) = (1+t)^2 + t`, the left Riemann sum falls short by about `(dt/2)(f(1)-f(0)) = 0.02*4 = 0.08`. The `I_21` variance adds back `Nt*dt^2/2 = 0.02`. Net deficit is about 0.06, or 2.1% of 17/6. That matches the kernel's 0.2131 of the 10% tolerance.
- Equivalently, it is the cross-covariance `2*E[S_k]*mu*Cov(dW1, int (s-t_k)dW1) = mu*E[S_k]*dt^2` per step, which is missing from the scheme. The omitted drift-noise integral's own variance (`Nt*dt^3/3 = 5e-4`) is negligible here.
- The only part of the omitted integral that correlates with `dW1_k` and `S_k` is its conditional mean `E[I_01|dW1] = 0.5*dt*dW1`. Adding it gives the weight `S_k + 0.5*mu*dt` and removes essentially all of the bias. That is exactly plan 1's scheme, and it would make this plan a copy of it.
- The remaining option is the full integral assembled from `dW_sub`. It reproduces the sub-grid reference to rounding and makes the terminal state `dt`-invariant, so R23 excludes the run.
- I found no third option. Assembling a coarse partial quadrature of the integral (a fixed number of sub-blocks) only recovers a fraction of the needed covariance (2 blocks recover half), and it mixes the two excluded behaviours in an ad hoc way. I judged that worse than declaring the floor.
- So this plan keeps its bias of about 2% in the U variance, which is inside the 10% tolerance. Plan 1 carries the bias-free variant of the same order. The left-point `I21` sum and the `S_T` channel self-check are unchanged.

## Evaluation

<metrics>
kind: sde
score: 9
certified: false
provenance: analytic
estimated_rel_error: 0.0121
error_is_estimate: false
observed_order: 0.9827
observed_order_se: 0.0307
order_floor: 1.0000
order_band: one-sided below the declared floor (an overshoot is flagged, not failed)
design_order: 1.0000
scheme_family: milstein
invariants_ok: true
constraints_ok: true
cross_check_ok: unavailable
functionals_ok: unavailable
invariant_max_drift: 3.192e-03
invariant_worst_headroom: 0.4959
reference_outcome: validated
mean_rel_err: 1.548e-03
variance_rel_err: 0.0121
moment_bias_binding: variance
moment_bias_frac_tol: 0.2131
moment_bias_actionable: true
order_check: true
mc_se_rel: 0.0209
resolved: true
dynkin_z: 2.0342
dynkin_bar: 2.4284
ci_mult: 2.0000
resolution_evidence: not_used_as_accuracy_evidence
solver_sha256: 88edd7c60c28cce05fede7e3994fb9d3d5d7fd34c55e71f786b76062f34e792c
solve_time_s: 1.4610
wall_time_s: 23.9700
cost_source: cpu
</metrics>

<review score=9>

**Score: 9/10**

Bound by `strong_gate_not_measured` (cap 9), the designed ceiling for this problem. The rubric itself now reaches its 10 row: every moment is within tolerance, the MC is resolved, the order study clears this plan's own design rate of 1.0, and the Dynkin check **ran and passed** (`dynkin_ok: true`). Nothing about this score is a solver defect. `certified` is false only because of that same cap.

### Numerical Accuracy
- provenance:              analytic  (analytic_moments, validated against the spec's moment ODE)
- mean_rel_err:            S: 1.55e-03 (2.0031 vs 2, se 0.0111, 0.28 se). U: absolute 0.0429 vs 0, se 0.0187 (2.29 se), inside the family-wise bar of 2.53 se. U's exact mean is 0, below `near_zero_mean_threshold`, so this check is absolute
- variance_rel_err:        S: 1.21e-02 (0.9879 vs 1, se 0.0157). U: 9.86e-03 (2.8054 vs 2.8333, se 0.0585). Tolerance 10%, ci_mult 2.0
- mc_se_rel:               0.0209   resolved: true (2 x se half-width 3.1% for S and 4.1% for U against a 5% bar, 8000 paths)
- functional_rel_error:    not applicable, the spec declares no functional
- observed_order:          0.9827 +/- 0.0307 (median 0.9806), read against this plan's own design order 1.0 (`expected_strong_order_milstein`; `order_floor` 1.0, `design_order` 1.0, one-sided below the floor), not the pool-wide 0.5. It is the U component (s0 0.01109, s1 0.00561). S is integrated exactly (ladder differences at roundoff) and its order abstained as `unresolved`, which costs nothing
- weak_order:              0.9902 (x^2 observable; x and tanh sit at the noise floor and were discarded)
- dynkin:                  **ran, ok.** First seed `z` = 2.034 against bar 2.428 (family-wise, ci_mult 2.0, 3 functions). That was 0.394 from the bar, inside the 0.7 margin, so it was replicated: seeds 42 / 3444837047 / 2669555309 gave worst-`z` 2.034 / 1.248 / 1.895, **median 1.895**, which decides it (`dynkin_replicated: true`). Per function on seed 42: `X0*X1` z 2.03, `X1^2` z 0.62, `tanh(X0+X1)` z 0.51; all three pass
- terminal_law:            not declared
- shape:                   U skewness 0.113 +/- 0.056, excess kurtosis 1.485 +/- 0.171 (spec predicts 1.4444, so the variance mixture is reproduced). S skewness 0.042 +/- 0.028, excess kurtosis 0.032 +/- 0.055
- constraints:             finite: pass (gated). estimator_stability: pass (0.0032 vs bar 0.01, headroom 0.50)
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                strong_gate_not_measured (cap 9). Other caps: rubric 10, Tier A 10

### Reading the findings

1. **The Dynkin verdict, which was the point of this pass.** The check now runs, because the initial condition `[1.0, 0.0]` is read from the top level. It reports `dynkin_ok: true`, `dynkin_z` 2.034 against `dynkin_bar` 2.428, median 1.895 over three seeds. This is the one check that is sensitive to a spurious additive drift in U, which both moment checks cannot see (U's mean is skipped as near zero, and the variance subtracts the mean). It did not find one. Three honest qualifications, none of which changes the verdict:
   - **`phi = X0*X1` is the marginal function and the only one near the bar.** Its generator residual is 0.0931 / 0.0942 / 0.0944 across the three step sizes. It is *not shrinking* with `dt`, so the extrapolated residual 0.0946 has `se` 0.0465, and `z` is 2.03. The finer residual difference (-1.9e-4, se 2.2e-4) is inside its own noise, so no rate could be read and the extrapolation **assumed order 1** (`extrapolation_order_source: assumed_1`). The other two functions measured their orders (1.03 and 0.69).
   - **Replication did not make it more comfortable.** The median over seeds (1.895) is below the bar and decides the verdict, but the three-seed *pooled* estimate is 2.99, above the bar, and `pooled_agrees` is false. The decision rule is the median, so the verdict stands, and the plan was marginal. The ranking should not read this as a wide margin.
   - **What it does not establish.** The generator is built from the spec's own drift and diffusion, so a drift mis-transcribed from `problem.md` would fool it.
2. **`strong_gate_not_measured` (cap 9) is the designed ceiling.** The 2% pathwise gate is held by the independent external grader and not by `verifylib`, which has no pathwise reference, so `strong_rel_err` is null. The spec's independent measurement of this recipe is 1.4e-02, inside the gate. This evaluation does not establish that.
3. **Order check.** U reads 0.9827 +/- 0.0307 with a clean factor-of-two ratio. The left-point `I_21` assembly delivers the claimed order-1 rate, where an increment-only scheme reads 0.5. The `levy_area_treatment: sub-increments` declaration backs the `milstein` claim, so no `milstein_claim` cap applied. The solver consumes the supplied `dW` and `dW_sub`.
4. **Variance bias in U, unchanged from last pass.** The ladder reports `variance bias_rel` 0.0213 (0.213 of the 10% tolerance, `shift_in_se` 23.8, so real but not binding). The graded 8000-path U variance error is 0.99% (se 2.1%). The bias is the left-point weight on `S_k dW1_k` (a deficit of about 0.06 on 17/6), not the `dt^3/3` residual. The only bias-removing terms are plan 1's conditional-mean projection `E[I_01|dW1] = 0.5*mu*dt*dW1` or the full assembly from `dW_sub`, which R23 excludes. Reported only, not scored.

No `agent_cap` applied. `solver.py` (sha256 unchanged, 88edd7c6...) implements what `SOLUTION.md` describes: left-point Ito sum over `dW_sub`, exact S update, supplied increments consumed.

### Feedback for solver
- No code change needed. The moments, the strong order against 1.0, the Dynkin check, the constraints and the invariants all pass. Do not alter the scheme.
- Keep the left-point `I21` sum and the `S_T == S_0 + mu*T + sum(dW[:, :, 1])` self-check. The channel mapping is the one error no moment check detects.
- Documentation only, still standing, and not a cap: in Implementation Notes, say which quantity each bias source belongs to. The `dt^3/3` residual (the omitted `mu * int (s-t_k) dW1`) is the source of the **pathwise strong error** (1.4e-02). The left-point `S_k dW1_k` weight is the source of the **2.1% variance bias** in U. The earlier paragraph at lines 94-98 attributes the variance bias to the first, and should match Refine cycle 2.
- Documentation only, still standing: the `dW_sub is None` branch (`0.5*w1*w2`) is a conditional-mean surrogate, not a Levy-area assembly. It is fine as a standalone fallback, but it carries only the order-0.5 behaviour on a pathwise problem. State that in the Notes.
- Nothing is outstanding on the spec side for Dynkin. The kernel cannot measure `strong_rel_err`, so the 9 ceiling stays by design.

</review>
