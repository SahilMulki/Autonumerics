# REPORT — `sde_geometric_asian`

**Problem type:** SDE — scalar Black–Scholes geometric Brownian motion
(`sde_family: geometric_brownian_motion`), `dS = r S dt + σ S dW`, `S_0 = 100`, `r = 0.05`,
`σ = 0.2`, `T = 1`. One state, one noise channel, multiplicative noise, non-stiff,
Milstein-eligible. Graded on two path functionals of the **geometric** average
`A = exp((1/T)∫₀ᵀ ln S dt)`: `call_price` (1% gate) and `geo_average_variance` (3% gate).

**Path:** A — the spec carries a closed form, and it was **validated** (not merely quoted).

## Provenance of the scoring

The winner scores **10/10, `certified: true`, provenance `analytic`** — the strongest tier.
The reference is the exact GBM terminal moments (`E[S(T)] = S_0 e^{rT} = 105.1271096376`,
`Var[S(T)] = 451.0288078158`), and `reference_outcome: validated`: the kernel checked them against
the spec's own moment ODE, which closes exactly for GBM. The two graded functionals likewise have
exact closed forms — `ln A` is Gaussian with `m = 4.6201701859880915`, `s² = σ²T/3`, giving
`call_price = 5.546818633789223` and `geo_average_variance = 140.17051781155084`. This is a 10
earned against closed-form ground truth, not against the solver's own convergence.

## Ambiguity recorded by the formulator

One, and it is worth the reader's attention because it is a place the **spec asserts more than
`problem.md` does**:

> **R21** — quote: *"this problem has no terminal-moment formula, so the quantity of interest below
> is the whole grade"*. `spec_path: evaluation_thresholds.functionals, analytic_moments`.

The formulator read this as a statement about **what the independent check grades** (the two
functionals carry the whole gate) rather than as a mathematical claim, and then declared exact
terminal moments anyway — `S(T)` under the stated GBM dynamics does have them — validating them
with the exactly-closing moment ODE. This cannot relax the problem, since the functional gates are
independent of the moment tolerances, and it strengthens the evidence tier. But the reader should
know that `analytic_moments` asserts ground truth the problem statement declines to assert, and
that the moment errors reported below rest on that assertion. The functional errors do not.

## Plans

| # | plan | scheme | score | cert | provenance | `call_price` rel err (gate 1%) | state-variance rel err | observed order | iter |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `1-log-exact-bridge-conditioned` | log-exact per-step step of `ln S` consuming `dW`, with the within-step Brownian-bridge part of `∫ln S dt` integrated out in closed form; antithetic pairing (`scheme_family: other`) | **10** | yes | `analytic` | **1.537e-03** | 5.206e-04 | unresolved (see below) | 2 |
| 2 | `2-milstein-trapezoid` | Milstein on `S` + `O(dt²)` trapezoidal quadrature of the log-average, plus a terminal-price control variate added in cycle 2 (`scheme_family: milstein`) | **10** | yes | `analytic` | 1.666e-03 | 5.325e-03 | 0.9882 ± 0.0078 vs design 1.0 | 2 |
| 3 | `3-euler-maruyama-trapezoid` | plain Euler–Maruyama + the same trapezoidal quadrature (`scheme_family: euler-maruyama`) | 8 | no | `analytic` | 2.222e-03 | 6.130e-03 | 0.4964 ± 0.0112 vs design 0.5 | 2 |

All three errors above are **measured against the closed form**, not estimated
(`error_is_estimate: false` on every plan). All three are MC-resolved (`resolved: true`,
`mc_se_rel ≈ 2.6e-03`). No plan recorded an `agent_cap`, and no `grading_violation` occurred —
every plan's `solver.py` hash was unchanged across its evaluation and matches the `solver_sha256`
in its own metrics block.

### The accuracy number that matters

The spec declares gated functionals, so the accuracy the problem asks about is the functional's,
not the state moment's. For the winner:

- `functional_name: call_price`, `functional_rel_tol: 0.0100`
- **`functional_rel_error: 1.537e-03`** — about 6.5× inside the gate
- split: `functional_bias_rel: 1.087e-05` (what `dt` costs) + `functional_se_rel: 1.526e-03`
  (what the path count costs) → **`functional_limited_by: sampling`**

That split is the plan's whole point: conditioning on the stepped endpoints removes the quadrature
bias in closed form, so essentially none of the error is discretisation. The ladder did **not**
abstain here and no `exact_sampling_components` was declared — the step consumes the supplied `dW`
at every step — so the bias term is a real measurement rather than a skipped one.
`geo_average_variance` is also clean, at a 1.1e-04 ladder error against its 3% gate.

### Dynkin — and the spec repair that made it reachable

Round 1 held **the entire pool at 9, uncertified**, for a reason in none of the solvers:
`dynkin_ok` reached no verdict, because the kernel's Dynkin reader could not find an initial state.
The spec declared `parameters.S_0`, and the reader takes `initial_condition.value` or
`parameters.X_0`/`Y_0`/`Z_0`. Two evaluators diagnosed this independently and both attributed it to
the spec, not the solver. The formulator was re-dispatched once with the finding and added a
top-level `initial_condition` block with `value: ["S_0"]` — binding to the existing parameter, so
there is no second source of truth for `100.0` — plus an extension of ledger entry R2's
`spec_path`. **No declared value moved**, so the round-1 runs stayed valid, and `check-spec` exits 0
before and after. In round 2 the check ran and passed on both leading plans: `dynkin_z = 0.3312`
(plan 1) and `0.3346` (plan 2) against a bar of `2.4284`, single-seed and not marginal. That repair
is what turned two 9s into two certified 10s.

## Best plan

**`1-log-exact-bridge-conditioned`.** The ranker's verdict, verbatim:

```
best: 1-log-exact-bridge-conditioned   decided_by: sqi
```

Score and provenance tie with plan 2 (both 10, both `analytic`), so the quality index separated
them — and **it separated them narrowly: 0.432 against 0.423.** The index was restricted to the
components the whole pool measured, and the ranker says which it dropped:

```
note: the index above is over accuracy, structure only. order, consistency did not resolve on
every plan in this pool, and a component no comparison was made on does not decide one
```

So the winner is chosen on accuracy and structural margin alone, on a ~2% index gap, with the
order component — the one place plan 2 has hard evidence plan 1 structurally cannot produce —
excluded for not being comparable across the pool. Read that as "the two leading plans are close to
indistinguishable on what this pool could measure in common", and prefer plan 1 on the strength of
its bias term (1.1e-05 against plan 2's 3.3e-04) rather than on the index margin.

## Verification gaps

`cli.py gate` over the finished workspace: **no findings**. `check-spec`: no findings. No
`GUARDRAIL_UNSATISFIED` file. The gaps below are therefore not gate failures — they are limits on
what this run established.

1. **The ranker's own warning about the winner, quoted:**

   ```
   warning: 1-log-exact-bridge-conditioned reports estimated_rel_error 0.000521 against a
   Monte-Carlo standard error of 0.00258 (5x larger) -- its accuracy is unmeasured at this path
   count, not good
   ```

   This is about the **state-variance** statistic (`estimated_rel_error = variance_rel_err =
   5.206e-04`), which sits inside its own sampling noise and so is not located. The graded
   functional *is* resolved — `functional_rel_error 1.537e-03` against `functional_se_rel
   1.526e-03`, a ratio of ~1.0, and the kernel records `resolved: true` and
   `resolution_evidence: not_used_as_accuracy_evidence`. The honest reading: the winner's state
   variance is not measured as accurate, it is measured as *inside the noise*, and the accuracy
   claim rests on the functional.

2. **The winner's order study abstained.** `order_check: unresolved`. The pathwise differences
   across refinement levels are at roundoff (2.6e-13 against a 1.05e-10 floor), because log-exact
   stepping integrates `ln S` exactly on a shared Brownian path. Per the rubric this is a property
   of the estimator and costs the plan nothing — but it means **no observed strong order was
   established for the shipped plan**. Plan 2, which does carry an order measurement
   (0.9882 ± 0.0078 against its design 1.0), is the pool's evidence that the Brownian increments
   are consumed correctly by a stepping scheme at all; plan 3 corroborates at 0.4964 ± 0.0112
   against 0.5. An abstention is not a pass.

3. **Dynkin cannot catch a mis-transcribed equation.** Both evaluators flagged this and it is the
   structural limit of the whole run: the generator the check builds comes from the spec's *own*
   drift and diffusion, so if `problem.md` was misread into `problem_spec.json`, every plan is wrong
   identically and the check passes unanimously. What covers it here, partially: the exact-moment
   match against independently-derived GBM formulas and the ungated `martingale` invariant on
   `exp(-rT)·S(T)` against `S_0`. Neither is spec-independent.

4. **Dynkin ran single-seed** on both certified plans. It was not replicated, on the grounds that
   `z ≈ 0.33` against a bar of 2.43 is not marginal.

5. **`cross_check_ok: unavailable`** on all three plans — the spec declares no ordering between
   functionals, so nothing was checked. A skip, not a failure.

6. **No cross-plan agreement was priced in.** `cross_plan_agreement: unavailable`. This is Path A,
   so the agreement gate is not required and was not run; the two certified plans are nonetheless
   independent schemes of different families (`other` and `milstein`) whose `call_price` estimates
   agree to 1.3e-04 relative (5.5446 and 5.5448 before the control variate; both ~1.6e-03 from the
   closed form). That agreement is reported here as context, not as a recorded break.

7. **`invariants_not_reported`: none** — every declared invariant (`positivity`, `finite`,
   `martingale`, `estimator_stability`) was evaluated and passed, with
   `invariant_worst_headroom: 1.43` decades on the winner.

8. **`surrogate_ok: unavailable`** on all plans — expected on Path A, where the analytic route
   supplied the reference. Not a gap in the result, recorded for completeness.

## Plans that did not certify

- **`3-euler-maruyama-trapezoid` — 8/10, `stopped` (plateaued at `iter: 2`).** Its moments and
  *both* gated functionals are clean (`call_price` 2.222e-03 against the 1% gate), but the
  terminal-law KS test on `S(T)` fails: **D = 2.08e-3 against a 1.45e-3 critical value, 1.44×
  critical.** Plain Euler on a multiplicative equation at the contract `dt = 0.04` under-skews
  `S(T)` — skewness 0.585 against 0.614 (5.7 se low), excess kurtosis 0.603 against 0.678 (3.5 se
  low). This is a `dt` bias, so more paths cannot reach it, and because the moments check failed the
  Dynkin check was capped before it ran (`dynkin: not_run` — a skip, not a failure). In cycle 2 the
  solver was told that the two available fixes (the Milstein correction, or log-exact stepping)
  would turn the plan into a copy of plan 2 or plan 1, and it chose to change nothing and document
  the ceiling instead. That is the correct call for a deliberate baseline, and the 8 is the
  measurement: **Euler at this step cannot match the terminal distribution, while still pricing both
  functionals inside their gates.** The pool's two prices isolate exactly that — the Euler weak bias
  is visible in the distribution's shape and invisible in the functional, because the trapezoid
  quadrature absorbs it.

## Phase 2b

**Did not run.** Both certified plans carry `moment_bias_actionable: false`
(`moment_bias_frac_tol: 4.5e-13` on the winner, `0.0418` on plan 2, both against
`moment_bias_binding: variance`), so there was no measured removable bias to spend a polish cycle
on. The winner ships the file that was scored, hash `a39c96b3…`, unmodified.

## Cost (diagnostic only — not a ranking input)

`solve_time_s` at the graded call: 0.351 s (plan 1), 0.380 s (plan 2), 0.359 s (plan 3), all
`cost_source: cpu`. The three are within 8% of each other, so cost would not have separated this
pool even if it were allowed to.
