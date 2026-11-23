# REPORT — `sde_clark_cameron`

**Problem type:** SDE, 2-dimensional state `(S, U)` driven by **two independent** Brownian channels —
the Clark–Cameron (1980) system, `dS = mu dt + dW2`, `dU = S dW1`, with `mu = 1`, `T = 1`.
Multiplicative noise (`S` multiplies `dW1`), constant drift, globally Lipschitz, not stiff.

**Outcome:** best plan `2-levy-area-milstein`, **score 9/10**, provenance `analytic`,
**`certified: false`**. No plan certified, and that is the designed outcome on this problem rather
than a pool failure — see *The ceiling* below. Read that section before reading the score.

---

## Provenance of the scoring

**`analytic` — validated, Tier A.** Both components have exact closed-form moments and the kernel's
reference check **validated** them against the spec's own moment ODE (`reference_outcome: validated`),
rather than accepting them on a quote. The hierarchy closes exactly:

- `E[S] = S_0 + mu*t`, `Var[S] = t`
- `E[U] = 0`, `Var[U] = ((S_0+mu*t)^3 - S_0^3)/(3*mu) + t^2/2`, which is `17/6` at `T = 1`
- `Cov(S, U) = 0` exactly

This is the strongest provenance tier available, and the moment evidence behind the 9 is real. What
it does **not** cover is the quantity this problem is actually graded on — the next section.

---

## The ceiling: why a 9 is the maximum, and what it is hiding

`problem.md`'s only stated accuracy criterion (ledger entries **R22** / **R7**) is a **2% relative
pathwise strong-error gate**, measured path by path against a reference driven by the same Brownian
increments. The spec carries it correctly as `evaluation_thresholds.strong_rel_err_max = 0.02`, and
`verifylib/schema.py:check_strong_gate_carried` hard-errors if that quote is *not* carried — so the
declaration is enforced, not merely tolerated.

**But this pipeline cannot measure it.** Nothing in `verifylib/kernel/` ever writes
`summary["strong_rel_err"]`: it has exactly one reader (`score.py:820`) and one pass-through
(`sde/driver.py:826`) and **no producer**, because the pathwise reference belongs to the independent
external grader, which this pipeline must not read. So `strong_rel_err` is `None` unconditionally and
`score.py` applies `strong_gate_not_measured`, capping every plan at 9. The kernel's own comment
states the intent: *"the plan has not been shown wrong, it has not been shown right."*

**This is not a cosmetic cap — it is the axis on which this pool actually differs.** Independent
measurement by the formulator at the graded `dt = 0.04`, before any plan was written:

| scheme | relative strong error in `U` | vs the 2% gate |
|---|---|---|
| Lévy-area Milstein (`I_21` from sub-increments) | **1.4e-02** | inside |
| plain Euler–Maruyama (increment-only) | **8.9e-02** | **4.5× over** |

Both reproduce `S` to 8e-13, so the gate is entirely a statement about `U`. Euler–Maruyama *cannot*
clear this gate at the graded step — that is Clark–Cameron's own lower bound and the problem's
design, not a mis-set threshold. The Lévy-area route is the only passing one.

So the headline reads: **the moment evidence is Tier A and validated; the problem's own accuracy
criterion is unmeasured by this run.** The 9 is honest about exactly that.

---

## Ambiguities recorded by the formulator

Two entries were marked `ambiguous`. Both are places `problem.md` was underspecified and a
documented choice was made.

**R5 — which Brownian channel is which.** Quote: *"Simulate the two-dimensional Ito system driven by
**two independent** Brownian motions W1 and W2"*. `problem.md` says only that "the last axis indexes
the 2 independent noise sources" and never says which column of `dW` is `W1`. The formulator encoded
the natural reading, `dW[:, :, 0] = W1` and `dW[:, :, 1] = W2`, matching the naming order in the
equations.

**This is the highest-risk detail in the problem and no verification layer can catch it.** `W1` and
`W2` are i.i.d., so every moment, invariant and Dynkin check in the spec is invariant under a swap —
but the grading is pathwise on the same increments, so a swap yields ~100% strong error. It was
measured as invisible: the removed `martingale` probe read `z = -0.63` under a channel swap against
`+0.76` for the correct scheme, and Dynkin read `z = +0.52` against `+0.81`. Mitigation: all three
solvers carry a free self-check, `S_T == S_0 + mu*T + dW[:, :, 1].sum(axis=1)`, and **all three report
it passing to ~2e-15**. That confirms the convention each solver *implemented* matches the one the
spec *declared*; it cannot confirm the spec's reading matches the grader's.

**R26 — no moment tolerances in the problem statement.** Quote: *"Scoring is independent of any
self-report"*. `problem.md` states the 2% pathwise gate and **no moment tolerances at all**. The
kernel's SDE rubric requires them, so the formulator supplied the template defaults (10% variance,
5% mean) against the exact moments, and set `num_paths: 8000` rather than copying the contract's
1000 — 1000 paths cannot resolve a 10% variance tolerance on `U` (the bar is
`4*(2/0.1)^2*(2 + 1.4444) = 5511`), which would have left every plan MC-inconclusive at 6 with no
lever. `graded_num_paths: 1000` carries the contract count for the audit. **The moment tolerances
every score below rests on are therefore a pipeline-supplied criterion, not one the problem asked
for.**

---

## Plans

All three plans consume the supplied `dW` pathwise and none declares `exact_sampling_components` —
deliberately, for the reason in *Verification gaps* below.

### `2-levy-area-milstein` — **best plan**, score 9/10

- **Scheme:** `U_{k+1} = U_k + S_k dW1_k + I_21`, with the Lévy-area term `I_21` assembled as the
  **left-point (Itô) sum** over the 256 sub-increments; `S` integrated exactly. `scheme_family:
  milstein`, `levy_area_treatment: sub-increments`, design order 1.0.
- **Score 9/10**, `certified: false`, provenance **`analytic`**, no `agent_cap`.
- **Error:** `estimated_rel_error` **1.21e-02** — this is a **state-moment** statistic
  (`mc_se_rel` 0.0209, `resolved: true`), *not* the pathwise error the problem is graded on. The spec
  declares no gated functional, so there is no `functional_rel_error` to lead with; the honest
  accuracy statement for this problem is that its own criterion was unmeasured.
- **Moments:** `S` variance error 1.21e-02, `U` variance error 9.86e-03, against a 10% tolerance.
  `U`'s mean is 0.0429 ± 0.0187 against an exact 0, inside the family-wise bar.
- **Observed order:** **0.9827 ± 0.0307** (median 0.9806), read against its **own** design rate of
  1.0 (`expected_strong_order_milstein`), not the pool-wide 0.5. `S`'s order abstained as
  `unresolved` because `S` is integrated exactly, which costs the plan nothing.
- **Dynkin:** `dynkin_ok: true`.
- **iter:** 3 (scores 5 → 9 → 9), `stopped` on the plateau rule. `solver.py` was **never changed** —
  both cycles were answered with reasoning rather than edits.
- **Measured bias, not removed:** `moment_bias_actionable: true`, `moment_bias_binding: variance`,
  `moment_bias_frac_tol` **0.2131** — 2.1% of `U`'s variance, 21% of its own tolerance. The solver
  answered the bias instruction with **"the current sub-step is the floor"** and the evaluator
  independently re-derived and confirmed it. The bias is the **left-point weight on `S_k dW1_k`**
  (the left Riemann sum of `E[S^2]` undershoots by 0.08, the `dt^2/2` term adds back 0.02, net 0.06 =
  2.1% of 17/6, matching the ladder) — *not* the `dt^3/3` residual the plan's notes originally
  blamed, which drives the pathwise error instead. The only bias-removing terms available are plan
  1's conditional-mean projection (which would make this plan a copy of plan 1) or the full
  sub-increment assembly (excluded by R23). Neither agent found a third option. **This bias is real
  and was knowingly left in place.**

### `1-levy-area-milstein-drift-corrected` — score 9/10

- **Scheme:** as plan 2, **plus** the increment-measurable conditional mean of the drift-noise
  integral, `E[I_01 | dW1] = 0.5*dt*dW1`. `scheme_family: milstein`, design order 1.0.
- **Score 9/10**, `certified: false`, provenance **`analytic`**, no `agent_cap`.
- **Error:** `estimated_rel_error` 1.21e-02 (`mc_se_rel` 0.0208, `resolved: true`) — again a state
  moment, not the graded pathwise quantity.
- **Moments:** `S` variance error 1.2%, `U` 1.1%, against 10%; `S`'s mean within 0.15%.
- **Observed order:** **0.9343 ± 0.0351** against its design rate of 1.0 — about 1.9 se under, inside
  the band, check passes.
- **Dynkin:** `dynkin_ok: true`. **No `moment_bias_actionable`** — alone in the pool, it carries no
  measured removable discretisation bias in a graded moment.
- **iter:** 3 (5 → 9 → 9), `stopped` on the plateau rule. `solver.py` never changed.
- **Why it matters that this plan lost:** its extra term is the *only* kernel-invisible difference in
  the pool, and the plan derives it as roughly halving the pathwise strong error (~6.9e-03 against
  plan 2's 1.4e-02). It stops deliberately one term short of the R23 trap. See *The winner is a
  one-standard-error call*.

### `3-euler-maruyama-control` — score 5/10 (kernel 9, capped to 5)

- **Scheme:** increment-only Euler–Maruyama, no Lévy-area term. `scheme_family: euler-maruyama`,
  design order 0.5. Written **as the pool's control**, declared in its own frontmatter as expected to
  miss the 2% gate and not to win a tie.
- **Kernel score 9 → `agent_cap` to 5**, provenance `analytic`, `certified: false`. **The cap is a
  finding and the reason is quoted in full:**

  > "Pool control, declared EM at strong order 0.5. The problem states a 2% pathwise strong-error
  > gate; the kernel does not measure it, but its own CRN ladder shows the dt=0.04 vs dt=0.02
  > pathwise difference in U is 0.0788 against terminal scale 1.246 (6.3%, 3x the gate) and the
  > formulator measured 8.9e-2 (4.5x). Order 0.4965 clearing its own floor of 0.5 is conformance to a
  > low design rate, not accuracy. Moment-only 9 would overclaim."

- **Everything the kernel measured, it passed:** moments inside tolerance (`U` variance error 2.0%),
  `resolved: true`, constraints and invariants clean, and order **0.4965 ± 0.0348** against its own
  floor of 0.5 — it *clears* its floor while the two correct plans sit slightly *under* theirs.
- **`moment_bias_frac_tol` 0.3874** — 3.9% variance bias, 39% of tolerance. The solver answered that
  the increment-only sub-step is the floor for this plan, since any reduction requires the Lévy-area
  term and would turn the control into a duplicate of plan 2. It also corrected its notes, which had
  claimed the omission cost the variance only 0.7%, to the measured ~4%.
- **iter:** 2 (5 → 5), `stopped` on the plateau rule.

**This plan is the pool's most important negative result.** It is a scheme known to be 4.5× outside
the problem's accuracy gate, and the kernel scored it **9** — tying the two plans that sit inside the
gate. Only the evaluator's downward cap kept the pool from inverting. Its *own* quality index is the
highest in the pool (0.625, against 0.536 and 0.514), precisely because order fidelity rewards
conformance to a low design rate; `score` being the first ranking key is the only thing that kept it
last.

---

## Best plan and how it was chosen

The ranker's verdict, quoted verbatim:

```
best: 2-levy-area-milstein   decided_by: sqi
```

```
note: the index above is over order, structure only. accuracy, consistency did not resolve on
every plan in this pool, and a component no comparison was made on does not decide one -- see
ranking.comparable_keys. Each plan's own index, over whatever it measured, is
2-levy-area-milstein=0.536, 1-levy-area-milstein-drift-corrected=0.514,
3-euler-maruyama-control=0.625.
```

| plan | score | cert | provenance | sqi (restricted) | ord | str | solve_s |
|---|---|---|---|---|---|---|---|
| `2-levy-area-milstein` | 9.0 | no | analytic | **0.619** | 0.983 | 0.165 | 1.461 |
| `1-levy-area-milstein-drift-corrected` | 9.0 | no | analytic | 0.593 | 0.934 | 0.165 | 1.383 |
| `3-euler-maruyama-control` | 5.0 | no | analytic | 0.625 | 0.993 | 0.165 | 1.048 |

Cost is reported as a diagnostic and was not ranked on.

### The winner is a one-standard-error call, and the unmeasured axis points the other way

This needs stating plainly rather than being left in the table. `decided_by: sqi` is not the `tie` or
`iter` default, but it is close to one in substance:

1. **The deciding margin is inside the noise.** Accuracy and consistency did not resolve pool-wide,
   so `ranking.comparable_keys` restricted the index to **order and structure**; structure is
   identical at 0.165. The two 9s were therefore separated **on order fidelity alone**: 0.983 against
   0.934. That is a gap of 0.048 between estimates whose own standard errors are 0.031 and 0.035 —
   combined ≈ 0.047, so **about one standard error**. The ranker did not resolve these two plans apart.
2. **The problem's own criterion favours the runner-up.** The one real difference between the two is
   plan 1's conditional-mean drift projection, and the evidence on the axis the problem actually
   grades — pathwise strong error — points to plan 1 (plan 1's derivation: ~6.9e-03, against ~1.4e-02
   for plan 2's recipe, which the formulator measured independently). The kernel cannot see this axis,
   so it played no part in the ranking.
3. **One more measurement was spent and came back inconclusive.** `cli.py compare` on the top two
   (recorded in `agreements.json`): *"INCONCLUSIVE ... differ by 2.000e+00 in moment space on
   independent seeds and 1.142e-02 on the shared seed (worst tol 0.05) at num_paths=8000, dt=0.04,
   seeds 42 / 3444837047; families milstein / milstein -- the Monte-Carlo interval straddles the
   tolerance, so this is neither a break nor a cap."* Run for information only: this is Path A, so the
   agreement gate is not required, and both plans are the same family, so it could not have been a
   circularity break regardless.

**Recommendation.** `2-levy-area-milstein` is the ranker's pick and is reported as the winner, and it
is a sound, Tier-A-validated, order-1 scheme. But a reader choosing between these two plans on
*accuracy* should know the selection did not measure accuracy: the two are statistically
indistinguishable on everything the kernel resolved, and the one piece of evidence on the problem's
stated criterion favours `1-levy-area-milstein-drift-corrected`. **If the pathwise 2% gate is what
matters to you, prefer plan 1.** Neither plan was shown to pass it by this run.

### Phase 2b

Did not run. It fires only on a loop that exits with a **certified** winner, and no plan certified —
the bias instruction was reachable through ordinary eligibility inside the loop instead, and both
plans that carried `moment_bias_actionable: true` received it and answered it (see each plan above).

---

## Verification gaps

Each of these bounds what the run established.

1. **The problem's only stated accuracy criterion was never measured.** The 2% relative pathwise
   strong-error gate (R22/R7) is correctly declared and structurally unmeasurable by this pipeline:
   `strong_rel_err` has a reader and no producer anywhere in `verifylib/kernel/`, because the
   pathwise reference belongs to the independent external grader. Every plan is capped at 9 by
   `strong_gate_not_measured`. **No plan on this problem can certify**, and lifting the cap by
   removing `strong_rel_err_max` would be an overclaim — it was not done. On the independent evidence
   available, the two Lévy-area plans sit inside the gate and the control sits 4.5× outside it, but
   **this run establishes none of that**.

2. **A `martingale` invariant was removed from the spec mid-run, after measurement.** On the first
   evaluation round all three plans scored **5**, every one pinned by the ungated `martingale`
   invariant on `U` reading `z ≈ 2.29–2.34` against a 2.0-se bar. It was removed as a **zero-power
   probe** — not to raise a score — and the decision was measured before being taken, by the agent
   that owned the spec, with the rationale and the table recorded in `verification.note` dated and
   labelled. The finding: `E[U_T] = 0` holds **exactly in exact arithmetic at every `dt`** for every
   admissible scheme here (`S` is `W2`-measurable, `U`'s increments are `W1`), so the null is
   identically true and the check cannot fail for any reason connected to the solver. Measured at
   400 000 paths, it read `z` of **+0.76 / +0.70 / −0.63 / +0.77** for the correct scheme, plain
   Euler–Maruyama (4.5× over the gate), a **W1/W2 channel swap**, and an anticipating update — i.e.
   indistinguishable, including on the one catastrophic error available on this problem. The only
   mode it detected was a spurious additive drift (`z = +15.9`), which the Dynkin test function
   `φ = X[:, 0]*X[:, 1]` already covers (residual predicted 0.060 analytically, measured 0.066).
   `estimator_stability` was kept, explicitly because its null is *not* identically true. **A reader
   who disagrees with that reasoning should treat the three 5s as the measurement and this run's 9s
   as resting on a spec edited mid-run.** The seed (42) and `num_paths` (8000) were deliberately left
   untouched, since the bar scales with the standard error and raising paths only redraws the test.

3. **Dynkin passes, but on a narrow margin, and the one function that matters is the one nearest the
   bar.** `dynkin_ok: true` on both candidate plans, `dynkin_z` **2.0342** against `dynkin_bar`
   **2.4284**. The first seed landed inside the 0.7 margin, so the kernel replicated on 3 seeds
   (`dynkin_replicated: true`); worst-`z` per seed was 2.034 / 1.252 / 1.896 and the **median 1.896**
   is what decides it. The **3-seed pooled `z` is 2.99, above the bar, and `pooled_agrees` is false**
   — so this is a soft pass on the median rule, not a comfortable one. `φ = X[:, 0]*X[:, 1]` is
   simultaneously the function nearest the bar (`z = 2.03`, versus 0.63 and 0.69 for the other two)
   and **the only test in the suite covering a spurious additive drift in `U`** — a mode both moment
   checks are structurally blind to, since `U`'s exact mean of 0 falls under
   `near_zero_mean_threshold` so the mean check is skipped, and the empirical variance subtracts the
   mean. Its residual (~0.0946) is **flat across `dt`**, roughly 2 se above zero, with the same sign
   on every seed, and `U`'s empirical mean of +0.043 leans the same way. The evaluator's reading is
   that this is consistent with a seed-42 excursion, since every term of the `U` update has zero
   conditional mean by derivation — and noted the discriminating test: a true drift would grow with
   more Dynkin paths and an excursion would not. **That test was not run.** No extrapolation rate
   could be read (the finer residual differences were inside their own Monte-Carlo noise), so the
   check assumed order 1.

4. **Dynkin abstained entirely for the first two evaluation rounds.** `dynkin_ok: unavailable` on all
   three plans across two rounds, through two distinct spec defects: first
   `initial_condition.value` written as the strings `["S_0", "U_0"]`, then — after that was made
   numeric — the block sitting at `verification.initial_condition` while
   `verifylib/kernel/sde/dynkin.py:203` reads `spec.get("initial_condition")` at the **top level**.
   Only the second was the real defect: `_initial_scalar` does resolve a string name against
   `parameters`, so the original spelling would have worked had the block been in the right place.
   Fixed by moving it to the top level; `generator` and `dynkin_test_functions` were verified at the
   reader to be correctly placed under `verification`. **Plan 3 was never re-evaluated after this
   fix** — it had already `stopped` on the plateau rule — so its recorded metrics carry
   `dynkin_ok: unavailable`, and the generator check was never run against the control.

5. **A conditionally tractable component is not integrated out — a real ceiling, not a missed
   technique.** The pool gate warns that state component 1 (`U`) is conditionally tractable and no
   plan declares `exact_sampling_components: 1`. The plan-creator was re-dispatched once and declined
   to add a plan, with this on the record: `U`'s drift is zero and its diffusion row `[S, 0]` is
   degree zero in `U`, and `S`'s coefficients never read `U`, so conditionally on the `W2` path
   `U_T` is exactly `N(0, ∫S^2 ds)`. But **R23** grades `terminal_paths[:, 1]` path by path against a
   reference on the same `W1` increments, and `E[U_T | W2] = 0` for every path — so a
   Rao-Blackwellised `U` leg scores **exactly 100%** relative strong error. Conditioning on only the
   `W2` sub-path plus the supplied `W1` *step* increments discards the sub-increment information that
   resolves the Lévy area, returning to `O(√dt)` and Clark–Cameron's lower bound (~4.5× the gate);
   conditioning on the **full** supplied information reproduces the sub-grid reference to rounding and
   is then `dt`-invariant, which **R23 excludes from the comparison rather than scoring**. There is no
   admissible middle. Note also that the variance the technique would remove is `U`'s *sampling*
   variance, which bears on the R26-supplied moment tolerances — already resolved at 8000 paths — and
   not on the pathwise error the problem is graded on.

6. **`dt`-invariance (R23) is an active trap that shaped every plan in the pool, and was not
   independently verified.** For this linear system, assembling `I_01` in full from `dW_sub`
   reproduces the grading reference to rounding, making terminal states identical at `dt`, `dt/2` and
   `dt/4` — which R23 excludes rather than scores. All three plans document steering clear of it, and
   plan 1 stops deliberately one term short. Plan 1's evaluator confirmed its ladder shows genuine
   `dt`-dependence, so the exclusion does not apply to it. **The same confirmation was not separately
   recorded for plans 2 and 3.**

7. **The moment tolerances the scores rest on were supplied by the pipeline, not the problem.** See
   R26 above. `problem.md` sets no moment tolerances; the 10% / 5% defaults and `num_paths: 8000` are
   pipeline choices, and the contract's own `graded_num_paths: 1000` cannot resolve the variance
   tolerance on `U` at all.

8. **The W1/W2 channel convention cannot be verified by anything in this pipeline.** See R5 above.
   Confirmed consistent between spec and solvers to ~2e-15; **unconfirmed against the grader**. A swap
   is worth ~100% strong error and is invisible to every check the suite contains.

9. **Common-mode spec dependence.** All three plans descend from one formulator's spec and one
   operator transcription. Dynkin's generator is built from the spec's **own** declared drift and
   diffusion, so it validates the paths against the equation *as transcribed* and would not catch a
   drift or diffusion mis-read from `problem.md`. No independent check of the transcription exists in
   this run.

10. **No `invariants_not_reported` entries**, and `invariants_ok: true` on all three plans
    (`estimator_stability` drift 3.192e-03 against a 1e-2 bar; `finite` clean). The surviving
    invariant set is small because of gap 2.

11. **A leakage guardrail fired and was repaired.** The Stop gate flagged error-severity leakage in
    plans 2 and 3: both `SOLUTION.md` files named the independent external grader by path in
    evaluator prose. **The citation originated in the conductor's own dispatch prompts**, not in the
    agents' analysis. One sentence in each file was redacted to remove the path while preserving every
    claim; **no measurement, score or metrics value was altered**, and the workspace re-checks clean.
    Later dispatches carried an explicit instruction not to name those paths.

**Final gate:** `cli.py gate` exits **0** over the finished workspace — no error-severity findings.
The single warning is the conditional-tractability finding recorded as gap 5. No
`GUARDRAIL_UNSATISFIED` file was produced.
