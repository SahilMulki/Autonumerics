# REPORT — `sde_kubo_oscillator`

**Problem type:** SDE, family `kubo_oscillator` — a 2-state, 1-noise-channel **Stratonovich**
rotation, `a = sigma = 1`, `T = 1`, `X(0) = (1, 0)`. Linear, non-stiff, multiplicative noise. The
discriminator of the problem is the calculus convention: the Stratonovich flow is an exact rotation,
and the equivalent Itô form carries the correction `0.5*(Dg)g = -0.5*sigma^2*X`.

**Outcome: no plan certified.** All three plans plateaued at `iter: 2`. The best plan is
`2-stratonovich-heun` at **9/10**, provenance `analytic`. `max_iter` was never reached — the plateau
early-stop ended every plan.

---

## Provenance of the scoring — read this first

All three plans were scored on **Tier A `analytic`** evidence: the closed-form moments, and
`reference_outcome: validated` on every plan, so the closed form was machine-checked against the
spec's own moment hierarchy rather than accepted on provenance. The formulator also cross-checked
the moments three independent ways before declaring them (closed form, the exact 5-moment ODE at
`rtol 1e-12`, and a 4e6-path evaluation of the exact flow; the first two agree to 1e-9). At `t = 1`
the reference is mean `(0.3277099, 0.5103780)`, variance `(0.3644465, 0.2676740)`.

So the accuracy numbers below are measured against an exact, validated reference — not against the
plans' own convergence behaviour. **The 9 is not an accuracy shortfall.** Every moment, MC-resolution
and constraint check on the winner passed with room to spare; the single point is withheld by the
rubric's 9 row because the Dynkin generator check never reached a verdict. See *Verification gaps*.

## Ambiguities recorded in Phase 1

One, and it is about the grading instrument rather than the mathematics.

> **R18** — quote: *"the harness evaluates its own exact solution on the very increments it handed
> you and gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))`
> over paths at < 20%"*

**What the formulator chose:** `evaluation_thresholds` has no key for a *pathwise strong-error*
tolerance — the SDE accuracy keys are moment-space, and `functionals` takes only scalars the solver
itself reports, which this is not. It was carried instead as (a) the exact pathwise reference in
`exact_pathwise_solution`, (b) the strong-order floors 0.5 / 1.0 that the CRN ladder measures, and
(c) the moment tolerances, which a solver at 20% pathwise strong error comfortably clears. **The 20%
number is recorded in the spec but is not enforced as its own numeric gate.** Nothing in this run
measured that statistic directly, so if the external harness grades on it, this run does not report
a number against it.

Two deliberate non-ambiguities worth flagging to the reader, both recorded as `mapped`:

- **R9** — `graded_num_paths: 2000` is the contract's path count, but the kernel was given
  `num_paths: 50000`. 2000 paths cannot resolve a 10% variance tolerance, so pinning the kernel
  there would have left every plan MC-inconclusive at 6 with no lever. Every accuracy number below
  is therefore at 50k paths, not at the graded 2k.
- **R7** — amplitude conservation is gated at a **loose 0.25 absolute** bound on `||X| - 1|`, not at
  1e-8. A discretisation conserves it only to its own order, while the Itô-misreading failure mode
  grows `|X|` by `exp(0.5*sigma^2*T) = 1.65` per unit time.

---

## Plans

### `2-stratonovich-heun` — **9/10**, `analytic`, not certified — BEST

- **Scheme as shipped:** a 4-stage explicit, derivative-free Stratonovich Runge–Kutta step on the
  stated coefficients (whole-step `dt` and `dW`, no `Dg·g` formed). It began the run as a two-stage
  Heun predictor–corrector and was rewritten on the bias instruction in cycle 2.
- **Accuracy (no functional is declared in this spec, so these state moments are the accuracy
  statistic):** worst mean relative error **6.298e-03** against a 5e-02 tolerance; worst variance
  relative error **1.02e-02** against 1e-01. `estimated_rel_error: 1.02e-02`, `error_is_estimate:
  false` — **measured against the validated closed form**, not estimated. MC-resolved
  (`resolved: true`, `mc_se_rel: 6.950e-03`, CI widths 0.0100 / 0.0138 against a 0.05 bar at 50k
  paths).
- **Observed order:** **1.9211 ± 0.0131** against `order_floor: 1.0` and `design_order: 1.0`. The
  band is one-sided below the floor, so the overshoot is **flagged, not failed**
  (`strong_order_above_band: true`, `orders_ok: true`); the weak order of 2.03 agrees. Read the
  design claim as 1.0 and 1.92 as what this linear single-noise flow showed.
- **Constraints:** `constraints_ok: true`, `invariants_ok: true`. `max||X|-1|` fell from 0.155 to
  **0.0077** against the gated 0.25.
- **Residual `dt` bias:** `moment_bias_binding: mean` at `moment_bias_frac_tol: 1.838e-03` —
  0.18% of its own tolerance, and `moment_bias_actionable: false`. There is essentially no
  discretisation bias left to remove at the graded step.
- **iter:** 2, then plateaued (9 → 9).
- **Cost (diagnostic, not ranked on):** `solve_time_s` 0.146 s.

### `1-stratonovich-implicit-midpoint` — **3/10**, `analytic`, not certified

- **Scheme as shipped:** a Yoshida triple-jump composition of three Cayley maps, sub-angles
  `g1*theta`, `g2*theta`, `g1*theta` with `g1 = 1/(2 - 2^(1/3))`, `g2 = 1 - 2*g1`, cancelling the
  single Cayley map's cubic angle error. Started the run as one Cayley map (the Stratonovich
  implicit midpoint rule in closed 2×2 form) and was rewritten on the bias instruction in cycle 2.
- **It scored 9 in cycle 1 and 3 in cycle 2, and its accuracy improved across that regression.**
  Worst moment error went 2.34e-02 → **1.07e-02**, the mean's `dt` bias went 0.2587 → **2.842e-03**
  of its tolerance, and `max||X|-1|` is **4e-15**. Observed order **1.8328 ± 0.0138** (overshoot
  flagged, not failed). It is, on every accuracy and structural statistic the kernel reports, the
  most accurate plan in the pool.
- **What cost it six points:** with the initial condition finally readable (see *The spec repair*),
  the Dynkin generator check reached a verdict and **failed**. Seed 42 gave `dynkin_z: 2.3124`
  against `dynkin_bar: 2.5310` — inside the bar but within 0.22 of it, which triggered a 3-seed
  replication: per-seed worst z of 2.31 / 3.20 / 4.39, median 3.20, pooled **5.72**.
- **Which statistic failed, and why the evaluator thinks it is an instrument artifact:** the failing
  test function is `X1^2 + X2^2`, whose extrapolated generator residual is **~1e-11 ± 3.5e-12** —
  roundoff scale, against the order study's own roundoff floor of 6.6e-13 on these terminal values.
  The Cayley composition conserves `|X|` exactly, so that residual is floating-point noise divided
  by a floating-point standard error. The three test functions that actually carry signal sit well
  inside noise (z = −1.52, +0.55, −1.65 for `X1`, `X1*X2`, `tanh X1`), and a genuine drift error
  would put the residual near 1e-1 rather than the 1e-3 observed. The evaluator's judgement, which
  the conductor records rather than acts on: a residual at roundoff scale should abstain as
  `unresolved` rather than fail, and had it abstained this plan would have scored 10. **No
  `agent_cap` was applied and the 3 is what the kernel computed, so the 3 is what is reported.**
- **iter:** 2, then plateaued on the regression (9 → 3). The cycle-1 file that scored 9 is preserved
  beside it as `solver.scored.py`; `solver.py` is the cycle-2 file the 3 belongs to
  (`solver_sha256: e8cd2fe3...`), so the artifact on disk and the score in STATE.md match.

### `3-euler-maruyama` — **3/10**, `analytic`, not certified

- **Scheme:** Euler–Maruyama on the Itô form *including* the `-0.5*sigma^2*X`
  Stratonovich-to-Itô correction. `design_order: 0.5`. `solver.py` is byte-identical across both
  cycles (`af56c6c3...`).
- **What bound the score:** the **gated** `amplitude_conservation` constraint fails —
  `constraints_ok: false`, `invariants_ok: false`, `invariant_worst_headroom: -6.8461`. 7.016% of
  paths breach `||X| - 1| > 0.25` at 50k paths, and 7.05% / 6.35% / 6.50% / 6.60% over four seeds at
  the **graded 2000 paths** (binomial s.e. ≈ 0.55%), so the breach is a property of the scheme at
  `dt = 0.04`, `sigma = 1`, not of the path count. The kernel recorded a `gate: true` structural
  violation: rubric cap 4, certification cap 3.
- **The moments pass and are resolved**: mean 0.0183 / 0.0358 against 5e-02, variance 0.0604 /
  0.0532 against 1e-01, `resolved: true`. The Itô correction is present and correct — this is not a
  calculus-convention error, and raising `num_paths` would not help.
- **A deliberate, documented decision not to rescue the plan** (Implementation Note 9): the
  Itô–Milstein correction would likely clear the gate, but with one noise channel, constant `Dg` and
  `J^2 = -I` it reduces to `I + theta*J + 0.5*theta^2*J^2`, a near-duplicate of plan 2's step, and
  would remove the pool's only order-0.5 anchor. Nothing was clipped, renormalised or projected to
  satisfy the gate. The conductor endorses recording this as the measured, intended behaviour of a
  strong-order-0.5 scheme on this problem rather than as a concealed defect.
- **iter:** 2, then plateaued (3 → 3).

---

## Best plan and why

```
best: 2-stratonovich-heun   decided_by: score
```

`decided_by: score` — the ranker separated the pool on the first key and never reached provenance or
the quality index. All three plans carry provenance `analytic`.

**The reader should know what that conceals.** `cli.py rank` also printed:

> note: the index above is over structure only. accuracy, order, consistency did not resolve on
> every plan in this pool, and a component no comparison was made on does not decide one — see
> `ranking.comparable_keys`. Each plan's own index, over whatever it measured, is
> `2-stratonovich-heun=0.463`, `1-stratonovich-implicit-midpoint=0.478`, `3-euler-maruyama=0.163`.

So on each plan's own quality index, **plan 1 (0.478) edges plan 2 (0.463)**, and on every accuracy
and structural statistic the kernel reports, plan 1 is the better solver: a smaller residual `dt`
bias (2.8e-03 vs 1.8e-03 of tolerance is a wash, but 1.07e-02 vs 1.02e-02 worst moment error is
also a wash, and `max||X|-1|` of 4e-15 vs 7.7e-03 is not). Plan 2 wins on `score`, and the six-point
gap between them is entirely the Dynkin verdict described above. **The recommendation is plan 2
because the score is the pipeline's verdict and the conductor does not overturn it — but a reader
choosing a scheme for this problem should read plan 1's Dynkin finding before accepting that
ordering.**

Phase 2b (the winner's one bias-removal cycle) **did not run**: it fires only on a plan that left the
loop `certified: true`, and none did. Both order-1 plans did take the in-loop bias instruction, and
their outcomes are recorded above — one accepted cleanly, one improved its accuracy and lost six
points on an unrelated check.

---

## Verification gaps

These bound what the run established.

1. **`dynkin_ok` never reached a verdict on the winner — this is the one point.** Plan 2's Dynkin
   outcome is **`unresolved`**, not a pass. Its `dynkin_z: 5.0943` against `dynkin_bar: 2.5310` sits
   on the same `X1^2 + X2^2` statistic as plan 1's failure, with the same cause: the generator
   residual is essentially zero (−5.3e-6 at the finest level, extrapolated −9.5e-08), its measured
   rate is 2.03 against the assumed first order, and the extrapolated limit moves by 5.5 paired
   standard errors between the two level windows — the levels are not in one asymptotic regime. The
   kernel abstained rather than failing it. **Nothing in this run establishes that the winner's
   paths came from this SDE.** Exact terminal moments cannot stand in: two processes can share two
   moments, and a mis-transcribed coefficient would pass the moment check identically. This matters
   more here than on a typical problem, because the Stratonovich-vs-Itô drift is precisely what the
   Dynkin residual measures and precisely what this problem is built to discriminate.
2. **The Dynkin instrument appears to be the limiting factor on both order-1 plans, in opposite
   directions.** Plan 1 *failed* at roundoff scale (residual ~1e-11 ± 3.5e-12) and plan 2
   *abstained* on a non-asymptotic ladder, both on the conserved quantity `X1^2 + X2^2`. Both
   evaluators reached the same diagnosis independently, and the conductor records it as a finding
   about the check rather than about the schemes. It is a candidate kernel defect: a residual whose
   magnitude and standard error are both at floating-point noise should abstain, not fail.
3. **`second_moment_sum` is declared and never evaluated.** `invariants_not_reported:
   ['second_moment_sum']` on all three plans — the spec declares `E[X1^2] + E[X2^2] = 1` and carries
   no `tol` for it because nothing evaluates it. `invariants_ok: true` on the winner is clean over
   the invariants the kernel *could* evaluate; this one is a property the run did not establish.
4. **The pathwise strong-error gate of R18 was never measured as a number.** See *Ambiguities*. The
   strong-order ladder (1.92 ± 0.013) and the moment tolerances stand in for it.
5. **No functional is declared in this spec**, so `functionals_ok: unavailable` and
   `cross_check_ok: unavailable` on every plan. The accuracy numbers reported are state moments, and
   they are the problem's own graded quantities here — there is no functional being substituted for.
6. **Accuracy is reported at 50 000 paths, not at the graded 2 000** (R9 above). The only statistic
   measured at the graded count is plan 3's amplitude breach fraction.
7. **Plan 3's order ladder and Dynkin check are `not_run`**, skipped for cost once the structural
   gate failed. That is neither a pass nor a failure: its strong order and generator consistency are
   unmeasured.
8. **`estimator_stability` is the tightest passing check in the pool and it barely passes.** The
   winner's `invariant_max_drift` is 8.582e-03 against a 1e-02 bar — `invariant_worst_headroom:
   0.0664`, i.e. **0.066 decades**. Plan 1 is the same at 0.0677. The evaluator's reading is that
   the drift is Monte-Carlo noise at 50k paths rather than scheme error (the moment biases are below
   2e-04 relative), but that was not tested by varying the path count.
9. **`surrogate_ok` is unavailable** on all plans — expected on Path A, where the closed form is the
   reference, and recorded for completeness.
10. **`2-stratonovich-heun` ships a stale scheme label.** Its frontmatter declares
    `scheme_family: milstein` and its `scheme:` title line still names Heun, while the code is a
    4-stage derivative-free Stratonovich RK with no Milstein correction. The evaluator called it a
    documentation finding and it carried **no score cost** (the spec sets `milstein_eligible: true`,
    so no `milstein_claim` cap fired, and `milstein` is what selects the one-sided order-1 floor the
    scheme is correctly judged against). It is recorded here because `scheme_family` is what the
    order check reads and nothing downstream re-derives the family from the code.

**Final gate:** `uv run python verifylib/cli.py gate workspace/sde_kubo_oscillator` → `ok: no
findings`, exit 0. The pool gate at plan-creation time also returned `ok: no findings` — no missing
`terminal_law` draw, no unintegrated conditionally-tractable component, no untreated boundary, no
duplicated scheme. No `GUARDRAIL_UNSATISFIED` file was written.

---

## The spec repair, and what it cost

Recorded here because it changed two plans' scores and is the run's most consequential event.

Round 1 scored both order-1 plans at 9, uncertified, with `dynkin_ok: unavailable` — and both
evaluators independently reported the same upstream cause. `verifylib/kernel/sde/dynkin.py:203`
reads a **top-level** `initial_condition`, and its only other route reads `parameters.X_0/Y_0/Z_0`;
the spec carried `parameters.X1_0/X2_0` and a nested `verification.initial_condition`, so neither
route hit and the kernel set `random_initial_condition: true`. The analytic rubric's 9 row then bound
the entire pool with no lever available to any solver.

The formulator was re-dispatched **once** (its single allowed re-dispatch) to repair only that. It
confirmed the cause in source, added a top-level `initial_condition: {law: "deterministic", value:
[1.0, 0.0]}` as an explicit restatement of the same deterministic state — keeping
`parameters.X1_0/X2_0` and `verification.initial_condition` untouched — amended R3's `spec_path` to
point at all three, and verified that `initial_moments()` now returns `x0 = [1.0, 0.0]` with
`random_initial_condition: false`. No mathematics, parameter, tolerance or ledger status changed; the
tally stayed 18 `mapped` / 1 `ambiguous` / 0 `dropped`, and `check-spec` still exits 0.

The repair worked as intended — Dynkin reached a verdict on both order-1 plans for the first time.
Its effect on the scores was not what was intended: plan 2 held at 9 with an `unresolved` verdict,
and plan 1 fell from 9 to 3 on a failed verdict whose residual is at roundoff. The run therefore
ends with better solvers and a lower top score than it had after round 1, and the reason is item 2
above rather than anything either solver did.
