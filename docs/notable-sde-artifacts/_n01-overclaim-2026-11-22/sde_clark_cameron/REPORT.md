# sde_clark_cameron — Conductor Report

**Problem type:** SDE, family `clark_cameron` — the canonical non-commutative-noise system of
Clark & Cameron (1980):

```
dS = mu dt + dW2,   S(0) = 1
dU = S dW1,         U(0) = 0        mu = 1, T = 1
```

2-dimensional state, 2-dimensional noise, multiplicative (the diffusion row for `U` reads `S`),
non-commutative. `milstein_eligible: false` with reason `levy_area`: the `U` correction is the
iterated integral `I_21`, so a scheme claiming order 1.0 must build it from the 256 declared
sub-increments per step.

**Evidence path:** A (closed form). **Winner provenance: `analytic`** — the exact terminal moments,
**validated** by the kernel against the spec's own declared structure (`reference_outcome:
validated`), not accepted on a quote. All three plans were scored against
`E[S_T] = 2`, `Var S_T = 1`, `E[U_T] = 0`, `Var U_T = 17/6`. This is the strongest tier the
pipeline has; no surrogate, manufactured or self-convergence fallback was used anywhere.

---

## Ambiguities recorded at formulation

Both are the same gap, and the reader should know about it because it is the one place where the
problem the pipeline scored differs in *form* from the problem as stated.

- **R7** — quoted from `problem.md`: *"This problem is graded **path by path**, not in distribution:
  the harness compares your terminal states to a reference solution driven by the very same Brownian
  path."* The formulator's choice: there is no `evaluation_thresholds` key carrying a pathwise
  strong-error tolerance against an external fine-grid reference, so it was carried as
  `rel_l2_err_max = 0.02` with the pathwise definition stated in `implementation_notes`.
- **R18** — quoted: *"the harness gates the relative strong error
  `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 2%"*. Same gap; `0.02`
  recorded verbatim at the right magnitude under a metric the kernel can evaluate.

The formulator's argument for calling these `ambiguous` rather than `dropped` was that both readings
separate the same two scheme families here, because the only pathwise error source in `U` is the
omitted Lévy area — which is exactly what the CRN strong-order ladder measures. **That argument is
sound as far as the ladder goes, and it is load-bearing for how this report should be read:** the
kernel never evaluated the 2% pathwise gate itself. It measured moment errors and a strong-order
rate. See *Verification gaps*, which is where the consequence lands.

Ledger tally: **21 entries, 19 `mapped`, 2 `ambiguous`, 0 `dropped`.** `check-spec`: `ok: no findings`.

---

## Plans

All three plans were implemented, all three certified at **10/10** with provenance `analytic`, and
every solver consumed the supplied `dW` (no `exact_sampling_components` declared anywhere, by
design — see *Pool findings*).

| # | Plan | Scheme | Score | Cert | Provenance | est_rel_err | measured? | Strong order (declared floor) | iter |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `1-levy-area-milstein-midpoint-drift` | Lévy-area Milstein, `scheme_family: milstein`, `levy_area_treatment: sub-increments`; exact `S` leg, `U += (S_n + mu·dt/2)·dW1 + I_21` | **10** | yes | `analytic` | **5.667e-03** | measured, **but below its own error bar** (see below) | **0.9932 ± 0.0110** (1.0) | 2 |
| 2 | `2-levy-area-milstein` | Lévy-area Milstein, the spec's `milstein_correction` implemented literally: frozen integrand `U += S_n·dW1 + I_21` | **10** | yes | `analytic` | 0.0261 | measured | **0.9922 ± 0.0096** (1.0) | 2 |
| 3 | `3-euler-maruyama-control` | Plain 2-noise Euler–Maruyama on the step increments alone, no `I_21`; `scheme_family: euler-maruyama` | **10** | yes | `analytic` | 0.0337 | measured | **0.4868 ± 0.0110** (0.5) | 2 |

No `agent_cap` was applied to any plan. No `grading_violation`: every `solver.py` hash was identical
before and after every evaluator pass, and each plan's `solver_sha256` in its `<metrics>` block
matches the file on disk.

Key shared metrics, last pass: `resolved: true` on all three, `mc_se_rel` 8.28e-03 – 8.36e-03 at
50 000 paths; `constraints_ok: true` (`finite`); `invariants_ok: true`; `dynkin_ok: true` with
`dynkin_z` 1.3946 / 1.3980 / 0.9392 against a family-wise bar of 2.4284; `order_check: true`.
`solve_time_s` 2.14 / 2.24 / 1.42 s (reported as a diagnostic; cost is not a ranking key).

**The strong-order ladder is the pool's real result.** It separates the two Lévy-area schemes
(≈0.993, ≈0.992) from the Euler control (0.487) with standard errors of ~0.01 — tight enough to
resolve the families apart. That is the Clark–Cameron bound measured rather than assumed, and it is
what the spec's twin floors (`expected_strong_order: 0.5` beside
`expected_strong_order_milstein: 1.0`) were declared to discriminate.

---

## Best plan

The ranker's verdict, quoted verbatim:

```
best: 3-euler-maruyama-control   decided_by: sqi
note: the index above is over order, structure, consistency only. accuracy did not resolve on
every plan in this pool, and a component no comparison was made on does not decide one -- see
ranking.comparable_keys. Each plan's own index, over whatever it measured, is
3-euler-maruyama-control=0.796, 1-levy-area-milstein-midpoint-drift=0.892,
2-levy-area-milstein=0.837.
```

`best_plan` is recorded as `3-euler-maruyama-control` because that is what the ranker returned and
the conductor does not rank by hand. **The reader should not stop there, and this is the most
important finding in this report.**

`decided_by: sqi` — but the index that decided it was computed over **three** components, not four.
Accuracy was dropped from the pool comparison because it did not resolve on plan 1: that plan's
`estimated_rel_error` of **5.667e-03 sits below its own `mc_se_rel` of 8.323e-03**, so the kernel
correctly declined to locate an error inside its own noise, and `ranking.comparable_keys` then
restricted the comparison to the components every plan measured. The restricted index reads
0.918 / 0.892 / 0.892 and the Euler control wins it on the residual-slope term (0.613 vs 0.426).
Each plan's own unrestricted index runs the other way: **0.796 / 0.892 / 0.837**, i.e. plan 1 first
and the control last.

Two things compound this, and both are properties of the instruments rather than of the schemes:

1. **Order fidelity is charged against each plan's own *design* order.** The control declares
   design order 0.5 and measured 0.4868, scoring 0.974; plan 1 declares 1.0 and measured 0.9932,
   scoring 0.993. A scheme designed to converge slowly gets near-full marks for converging exactly
   as slowly as designed. As a fidelity measure that is correct and intended. It does mean `sqi`
   cannot see that order 1.0 is *better here* than order 0.5 — and on this problem that gap is the
   entire point.
2. **The problem's own gate is pathwise, and the kernel did not evaluate it.** Per R7/R18 the
   harness gates relative strong error at **2%**. The plan-creator's design estimates were ≈0.7%
   for plan 1, ≈1.4% for plan 2, and **≈8.4% for the Euler control** — i.e. the control is expected
   to *fail* the gate the problem is actually scored on, and its measured strong order of 0.487 is
   consistent with that estimate. The pathwise error is not among the numbers the `sqi` comparison
   was able to range over.

This is the `sde_sciml_stiffquad_mild` failure mode in near-pure form — the best plan losing for
being too accurate to resolve — reached through `comparable_keys`, the mechanism added to prevent
it. Restricting the comparison to components the whole pool measured is right in general; here the
component it had to drop was the only one separating the pool, and what remained handed the decision
to a residual slope.

**On the evidence actually measured, `1-levy-area-milstein-midpoint-drift` is the plan to ship:**
highest unrestricted quality index (0.892), lowest moment error (5.667e-03, ~4.6× and ~5.9× below
the other two), strong order 0.9932 ± 0.0110 against the Lévy-area floor of 1.0, clean Dynkin, and
the lowest design-estimated pathwise error (≈0.7%) against the 2% gate the problem states. Plan 2 is
the same family with a larger error constant and corroborates it at the same order. The control's 10
is a true certification — it genuinely clears every gate the *spec's* moment tolerances impose — and
it is nonetheless the pool's weakest scheme for this problem.

The pipeline could not settle this by one more measurement: `num_paths` is fixed by the contract at
50 000, so no solver cycle can shrink `mc_se_rel` enough to resolve plan 1's accuracy, and all three
plans were certified and therefore ineligible for further cycles.

---

## Phase 2b — the winner's bias-removal cycle

It fired, on the ranker's winner (`3-euler-maruyama-control`, `moment_bias_actionable: true`).

| | |
|---|---|
| Binding moment | `variance` (of `U`) |
| `moment_bias_frac_tol` before | **0.2880** |
| `moment_bias_frac_tol` after | **0.2880 (unchanged)** |
| Outcome | **no change — `polish_reverted: true`** |

The certified artifact was preserved first (`solver.certified.py`, `SOLUTION.certified.md`). The
solver declined to change anything and gave a reason, which is a result and is recorded as such:

- The bias sits in the `U` update's **left-endpoint quadrature**, `U += S_n·dW1`. Since `S` is driven
  only by `W2` and is independent of `W1`, this yields `E[U²]` as a left-endpoint Riemann sum of
  `E[S_t²] = (1+t)² + t`, which rises from 1 to 5 on `[0,1]`; the predicted bias
  `≈ −(dt/2)·(5−1) = −0.08` at `dt = 0.04` is ≈2.8% of 17/6 and is consistent with the kernel's
  0.288-of-tolerance reading.
- **It is not caused by the omitted Lévy area**: the `I_21` terms are mean-zero and contribute
  nothing to the variance bias.
- About −0.06 of the −0.08 is removable by a drift-midpoint shift (`S_n + mu·dt/2`, no `dW_sub`
  needed) — which is precisely what plan 1 already does. The remaining −0.02 needs within-step
  `W2` information, i.e. sub-increments.
- The solver declined because that shift would stop the plan being the plain Euler–Maruyama control
  the pool is built around, and the conductor's dispatch had explicitly ruled out turning it into a
  Lévy-area scheme (that would duplicate plan 2).

Since the file was byte-identical afterwards, no evaluator pass was spent re-measuring it: an
unchanged solver cannot report a strictly lower `moment_bias_frac_tol`, so Phase 2b's accept
condition was already determined. Nothing needed restoring — `solver.py` still hashes to
`e79300b8…`, matching its `<metrics>` `solver_sha256`.

Note what this says in context: the bias the kernel measured on the control was real, correctly
located, and **already removed in plan 1** by the midpoint drift term. The instrument pointed at a
genuine defect; the fix for it is a different plan in the same pool.

---

## Pool findings

The pre-solve pool gate raised one warning, and it is unresolved by choice. `cli.py gate` reports
(both before any solver ran and over the finished workspace):

> state component 1 is **conditionally tractable** … No plan in the pool integrates it out — none
> declares `exact_sampling_components: 1`. Conditional Monte Carlo (Rao-Blackwellisation) removes
> this component's sampling variance entirely … At least one plan should condition on the driving
> components and integrate component 1 out.

The plan-creator was re-dispatched once for such a plan and **declined**, without disputing the
detection. Its reasons, which the conductor accepted:

1. The technique is genuinely available in distribution — conditionally on the `W2` path,
   `U(T) = ∫S dW1` is exactly centred Gaussian with variance `V = ∫S² ds`, and `V` is computable
   per path from `dW_sub`. But **this problem's gate is pathwise against a shared Brownian path**,
   and conditioning replaces that path's realised `U(T) = √V·Z_true` with `√V·Z` for a fresh,
   independent `Z`: right conditional law, wrong draw. The resulting error is
   `E[(U_cond − U_true)²] = 2·E[V] = 2·(17/6)`, ≈141% relative on the `U` column against a 2% bar.
   Conditioning does not reduce that error; it *is* the error.
2. `exact_sampling_components: 1` makes the kernel abstain on component 1 — and component 1 is
   where this entire problem lives, since `I_21` enters `U` and nothing else while `S` is exact
   under every plan in the pool. The declaration would have forfeited the strong-order ladder that
   is the pool's **only** discriminator, which is what the warning's silence would have cost.
3. `Var(U_T) = 17/6` is already closed form, so the estimator's moment-side output converges to a
   number the spec states exactly.

**Recorded as the pool's verification ceiling:** no plan here removes `U`'s sampling variance
structurally, and the variance-reduction axis was deliberately left unexplored in favour of
preserving the order measurement.

---

## Verification gaps

These bound what this run established.

1. **The 2% pathwise strong-error gate (R7, R18) was never evaluated by the pipeline.** This is the
   criterion `problem.md` states the problem is scored on, and no `evaluation_thresholds` key
   carries it, so the kernel measured moment errors and a convergence *rate* instead. The rate is
   strong evidence about the schemes and it is not the gate. Every "10" in this report is a
   moment-space-plus-order certification; the pathwise numbers quoted for the three plans
   (≈0.7% / ≈1.4% / ≈8.4%) are the **plan-creator's design estimates, not measurements**. In
   particular, the ranker's winner is the one plan expected to fail that gate.
2. **Accuracy did not resolve on plan 1.** `estimated_rel_error` 5.667e-03 < `mc_se_rel` 8.323e-03.
   The accuracy of the pool's best plan was **unmeasured at the path count it ran** — not measured
   and found good. The consequence for selection is in *Best plan* above. (All three plans do carry
   `resolved: true`, which is the separate moment-space variance claim; the two are different
   questions and neither substitutes for the other.)
3. **Three declared invariants were never checked** — `invariants_not_reported:
   ['symmetry_of_U', 'zero_cross_moment', 'martingale']`, on all three plans. `invariants_ok: true`
   sits beside them and does not cover them. `zero_cross_moment` is the material loss: the
   formulator declared it specifically as the check that would catch a solver swapping the two noise
   channels while keeping both marginal variances right. Each solver verified channel order
   informally instead (sample `corr(S,U)` ≈ −6e-4 to −6.8e-3 against an SE of ~4.5e-3), which is
   reassuring and is not the declared check. `martingale` on `U` was declared ungated deliberately:
   its bar is `ci_mult × SE`, so gating an exactly-true property would fail a correct solver roughly
   one seed in twenty into a score of 4.
4. **Pool gate warning unresolved** — the conditional-MC / Rao-Blackwellisation technique is
   structurally available on component 1 and no plan implements it. See *Pool findings* for why, and
   read it as a declared ceiling rather than an oversight.
5. **`cross_check_ok: unavailable` and `functionals_ok: unavailable`** on all three plans: the spec
   declares no gated functional and no functional ordering, so neither check had anything to run.
   Accuracy therefore ranks on `estimated_rel_error` (a state-moment proxy) rather than on a graded
   quantity of interest — there is no functional on this problem whose error could have been
   reported instead.
6. **The kernel's reference side ran coarser in the iterated integral than the spec advertises.**
   `sub_increments_per_step` is declared 256; to fit a 1 GiB memory cap the kernel cut it to **53**
   on the graded run and **64 → 33** in the order ladder. This affects the kernel's reference, not
   any solver. The formulator's assessment is that the Lévy-area approximation error at 53 sub-steps
   remains well below the `O(dt)` term the ladder reads, so the order estimates stand; the margin is
   nonetheless smaller than the declared 256 implies. If it ever needs tightening, the lever is
   `num_paths` (50 000, against a resolution bar of ~11 100 from `U`'s measured excess kurtosis of
   1.46), not the sub-increment count.
7. **`terminal_law` abstained by design.** `S_T` is exactly normal but `U_T` is a leptokurtic
   variance mixture of centred normals matching none of the three implemented laws, so the
   distributional tiers were not declared and no KS-type shape test ran. `kolmogorov` is null (the
   moment hierarchy closes exactly, so a degenerate 2-D solve would be strictly weaker and costlier)
   and `stationary_density.ergodic` is false (neither component is mean-reverting) — both with
   stated reasons.
8. **What the Dynkin pass does not establish.** The generator is built from the spec's *own* drift
   and diffusion, so a drift or diffusion mis-transcribed from `problem.md` into
   `problem_spec.json` would pass unanimously on all three plans. Two of the three evaluators said
   so explicitly. Every plan in this pool descends from one spec; the external harness is the only
   spec-independent check.
9. **One solver carries a latent fallback defect, never exercised.** Plan 1's (and plan 2's)
   `dW_sub is None` fallback sets `I21 = 0.5*dW1*dW2`, which is not an Itô Lévy area and would bias
   `Var U` if ever reached; the correct fallback is `I21 = 0`. It did not run in any scored
   evaluation. Flagged as optional by the evaluator and left unfixed — it would matter if the harness
   ever calls the solver without sub-increments.

**Process note, recorded because it shaped the run.** The first evaluation round returned
9 / uncertified on *all three* plans, with `dynkin_ok: unavailable` and no `dynkin_z` emitted: the
spec's `verification.initial_condition.value` carried the strings `["S_0","U_0"]`, so the kernel
could not form `E[φ(X_0)]`. No solver change could have fixed it. The formulator was re-dispatched;
its first repair wrote numeric values into the *nested* `verification.initial_condition`, which left
the score unchanged because `verifylib/kernel/sde/dynkin.py` (`initial_moments`, line 203) reads
`spec.get("initial_condition")` at the **top level**. The conductor verified that against the kernel
source directly rather than on the evaluators' report, and a second dispatch moved the key to the
top level; all three plans then certified at 10. `check-spec` returned `ok: no findings` at every
stage, including both wrong placements — **it validates the block's shape, not its location**, which
is a real gap in that gate, and the reason a clean spec gate sat beside an uncertifiable pool for two
rounds. No `solver.py` changed across any of these re-evaluations (hashes identical throughout), so
they were not refine cycles and were not counted as iterations.

`GUARDRAIL_UNSATISFIED` is absent. Final `cli.py gate` over the finished workspace: exit 0, the one
pool warning above.
