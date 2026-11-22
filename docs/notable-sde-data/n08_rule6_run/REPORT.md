# REPORT — `sde_three_halves_vol`

**Problem type:** SDE — scalar, multiplicative, non-linear, stiff. Family recorded as
`three_halves_stochastic_volatility`:

```
dV = kappa * V * (theta - V) dt + epsilon * V**1.5 dW,   V(0) = 0.15
kappa = 22.84,  theta = 0.218,  epsilon = 8.56,  T = 0.1
```

**Path B** — `analytic_moments.has_analytic_solution: false`. There is no closed form for V's own
terminal moments: the moment hierarchy does not close (quadratic drift, cubic `g²`), and V's moments
are genuinely ill-conditioned — the invariant law is `InverseGamma(2.6234, 0.1359)`, so `E[V³]` is
infinite and V's sample variance never settles.

**Outcome: no plan certified. All three plans finished at score 9/10, provenance `surrogate`, and
the ceiling is a spec-level defect rather than a solver defect** (see *Why nothing certified*).
Ranked winner: **`1-lamperti-strang-split`**.

---

## Provenance of the scoring

`surrogate` on all three plans. **Read this before the scores.** The evidence splits in two, and the
halves point in opposite directions:

- **The graded quantities rest on an exact reference.** `Y = 1/V` is exactly a CIR process
  (`a = kappa*theta = 4.97912`, `b = (kappa + epsilon²)/(kappa*theta) = 19.30333`, `sigma_Y = epsilon`,
  `Y_0 = 6.6667`; Feller ratio 2.6234 > 1, which is what licenses `problem.md`'s strict-positivity
  claim). The formulator derived `inv_var_mean = 11.622786285052442` and
  `inv_var_variance = 45.23510647598556` from it and then validated them two independent ways that do
  not use the identity — a second derivation through the Lamperti transform `X = V^{-1/2}`, and a
  direct Kolmogorov solve of the V-equation as stated (11.6244 and 180.40 for the two raw reciprocal
  moments at the finest grid, converging on 11.6228 and 180.3243). Both gated `martingale`
  constraints are checked against those values and both pass on all three final plans.
- **V's own moments rest on nothing usable.** The only surrogate route the kernel obtained is
  `verification.stationary_density` at `T_stat = 6`, and that reference does not describe the graded
  horizon — see below. The `mean_rel_err` of 0.56–0.58 and `variance_rel_err` of 0.70–2.26 in every
  metrics block measure the distance between the transient law at T = 0.1 and the stationary law, not
  solver error. **They must not be read as accuracy.**

So a 9 here is: the graded functionals are clean against a twice-validated exact reference, the orders
and the generator check pass, every gated constraint passes — and the one comparison the rubric needs
to certify could not be made.

## Ambiguities recorded in Phase 1

One, and the formulator's choice was documented:

- **R14** — quote: `def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict`.
  `problem.md` calls for "exactly this function"; the project's SDE contract adds two optional kwargs
  (`dW=None`, `observables=None`) that the kernel needs for the common-random-number strong-order
  study and the Dynkin generator check. **Chosen:** keep the four `problem.md` parameters with their
  exact names and semantics and add the two optional kwargs defaulting to `None`, so a call with only
  the stated arguments reproduces the stated behaviour exactly. Flagged because it adds to a signature
  `problem.md` calls exact; without it no plan could be certified above `self_convergence`.

No requirement was `dropped` (25 entries, R1–R25), and `cli.py check-spec` exited 0.

## Why nothing certified

Three evaluators reached this independently, and two of them backed it with their own seed studies:

`verification.stationary_density` is declared with `T_stat = 6` as the surrogate for V's moments, but
the graded time is **T = 0.1** and the reciprocal process relaxes on `1/(kappa*theta) = 0.2` — about
half a relaxation time. V is still transient there (solver mean ≈ 0.132 against the stationary
reference's 0.0838). The kernel compared the two anyway, because the route passed its trust guard, and
the comparison therefore cannot decide: `moments_ok` and `variance_ok` come back `unavailable`,
`resolved` is `false`, and `certified` is `false` on every plan. The rubric's 9 row — "the orders,
Dynkin and the constraints all hold and the Richardson moments are stable across the ladder" — is what
each plan landed on.

No path count and no scheme change reaches this. V's own variance is structurally unresolvable here
(`E[V³]` infinite under the invariant law; measured excess kurtosis of V ran 182 to ~5e4 across plans),
and the spec's own notes say V's moments are not graded and have no trustworthy surrogate. **This is a
formulator/spec-declaration gap, and it capped the whole pool at 9.** The fix is at the spec: either
drop the `stationary_density` route for this transient horizon, or route the kernel's moment comparison
to the exact reciprocal-CIR reference the spec already carries.

## The accuracy that was measured

The spec declares two gated functionals, so these — not `estimated_rel_error` — are the error on the
quantity the problem asks about. Binding functional on all three plans: **`inv_var_variance`**,
tolerance `0.10`.

| plan | `functional_rel_error` | bias | sampling | limited by | `inv_var_mean` on the ladder (tol 0.05) |
|---|---|---|---|---|---|
| 1-lamperti-strang-split | **2.478e-03** | 4.58e-08 | 2.478e-03 | sampling | 1.7e-05 |
| 2-split-step-milstein-v | 8.660e-03 | 8.80e-07 | 8.659e-03 | sampling | 6.1e-06 |
| 3-truncated-milstein-log | 1.21e-02 | 2.282e-03 | 9.825e-03 | sampling | 8.6e-04 |

All three sit roughly 8–40× inside the gate, and all three are sampling-limited at the kernel's
diagnostic path count. The bias terms are **not** abstentions: no plan declared
`exact_sampling_components`, every plan consumes the supplied `dW`, and the `dt` ladder produced a real
bias estimate for each.

## Plans

### 1 — `1-lamperti-strang-split` (ranked winner)

- **Scheme** (`scheme_family: other`, `boundary_treatment: reflection`, design order 0.5): Lamperti
  transform `X = V^{-1/2}`, where the noise becomes constant and additive. Each step is a Strang
  composition in which **both sub-flows are exact** — the drift ODE linearises under `u = X²` to
  `du/dt = -a*u + 2c`, and the noise sub-step is an exact additive increment — with `X = |X|`
  reflection at the entrance boundary. Variance reduction: antithetic pairs plus a control variate
  built from the Brownian aggregate `S = sum(dW)` via `E[S² − T] = 0` and `E[S⁴ − 3T²] = 0`.
- **Score 9/10**, `certified: false`, provenance `surrogate`, **iter 3**, no `agent_cap`.
- **Graded accuracy:** `inv_var_variance` 2.478e-03 against 0.10 (bias 4.6e-08 + sampling 2.478e-03);
  `inv_var_mean` 1.7e-05 relative. Values 11.62259 and 45.2974 against 11.622786 and 45.235106.
- **`estimated_rel_error: 0.7516` — this is an estimate, and it is the invalid stationary comparison on
  V's own variance. It is not this plan's error.**
- **Observed strong order** 0.886 ± 0.098 against a floor and design rate of 0.5 (one-sided; an
  overshoot is flagged, not failed). `order_check: true`.
- **Dynkin** passes: z = 1.84 against a bar of 2.43, replicated to 3 seeds (median z 1.50).
- Constraints and invariants all pass; `invariant_max_drift` 0.333 with 0.31 decades of headroom —
  the best margin in the pool. `solve_time_s` 0.115.
- **Independently seed-studied by its evaluator** (200 seeds at 50 000 paths): the control variate is
  unbiased (−9.5e-05 on the mean, +3.9e-03 on the variance, each under 1 s.e. of the exact value) and
  its reported standard errors are 0.973 and 0.989 of the empirical between-seed sd, so the error bars
  are not optimistic. The variate cuts the sampling error 5.7× on the mean and 3.8× on the variance.
- **Disclosure the evaluator asked be carried:** the returned `functionals` are the control-variate-
  adjusted estimators, not the plain `mean(1/V_T)` / `var(1/V_T)` of the returned `terminal_paths`
  (at 50 000 paths the plain values are 11.6167 and 45.0455, i.e. 0.20 and 2.2 plain standard errors
  away). Same data, different estimator.
- **A contract violation was found and fixed inside the run.** In its round-2 cycle this plan pooled
  the two functionals over 8 antithetic blocks, which cut the sampling term but computed them from 8×
  the declared path budget rather than from the `terminal_paths` the call returns. The evaluator
  flagged it as a contract finding; the round-3 cycle reverted it and replaced it with the
  no-extra-path control variate above. The final solver honours the contract. Recorded because it
  bears on how the earlier numbers in `SOLUTION.md` should be read.

### 2 — `2-split-step-milstein-v`

- **Scheme** (`scheme_family: strang-splitting`, design order 0.5): Strang splitting in the original
  variable — exact logistic drift half-flows (linear in `u = 1/V` after the Stratonovich conversion
  with `c = kappa + 0.75*epsilon²`) composed with the **exact additive-noise flow** in `Z = V^{-1/2}`.
  Positivity is algebraic: `u` is bounded below by `(c/r)(1 − e^{−r dt/2}) > 0`, with no clip, no
  truncation and no cap.
- **Score 9/10**, `certified: false`, provenance `surrogate`, **iter 3**, no `agent_cap`.
- **This plan was repaired by the bias instruction.** Round 1 scored **3/10**: both gated `martingale`
  constraints failed on a real `dt` bias in its Milstein noise sub-step (`E[1/V]` off 0.104 against a
  2-s.e. bar of 0.059; `E[(1/V)²]` off 3.77 against 1.89). The kernel named the moment —
  `moment_bias_binding: mean`, `moment_bias_frac_tol: 0.2736`, `moment_bias_actionable: true` — and the
  solver traced the bias to the Milstein factor `1 + s + 0.75s²`, which only approximates the
  multiplicative noise flow to O(s³) while `|s|` reaches ≈0.5 in the small-V tail that dominates the
  reciprocal moments. Replacing that sub-step with its exact flow took the score 3 → 9 and both gates
  now pass. **`scheme_family` was re-declared from `milstein` to `strang-splitting`** to match what was
  written, as required.
- **Graded accuracy:** `inv_var_variance` 8.660e-03 against 0.10; `inv_var_mean` 6.1e-06.
  `estimated_rel_error: 0.6990` is again the invalid stationary comparison, not this plan's error.
- **Observed strong order** 0.886 ± 0.098 against 0.5. Dynkin median z 1.50 over 3 seeds, bar 2.43.
- Constraints and invariants pass, but **the martingale headroom is the thinnest in the pool**: worst
  drift 1.360 against a bar of 1.930, 0.15 decades. `solve_time_s` 0.109.
- Its round-3 cycle also fixed a `SOLUTION.md`/code mismatch the previous review flagged as cap-worthy
  (the body still described a truncated-Milstein sub-step with a `V_cap`, and the solver returned a
  `cap_activations` field hardcoded to zero). Both are gone and the evaluator confirmed body and code
  now agree.
- It measured antithetic pairing and **rejected it with evidence**: 2.7× better on `E[1/V]`, worse on
  the binding `Var[1/V]` (0.46 against 0.40 at 50 000 paths). The evaluator agreed that call was right.

### 3 — `3-truncated-milstein-log`

- **Scheme** (`scheme_family: tamed-milstein`, design order 1.0): fully explicit Mao-truncated Milstein
  in `L = log V`, truncation radius `R = dt^{-1/4}`, positivity free through `V = exp(L)`. Its role in
  the pool is an independent-coordinate cross-check — it is the only plan advancing the stiff quadratic
  drift explicitly. Its round-2 cycle raised that drift sub-step from explicit Euler to explicit
  trapezoid (Heun); its round-3 cycle added antithetic pairing.
- **Score 9/10**, `certified: false`, provenance `surrogate`, **iter 3**, no `agent_cap`.
- **Graded accuracy:** `inv_var_variance` 1.21e-02 against 0.10 (bias 2.282e-03 + sampling 9.825e-03) —
  the pool's largest, and the only plan whose bias term is a material share of its error.
  `inv_var_mean` 8.6e-04.
- **`estimated_rel_error: 2.2648` is the invalid stationary comparison on V's variance** and, on this
  plan, is additionally meaningless because V's own sample variance has infinite variance in the log
  coordinate's heavy tail (its evaluator measured excess kurtosis ≈5e4, and `mc_se_rel` is 0.3777 —
  larger than several plans' reported errors). **The accuracy of this plan's V-moments was not measured
  at the path count it ran.** The graded functionals above are what was measured.
- **It carries a measured, real `dt` bias**, confirmed by its evaluator over 300 seeds at two path
  counts: `E[1/V]` sits +0.364% over the exact value (≈600 standard errors, identical at 50 000 and
  200 000 paths and under i.i.d. noise, so it is `dt` bias and not seed noise), and `Var[1/V]` carries a
  matching −0.90%. Both are far inside their gates. The kernel also flags
  `moment_bias_binding: mean`, `moment_bias_frac_tol: 0.2616`, `moment_bias_actionable: true`. The
  solver identified the first-order noise/Milstein sub-step as the remaining floor for this scheme and
  measured three drift treatments under common random numbers before settling on Heun.
- **Observed strong order 0.7999 ± 0.1281 against a floor and design rate of 1.0** — the only plan in
  the pool reading below its design rate. `order_check` still passed (one-sided band, median-based
  order 1.02), but it is the weakest order evidence here.
- Its evaluator verified the paired-sample error bars over 300 seeds: reported s.e. matches the
  empirical between-seed sd within ≈5%, and centred 2-s.e. coverage is 0.947 for both functionals.
  It also measured that the antithetic pairing **does not help the binding functional** — `Var[1/V]`
  sd went 0.387 → 0.435 at 50 000 paths and 0.178 → 0.215 at 200 000. `solve_time_s` 0.125.

## Best plan recommendation

**`1-lamperti-strang-split`.** The ranker's line, verbatim:

```
best: 1-lamperti-strang-split   decided_by: sqi
```

All three plans tie on the first two keys — score 9.0 and provenance `surrogate` — so the quality index
separated them, and it did so on a real margin rather than inside the noise: `sqi` 0.640 / 0.523 /
0.438, driven mainly by accuracy headroom (0.803 / 0.531 / 0.459) with the full four components
comparable across the pool. That ordering is the same one the graded functional errors give directly
(2.478e-03 / 8.660e-03 / 1.21e-02), and plan 1 also has the best structural margin (0.103 against 0.051
and 0.044) and the only order reading at or above its design rate. Plan 3 is last on every component
that separates them. Cost was reported and not ranked on (0.115 / 0.109 / 0.125 s — indistinguishable
anyway).

Qualifier on the winner's margin: its accuracy advantage is partly estimator and partly scheme. The
control variate buys 3.8–5.7× of it at no extra paths, and its evaluator verified that reduction is
unbiased and honestly priced. Its `dt` bias (4.6e-08) is genuinely four to five orders of magnitude
below the other two, because both of its sub-flows are exact.

## Unmeasured accuracy — stated as such

**Every plan in this pool records `resolved: false`**, and on all three `mc_se_rel` (0.0607 / 0.0439 /
0.3777) is larger than the graded functional error it reports. That is a statement about V's **own**
moments: at the path count the kernel ran, V's variance estimate does not separate from its reference by
more than the sampling noise, and it structurally cannot, because `E[V³]` is infinite under the invariant
law. **The accuracy of V's own mean and variance was not measured on any plan, and must not be reported
as good.** What *was* measured is the graded reciprocal functionals, against a twice-validated exact
reference, and those are the numbers in the tables above.

All three plans were marked **`stopped unresolved`**, not plateaued. Each had completed ≥2 cycles with
no score improvement, and each had already been given one retry under the unresolved rule — plan 1 and
plan 3 explicitly (`functional_limited_by: sampling` → variance reduction at no extra path cost), plan 2
in the same round — and the following evaluation still returned `resolved: false`. That is "nobody found
out", not "this is as good as the scheme gets". Nothing in the schemes was left unfixed; the measurement
was never available.

## Path B — the agreement gate

The gate was run on every pair, and re-run after every solver change (stale comparisons are ignored by
the kernel, and one earlier verdict did flip when the solvers changed).

| pair | families | verdict | moment space (tol 0.10) | worst graded functional |
|---|---|---|---|---|
| 1 vs 2 | other / strang-splitting | **INCONCLUSIVE** (exit 3) | 3.282e-02 independent, 3.050e-02 shared seed | `inv_var_mean` 2.108e-03 relative = **0.81 sigma** — within sampling error |
| 1 vs 3 | other / tamed-milstein | **INCONCLUSIVE** (exit 3) | 1.846e-01 independent, 6.033e-01 shared | `inv_var_mean` 3.271e-03 relative = **3.42 sigma** — above the 3-sigma bar, no break earned |
| 2 vs 3 | strang-splitting / tamed-milstein | **INCONCLUSIVE** (exit 3) | 1.978e-01 independent | `inv_var_mean` 8.650e-03 relative = 2.38 sigma — within sampling error |

All at `num_paths = 50000`, `dt = 0.004`, seeds 42 / 3444837047; recorded in `agreements.json`.

- **No `agreement_cap` is in force.** An earlier round did produce one: on the round-1 solvers,
  `1 vs 3` came back **DISAGREE (exit 2)** at 3.87 sigma on `inv_var_mean`, and both plans were capped
  at 8 until a refine cycle broke the tie. Plan 3's Heun drift did break it — the pair re-ran
  inconclusive on the round-2 solvers — so the cap was lifted, and the final solvers carry none.
- **No second circularity break was earned.** An inconclusive comparison is not an agreement. Every
  pair's moment-space interval straddles the tolerance, because the quantity being compared there is
  V's own ill-conditioned variance. The pool is three distinct families, so `single_family_pool` does
  not apply — the pool had what it needed and the *measurement* is what could not be made.
- **What the gate does not establish, even where it agreed.** All three plans read the same
  formulator's operator out of the same `problem_spec.json`, so a mis-transcribed equation would pass
  unanimously. The independently-authored harness is the only spec-independent check on this run.
  Here the risk is partly offset: the formulator validated the reciprocal-CIR reference two ways that do
  not use the identity, including a direct Kolmogorov solve of the V-equation **as stated in the spec**,
  which is a check on the transcription itself.

## Phase 2b

Did not run. No plan certified, so the bias instruction was reachable through ordinary eligibility
inside the loop — and it was used: plan 2 took it in round 2 (3 → 9, both martingale gates repaired) and
plan 3's round-2 drift-order cycle came out of the same diagnosis. Plan 3's final block still carries
`moment_bias_actionable: true` (`mean`, 0.2616 of its own tolerance), which is a live, unspent repair on
the pool's third-ranked plan; the winner's bias term is 4.6e-08 and has nothing left to remove.

## Verification gaps

These bound what the run established.

1. **The surrogate for V's own moments is invalid at the graded horizon, and it capped the entire pool
   at 9 uncertified.** `verification.stationary_density` is declared with `T_stat = 6`; the graded time
   is T = 0.1, about half the reciprocal process's relaxation time of 0.2. `moments_ok` / `variance_ok`
   are `unavailable` and `resolved` is `false` on all three plans as a direct result. The rubric's
   certification row needs that comparison and never got it. Spec-level, not solver-level; all three
   evaluators reported it independently and two backed it with seed studies.
2. **`moments_ok` unavailable → the moment half of the SDE rubric never produced a verdict.** The
   `mean_rel_err` (0.56–0.58) and `variance_rel_err` (0.70–2.26) in every metrics block are
   transient-vs-stationary distance, not error. Never quote them as accuracy.
3. **`resolved: false` on all three plans**; `mc_se_rel` exceeds each plan's reported graded error. V's
   own moment accuracy is unmeasured at the path count run, and is structurally unmeasurable here
   (`E[V³]` infinite). See *Unmeasured accuracy* above.
4. **No `terminal_law` was declared, so the KS shape test never ran.** This was a deliberate formulator
   trade and the reasoning is sound: the exact terminal law is the *reciprocal* of a scaled noncentral
   χ² (`df = 5.24683`, `nc = 2.80817`, `scale = 1.44293`), which the registry does not implement, and
   declaring `noncentral_chi2` for the state V would have been a false claim. The distributional content
   is carried instead by the two exact reciprocal-moment references. The cost is real: **no
   distributional check was performed on any plan** — only two moments of `1/V`, plus Dynkin.
5. **`cross_check_ok: unavailable` on all three plans** — the spec declares no ordering between
   functionals, so that check had nothing to test.
6. **`cross_plan_agreement` read `unavailable` inside every kernel run**, because the stored comparison
   was always stale relative to the solver being graded (the solvers changed every round). The
   agreement evidence in this report comes from the conductor's own `cli.py compare` runs against the
   final hashes, not from the kernel's reading of it, so **no plan's score was ever priced on a
   recorded agreement.**
7. **The moment-ODE and Kolmogorov surrogate routes were attempted and are unavailable.** The moment
   hierarchy does not close; the formulator additionally built a truncated backward Kolmogorov solve for
   `E[V_T]`/`E[V_T²]` and **rejected it**, because the answer moved from 0.1204 to 0.1699 as `Vmax` went
   8 → 27, failing its own truncation test. Declaring it would have handed the evaluator a bad reference.
   Recorded as a route deliberately not taken, which is the right call and still a gap.
8. **The stationary-density route's variance leg abstains by construction** (`alpha = 2.6234`, so only
   the stationary mean is estimable). Noted in the spec in advance.
9. **`cli.py check-spec` and the final `cli.py gate` both exit 0, with two warnings**, folded in here:
   - `reference [analytic_solution]: reference check unavailable: no analytic moments claimed` — expected
     on Path B; `analytic_moments` is correctly null because there is no formula for V's moments, so
     there was nothing for the reference check to validate. The two reciprocal-moment references are
     carried as gated `martingale` constraints instead, and the formulator's own two-way validation of
     them is recorded in `verification.reciprocal_cir_identity`.
   - `state_clip [1-lamperti-strang-split/solver.py]: the state is clipped to itself as a statement:
     ['X = abs(... X ...) at line 57']` — this is the winner's **declared reflecting boundary**, not a
     hidden clip. `boundary_treatment: reflection` is in the plan's frontmatter and the reflection is
     documented throughout its `SOLUTION.md`, which is what the warning asks for. Recorded rather than
     dismissed: the scheme's positivity at the `X = 0` entrance boundary does rest on that reflection,
     and it fired (11 path-steps at 200 000 paths, 3 at 50 000 — against the plan's original prediction
     of none).
10. **Three plan-level predictions were wrong and were corrected from measurement, not from reasoning**
    — plan 1's "no reflections", plan 2's "the `V_cap` never activates" (it fired on ≈0.09% of noise
    sub-steps before that sub-step was replaced), plan 3's "no truncation activations" (1061 path-steps
    at 200 000 paths, falling to ≈355 after the Heun drift). Plan 3's scheme is therefore partly held
    together by its truncation rather than merely insured by it, and the effect of that freezing on the
    graded reciprocal moments was **not** measured.
11. **One operational defect worth recording for the pipeline, not for this problem's result.** In
    round 1, two of the three concurrently-running evaluators had their kernel output file clobbered by
    a sibling sharing the scratchpad, and each initially read a `solver_sha256` belonging to a different
    plan — one of them drafted a review from those numbers before catching it. Both detected it via the
    hash and re-ran under unique filenames; later rounds were dispatched with an explicit hash to
    confirm. Every score in this report was verified against the live `solver.py` hash by the conductor
    independently, and all three match (`dbc24d24…`, `00b859ae…`, `757534e9…`). No `grading_violation`
    occurred: no evaluator modified a solver it graded.

No `GUARDRAIL_UNSATISFIED` file was written. No plan carries an `agent_cap`.
