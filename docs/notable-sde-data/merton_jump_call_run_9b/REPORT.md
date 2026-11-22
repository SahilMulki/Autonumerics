# REPORT — sde_merton_jump_call

**Problem type:** SDE, family `merton_jump_diffusion` — scalar, one noise channel,
multiplicative lognormal diffusion plus a compensated compound-Poisson jump component.
**Path A** (`analytic_moments.has_analytic_solution: true`).

**Best plan:** `1-jump-adapted-milstein-control-variate` — score **9/10**, `certified: false`,
provenance **`analytic`**.

## Provenance of the scoring

`analytic`, `reference_outcome: validated`. The winner was graded against closed forms the
formulator **derived and independently checked**, not quoted: `E[X_T] = X_0 e^{rT}` and
`Var[X_T] = X_0^2(e^{(2(r-\lambda\kappa)+\sigma^2+\lambda\eta_2)T} - e^{2rT})`, cross-checked by an
exactly-closing moment ODE in `verification.moment_ode` built from the generator rather than from
the moment formulas, and the graded call price against Merton's Poisson-mixture series. This is the
strongest provenance tier. The score of 9 rather than 10 is **not** an accuracy shortfall — see
*Verification gaps*.

## Ambiguity recorded by the formulator

**R18**, quoted verbatim from `problem.md`: *"this problem has no terminal-moment formula, so the
quantity of interest below is the whole grade"*.

The formulator did not carry this across as written, and flagged it rather than overriding it
silently. Its finding: the assertion is false — the compensated jump diffusion has affine drift and
a multiplicative jump multiplier, so the moment hierarchy closes exactly, and both terminal moments
are available in closed form (derived above, and confirmed against a 4e6-path exact-law Monte Carlo
before being written into the spec). It therefore set `has_analytic_solution: true`. The substance
of the sentence is preserved: `call_price` remains the gated functional at a 1% tolerance and the
moments are scored against the independently integrated `moment_ode`, not against any solver's own
numbers. **A reader who meant the sentence literally should know the pipeline graded two moments it
says do not exist, in addition to the call price it does ask about.**

## The graded functional — the accuracy number that matters

The spec declares a gated quantity of interest, so this, not the state-moment proxy, is the
winner's error:

| | winner (plan 1) |
|---|---|
| `functional_name` | `call_price` |
| `functional_rel_error` | **4.530e-04** against `functional_rel_tol` **1.00e-02** |
| split | `functional_bias_rel` 5.289e-06 / `functional_se_rel` 4.477e-04 |
| `functional_limited_by` | **sampling** |

The price is 22× inside its gate, and essentially all of the remaining error is sampling: the `dt`
bias is 5.3e-06, two orders below the sampling term, so a smaller step buys nothing here and only
more paths would tighten it. The bias term is a real measured ladder result, not an abstention — no
plan declared `exact_sampling_components`.

## Plans

| plan | scheme | score | cert | provenance | est_rel_err (state moment) | functional err | observed order | iter |
|---|---|---|---|---|---|---|---|---|
| 1-jump-adapted-milstein-control-variate | jump-adapted Milstein, jump times inserted as sub-step boundaries by Brownian bridge; jump-free GBM call as an analytic control variate (`scheme_family: milstein`, design order 1.0) | **9** | no | `analytic` | 2.289e-03 (measured, `error_is_estimate: false`) | 4.530e-04 / 1e-02 | **0.9969 ± 0.00116** vs floor 1.0 | 1 |
| 3-euler-maruyama-grid-jumps | plain Euler–Maruyama, jumps applied at grid points, no variance reduction (`scheme_family: euler-maruyama`, design order 0.5) | 3 | no | `analytic` | 1.762e-03 (measured) | 2.146e-03 / 1e-02 | 0.4930 ± 0.00492 vs floor 0.5 | 1 |
| 2-jump-adapted-exponential-euler | exact exponential log-coordinate flow between jumps, jumps at their exact times, antithetic Brownian pairs (`scheme_family: other`, design order 0.5) | 3 | no | `analytic` | 1.240e-02 (measured) | 1.250e-03 / 1e-02 | **not resolved** — CRN differences at roundoff (4.55e-14 vs a 1.01e-10 floor), the signature of an exact continuous flow | 1 |

Last-pass detail, all three plans: `resolved: true`, `constraints_ok: true`, `invariants_ok: true`,
`functionals_ok: true`, no `agent_cap`, no `invariants_not_reported`. The gated discounted-martingale
identity `e^{-rT}X` passes on all three — drift 5.021e-03 (plan 1), 3.409e-03 (plan 3) and 0.0333
(plan 2) against a bar of 5.7e-02 — which is the check that would have caught the likeliest bug on
this problem, since κ is **negative** here so `r - λκ = +0.10501` is *larger* than `r`. Positivity
and finiteness hold everywhere (minimum terminal path 3.26 / 3.458 / 5.44).

Plans 2 and 3 were **not** left unfinished and did not plateau on accuracy: both scored 3 solely on
the Dynkin row described below, and when each was dispatched a second solver cycle, all three
solvers returned `solver.py` **byte-identical** (hash matching the `solver_sha256` in their own
metrics blocks), on the ground that no solver change can reach the finding without falsifying the
compensated drift. Re-grading identical bytes against an unchanged spec yields no new evidence, so
each plan is terminal at `iter: 1`. No plan certified, so Phase 2b (the winner's bias-removal cycle)
did not fire; the winner's `moment_bias_actionable` is `false` in any case
(`moment_bias_binding: variance`, `moment_bias_frac_tol: 8.565e-03`).

## Best plan and why

The ranker's line, verbatim:

```
best: 1-jump-adapted-milstein-control-variate   decided_by: score
```

`decided_by: score` — the pool was separated by the score itself (9 against 3 and 3), not by a
tiebreak, so this is a measurement rather than a default. The restricted quality index agrees
(0.470 / 0.305 / 0.267 over accuracy, structure and consistency). The ranker notes that `order` was
dropped from the compared index because it did not resolve on every plan in the pool (plan 2's
continuous flow is exact, so there is no discretisation error to measure a rate on);
per-plan unrestricted indices are 0.616 / 0.494 / 0.267. Cost is reported and not ranked on: solve
times were 0.475 s / 0.256 s / 0.325 s.

Plan 1 also wins on the number the problem is graded on: its control variate cuts the call price's
standard error ~4.4× (8.69e-03 → 1.97e-03), giving the lowest `functional_rel_error` in the pool.

## Verification gaps

These bound what this run established.

1. **The Dynkin generator check cannot express this problem, and that — not accuracy — is what
   capped every plan.** This is the run's central finding.

   All three plans report `dynkin_z ≈ 72.07` against `dynkin_bar: 2.43`, with a **`dt`-independent**
   residual of ≈ −1.369 on `φ = X` (−1.3700, −1.3692, −1.3689, −1.3687 across four ladder levels).
   That matches the missing jump term `λκ·E[X]·T = −0.055 × 101 × 0.25 ≈ −1.39` to within ~2%. The
   kernel builds its generator as `f φ' + ½g²φ''` from `drift_expression` / `diffusion_expression`,
   i.e. the **diffusion-only** generator; Merton's non-local term `λE[φ(xY) − φ(x)]` has no
   representable home in the schema. On `φ = x` the omitted term is `−0.055x` against a declared
   drift coefficient of `+0.105`, so the diffusion-only generator predicts `E[X_T]` growing at 0.105
   where the true process grows at `r = 0.05`. **The check therefore fires identically on every
   correct solver for this problem**, which is why the residual is the same to four digits across
   three unrelated schemes.

   The formulator anticipated this and set `verification.dynkin_test_functions: null`, expecting that
   to make the check a skip (its `dynkin_note` says so, and predicts a ceiling of 9). **That
   expectation is wrong about the kernel**: `verifylib/kernel/sde/dynkin.py:66` `test_functions()`
   falls through to `DEFAULT_TEST_FUNCTIONS` (`X`, `X**2`, `tanh`) whenever the declared list is
   absent or empty, so the check ran anyway. Nor is there a spec route to a jump-aware generator:
   `verification.generator` (`_vector_payload`, same file) carries only `drift` and
   `diffusion_matrix`. Consequence under the project manual's SDE analytic rubric:

   - plan 1 landed on the **9** row (`dynkin_ok` never reached a verdict — the extrapolated limit
     moved by 4.0 and 3.2 paired standard errors between the ladder's two windows, so the levels are
     not in one asymptotic regime: *unresolved*, "uncertified, not suspect");
   - plans 2 and 3 landed on the **3** row (`dynkin_ok: false`), on margins of ~69.6 to the bar, so
     the verdict is not seed-sensitive.

   **No plan on this spec can certify, and the two 3s are not statements about those solvers.** Both
   evaluators said so explicitly and neither applied an `agent_cap`; the conductor did not edit
   `problem_spec.json`, cap any score, or re-dispatch the formulator, because there is no field in
   the current schema that would express either the skip or the jump term. Fixing this needs a
   kernel change (a jump-aware generator, or an honoured skip) and is outside what this run could
   establish. The generator evidence Dynkin would have supplied is replaced, partially, by the
   exactly-closing `moment_ode`, the gated discounted-martingale identity (an identity of the *full*
   generator, so it does see the jump term, and it passes on all three plans), and the gated
   `call_price` functional, which is sensitive to the whole terminal distribution rather than two
   moments.

2. **The winner's state-moment `estimated_rel_error` is not a measurement.** `cli.py rank` warns
   that plan 1's `estimated_rel_error` of 2.289e-03 sits against an `mc_se_rel` of 6.766e-03 — 3×
   larger — and plan 3's 1.762e-03 against 6.763e-03, 4× larger. Those two state-moment errors are
   **unmeasured at the path count that ran, not good**. The kernel's own `resolved: true` is a
   variance-space claim and does not rescue them. The accuracy claim this report makes rests on the
   **functional** instead (`functional_rel_error` 4.530e-04 with a resolved bias/sampling split),
   which is the quantity the problem is scored on and which the ranker's accuracy component reads.

3. **No `terminal_law` distributional test.** The formulator set it `null` deliberately: `X(T)` is a
   Poisson **mixture** of lognormals, which is none of the implemented laws, and at `λT = 0.025`
   about 2.5% of paths carry a jump of median size `e^{-0.9} ≈ 0.41`, so a KS test against a single
   lognormal would reject a correct solver on the mixture tail. The run therefore has **no
   shape test on the terminal distribution** — the gated call price is the only check sensitive to
   the distribution beyond its first two moments.

4. **`cross_check_ok: unavailable`** on all three plans (the spec declares no ordering between
   functionals), and `surrogate_ok: unavailable` — expected on Path A, where the closed form was
   used instead.

5. **Final `cli.py gate` warning on the winner, which I examined and judge a false positive.** The
   gate (exit 0, warning only) reports `state_clip` at `solver.py:88`:
   `u = clip(... u ...)`. That line is `u = np.clip(u, 0.0, 1.0)` where `u` is the **within-step
   fractional jump time** used by the Brownian-bridge split — an interpolation parameter that must
   lie in [0, 1] — not the state. The state `X` is never clipped in that solver; positivity comes
   from the multiplicative step form. The scanner matched on the variable name. Recorded here rather
   than dismissed silently.

6. **Order study on plan 2 is `unresolved`, not passed.** Its CRN level-to-level differences are at
   roundoff because the continuous flow is integrated exactly, so its strong order is not measurable
   by this ladder. Per the rubric that abstention costs the plan nothing, but it also establishes
   nothing: plan 2's convergence rate is unknown, not confirmed.

7. **Single seed on Dynkin.** Neither Dynkin verdict was replicated across seeds. The margins
   (|z − bar| ≈ 69.6) make seed sensitivity implausible, and the diagnosis in gap 1 is independent of
   the seed, but the replication was not run.

## Bottom line

Plan 1 — jump-adapted Milstein with a Brownian-bridge jump split, a `dt`-independent jump stream,
the correct compensated drift `a = r − λκ = +0.10501`, and an analytically priced GBM control
variate — prices the call to **0.045%** against a **1%** gate, hits its design strong order
(0.9969 ± 0.0012 against 1.0), and passes every structural gate and invariant the spec declares. It
is uncertified, and the single reason is that this spec's generator check cannot represent a jump
process, which is a limitation of the verification kernel rather than of the solver.
