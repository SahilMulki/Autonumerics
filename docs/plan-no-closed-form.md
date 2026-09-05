# Verification Without a Closed Form — Implementation Plan

**Status:** proposal, rev 2
**Scope:** how a solution is scored when `analytic_solution` is `null` (PDE) or
`analytic_moments.has_analytic_solution` is `false` (SDE)
**Related:** [plan-hallucination-guardrails.md](plan-hallucination-guardrails.md),
[plan-blind-evaluator.md](plan-blind-evaluator.md)

Provenance labels: **[A]** = already in `references/verification_manual.md`, **[B]** = adapted from
the AutoNumerics paper-release pipeline, **[N]** = new in this plan, **[C]** = corrected in rev 2
against the shipped code.

---

## 0. What changed, and why

### Rev 1 → rev 2 (review against the shipped guardrails work)

Rev 1 was written before `verifylib/` existed. Five of its claims are now false, and three of its
mechanisms do not work as specified. Recording them rather than silently rewriting:

| # | Rev 1 said | Correction |
|---|---|---|
| 1 | `verifylib/` is new; §8a lists its tree | It exists: 3,672 lines, 136 passing tests, and `operator.py` and `tests/test_canaries.py` are **already taken** by different content. The kernel is an *extension*, and the tree collides — see §9a |
| 2 | §3 defines the `verification.operator` schema | Already shipped — schema-gated in `verifylib/schema.py`, required by `agents/formulator.md`, present in both PDE templates, documented as manual §2b. This plan consumes it; §3 is now about *what D1 does with it* |
| 3 | D1 gates: "D1 failure → 3" | No pass criterion was ever defined, and no fixed magnitude threshold can exist — §5 |
| 4 | "`override` must support returning `u(T-dt)`" | `override` is an *input* dict. Getting a second snapshot needs a **return**-contract change — §5c |
| 5 | KS is `SELF_ONLY`; "under this design it certifies" at `manufactured` | It already reports `provenance: manufactured` and scores 10, via the `degenerate_limit` already in its spec. `SELF_ONLY` is the *benchmark's* verdict about independent verification, a different axis entirely — §8 |

Three mechanisms were rebuilt: D1 became a **slope test** (§5a), the Tier-B-order trick acquired the
preconditions that stop it understating error (§7a), and D2 acquired a **closed invariant-name
registry** (§6), without which a gated invariant the kernel does not recognise reads as a pass.

The helper names in rev 1's own examples (`dxx`, `dxxxx`, `grad`, `div`, `adv`) do not exist in
`verifylib/operator.py`, and manual §2b explicitly *refuses* `adv`. Every example here is written
against the shipped namespace.

**Two decisions taken since this rev's first draft**, both closing questions it left open:
`p_design` for the slope test comes from `verification.theoretical_order` (§12); and the benchmark's
KS blind spot is fixed by a harness-owned reference rather than by routing the kernel through
`verify.py` — which removes `benchmark/` from this plan's scope entirely (§8c–d).

---

## 1. Why this exists

**The evaluator agent writes `evaluate.py` fresh every cycle** — nested ladders, Richardson, GCI
guards, asymptotic tests, MMS probe validation. That is a nontrivial numerical program regenerated on
every pass, and nothing re-checks its arithmetic. `benchmark/verify.py` re-checks the *solver's final
answer* against independent ground truth, so a miscomputed GCI is invisible wherever ground truth is
absent — and invisible *everywhere* as a measurement of the evaluator itself.

That is not hypothetical. The four plans under `workspace/pde_kuramoto_sivashinsky/` report:

| Plan | `observed_order` | `estimated_rel_error` | Score |
|---|---|---|---|
| `1-etdrk4-spectral` | 10.467 | 2.538e-06 | 10 |
| `2-imex-sbdf3-spectral` | 10.925 | 1.345e-06 | 10 |
| `3-ifrk4-spectral` | **0.003** | **6.695e-11** | 10 |

Same problem, same spec, four independently regenerated evaluators, orders spanning four orders of
magnitude and error estimates spanning five — all certified, all tagged `manufactured`, nothing
flagged. Whatever those numbers measure, it is not reproducible, and no downstream consumer can tell
which of them to believe.

**This is also what makes the guardrails plan's Layer 5 worth anything.** Its numeric-claim binding
checks that a review's prose is backed by its `<metrics>` block; the same LLM writes both, so today it
catches transcription drift and nothing else. Guardrails §14 C6 states the consequence plainly:
binding "starts catching fabrication rather than only transcription drift" the moment metrics are
**kernel-computed**. That upgrade applies to all 50 benchmark problems and every user problem — not
to the no-closed-form population, which in the benchmark is exactly one problem (§8).

**The paper-release pipeline** offers the other half of the design. Its term-balanced residual is
fixed Python and cannot be hallucinated — but it is a hard-coded allowlist of 8 problem IDs, it never
checks convergence order, and it differences the field with the solver's own stencil family, which
makes agreement a tautology.

The two flaws are mirror images: general-but-regenerated versus deterministic-but-hard-coded. This
plan takes the coverage of the first and the trust model of the second.

---

## 2. Architecture — split the evaluator in two **[N]**

| Layer | Implementation | Responsibility |
|---|---|---|
| **Kernel** (`verifylib/kernel/`) | Fixed, human-written, version-controlled, under test. **Never regenerated.** | Every numerical check: ladders, Richardson/GCI, D1 residual, invariants, MMS driver, temporal isolation, Dynkin, CRN. Emits one machine-readable metrics JSON, **including `score` and `provenance`** (§2c) |
| **Evaluator agent** | `agents/evaluator-{pde,sde}.md` | *Selects and configures* checks, then **diagnoses** — turns kernel output into specific solver feedback. Writes no arithmetic and **assigns no score** |

Everything in `verification_manual.md` survives. The surface on which it can be miscomputed does not.
Parts I and II stop being instructions to an LLM and become the **specification for the kernel** —
a better use for them, since they already read like one.

### 2a. The entry point is the CLI, not an `evaluate.py` **[C]**

Rev 1 had `evaluate.py` shrink to a `verifylib.run(...)` call. It should disappear instead:

- `import verifylib` only resolves inside this repo. On a user's own problem the package lives at
  `${CLAUDE_PLUGIN_ROOT}`, and every generated `evaluate.py` would have to rediscover that.
- `verifylib/cli.py` already solves it — absolute-path invocation from any cwd, plus the `sys.path`
  surgery that stops `verifylib/operator.py` shadowing the stdlib `operator` module for the whole
  interpreter (a trap the kernel will hit the moment it imports `functools`).
- "No `evaluate.py` contains numerical logic" is not a checkable criterion. "No `evaluate.py` exists"
  is.

So: **`cli.py evaluate <plan_dir> [--json]`**, one more subcommand beside `check-spec`, `gate` and
`hook`. The evaluator agent runs it and reads the JSON.

### 2b. The kernel runs solver code, so it runs sandboxed **[N]**

The kernel imports and executes `solver.py` up a three-level ladder — in 3-D the top level is 32× the
base. `benchmark/runner.py` already provides exactly the isolation this needs (wall-clock timeout,
`RLIMIT_AS` memory cap, numpy+stdlib-only import policy, leakage scan before import) and already
returns `status="crashed"` with a reason rather than hanging. The kernel reuses it.

This is not belt-and-braces. The guardrails work measured the analogous failure: an unbounded Stop
hook fired 8 times and the run died at max-turns. A hung top-level solve burns the same budget with
less to show for it.

### 2c. The kernel emits the score, not just the metrics **[N]**

The scoring rubrics are **decision tables with no judgement in them**. Manual §12 reads
`verified and converged and order_ok and invariants_ok and constraints_ok and temporal_ok -> 10`;
§23 is its SDE twin; §4c's certification table below is the same shape. Every input is a boolean or a
comparison the kernel already computed.

Handing those booleans to an LLM and asking it to apply the table is a pure loss: it adds no
information and adds a drift surface. §1's KS table is what that drift looks like in practice —
`observed_order: 0.003` and `observed_order: 10.9` both arriving at 10.

So the kernel emits `score` and `provenance` alongside the metrics, and the agent writes the review
prose and the solver feedback — which is the part that actually needs a language model, and the part
that drives the five refine cycles. Three consequences worth stating:

- **`commands/conductor.md` ranks on a computed field**, not on a number parsed out of prose an agent
  wrote about its own work.
- **Guardrails Layer 5 gets its strongest form.** `review.py` already binds keyed claims in the
  review's *Numerical Accuracy* section against `<metrics>`; with a kernel-emitted score the review's
  `Score: N/10` headline becomes an **exact** match against `metrics.score`, not a tolerance check.
- **The rubric becomes code.** Changing it means editing `kernel/score.py` and its tests rather than
  a markdown table. That is a real cost, and it is the point: a rubric that can be reinterpreted per
  cycle is not a rubric.

**The agent keeps one authority, and it is asymmetric: it may cap the score downward, never raise
it.** A cap requires a machine-readable reason recorded in the metrics (`agent_cap: {score, reason}`).
This is the escape hatch for what no check measures — `SOLUTION.md` describing a different scheme than
`solver.py` implements, an answer smuggled in as a lookup table, a plan that "converges" by returning
its input. It is also exactly the shape `benchmark/verify.py`'s `ledger_audit` already uses
(`score_cap: 3`), so it introduces no new concept. Asymmetry is what preserves the anti-drift
property: an agent that cannot inflate cannot launder a 4 into a 10, and one that can deflate can
still stop a wrong 10 from shipping.

---

## 3. Consuming the operator declaration **[C, was N]**

> **Ownership.** `verification.operator` is **defined and required by**
> [plan-hallucination-guardrails.md](plan-hallucination-guardrails.md) §4d, and is already shipped:
> validated in `verifylib/schema.py:check_operator_declaration`, required by `agents/formulator.md`,
> present in both PDE templates, documented as `verification_manual.md` §2b. This plan **consumes**
> it. Do not re-specify the shape here; if it needs to change, change it there.

What D1 adds is the discretization rules for applying it to *solver output* rather than to a formula.

### 3a. The helper namespace, as actually shipped **[C]**

`verifylib/operator.py:build_namespace` supplies, for every declared field `u` and every axis `v` in
`spatial_variables`:

| Kind | Names |
|---|---|
| Field / time | `u`, `u_t`, `u_tt` |
| First derivative | `u_v`, `grad_u` (array; `grad_u_0`, `grad_u_1`, … per component) |
| Second derivative | `u_vv`, `u_vw` (mixed, both spellings), `lap_u`, `lap_lap_u`, `lap_u3` |
| Callables | `dt(·)`, `lap(·)`, `lap2(·)`, `d_v(·)`, `d_vv(·)`, `dv(·)` |

Non-Cartesian axes come free — `["S", "v"]` yields `u_SS`, `u_Sv`, `u_vv`. There is deliberately **no**
`adv_u`, `grad_p`, `lorentz_u` (manual §2b: a composite means something different in every system
that uses it), **no** `dxx` (it is `d_xx`), and **no** fourth-derivative callable — in 1-D the
biharmonic is `lap2(u)` or the name `lap_lap_u`.

`dt(a)` resolves by array identity against the `time_derivs` the kernel passes in, so it is defined on
a declared field only. Write `dt(u)`, never `dt(u*v)`.

Worked, against the real `pde_kuramoto_sivashinsky` spec:

```json
"operator": {
  "fields": ["u"],
  "terms": {"u_t": "dt(u)", "burgers": "gamma*u*d_x(u)",
            "u_xx": "mu*lap(u)", "u_xxxx": "nu*lap2(u)"},
  "source": "0"
}
```

This is the same operator its `mms_probe.operator_check` already declares
(`"u_t + gamma*u*grad_u + mu*lap_u + nu*lap(lap_u)"`), term-decomposed so the balanced denominator is
computable — which is the only thing `terms` exists for.

### 3b. Three discretization rules

1. **Higher order than the solver used.** **[A]** — manual §9 states this for the steady-state
   residual; apply it to the time-dependent case. **This is currently false in the code** [C]:
   `operator.py`'s `d1`/`d2` are 4th-order, and `workspace/` already holds
   `3-fd-compact-4th-order`, `4-fd6-imex-sbdf3` and several spectral plans. §9b adds 6th/8th-order
   and spectral variants.
2. **A different discretization *family* than the solver's.** **[N]** — if the solver was spectral,
   difference with high-order FD, and vice versa. This prevents the paper-release pipeline's
   `5.03e-15` viscous-Burgers artifact: reusing the solver's own stencil makes agreement a tautology.

   **Both rules need an input that does not exist yet** [C]. Rev 1 said "read
   `plan.spatial_discretization.scheme`"; there is no such field anywhere in the repo. What exists is
   a free-text `scheme:` in `SOLUTION.md`'s YAML frontmatter, with 30+ distinct values across
   `workspace/` (`finite-difference-implicit-adi`, `spectral-imex`,
   `finite-difference-implicit (Craig-Sneyd ADI, monotone rotated cross stencil, ...)`), and none of
   them state the spatial *order* — which rule 1 needs in order to exceed it. So Phase 0 adds two
   declared frontmatter fields, written by `plan-creator-{pde,sde}.md`:

   ```yaml
   scheme_family: fd | spectral | fv | fe | other
   spatial_order: 2
   ```

   The kernel parses the frontmatter block, never greps for `^scheme:` — a body line beginning with
   `scheme:` already exists in `pde_convection_diffusion_bl/plans/1-exponential-fitting/SOLUTION.md`
   and a grep picks it up. Where the fields are absent (every plan written before Phase 0), the
   kernel defaults to 8th-order FD, records `stencil_choice: default`, and the §5b
   insensitivity guard catches the case where that was not enough.
3. **The time term comes from stored snapshots, not a second solve.** **[C]** — §5c.

### 3c. SDE side — derived, and the contract already exists **[C]**

No spec surface and **no contract change**. The generator `Lf = f·φ' + ½·g²·φ''` is built from
`drift_expression` and `diffusion_expression`; `solve_sde` already accepts `observables` and returns
`path_integrals` (manual §18, `agents/solver-sde.md`), which is exactly what Dynkin needs. The
formulator supplies `dynkin_test_functions`, which it already does.

---

## 4. The ladder

Two structural changes from manual §1: **D1 and D2 move out of the scoring tiers into a gate layer**,
and **cross-plan consensus leaves the ladder entirely** (it is a cross-plan comparison, so it belongs
to ranking — see `plan-blind-evaluator.md`).

### 4a. Scoring tiers **[C — A⁻ added]**

Rev 1's table omitted **A⁻**, which the guardrails work shipped (manual §1, guardrails §14 C4). The
total provenance order is `analytic` > `analytic_unvalidated` > `surrogate` > `manufactured` >
`manufactured_partial` > `self_convergence` > `none`.

| Tier | PDE evidence | SDE evidence | Certifies | Tag |
|---|---|---|---|---|
| **A** | Closed form, **validated** by the reference check | Exact moments, validated against the moment ODE | The answer is right | `analytic` |
| **A⁻** | Closed form accepted on provenance — verbatim quote, or the check could not run | same | The answer is right *if the formula is* | `analytic_unvalidated` |
| **A′** | — | Moment ODE (`closes_exactly`), Kolmogorov solve, stationary density — §15–17 | Right to a controlled tolerance | `surrogate` |
| **B** | MMS (probe numerically validated) or degenerate limit — §5–6 | Degenerate limit (σ→0; linear-coefficient limit with exact moments) | The **scheme and code** are right | `manufactured` |
| **C** | Richardson + GCI + asymptotic guards — §3–4 | CRN ladder + strong/weak order — §18–19 | It converges, and by how much it is off | `self_convergence` |
| — | Nothing runnable | Nothing runnable | — | `none` |

### 4b. Gate layer — runs every cycle, never raises a score, can always sink one **[N]**

| Gate | PDE | SDE |
|---|---|---|
| **D1** — operator residual | Term-balanced residual from `verification.operator`, as a **slope test** with the guards in §5 | Dynkin identity, Richardson-extrapolated in `dt`, tested against the MC confidence interval of zero — §20 |
| **D2** — structure | Registry invariants with `gate: true`, BC/IC enforcement, positivity, symmetry, divergence — §6 | Positivity, support, martingale, finiteness, estimator stability — §21; MC-resolution — §14 |

### 4c. Certification rule **[A, tightened]**

> **A 10 without a closed form requires two independent circularity breaks.**

B and D1 break the circle in different directions. MMS checks the *discretization* against a known
solution of a *modified* problem; D1 checks the *produced field* against the *stated* operator on the
real problem, where the shock, layer or stiffness the manufactured problem lacks actually lives.

**They are independent only given a correct operator** [C] — a misread equation fools both. That is
why D1 counts as a break only when the operator has been independently validated (§5d), and why the
requirements ledger's `equation` entry must name `verification.operator` in its `spec_path`
(guardrails §4g). Rev 1 asserted the independence without the precondition.

| Condition | Max score | Tag |
|---|---|---|
| Tier A (`validated` / `validated_off_singularity`), gates clean | **10** | `analytic` |
| Tier A⁻ via the **quote** route (`reference_outcome: quoted`), gates clean | **10** | `analytic_unvalidated` |
| Tier A⁻ because the check **could not run** (`unavailable`), gates clean | **9** | `analytic_unvalidated` |
| Tier A′ clean, gates clean | **10** | `surrogate` |
| Tier B **and** D1 clean *with a validated operator*, C clean, gates clean | **10** | `manufactured` |
| Tier B **or** D1 alone, C clean, gates clean | **9** | `manufactured_partial` |
| C + gates only | **8** | `self_convergence` |
| Any `gate: true` D2 violation, or D1 `stalled` with a validated operator | **3** | (tag unchanged) |
| No test runnable | **2** | `none` |

**This table is implemented in `kernel/score.py`, not read by an agent** (§2c). Every row is a
conjunction of booleans the kernel already has.

The A⁻ split is new [N] and closes the hole guardrails §14 C4 opened but did not price: a *quote* is
evidence external to the pipeline (someone wrote that formula in `problem.md`, and
`schema.check_source_text` confirms it contains arithmetic); "the check could not run" is no evidence
at all, and should not buy the same score. The score rule reads `reference_outcome` from the metrics,
not the provenance tag — the tag stays coarse so C4's total order is untouched.

Preserved unchanged from the manual: a check skipped for cost never counts against a plan; a faulty
MMS probe is skipped, not failed; a missing reference is not a solver crash; never score on
plausibility.

---

## 5. D1 — the operator residual, and why it cannot be a magnitude test **[C]**

### 5a. The slope test

Rev 1 assigned D1 a hard gate and never said what failure was. No fixed threshold can exist, and its
own §5 table says so ("residual magnitude depends on operator conditioning, grid, and stencil order.
There is no universal passing value"). Concretely: the only calibrated number in the repo is
`reference.TOL = 1e-6`, measured for *a closed form on a 1025-point grid whose own discretization
error is 1e-10*. A solver's residual is dominated by its own truncation error, `O(h^p) + O(dt^q)`; it
does not approach zero at fixed resolution. Any constant threshold either fails every correct
coarse-grid solver or passes everything.

**The SDE half of this plan already solved this.** Manual §20, for Dynkin: *"the trapezoid path
integral carries its own `O(dt)` bias, so the residual does not go to zero at fixed `dt` — the
meaningful test is that it shrinks at the expected rate."* Do the same for D1.

```python
r(h)  = median over the interior of  |sum(terms) - source| / max_k |term_k|
p_res = log2(r(h) / r(h/2))          # both grids are already on the ladder
```

| Outcome | Condition | Effect |
|---|---|---|
| `clean` | `p_res >= p_design - slack`, **or** `r` has reached the round-off floor | Counts as a circularity break |
| `stalled` | `r` does not fall and is not at round-off | **Gate** (score 3) when the operator is validated; warning naming both suspects otherwise |
| `unresolved` | The two kernel stencils disagree (§5b) — the check, not the solver, is the limit | Skipped; never counts against the plan |
| `unavailable` | No snapshots, no operator, or an unsupported masked domain | Skipped |

The round-off branch is manual §4's `not_at_roundoff` guard, and it is not hypothetical: KS's spec
note documents a spectral scheme collapsing `d21` to round-off on the 64/127/253 ladder. `p_design`
is `verification.theoretical_order` — **not** `evaluation_thresholds.min_spatial_order`, which is
`0.0` on that same spec and would make every guard vacuous. That fallback order is itself transcribed
from the KS spec's note, where a previous evaluator had to be warned about it in prose.

Magnitude is still reported, as a diagnostic and a cross-plan ranking signal on a fixed grid. It just
does not decide anything on its own.

### 5b. The stencil-insensitivity guard **[N]**

Rule 1 (§3b) says the kernel differences at higher order than the solver. When it does not — a 6th-
order compact or spectral solver against 4th-order kernel stencils — the residual measures the
*kernel's* error and the slope test reads the kernel's order, not the solver's.

So compute `r` twice, with two stencil orders (FD6 and FD8, or FD8 and spectral on a periodic grid).
If they disagree by more than a factor of ~2, the residual is check-limited: report `unresolved` and
skip. This is the same escalation idiom `reference.PROBE_N = (257, 513, 1025)` already uses to
separate "under-resolved" from "wrong", and it is what keeps D1 from punishing the best solvers in
the population.

### 5c. The two-snapshot time term needs a **return**-contract change **[C]**

Rev 1 asked `override` to "support returning `u(T-dt)`". `override` is an input dict of problem
modifications (manual §7); it has no return channel. The two routes that exist today both fail:

- **A second solve at `params={"t_final": T-dt}`** — the two runs pick different internal step
  sequences, so each approximates its own exact solution to `O(dt^p)` *independently*. The difference
  quotient then carries `O(dt^p)/dt = O(dt^{p-1})`, which for a first-order scheme is `O(1)`.
  Unusable, and it costs a full extra solve.
- **The solver's own last step** — `(uⁿ − uⁿ⁻¹)/dt` is what the scheme computes, so the time term is
  consistent by construction and D1 degenerates to a test of the spatial operator alone.

The fix is a new optional **return** key, documented in manual §7 beside `invariant_trace`:

```python
"snapshots": [{"t": float, "fields": {name: array}}, ...]   # the last K states, ending at t_final
```

The kernel forms the time derivative by Fornberg weights over the returned `t` values — arbitrary
spacing, so an adaptive final step is fine — at order `K-1`. `K >= 3` gives `O(dt^2)`; `K = 2` runs
but records `time_term_order: 1` and its own truncation floor in the metrics. Absent → D1 is
`unavailable` and **skipped, not failed**, per the manual's standing rule.

This is what makes "D1 is free" true: the snapshots ride along with a solve already performed, and the
residual is a handful of stencil applications on arrays already in memory.

### 5d. Validating the operator itself **[N]**

D1's residual and the solver both descend from `verification.operator`. If it is wrong, both are
wrong consistently. Three routes validate it, in descending order of strength:

1. **The reference check** (Path A) — already shipped; a wrong operator makes a correct closed form
   fail.
2. **The MMS probe** (Path B) — applying the operator to `mms_probe.exact` must reproduce
   `mms_probe.source`. `reference.residual_field` already takes an `exprs=` override, so this is a
   call, not an implementation.
3. **The degenerate limit** (Path B, no MMS) — apply the operator with `params` set to the degenerate
   values to `degenerate_limit.exact`; the residual must vanish. This covers KS, whose γ→0 two-mode
   solution is already in the spec.

None of them catch an operator that is wrong *in the same way the problem statement was misread*.
That is the requirements ledger's job, and it is the only check that reaches outside the spec.

### 5e. Gating on the median, and masked domains **[C]**

`reference.check_operator_residual` gates on the **median**, not the max, and records spatial
concentration as a diagnostic. That was not a stylistic choice: the max-plus-"singular set < 2%"
design was measured to be fragile (real singular cases land at 0.8/1.2/1.6%) and could not classify
`pde_black_scholes_call` at all, whose kink keeps ~4% of the domain above tolerance on an exactly
correct formula. Solver fields have shocks and layers too — `burgers_inviscid`,
`convection_diffusion_bl`. D1 inherits the median rule; rev 1 said nothing and would have inherited
the design already known to fail.

`operator.py`'s stencils are `np.roll`-based with a rectangular `STENCIL_HALO = 4` interior mask.
That is wrong on `poisson_lshape` and on the masked domains the benchmark harness supports. D1 must
intersect the interior mask with the spec's domain mask, or report `unavailable` — never quietly
difference across a hole.

### 5f. The three trivial-attractor guards **[N, retained from rev 1]**

A residual-only metric has a defect that is live in the paper-release pipeline:

> **A residual rewards any field sitting on a stable steady state of the operator, whether or not it
> is the state the initial condition actually evolves to.**

Gray–Scott is the sharpest case: `u ≡ 1, v ≡ 0` is an exact homogeneous solution. A solver that
over-diffuses and destroys the pattern satisfies every term to near machine precision, having
eliminated the only phenomenon the problem is about. The degenerate case is worse: if `u → 0`,
numerator and denominator both vanish and the epsilon in the denominator leaves you dividing zero by
`1e-300` — a perfect score for solving nothing.

```python
nontrivial    = rms(u_T) > triv_tol * rms(u_0)                # not collapsed
ic_consistent = rel_err(solve_to(t=0), u_0) < 1e-10           # the run started where it should
structural    = all(gate_invariants_pass)                     # D2 — §6
```

All three are cheap, all are computed by the kernel, and all gate.

### 5g. Why not residual-only, in one table **[A]**

Each rung answers a specific direction in which a residual is blind. The blindness is structural — a
consequence of testing the PDE rather than the initial-boundary value problem, on a fixed grid, at a
single instant.

| Residual blind spot | Why | Covered by |
|---|---|---|
| No error bound | `‖e‖ ≤ ‖L_h⁻¹‖·‖r‖`; the stability constant is ~`1/eps` for convection-dominated, unbounded near a Helmholtz resonance, growing with `T` for hyperbolic. Estimating it requires refinement | **C** |
| No calibrated threshold | Magnitude depends on conditioning, grid and stencil order | **§5a** (require it to *fall*), **C** |
| Amplitude / phase error | For any linear homogeneous PDE `c·u` is also a solution; for translation-invariant problems so is `u(x−s)`. Residual is **identically zero** on both. Phase lag is the dominant error of every advection scheme | **D2** invariants, **C** at full `T` |
| Boundary conditions | A stencil needs ghost points, so the residual is interior-only. BC violation lives exactly where it cannot be computed | **D2** BC check |
| Under-resolution | An upwind scheme solves `u_t + a·u_x = (eps + eps_num)·u_xx`. On a coarse grid every derivative is small and mutually consistent, the residual small — and the answer 30% wrong | **C** |
| Wrong operator in the spec | Solver and residual come from one declaration; both solve the wrong problem, self-consistently | **§5d**, and the requirements ledger |
| SDEs | No pointwise residual exists for a Monte Carlo path solver | §18–20 |

**Note on short-time Richardson.** Rev 1 correctly rejected it as a general guard — shortening the
horizon removes precisely the accumulated error you are hunting. It is correct *only* on a chaotic
problem where full-`T` convergence is meaningless (§8).

---

## 6. D2 — invariants need a closed registry **[N]**

Real gated invariants are **catalogue entries, not expressions**:

```json
{"name": "divergence_free",  "gate": true, "tol": 1e-08, "fields": ["u", "v"]}
{"name": "energy_bounded",   "gate": true, "bound": 100.0, "requires_trace": true}
{"name": "symmetry",         "gate": false, "axis": 0, "parity": "odd", "tol": 1e-08}
```

`schema.py` today validates only entries carrying an `expr` (`_GATED_EXPR_KEYS`), so
`{"name": "physically_reasonable", "gate": true}` passes the schema, matches nothing in the kernel,
and — unless the kernel is careful — renders as a pass. A gate nobody implements is worse than no
gate, because it reads as evidence.

Three changes:

1. **`kernel/invariants.py` owns a closed registry** of the manual §8 names, each with its parameter
   shape (`tol`, `bound`, `axis`, `parity`, `fields`, `requires_trace`).
2. **`schema.py` errors on an unknown gated invariant name** — the registry is importable, so this is
   a set membership. Ungated entries stay free-form notes.
3. **A declared gate the kernel cannot evaluate reports `not_reported`, never `pass`**, and caps the
   score at 9 — which is already the manual's rule for a missing `invariant_trace`, generalised.

---

## 7. Cost **[C]**

Using manual §24's model (each ladder level costs `2^(d+2)`× the one below, for an explicit scheme):

| Path B evaluation | 1-D | 3-D |
|---|---|---|
| **Today:** 3-level ladder + MMS at certification | 73 + 9 = **82** | 1057 + 33 = **1090** |
| **Merged, trick applicable:** 2-level ladder + MMS supplying the order | 9 + 9 = **18** | 33 + 33 = **66** |
| **Merged, trick not applicable:** unchanged ladder + D1 | 73 + 9 = **82** | 1057 + 33 = **1090** |

Rev 1 quoted only the middle row and called it a 4.6×/16.5× saving. The honest claim is that the
saving applies **when the preconditions in §7a hold**, and that D1 and D2 are free either way.

**D1 is free** — with §5c's snapshot contract it needs no extra solve, and the residual is stencil
applications on arrays already in memory, roughly `1e-3` of a solve. Operator validation (§5d) is a
formula evaluation on a grid. **Path A is unchanged** (already 2 levels).

**On guardrails §14 C8** — "re-run the cost table once §4's outcome distribution is known". It is
known: 16 of 20 Tier-A PDE specs validate, and the other 4 report `unavailable`, which makes them
A⁻, not Path B. The in-repo caseload does not move. The population that grows is user-brought
problems, where no closed form is the normal case and this whole plan is the only scoring route.

### 7a. The Tier-B-measured-order trick, with the preconditions it needs **[C]**

When a Tier-B probe has already measured the order at two grids, the main ladder can run 2 levels and
plug the probe order into the GCI:

```python
e_est = d10 / (2**p_probe - 1)      # p from the MMS probe, not from a third grid
```

**Rev 1 got the error direction wrong.** `p_probe` too high *understates* `e_est` — using `p=4` where
the real order is 1 underestimates by 15× — which is exactly the direction that turns a fail into a
pass. And `p_probe` is measured on a *smooth manufactured* problem, while the real problem may be
shock-, layer- or BC-limited. The KS spec makes the gap concrete: `mms_probe.expected_order` is
**2.0** while `theoretical_order` is **4.0**.

Three preconditions, all required:

1. `p_probe` is clamped to `min(p_probe, theoretical_order)`.
2. The problem is not declared to carry a shock, kink or layer (`verification.structural_facts`), and
   the probe order agrees with the theoretical order to within the asymptotic band.
3. The GCI uses **`Fs = 3.0`**, not 1.25 — Roache's factor for a study whose order was *imported*
   rather than measured on the ladder itself. Rev 1 kept 1.25, which compounds the understatement.

**And §4a's Tier C must be relaxed accordingly** [C]: it is defined as "Richardson + GCI + asymptotic
guards", and the guards (`shrinking`, `p_sane`, `not_at_roundoff`) need three grids. On the 2-level
path they are supplied by the probe's own pair, and the metrics record
`asymptotic_source: probe`. Where neither is available, Path B stays at 3 levels. Rev 1 removed the
third level and left the guards in the tier definition — a contradiction.

---

## 8. Chaotic problems, and what KS actually shows **[C]**

### 8a. The flag

`chaotic: true` already has a *consumer* — `reference.check_reference` short-circuits on it
(guardrails §14 C9) — but no producer: it is absent from `schema.py`, from `formulator.md`, and from
the KS spec itself. Phase 0 ships the producer side. When set:

- Tier **C** at full `T` is **waived, not failed** — trajectories separate past the Lyapunov time, so
  the asymptotic guards cannot pass and their failure carries no information about the scheme.
- Tier **C** runs instead at a declared `chaotic_T_ref` inside the predictable window.
- Tiers **B**, **D1**, **D2** run unchanged at full `T`.
- The waiver is recorded as `order_check: waived_chaotic` so it never reads as a pass.

**`chaotic: true` is a self-relaxation vector** [N]. It waives the order check, which is precisely the
threat shape guardrails Layer 2 exists for: a formulator that sets it escapes Tier C entirely. So it
requires a `requirements` ledger entry quoting `problem.md` (KS's statement says "This equation is
chaotic and has no closed-form solution" — the quote exists), and `chaotic_T_ref` must be declared
rather than chosen by the evaluator at scoring time.

### 8b. KS is not the example rev 1 thought it was

Rev 1 said KS "is currently `SELF_ONLY`… under this design it certifies", and made "KS certifies at 10
with `manufactured`" a phase gate. **It already does** — all four plans report
`provenance: manufactured` today, via the `degenerate_limit` already in the spec (γ→0 leaves the
linear KS equation, exact mode by mode). That degenerate limit is also strictly better than the
`exact_spectral` rev 1 proposed to add, which was neither a schema-known key nor an evaluable
expression (`uhat_k(0)` is undefined in the namespace).

`SELF_ONLY` is `benchmark/report.py`'s verdict about **the benchmark's independent verification**,
driven by `has_ground_truth=False`. `manufactured` is the **pipeline evaluator's** provenance. Two
different axes; rev 1 conflated them.

What KS really demonstrates is §1's table: four evaluators, four irreconcilable numbers, four 10s.
That is the case for the kernel, and it does not depend on the absence of a closed form at all.

### 8c. The benchmark is a different layer, and needs its own remedy **[C]**

Of the 50 benchmark problems: 47 have ground truth, 2 are `ground_truth_kind: "stability"`, and
`pde_kuramoto_sivashinsky` is the **only** no-ground-truth case. Rev 1's acceptance criteria 4, 5 and
6 and its Phase 5 gate all rest on that single problem.

That is a fact about the benchmark, not about this plan, because the two operate at different layers:

| Layer | Owns | Scored against | Population |
|---|---|---|---|
| **The pipeline** — formulator → plan-creator → solver → evaluator | `workspace/{slug}/` | its own `problem_spec.json` | any problem a user brings |
| **`benchmark/verify.py`** — grades the pipeline | `benchmark/problems.py` | harness-owned ground truth, never a pipeline artifact | the 50 benchmark problems |

**This plan fixes the first layer, for every problem.** That is its whole justification, and it does
not depend on the no-closed-form population being large. §1's KS table happens to be a Path-B
problem, but the same regenerated arithmetic scores the 47 Path-A ones.

**It cannot fix the second layer, and must not try.** The kernel reads `verification.operator`, which
the pipeline's own formulator wrote. Running it inside `verify.py` would grade the pipeline against a
field the pipeline produced — precisely the circularity `verify.py` exists to break. A structural
check on the pipeline's own declaration is a *self*-check, however deterministic the code performing
it is.

So the benchmark's one blind spot gets the benchmark's own remedy: **give KS a harness-owned
reference.** `ground_truth_kind: "reference"` already exists and is used by `sde_ginzburg_landau_s4`
and `s6`, where the harness ships `moments(T) -> (value, standard_error)` from a discretization-free
Monte Carlo computed offline. The PDE analogue is a high-resolution spectral KS solve, run once,
offline, with `u(x, t = 50)` stored in `problems.py`.

Chaos does not prevent this. At `L = 32*pi` the largest Lyapunov exponent is ~0.05-0.1, so error
amplifies by `e^2.5` to `e^5` over `t = 50`: a solve converged to 1e-12 is still good to ~1e-10, six
orders below the problem's own 1% target. Confirm it the way any reference is confirmed — two
independent high-resolution schemes (ETDRK4 and IMEX-SBDF3 at large `N`) agreeing to that tolerance.
`benchmark/validate_ground_truth.py` is the existing home for exactly this work.

### 8d. Which removes `benchmark/` from this plan entirely **[C]**

Rev 2's first draft offered the reference as a "cheaper alternative" to routing D1/D2 through
`verify.py`, and floated a `SELF_STRUCTURAL` verdict. That was a false choice: the two fix different
layers and neither substitutes for the other. The correct consequence is simpler than either — with a
harness reference, KS is verified the way the other 47 problems are, `verify.py` needs no structural
path, and **this plan touches `benchmark/` not at all.**

One standing rule survives, in case the question returns: if a structural check is ever added to
`verify.py`, its operator must live in `benchmark/problems.py` alongside every other piece of ground
truth, and its verdict must be named for what it is. A verdict derived from the pipeline's own spec
is not `VERIFIED`.

---

## 9. Implementation — file by file **[C]**

### 9a. `verifylib/` exists; the kernel is a subpackage

136 tests pass today. `verifylib/operator.py` and `verifylib/tests/test_canaries.py` are **already
taken** by different content, so rev 1's tree collides with both. The kernel goes in a subpackage and
imports the shared modules rather than duplicating them:

```
verifylib/
├── operator.py        [existing — shared stencils + restricted namespace; §9b extends it]
├── reference.py       [existing — the Tier-A check; D1 reuses residual_field / _scaled_residual]
├── schema.py          [existing — gains the invariant-name and `chaotic` checks]
├── cli.py             [existing — gains the `evaluate` subcommand]
└── kernel/                                                              [NEW]
    ├── __init__.py         run(spec, plan_dir, config) -> metrics dict
    ├── ladder.py           refine / detect_periodic / build_ladder / restrict     [A §3]
    ├── richardson.py       GCI, asymptotic guards, probe-order variant            [A §4] [N]
    ├── residual.py         D1: slope test, stencil guard, trivial guards          [N]
    ├── mms.py              probe driver + numeric probe validation                [A §5]
    ├── degenerate.py       degenerate-limit driver                                [A §6]
    ├── invariants.py       the §8 catalogue **as a closed registry**              [A §8] [N]
    ├── temporal.py         dt_factor isolation                                    [A §10]
    ├── score.py             §4c + manual §12/§23 as a decision table              [N]
    ├── metrics.py           the emitted JSON schema, incl. the skip vocabulary    [N]
    └── sde/
        ├── surrogates.py   moment ODE, Kolmogorov + truncation guard, stationary  [A §15-17]
        ├── crn.py          common random numbers, strong/weak order               [A §18-19]
        ├── dynkin.py       generator construction, dt-Richardson, CI test         [A §20]
        └── constraints.py  positivity, support, martingale, estimator stability   [A §21]
```

Tests go in `verifylib/tests/test_kernel_*.py`; `test_canaries.py` **gains** the seeded-defect suite
(§9c) rather than being replaced — its two existing canaries test the guardrails layers.

`verifylib/__init__.py`'s docstring currently states the package "deliberately does not hold the
numerical verification kernel". Update it; it is the module-level statement of what this package is.

Almost all of the rest is transcription — `verification_manual.md` already contains the reference
implementations. The work is packaging, not derivation. The exceptions are `residual.py`,
`invariants.py`'s registry, and `metrics.py`, which are new.

### 9b. Higher-order stencils are added, not swapped **[N]**

`operator.py` gains `d1(f, axis, h, order=4)` / `d2(..., order=4)` with 6th- and 8th-order variants,
plus spectral differentiation for periodic axes. **The default stays 4** — `reference.py`'s `TOL`,
its `PROBE_N` escalation and the measured 16-of-20 distribution are all calibrated against the
current stencils, and changing them silently would move a result the guardrails plan pins with tests.
D1 opts into the higher orders explicitly.

### 9c. The canary suite, enumerated **[C]**

The kernel becomes the single point of failure for every score; rev 1 gave its test plan one line.
Each seeded defect names the tier that must catch it:

| Seeded defect | Must be caught by |
|---|---|
| Trivial collapse (field decays to a homogeneous steady state) | D1 `nontrivial` guard |
| Phase lag (correct amplitude, shifted solution) | D2 invariants / C at full `T` |
| Over-diffusion (upwind numerical viscosity on a coarse grid) | C |
| Wrong boundary condition applied | D2 BC check |
| A first-order scheme reporting second-order convergence | C asymptotic guards |
| MC under-resolution (too few paths, CI wider than tolerance) | §14 MC-resolution |
| A sign-flipped or dropped operator term | §5d operator validation |

Plus a negative control: an honest solver triggers none of them — the rule `test_canaries.py` already
enforces for the guardrails layers.

### 9d. Modified files

| File | Change |
|---|---|
| `agents/evaluator-pde.md` | Replace "write `evaluate.py`" with "run `cli.py evaluate`, then diagnose". **Delete the scoring rubric** — the kernel emits `score` (§2c); the agent transcribes it into the headline and may only cap downward with a reason. Keep the feedback-by-failure-mode table and the Key Rules, which are now the agent's whole job. Delete the inline numerical code and the ladder-length arithmetic |
| `agents/evaluator-sde.md` | Same. Path A/A′/B selection stays; the arithmetic and the score move to `kernel/` |
| `agents/solver-pde.md` | Document the `snapshots` return key (§5c). `override` is unchanged |
| `agents/solver-sde.md` | No change — `observables` / `path_integrals` already cover Dynkin |
| `agents/plan-creator-{pde,sde}.md` | Declare `scheme_family` and `spatial_order` in the `SOLUTION.md` frontmatter (§3b). Both rules in §3b are unimplementable without them |
| `agents/formulator.md` | Add `chaotic` + `chaotic_T_ref` with the ledger requirement (§8a). Require a degenerate limit whenever a parameter setting collapses the problem to something closed-form. `verification.operator` is already required |
| `references/verification_manual.md` | Reframe Parts I–II as the kernel's specification — §12 and §23 specifically become the specification for `kernel/score.py`. New: §26 D1 (the slope test, the guards, the snapshot contract), §27 the invariant registry, §28 the chaotic waiver. Add `snapshots` to §7's return contract. Move §11/§22 consensus out to the ranking plan |
| `references/project_manual.md` | Score convention: the §4c table, incl. the A⁻ split; add `manufactured_partial` to the provenance list |
| `verifylib/schema.py` | Unknown gated invariant name → error; `chaotic` type + ledger check |
| `verifylib/review.py` | The review's `Score: N/10` headline must equal `metrics.score` **exactly** — a new check, and the one that makes the kernel-emitted score binding rather than advisory |
| `verifylib/cli.py` | `evaluate <plan_dir> [--json]` |
| `benchmark/verify.py`, `benchmark/report.py` | **No change** (§8d). The benchmark's KS blind spot is fixed by a harness-owned reference in `problems.py` — separable work, not a deliverable of this plan |
| `commands/conductor.md` | Rank on the kernel's `score` / `provenance` fields rather than a number parsed from review prose. Parse `manufactured_partial` and `analytic_unvalidated`; rank per guardrails C4. **Sequence against the other two plans' conductor edits (C7)** |

The provenance vocabulary is now seven values and is read by `report.py`, `compare.py`, `run.py`,
`conductor.md`, `STATE.md` and `REPORT.md`. Change them in one pass.

---

## 10. Phasing **[C]**

Rev 1 had no phase for the contract changes, which meant every spec and solver written before them
would be legacy forever. They are cheap and they gate everything else.

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **0** | Contract and schema only: `snapshots` return key (manual §7 + `solver-pde.md`); `chaotic` + `chaotic_T_ref` in `schema.py` and `formulator.md`; the invariant-name registry + schema error; higher-order/spectral stencils in `operator.py` (opt-in); `scheme_family` + `spatial_order` in the plan frontmatter (§3b); the metrics skip vocabulary | All 136 existing tests still pass; all 23 PDE specs in `workspace/` still validate; no change to any measured reference-check outcome |
| **1** | `kernel/` ladder + richardson + invariants + metrics, PDE only. `cli.py evaluate` wired; evaluator agents rewritten | On **frozen artifacts**, the kernel reproduces each plan's metrics within stated tolerance — **and reconciles the four KS plans**, or reports why they cannot be reconciled. That reconciliation is the phase's real product |
| **2** | `residual.py` — D1 with the slope test, the stencil guard and the trivial guards; operator validation (§5d) | The trivial-collapse and sign-flipped-operator canaries are caught; the FD6/FD8 guard reports `unresolved` rather than failing on the spectral plans |
| **3** | The Tier-B-order trick with §7a's preconditions | Path B wall time falls measurably where the preconditions hold, and is unchanged where they do not — both measured, not assumed |
| **4** | `kernel/sde/` — surrogates, CRN, Dynkin, constraints | The 6 SDE problems in `workspace/` reproduce their metrics on frozen artifacts |
| **5** | *(Separable — not a dependency of 0–4, and touches no file this plan owns.)* Give KS a harness-owned reference in `benchmark/problems.py`: high-resolution spectral solve, stored `u(x, t=50)`, `ground_truth_kind: "reference"` | Two independent high-resolution schemes agree to ~1e-10, and KS's benchmark verdict is no longer `SELF_ONLY` |

Phase 0 is a day. Phases 1–2 deliver most of the value. Phase 4 is the largest single chunk.

Rev 1's Phase 1 gate ("existing problems reproduce their current scores") is too weak to keep: the
current scores include KS plan 3's `observed_order: 0.003` scoring 10. Reproducing that faithfully
would be a failure, not a pass.

---

## 11. Acceptance criteria **[C]**

1. No evaluator writes an `evaluate.py`. The kernel is invoked as `cli.py evaluate <plan_dir>`, and
   no plan directory created after Phase 1 contains that file.
2. Every `<metrics>` block is kernel-emitted, and `review.py`'s numeric binding finds backing for
   every number in the review's bound section — guardrails criterion 11, now meaningful (C6).
3. **No evaluator assigns a score.** The review's `Score: N/10` headline equals `metrics.score`
   exactly, enforced by `review.py`; the only agent-side deviation is a recorded `agent_cap` with a
   reason, and it is always downward. A test asserts that an agent-raised score is rejected.
4. On frozen artifacts the kernel reproduces each problem's metrics within stated tolerance, and the
   four KS plans either reconcile or are individually flagged. No two plans of one family report
   observed orders differing by more than the asymptotic band with both unflagged.
5. All seven §9c canaries are caught by the named tier, and an honest solver triggers none.
6. D1's slope test is `clean` on the correct solvers across the PDE population, and `unresolved`
   rather than failing on the spectral and 6th-order plans.
7. An operator corruption the reference check catches on Path A is also caught on Path B by the MMS
   or degenerate-limit validation route (§5d).
8. Path B wall time falls where §7a's preconditions hold and is unchanged where they do not; D1's
   overhead is under 1% of a Path B evaluation, measured.
9. `chaotic: true` without a supporting ledger entry is a schema error; `order_check:
   waived_chaotic` appears in the metrics and is never rendered as a pass.
10. An unknown gated invariant name is a schema error; a declared gate the kernel cannot evaluate
   reports `not_reported`, caps the score at 9, and is named in the review.
11. The kernel runs solver code under `runner.py`'s sandbox: a hanging or memory-hungry solver yields
    `crashed` with a reason, not a hung session.
12. `uv run --extra dev python -m pytest verifylib` stays green (136 tests today, plus the kernel's).
13. *(Separable, §8c.)* KS's benchmark verdict is no longer `SELF_ONLY` — via a harness-owned
    reference in `problems.py`, not via anything the kernel computes.

Dropped from rev 1: "KS reports a provenance other than `none`" — already true before this plan
(§8b), so it measures nothing.

---

## 12. Risks and open questions **[C]**

| Risk | Mitigation |
|---|---|
| The formulator writes a wrong `verification.operator`, making solver and residual self-consistently wrong | §5d's three validation routes, and D1 gates only when one of them has run. Does **not** catch a spec wrong in the same way `problem.md` was misread — that is the requirements ledger's job |
| The kernel becomes the new single point of failure | Fixed code, under test, with the §9c canary suite and the frozen-artifact reproduction gate. Strictly better than regenerated code with no test at all — which is the status quo, and §1 shows what it produces |
| **D1's slope test needs the residual at two resolutions** | Not a constraint in practice: every path runs at least 2 ladder levels, so one ratio always exists. What it does require is that the kernel request `snapshots` from **both** of the top two solves rather than only the finest — free, since the solver returns them on every call, but it must be specified. (Rev 2's first draft claimed the slope was unavailable on Path A; that was wrong) |
| "Different discretization family" is ambiguous for hybrid schemes (IMEX spectral-FD) | Default to high-order FD unless the solver is *purely* FD, then spectral on a periodic grid or a higher-order compact stencil otherwise. Record the choice in the metrics; the §5b guard catches the case where it did not help |
| Degenerate limits do not exist for every problem | Then Tier B is unavailable and the plan tops out at 9 with `manufactured_partial`, or 8 with `self_convergence`. The honest outcome, not a defect |
| **The snapshot contract adds solver burden and older solvers do not have it** | It is three lines beside the `invariant_trace` the contract already asks for, and its absence is `unavailable`, never a failure. Solvers written before Phase 0 keep scoring exactly as they do now |
| **The kernel scores a case its checks did not anticipate, and the agent cannot correct it upward** | Deliberate. The honest outcome of an unanticipated case is a lower score with a named gap, and a cap that only moves downward cannot launder a failure into a pass. If a pattern of correct solvers scoring low emerges, that is a rubric bug to fix in `kernel/score.py` with a test — visible, versioned, and fixed once, rather than re-litigated by an agent every cycle |
| Extra spec surface raises formulator burden | `chaotic` is one boolean plus a quote it already has. `operator` is already required and already shipped |

### 12a. Open at implementation time **[N]**

None of these block Phase 0. All three are cheap to decide while building and expensive to retrofit,
so decide them in the phase named.

1. **Config provenance — decide in Phase 1.** §2 has the agent select and configure checks (manual
   §24's conditional Stage-2 gates). Every value it picks — the ladder's base `N`, which Stage-2
   checks run this cycle, which tolerances apply — moves the outcome, and none of it is currently
   recorded. `metrics.py` should carry each config value **and its source** (`spec` / `agent` /
   `kernel_default`), so a misconfiguration is visible directly rather than inferred from an odd
   result. Cheap while `metrics.py` is being written; expensive once consumers exist.

2. **`order_check: false` versus Tier C — decide in Phase 1.** `pde_kuramoto_sivashinsky` sets
   `evaluation_thresholds.order_check: false` and `min_spatial_order: 0.0`, because its problem
   contract lists only a relative-L2 target and imposes no convergence-order requirement. Tier C is
   defined as "Richardson + GCI + asymptotic guards", which reads as a contradiction. Proposed
   resolution: the asymptotic guards still run, against `verification.theoretical_order` per §5a;
   `order_ok` is vacuously true; and C rests on the GCI alone. Encode that in `kernel/score.py`
   explicitly — an implicit reading of this is how a spec-level opt-out silently becomes a tier-level
   one.

3. **Phase 1's reproduction gate is partly unachievable, and not through any fault of the kernel.**
   A plan whose solver never implemented `override` has no MMS metric to reproduce, so its metric set
   legitimately differs. Make the pass condition explicit rather than hedged: **reproduce every
   metric whose inputs exist, and enumerate the plans where they do not, naming the missing input.**
   A silent partial reproduction reported as a pass is the failure mode to avoid, and it is the one
   this plan exists to prevent elsewhere.

**Decided** (open in rev 2's first draft): `p_design` for the slope test comes from
`verification.theoretical_order`, not from the plan's claimed order in `SOLUTION.md`. It is the
problem's property, so a plan cannot lower its own bar by claiming a lower order, and it is the value
manual §4's `p_sane` guard already uses — one source, one meaning. A scheme that deliberately
under-resolves is caught by C rather than D1 either way.

---

## 13. Provenance summary

| Element | Source |
|---|---|
| Ladder of evidence, tiers A/A⁻/A′/B/C, provenance tags, score ceilings | **[A]** §1, §12, §23 |
| MMS with numeric probe validation; faulty probe skipped not failed | **[A]** §5 |
| Degenerate limit as Tier-B evidence | **[A]** §6 |
| Nested ladder, Richardson, GCI, asymptotic guards | **[A]** §3–4 |
| Invariant catalogue, gate/non-gate distinction | **[A]** §8 |
| Temporal-error isolation | **[A]** §10 |
| SDE surrogates, CRN, weak/strong order, Dynkin, constraints | **[A]** §15–21 |
| Cost discipline, conditional Stage 2 gates | **[A]** §24 |
| `verification.operator`, its shape and its schema gate | **[guardrails §4d]** — shipped |
| Median-gated, term-balanced residual and its normalization | **[B]**, **[guardrails]** — shipped in `reference.py` |
| Candidate-reported residual structurally demoted | **[B]** |
| Deterministic kernel / agent split | **[N]** |
| Different-discretization-family rule | **[N]** |
| Trivial-attractor hole and the three D1 guards | **[N]** |
| D1 as a gate rather than a report | **[N]** |
| Two-circularity-breaks certification rule | **[N]**, tightening **[A]** |
| `chaotic` flag and the C waiver | **[N]**, consumer shipped by guardrails C9 |
| **D1 as a slope test rather than a magnitude test** | **[C]**, transplanting **[A]** §20's Dynkin rule |
| **Stencil-insensitivity guard** | **[C]**, transplanting `reference.PROBE_N`'s escalation |
| **`snapshots` return-contract extension** | **[C]**, replacing rev 1's unimplementable `override` route |
| **A⁻ score split (quoted vs unavailable)** | **[N]**, completing guardrails C4 |
| **Closed invariant-name registry** | **[N]** |
| **Tier-B-order trick preconditions (clamp, smoothness, `Fs = 3.0`)** | **[C]**, correcting rev 1's error direction |
| **CLI entry point instead of a generated `evaluate.py`** | **[C]** |
| **Sandboxed kernel execution** | **[N]**, reusing `benchmark/runner.py` |
