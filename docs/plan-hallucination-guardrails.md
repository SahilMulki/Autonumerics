# Hallucination Guardrails — Implementation Plan

**Status:** **implemented** (rev 6, 2026-09-04) — Phases 0–6 landed, 136 tests passing.
Seven rev-5 claims did not survive implementation; all are corrected below and recorded as
E23–E31. The most important: `pde_cahn_hilliard_2d` is **not** a spec defect.

> **Two things are outstanding, and neither is a code change.**
>
> 1. **Criterion 14 has not been run.** The whole battery on a problem outside `benchmark/` needs a
>    live `claude --plugin-dir` session; every component is tested but the end-to-end run is not.
>    This is the criterion that says Autonumerics is a tool rather than a benchmark harness, so the
>    claim should not be made until it passes. Procedure, and the one sub-check that testing the
>    risk already turned into a real fix, in §15.
> 2. **Two staged specs fail `check-spec` today** — `pde_porous_medium_2d` and
>    `pde_fractional_diffusion`, both missing `verification.operator`. Neither blocks anything; both
>    are pre-Phase-4 pipeline outputs that a re-formulation fixes. Details in §2.

**Scope:** Layers 1–5. Layers 6 and 7 are **deferred** (§9); a small, load-bearing part of 7 is
retained.
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-blind-evaluator.md](plan-blind-evaluator.md) — compatibility analysed in §14.

Provenance labels: **[A]** = already in this repo, **[B]** = adapted from the AutoNumerics
paper-release pipeline, **[N]** = new in this plan, **[C]** = corrected against measurement.

Every quantitative claim below was measured against this repo, most recently on 2026-08-31. Where a rev-2 claim did
not survive measurement it is marked **[C]** and the corrected number is given. Errata: §17.

---

## 0. What changed, and why

### Rev 1 → rev 2 (design review)

| Rev 1 claim | Outcome |
|---|---|
| Quote-or-null is the "highest consequence, smallest change" | **Wrong predicate.** It grades the formula's *origin*, not its *truth*: it accepts a wrong user-supplied formula and rejects a correct derived one. 0 of 29 `problem.md` state a closed form while 25 of 28 specs carry one, so it also nulls 25 specs |
| The AST scan catches the two observed leaks | **False for both.** One is transcribed numpy, one is a docstring — and the plan itself specified that comments are not access |
| The ledger re-audit probes the run's actual `eps` | **Not implementable.** A PDE solver returns `{numerical_solution, grid, t_final}`; no channel carries `eps` |

### Rev 5 → rev 6 (implementation)

Building it changed seven things. Two are ordinary; five matter.

| Rev 5 claim | Measured outcome | Fixed in |
|---|---|---|
| `pde_cahn_hilliard_2d` is a genuine spec defect — "the first real defect the guardrail found" | **No. It is a manufactured solution whose source is recorded in the top-level `source_term` field.** Bind it and the median residual falls from 0.247 to 1.2e-08. The guardrail as designed would have made a **false accusation of a spec defect** — the failure mode §16 singles out as invisible, because it looks exactly like a caught bug | §2, §4f, §10 D |
| Poisson / Helmholtz / Monge–Ampère are blocked because the source "exists only as prose inside `governing_equation`" | **`source_term` is a real, evaluable, top-level field**, and `residual_operator` already references it as a bare `f`. All three validate today, with no spec change, plus `anisotropic_diffusion` | §2, §4d |
| PDE coverage is 11 of 20 | **16 of 20** — 9 clean, 7 off a singularity | §2, criterion 2 |
| SDE coverage is 3 of 5; vector moment shapes need per-shape handling | **5 of 5.** Differentiating the claimed moments needs the state recovered from them, which fails when a state component appears in no claimed moment (`p12`). *Integrating* the declared moment ODE forward needs nothing from the claimed moments and works at any state dimension | §4e, criterion 7 |
| Apply expression-not-program to `drift_expression` and `diffusion_expression` | **Hard-fails 3 of 6 real SDE specs on correct content.** A vector problem's drift is legitimately prose — `"X @ F.T (row-major paths, X shape (num_paths, 2))"` describes a matrix action no scalar expression can. Split: answer-key fields must parse; guidance fields must merely not be a *program* | §4a |
| Test 2 compares the formula at `t = 0` against `initial_condition` | Two corrections. A **backward parabolic** problem states its condition at maturity, not at 0 — reading `pde_black_scholes_call` at 0 reports a 3.5% mismatch on an exactly correct spec. And an exact-match tolerance false-fails `pde_advection_1d`, whose non-periodic `initial_condition` differs from its periodic solution by 1.2e-04 — real, benign, two orders below the problem's own target | §4c |
| Numeric binding: every number in the Numerical Accuracy section | **Flags 28% of 206 archived reviews**, all on diagnostic context the pipeline computes but records nowhere. Bind the **claim** on each keyed line instead — the leading token, not every number. 0 false positives | §8a |

### Rev 2 → rev 3 (measurement)

*(rev 4 adds the production path — hooks ship with the plugin, the `Stop` hook is the gate on a
user's own problem — and lands Phase 0 in `verifylib/`. See E20–E21.)*

Rev 2 was prototyped and hook behaviour was tested empirically. Four of its claims did not survive.

| Rev 2 claim | Measured outcome | Fixed in |
|---|---|---|
| `PostToolUse` hooks with a **path matcher** block a bad write | **Both halves false.** `matcher` is tool-name only — a `"**/target.txt"` matcher never fires. And `PostToolUse` exit 2 does **not** block: the file is on disk and is not reverted. Only `PreToolUse` blocks | §3 |
| Singular sets are found by "points where the residual grows under refinement" | **Misclassifies in three directions**, including waiving a wrong formula as "singular" — the exact failure it exists to catch | §4f |
| The reference check runs on 17 of 20 PDE Tier-A specs | **11 of 20** with a real implementation. An operator *string* existing is not the check *working* | §2, §4d |
| SDE coverage is 5/5 | **3 of 5.** Two specs have vector-valued moment ODEs (5 coupled moments) needing per-shape handling | §4e |

Rev 2 also under-specified two things measurement exposed: the term-balanced denominator is
degenerate for single-term operators (§4c), and the rev-1 AST scan flags **124 of 137** pipeline
files, all false positives (§6b).

---

## 1. Threat model

Guardrails are organised by **what gets faked**, because that is how they actually fail.

| # | What gets faked | Consequence | Status |
|---|---|---|---|
| 1 | The ground truth | Everything downstream grades against a wrong reference | **in scope** |
| 2 | The problem itself | The pipeline solves something easier and scores 10/10 | **in scope** |
| 3 | Isolation of the answer key | The solver reads the answer it is graded on | **in scope** |
| 4 | The grade | A component grades its own output | **in scope, mostly already built** |
| 5 | The judgement | An LLM grader rewards plausibility instead of evidence | **in scope** |
| 6 | The write-up | The report claims more than the runs support | deferred — §9 |
| 7 | The guardrails themselves | A layer silently stops working after a refactor | deferred except §12 |

---

## 2. Measured baseline

| Quantity | Value |
|---|---|
| Specs in `workspace/` | 28 (22 PDE, 6 SDE) |
| Specs carrying a non-null closed form | **25** — 20/22 PDE, 5/6 SDE |
| `problem.md` files stating a closed form | **0 of 29** — they say "scored against a hidden reference solution" |
| Specs carrying `verification.operator` | 0 — the field is defined by this plan (§4d) and lands in Phase 4 |
| PDE specs carrying `mms_probe.operator_check` | 19 of 22 |
| Operator strings that AST-split into signed top-level terms | **19 of 19** |
| **PDE Tier-A claims the reference check validates** | **16 of 20** — 9 clean, 7 off a localized singularity |
| **SDE Tier-A claims the reference check validates** | **5 of 5** |
| Whole-workspace gate wall time | 2.3 s for all 29 specs |
| Pipeline files flagged by the rev-1 AST leakage rule | **124 of 137 (91%)**, all false positives |
| Pipeline files flagged by the narrowed rule | **0 of 137**, with the genuine patterns still caught |

### Why 16 of 20, and what blocks the other 4 **[C]**

Rev 2 counted specs that *have* an operator string; rev 3 counted what a first-cut implementation
could run. Neither is what the finished check reaches. Root causes, because they determine which are
fixable and by whom:

| Root cause | Specs | Owner |
|---|---|---|
| **Validated** (9) | `heat_1d`, `heat_2d`, `laplace_2d`, `wave_1d`, `wave_2d`, `poisson_2d`, `helmholtz_2d`, `monge_ampere_2d`, `anisotropic_diffusion` | — |
| **Validated off a localized singularity** (7) | `burgers_inviscid` (shock), `stefan_1d_similarity` (kink), `convection_diffusion_bl` (layer), `black_scholes_call` (kink at the strike), `cahn_hilliard_2d`, `advection_1d`, `fokker_planck_ou` (nodal lines, where every term vanishes together) | — |
| **Multi-field system on the legacy operator string** | `mhd_2d`, `navier_stokes_2d` | formulator — these specs predate `verification.operator`; the systems path itself is implemented and tested |
| **No operator string in the spec at all** | `porous_medium_2d` | formulator — `verification.operator` is now required |
| **Non-local operator — genuinely unavailable** | `fractional_diffusion` (Caputo derivative) | — |
| Already `null` | `heston_2d`, `kuramoto_sivashinsky` | — |

### Two staged specs fail the gate today **[N]**

Distinct from the coverage table above, and more operationally important: these are specs that
produce an **error-severity** finding, not merely an unvalidated one. `check-spec` exits 2 on them.

| Spec | Finding | Why it stands |
|---|---|---|
| `pde_porous_medium_2d` | `verification.operator` required | No operator information of any kind — no `operator`, no `mms_probe.operator_check`, no `residual_operator`. Its Barenblatt solution is checkable; nothing in the spec says against what |
| `pde_fractional_diffusion` | `verification.operator` required | A Caputo derivative is non-local, so the operator genuinely cannot be written as a local expression. The fix is `"operator": null` **with an `operator_note`** — the field distinguishes a recorded gap from a silent omission, and this spec currently records nothing |

**Neither is a bug in the guard, and neither blocks anything today.** Both specs are outputs of
previous pipeline runs, written before the field existed; `workspace/` is not tracked in git, and a
re-formulation under the Phase-4 `formulator.md` emits the field. The `Stop` gate scopes to the
active problem (§3), so a session working on a third problem is not blocked by either. What they
*would* do is fail the conductor's step-3b `check-spec` call if either problem were re-run without
re-formulating — which is the intended behaviour, and the reason they are named here rather than
waived.

**One near-miss worth recording.** A first cut of `schema.check_operator_declaration` accepted only
`mms_probe.operator_check` as the legacy fallback, while `reference.resolve_operator` also consults
`residual_operator`. That made three specs error, not two: `pde_anisotropic_diffusion` was rejected
by the gate while the reference check it gates for **validated it at 2.5e-10**. A gate that
contradicts the check it exists to enable is worse than no gate, because the contradiction is
invisible from either side alone. The two now read the same sources, and
`test_the_schema_gate_agrees_with_the_reference_check_on_legacy_sources` pins it.

**`pde_cahn_hilliard_2d` was not a spec defect, and this is the most important correction in rev 6.**
Rev 3 through rev 5 held it up as "the first real defect the guardrail found, and it was found
before the guardrail was built." Its claimed solution `0.2*sin(2πx)*sin(2πy)*cos(t)` does give a
median term-balanced residual of 0.247, flat across N = 129/257/513 — but against the **homogeneous**
operator. The solution is *manufactured*, and its source term is recorded in the spec's top-level
`source_term` field. Bind that and the median is **1.2e-08**.

Two things follow, and both are worth more than the finding would have been.

1. **The source path is the fix, not a workaround.** `source_term` is an evaluable field on five
   specs, and `verification.residual_operator` on eleven already references it as a bare `f`. Rev 3
   assumed the source "exists only as prose inside `governing_equation`" and never checked. Binding
   it validates Poisson, Helmholtz, Monge–Ampère and `anisotropic_diffusion` as well — four more
   specs, no spec change, which is where 11 → 16 comes from.
2. **The guardrail nearly produced the exact failure §16 warns about.** A false hard-fail is worse
   than a missed relaxation because it is invisible as a false positive: it looks exactly like a
   caught bug. Here it would have accused a correct spec of being defective, in a document that
   named it as the headline result. The regression test in
   `verifylib/tests/test_reference_pde.py::test_cahn_hilliard_is_a_manufactured_solution_not_a_defect`
   pins the correction, and it is the reason §4d's operator resolution consults every source the
   spec offers before concluding anything.

---

## 3. The enforcement model **[N, C]**

*Cross-cutting; every deterministic check depends on it, so it comes first.*

Rev 2 specified `PostToolUse` hooks with a path matcher, blocking on exit 2. Seven controlled runs
against `claude 2.1.226`; full table in
[`verifylib/tests/hookprobe/FINDINGS.md`](../verifylib/tests/hookprobe/FINDINGS.md), which should be
re-run after a Claude Code upgrade because the whole enforcement model rests on it.

| Test | Result |
|---|---|
| `"matcher": "**/target.txt"` alongside `"matcher": "Write"` | **Only `Write` fired.** `matcher` is tool-name only; there is no path matcher |
| `PostToolUse` exit 2 on a `Write` | stderr reached the agent; **the file was on disk and not reverted** |
| `PreToolUse` exit 2 on a `Write` | **The file was never created.** Payload has `tool_input`, no `tool_response` |
| **`Stop` hook exit 2** | **Blocks session end** — but fired 8 times and the run died at max-turns |
| Reason-fed retry (satisfiable constraint) | `REJECT` → 3.2 s → `PASS` |
| Hooks inside a Task-dispatched **subagent** | **Fire normally**, same REJECT→PASS cycle |
| **Plugin-declared hooks** (`<plugin>/hooks/hooks.json`, loaded via `--plugin-dir`) | **Load and fire** |
| **`${CLAUDE_PLUGIN_ROOT}` in a hook `command`** | **Expands** to the plugin directory; `CLAUDE_PROJECT_DIR` is separately the user's cwd |
| `Bash` heredoc (`cat > target.json`) against a `Write\|Edit` matcher | **No hook fired.** The file landed unvalidated |
| Hook contradicting an explicit user instruction | The agent **stopped, left the bad file on disk, and escalated to the user** |

Four consequences.

**(a) Path filtering happens inside the hook.** The payload carries `tool_input.file_path`, so this
costs three lines. But the matcher must be `Write|Edit|Bash` or a heredoc walks straight past it, and
the `Bash` branch has to inspect `tool_input.command` for a redirect target.

**(b) Blocking requires `PreToolUse`, which sees the wrong thing.** `PreToolUse` gets `tool_input`,
not the resulting file. For `Write` that is the whole content and is fine; for `Edit` it is
`old_string`/`new_string`, so a validator that must parse the finished `problem_spec.json` would have
to apply the edit itself to reconstruct it. Pay that cost only where blocking genuinely matters.

**(c) A hook is feedback, not a gate.** The observed failure is not the deadlock rev 2 planned for.
It is the opposite: the agent stops, leaves the invalid file in place, and asks the user. Interactively
that is correct. In a headless `claude -p` run **nobody answers** — the subagent returns, the file is
invalid, and the conductor never learns.

**(d) The `Stop` hook is the gate that works everywhere.** It is harness-enforced, fires at session
end, and blocks. It is the only mechanism that gates a run on *any* problem — including a user's own,
where `benchmark/verify.py` does not exist. But it looped 8 times and killed the run at max-turns, so
the attempt-counter bail-out is **mandatory**, not defensive.

> **Every guard runs in two places: as a hook, for fast in-loop correction; and as a gate that does
> not depend on an agent cooperating. On the benchmark that gate is `benchmark/verify.py`; on a
> user's own problem it is the `Stop` hook. The hook-on-write is an optimisation.**

### Assignment

| Guard | In-loop feedback | Gate (benchmark) | Gate (any user problem) |
|---|---|---|---|
| Spec schema + reference check (§4) | `PostToolUse` on `Write\|Edit\|Bash` → `problem_spec.json` | conductor pre-dispatch + `verify.py` | **`Stop` hook** |
| Leakage scan (§6) | `PreToolUse` on `Write\|Edit` → `solver.py`, `evaluate.py` (blocks) | `verify.import_solver`, `runner._child_main` | **`Stop` hook** |
| Ledger re-audit (§5) | — needs a completed run | `benchmark/verify.py` | **`Stop` hook** |
| Numeric-claim binding (§8) | `PostToolUse` on `SOLUTION.md` | conductor, when reading the score | **`Stop` hook** |

The `Stop` hook runs the same `verifylib` entry points over the workspace directory and refuses to
end the session while a `failed` outcome stands. On its 3rd firing for the same finding it exits 0,
writes `GUARDRAIL_UNSATISFIED` beside the offending file, and lets the session end — so a run that
cannot be fixed terminates with a recorded reason instead of burning the turn budget.

**It gates one problem, not the whole workspace. [C]** Implementation exposed a scope error in that
sentence: `workspace/` holds 29 problem directories from previous runs, and three carry pre-existing
findings. Sweeping all of them blocks a session working on a fourth, for something it did not do and
cannot fix — and then writes a marker the conductor reads as a block on the *current* problem. Both
the pipeline (`/conductor workspace/{slug}/problem.md`) and a user running Autonumerics on their own
problem work one problem at a time, so the gate scopes to the problem directory whose files were most
recently touched, and falls back to the root when there are no subdirectories — which is the shape a
scratch directory with a bare `problem.md` has, i.e. criterion 14's shape.

### Packaging and distribution

Hooks ship **with the plugin**, in `hooks/hooks.json`, not in `benchmark/pipeline-settings.json`.
That is what makes them apply to a user's own problem rather than only to benchmark runs, and it is
measured to work. Commands are addressed as:

```json
{"type": "command", "command": "${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py check-spec"}
```

`verifylib/cli.py` is the single entry point, invoked by absolute path so it does not depend on the
working directory. `pyproject.toml` keeps `package = false`; nothing needs installing. Agents that
call `verifylib` directly use the same form, matching the existing `${CLAUDE_PLUGIN_ROOT}/references/…`
convention.

`benchmark/pipeline-settings.json` keeps only its `deny` rules — those are benchmark-specific, since
`benchmark/` does not exist in a user's project.

---

## 4. Layer 1 — Fabricated ground truth

A wrong reference is the **one input that fails common-mode**: every plan for a problem descends
from one spec, so all plans are wrong identically. Cross-plan consensus confirms it, the solver
spends its iterations chasing feedback derived from it, the ranking orders on it, and `REPORT.md`
cites it. §1 of `verification_manual.md` says this of tiers C/D/E; it is equally true of a corrupted
Tier A.

### 4a. The expression-not-program check **[N]** — do this first

`analytic_solution.expression` must parse as a single evaluable expression:

```python
ast.parse(expr, mode="eval")     # SyntaxError on anything else
```

The observed Heston leak was a 2063-character `def _heston_call_price(...)` with Gauss–Legendre
quadrature in that field, plus a `notes` line telling the evaluator to "transcribe it verbatim into
evaluate.py". This rejects it instantly. One line, zero judgment, no false negatives.

**Two classes of field, because one rule measured badly. [C]** Applying `ast.parse(mode="eval")` to
`drift_expression` and `diffusion_expression` as rev 5 specified hard-fails 3 of 6 real SDE specs on
*correct* content: on a vector problem the drift is legitimately prose —
`"X @ F.T (row-major paths, X shape (num_paths, 2), F = [[F11, F12], [F21, F22]])"` describes a
matrix action that no scalar expression can express, and an agent reads it rather than evaluating it.

The threat §4a exists to stop is a **program** parked in a field the evaluator is told to transcribe.
So:

| Class | Fields | Rule |
|---|---|---|
| **Answer-key** — the pipeline evaluates these | `analytic_solution.expression` / `.fields.*`, `analytic_moments.*`, every `verification.*` expression, `verification.operator.*`, and any `constraints`/`invariants` entry with `gate: true` | Must parse as an expression |
| **Guidance** — an agent reads these | `drift_expression`, `diffusion_expression`, `diffusion_derivative_expression`, `diffusion_matrix_expression`, ungated `constraints`/`invariants` notes | Must not be a **program**: valid Python that is not a single expression |

The second rule is the sharper statement of the same idea. Prose ("`X ~ Normal(mu, sigma**2)`")
parses as neither an expression nor a program, so it passes; the Heston `def` parses as a program
and fails **in any field**. Measured over all 29 staged specs: **0 false positives**, with the
archived leak still caught. All 11 `gate: true` constraints parse; the one that does not is ungated,
and is a distributional statement rather than a check.

### 4b. The rule: validate, or quote, or `null` **[N]**

Rev 1 proposed *quote-or-null*; the design review proposed *graded provenance*
(`quoted`/`recalled`/`derived`). Both grade the formula's **origin**, and origin is either
externally checkable (quoted) or a self-report (everything else). Graded provenance is the worse of
the two: it asks a model to classify its own hallucination, and a model that misremembers a formula
does not experience it as misremembering — it will label a confabulated integral `recalled`,
sincerely. Quote-or-null at least has a machine-decidable predicate, but the predicate is orthogonal
to the property that matters, and quoting establishes **blame, not correctness**.

> **An `analytic_solution` (or `analytic_moments`) enters Tier A when it is validated against the
> problem's own declared structure. Where validation cannot run, a verbatim quote from `problem.md`
> is accepted in its place and the gap is recorded. Otherwise the field is `null`.**

Validation is primary; a quote is the **fallback for an unavailable check**, not a bypass. A quoted
formula that *fails* validation is still `null` — the case neither original rule caught, and the one
that matters in production, where there is no `verify.py` behind the user.

### 4c. The reference check — PDE **[N]**

Three tests, all on the *formula*, none on the solver:

| Test | Catches |
|---|---|
| Term-balanced operator residual ≈ 0 | Wrong PDE, wrong coefficient, dropped factor, sign flip |
| Formula at the initial time matches `initial_condition` | Right PDE, **wrong solution branch or mode** |
| Formula on the boundary matches `boundary_conditions.values` | Right PDE and IC, wrong BC selection |

**Test 2 needed two corrections that only appear once it runs on real specs. [C]**

*The initial time is not always zero.* `pde_black_scholes_call` is backward parabolic: its
"initial" condition is a **terminal payoff** at `T_maturity`, and it marches forward in
`tau = T_maturity - t`. Reading it at `t = 0` reports a 3.5% mismatch on an exactly correct spec.
The check evaluates at each time the spec *names* — `t0` for a similarity solution singular at the
origin, `T_maturity` for a backward problem — and records which one matched. Two declared
quantities, not a search.

*Exact agreement is the wrong bar.* `pde_advection_1d` declares a plain Gaussian
`initial_condition` on a **periodic** domain while its solution uses the periodic wrap; near the
seam they differ by **1.2e-04**. That is real — the IC as literally written is not periodic — but it
is benign, two orders below the problem's own 1e-2 target. So test 2 has two thresholds: below 1e-6
is `validated`, above 1e-3 is `failed`, and the band between is `inconsistent` — reported as a
warning, never fatal. A wrong branch or doubled mode differs by O(1), so the `failed` bar keeps three
orders of headroom.

Test 3 skips what it cannot read rather than guessing: a moving boundary (`"x=s(t)"`), an
`"outflow"` label, a `"note"` key. Saying the check does not reach there beats inventing a face.

Measured separation on real specs (N = 257, 4th-order stencils, `h_t = 1e-3`):

| Problem | correct | `pi**2` dropped | decay halved | sign flip | mode doubled |
|---|---|---|---|---|---|
| `pde_heat_1d` | 2.5e-10 | 8.99e-01 | 5.00e-01 | 2.00e+00 | 7.50e-01 |
| `pde_heat_2d` | 4.8e-09 | 8.99e-01 | 5.00e-01 | 2.00e+00 | 6.00e-01 |
| `pde_wave_1d` | 2.6e-10 | — | — | — | 7.50e-01 |
| `pde_laplace_2d` | 1.1e-11 | — | — | — | 7.50e-01 |

Eight to eleven orders of separation, deterministic, no self-report. This is **better calibrated
than a residual on solver output**, because evaluating a closed form is nearly free: put it on
4096 points, drive the check's own discretization error to 1e-10, and use a tight tolerance. You
cannot do that with a solver, which is why D1 in
[plan-no-closed-form.md](plan-no-closed-form.md) must gate rather than certify. Here the same
arithmetic certifies.

**Three implementation facts established by prototype, all of which rev 2 assumed without checking:**

1. **Term decomposition works on the existing single-string form.** Parse the operator with
   `ast.parse(mode="eval")` and walk the top-level `+`/`-` chain, collecting signed operands:
   `"u_t - alpha*lap_u"` → `[+u_t, -alpha*lap_u]`. **19 of 19 real operator strings split cleanly**,
   from 1 term (`lap_u`) to 7 (Heston). No spec change needed to get a term-balanced denominator.
2. **Single-term operators are degenerate and need one more level. [C]** For `lap_u` the residual
   *is* the only term, so `residual / max|term| ≡ 1` and Laplace fails on a correct harmonic
   solution. Fix: when the split yields fewer than two terms, decompose one level down into the
   constituent second derivatives (`u_xx`, `u_yy`). With that, `pde_laplace_2d` reports **1.1e-11**.
3. **The restricted namespace must be passed as globals, not locals.** Any claimed solution
   containing a lambda or comprehension raises `NameError` if the variables are supplied as the
   locals mapping — several real specs do. The namespace must also supply the special functions
   real formulas use (`erf`, `erfc`, `ndtr`); `pde_black_scholes_call` references `ndtr` and
   errors without it.

**Normalisation is term-balanced, never absolute** — `max|residual| / max_k max|term_k|`. Helmholtz
at large `k` and 100:1 anisotropy both have individually huge terms that cancel; an absolute
tolerance rejects correct formulas on both.

### 4d. Where the operator comes from — **this plan ships the field** **[C]**

The check resolves an operator from four sources, in descending order of what each can express:

| # | Source | Covers |
|---|---|---|
| 1 | `verification.operator` — the field this plan defines | everything: source terms, systems |
| 2 | `verification.residual_operator`, with `f` bound to the top-level `source_term` | 11 specs; source-bearing scalar problems |
| 3 | `mms_probe.operator_check`, minus `source_term` when the spec declares one | 19 specs; the homogeneous form, and manufactured solutions |
| 4 | nothing | `unavailable`, with the reason recorded |

**Rev 5 was wrong about routes 2 and 3, and the error mattered. [C]** It claimed
`pde_poisson_2d`'s source "exists only as prose inside `governing_equation`" and that
`residual_operator`'s bare `f` "is never defined anywhere evaluable". Neither is true:
`source_term` is a real top-level field carrying `"2 * np.pi**2 * np.sin(np.pi * x) * np.sin(np.pi * y)"`,
and binding it makes Poisson, Helmholtz, Monge–Ampère and `anisotropic_diffusion` all validate
today, on the legacy path, with no spec change. It is also what shows `cahn_hilliard_2d` to be a
manufactured solution rather than a defect (§2).

The ceiling that *is* real: `operator_check` cannot express a **per-equation system**. `mhd_2d` and
`navier_stokes_2d` declare `"u_t + adv_u + grad_p - nu*lap_u"`, where `adv_u` and `lorentz_u` are
composites meaning something different in every system that uses them. Supplying them generically
would be guessing, so the check reports `unavailable` and names the fix. Those two specs predate the
field; a spec written after Phase 4 declares `operator.equations` and is checked in full —
demonstrated on the real Taylor–Green solution, which validates at 1.8e-07 while a sign-flipped
pressure fails at 2.0.

**Decision: `verification.operator` is defined and required by this plan, not by
[plan-no-closed-form.md](plan-no-closed-form.md).** Rev 3 listed it as that plan's §3 deliverable and
therefore as a hard dependency on 5 of 20 specs. That ordering does not survive the requirement that
Autonumerics work on any problem a user brings: a fresh `problem.md` with a source term or a system
would get no reference check at all until a *different* plan landed. The field is a JSON shape plus a
formulator paragraph — small, and this plan is the one that needs it first.

The same ownership rule as C1: whoever lands first owns it. plan-no-closed-form **consumes** the
declaration for its D1 residual gate rather than defining it, exactly as it consumes
`verifylib.operator` for stencils. A cross-reference is added to that plan's §3 so the field cannot
be specified twice with different shapes.

#### The field

Scalar, with the source that `operator_check` has no room for:

```json
"operator": {
  "fields": ["u"],
  "terms": {"u_t": "dt(u)", "diffusion": "-alpha*lap(u)"},
  "source": "0"
}
```

System — one entry per equation, which is what MHD and Navier–Stokes need:

```json
"operator": {
  "fields": ["u", "v"],
  "equations": {
    "u": {"terms": {...}, "source": "0"},
    "v": {"terms": {...}, "source": "0"}
  },
  "combine": "rms"
}
```

The residual is `sum(terms) - source` per equation; systems combine by RMS. The term keys are the
decomposition the balanced denominator needs, supplied explicitly rather than recovered by AST
splitting — the split stays as the fallback path for legacy specs.

**Required on every new PDE spec, both paths** — with or without a closed form. `formulator.md` gains
the instruction and the two PDE templates gain the shape. A spec that cannot express its operator
(a Caputo derivative) declares `"operator": null` with a reason, which is a recorded gap rather than
a silent omission.

**This is a legacy-spec ceiling, not a design one.** The 5 blocked specs were written before the
field existed. A problem formulated after Phase 4 emits the operator with its source, so a user
bringing their own `problem.md` gets the full check on the first pass. The fallback exists to cover
`workspace/` as it stands, not to define what the check can do.

Helper namespace required by the real strings: `u`, `u_t`, `u_tt`, `grad_u` (and `grad_u[i]`),
`lap_u`, `lap_lap_u`, `lap(·)` as a callable, `u_xx`/`u_yy`/`u_xy` and the non-Cartesian analogues
(`u_SS`, `u_Sv`, `u_vv`), plus `adv_u`, `grad_p`, `lorentz_u` for the systems. Two vocabularies for
the same thing already coexist — `lap_lap_u` in Cahn–Hilliard, `lap(lap_u)` in Kuramoto–Sivashinsky
— so supply both. `pde_monge_ampere_2d` needs the mixed derivative `u_xy`, which
plan-no-closed-form's helper list omits; that is §14 C1.

### 4e. The reference check — SDE **[N, C]**

The moment ODE the spec declares — `state`, `rhs`, `initial` — is a **self-contained initial-value
problem**. Integrate it forward and compare the result against the claimed moments, through the
`mean_from` / `variance_from` map the spec also declares. Run it only where `closes_exactly` is
`true`; a closure approximation is not a reference.

**This replaces rev 5's differentiate-the-claimed-moments design, and it is why coverage is 5 of 5
rather than 3 of 5. [C]** Differentiating requires the *full state vector* recovered from the
claimed moments at each `t`, so it fails whenever a state component appears in no claimed moment —
`sde_multichannel_stiff_m13`'s `p12` is exactly that, with nothing to differentiate. Integrating
needs nothing from the claimed moments at all, so the state dimension stops mattering: it is one
code path for scalar and vector alike, and it is *more* accurate (≤ 2.1e-10 against the
differentiation route's 2e-11…2e-10 on a smaller set) because the integrator runs at rtol 1e-12,
six orders tighter than the comparison tolerance.

One trap worth naming, because pairing by name walks straight into it: `cross_from` is a
**covariance** (`"mxy - m1x*m1y"`) while `cross_moment.expression` is the raw `E[XY]`. Matching
those two reports a 1.0 mismatch on a perfectly consistent spec. The covariance is matched against
`cross_moment.covariance_value_at_T` instead — both say covariance, and the spec commits to that
number. Every pairing is by declared quantity; none is guessed.

Measured over 12 time points on the real specs:

| Spec | state | max relative mismatch | Verdict |
|---|---|---|---|
| `sde_bm_with_drift` | 2 | 1.9e-13 | `validated` |
| `sde_cir_feller_violated` | 2 | 2.9e-13 | `validated` |
| `sde_ornstein_uhlenbeck` | 2 | 4.1e-12 | `validated` |
| `sde_gbm_2d_high_corr` | **5** | 2.1e-10 | `validated` |
| `sde_multichannel_stiff_m13` | **5** | 1.5e-13 | `validated` |
| `sde_ornstein_uhlenbeck`, decay halved | 2 | **5.3e-01** | **caught** |
| `sde_ornstein_uhlenbeck`, variance `2*theta`→`theta` | 2 | **3.9e-01** | **caught** |
| `sde_bm_with_drift`, `mu`→`2*mu` | 2 | **2.5e-01** | **caught** |
| `sde_gbm_2d_high_corr`, `sigma1**2` halved | 5 | **5.1e-01** | **caught** |
| `sde_multichannel_stiff_m13`, `F12*t`→`2*F12*t` | 5 | **1.8e+00** | **caught** |

**Coverage is 5 of 5.** Nine orders between every correct spec and every corruption.

This is a consistency check between two independently written spec fields rather than an external
truth check (§4g), but a mis-recalled OU mean and a correctly transcribed OU moment ODE will not
agree, and that is the failure being hunted.

### 4f. Classifying a non-zero residual **[N, C]** — the rule rev 2 got wrong

This is the highest-risk mechanism in Layer 1 and rev 2's version does not work. It proposed:
*"points where the residual grows under refinement are singular."* Measured, that misclassifies in
three directions at once:

| Case | Rev-2 rule | Should be |
|---|---|---|
| `heat_1d` with `pi**2` dropped | **UNAVAILABLE** — the residual is flat, and "not falling" read as "growing" | **FAILED** |
| `heat_2d`, correct | **UNAVAILABLE** — residual already at 1e-9, so refining makes it noisier, not smaller | **VALIDATED** |
| `burgers_inviscid`, correct | **FAILED** — the singularity is one point in 129, so the *fraction* of growing points is ~0 while the *max* is O(1) | **UNAVAILABLE / validated off it** |

The first row is disqualifying on its own: the rule waives a wrong formula as "singular", which is
exactly what it exists to catch.

**The primary statistic is the median, not the max. [C]** An intermediate draft of this section
gated on `max` plus a "singular set < 2% of the domain" threshold. Measured, that threshold is
fragile — the three real singular cases land at 0.8%, 1.2% and 1.6%, a margin of 1.25× — and it
cannot classify `pde_black_scholes_call` at all, whose kink at the strike keeps 3.9% of the domain
"bad" even at N = 1025 while the solution is perfectly correct.

The median separates every case by seven orders with no threshold tuning:

| Case | **median** | q95 | max | Verdict |
|---|---|---|---|---|
| `heat_1d` correct | **2.5e-10** | 2.6e-10 | 3.1e-10 | `validated` |
| `heat_2d` correct | **4.8e-09** | 4.8e-09 | 4.9e-09 | `validated` |
| `wave_1d` correct | **2.4e-10** | 2.7e-10 | 2.9e-10 | `validated` |
| `laplace_2d` correct | **2.6e-12** | 8.3e-12 | 7.0e-11 | `validated` |
| `black_scholes_call` correct (kink at strike) | **5.7e-12** | 1.4e-02 | 1.1e+00 | `validated_off_singularity` |
| `burgers_inviscid` correct (shock) | **0.0** | 0.0 | 1.0e+00 | `validated_off_singularity` |
| `stefan_1d_similarity` correct (kink) | **1.1e-10** | 1.9e-10 | 1.3e+00 | `validated_off_singularity` |
| `convection_diffusion_bl` correct (layer) | **0.0** | 0.0 | 5.7e-03 | `validated_off_singularity` |
| `advection_1d` correct, under-resolved | **1.8e-05** | 5.0e-04 | 6.8e-01 | refine → 9.2e-07 at N=513 |
| `fokker_planck_ou` correct, under-resolved | **6.3e-05** | 1.2e-03 | 1.7e-03 | refine → 4.1e-06 at N=513 |
| `heat_1d`, `pi**2` dropped | **8.99e-01** | 8.99e-01 | 8.99e-01 | **`failed`** |
| `heat_1d`, mode doubled | **7.50e-01** | 7.50e-01 | 1.01e+00 | **`failed`** |
| `cahn_hilliard_2d` against the **homogeneous** operator | **2.47e-01** | 2.99e-01 | 1.03e+00 | `failed` — **and wrongly so**: it is a manufactured solution, and binding its declared `source_term` gives median **1.2e-08**. See §2 |

Correct-and-resolved medians are all ≤ 4.8e-09; every wrong formula is ≥ 2.5e-01. **Tolerance:
`median < 1e-6`**, which sits in a seven-order gap and leaves room for a coarse probe grid.

The rule, in full:

| Condition | Verdict |
|---|---|
| `median < tol` and `max < tol` | `validated` |
| `median < tol`, `max ≥ tol` | `validated_off_singularity` — record where, and how much of the domain |
| `median ≥ tol`, **falls** under refinement | re-probe at the finer grid; classify there |
| `median ≥ tol`, **flat** under refinement | **`failed`** |

Two signals still, but the roles change from the intermediate draft: the **median** is the gate and
**refinement** disambiguates under-resolution. Spatial concentration demotes from gate to
**diagnostic** — it says *where* the singularity is and how large it is, which belongs in the
recorded gap, but it no longer decides pass/fail and the fragile 2% threshold disappears.

This is also better than rev 2 claimed. The non-smooth cases are not merely waived: their median
residual is at or below the smooth cases', so they are genuinely *validated on the smooth
complement* — a real Tier A with a recorded gap, not an unchecked one.

`np.where` / `np.maximum` / `np.minimum` / `np.sign` / `np.heaviside` in the expression is a
syntactic tell for a piecewise solution, detectable by AST. Use it to *predict* where a singular set
will appear, never to excuse one — a formulator-set waiver flag would reintroduce exactly the
self-report §4b exists to eliminate.

### 4g. Outcomes and honest limits

| Outcome | Meaning | Consequence |
|---|---|---|
| `validated` | Residual, IC and BC all pass | Tier A |
| `validated_off_singularity` | Median passes; `max` does not. The singular set's size and location are recorded, not thresholded | Tier A, gap recorded in `<metrics>` and `REPORT.md` |
| `unavailable` | No operator string, no source term, non-local operator, or a systems form not yet supported | Tier A **iff** verbatim-quoted; else `null`. Gap recorded |
| `quoted` | Verbatim in `problem.md`, check not run | Tier A, gap recorded |
| `failed` | The check ran and the formula did not satisfy it | **`null`, unconditionally** — including a quoted formula |

**Common-mode formulator error.** The formulator writes the analytic solution *and* the operator
declaration in the same pass, from the same reading, in the same output. If it believes the equation
is `u_t = alpha*u_x`, it writes both consistently and the residual is ~0. This is not the "narrow
coincidence" rev 1 implied — same author, same context window. Mitigation: the requirements ledger
must carry an entry quoting the governing equation from `problem.md` with `spec_path` pointing at
the operator declaration, so the operator is itself ledger-audited. That is where this is caught,
and nowhere else.

**Two of the three PDE tests are consistency, not truth.** The IC and BC in the spec are also
formulator-transcribed. Checking the formula against them catches wrong-branch and wrong-BC errors —
real and common — but does not reach outside the spec. Only the operator residual can catch a
genuinely mis-recalled formula, and only to the extent the operator is right. Do not present all
three as equivalent evidence.

**The residual alone cannot see initial data.** `pde_advection_1d` demonstrates it: widening the
Gaussian produces a *different but still exact* solution of `u_t + a·u_x = 0`, and the residual
correctly stays small. Only the `t = 0` test catches that. This is the concrete argument for keeping
all three tests rather than the residual alone.

**Retained from rev 1, unchanged:** the prohibition on deriving a solution the problem supplies
**[B]**; `analytic_solution_source_text` required on the `quoted` route, checked for real formula
tokens **[B]**; the blocklist for prose posing as a formula **[B]**; the evidence requirement
extended to the whole `verification` block, since an MMS source is also a fabricable answer key
**[N]**; MMS probes validated numerically, a faulty probe **skipped, not failed** **[A]** §5.

One correction to rev 1's framing: a fabricated MMS source is **not** symmetric with a fabricated
closed form. A wrong MMS source is detectable — apply a high-order operator to `u_mms` and confirm
it reproduces `f`, which is what `operator_check` already does. A mis-recalled closed form was
detectable by nothing internal. That asymmetry is why Tier B sits where it does, and why §4c is the
change that matters most.

---

## 5. Layer 2 — Silent problem relaxation

**Already in place [A]:** the requirements ledger (verbatim `quote`, `status`, `spec_path`; "never
paraphrase — a paraphrase is where a relaxation hides"); the conductor halting at `phase: blocked`
on any `dropped` entry or a `mapped` entry with no `spec_path`.

The gap: the ledger proves the **spec** carries every constraint; nothing checks the **solver obeyed
it**.

### 5a. What is actually auditable **[C]**

Rev 1 assumed `probe_run(result, spec_path)` recovers the value the solver used. It cannot — a PDE
solver returns `{numerical_solution, grid, t_final}`.

| Requirement kind | Auditable from run output? | How |
|---|---|---|
| `domain` bounds | **yes** | `grid[axis][0]`, `grid[axis][-1]` |
| `evaluation.grid_N` | **yes** | array shape |
| time horizon `T` | **yes** | `t_final` — `[B]`'s time-horizon pinning, generalised |
| `boundary_condition` (Dirichlet) | **yes** | `u[0]`, `u[-1]` against declared values |
| `boundary_condition` (Neumann) | partly | one-sided difference at the edge |
| SDE `num_paths` / `dt` / `T` / `seed` | **yes** | already in `run_meta` |
| `parameter` | **no** | needs a source scan or a contract change |
| `initial_condition` | **no** | not recoverable from a terminal snapshot |
| `method` | **no** | prose |

The flagship `eps` example is in the "no" column. Two routes, weighted differently:

- **Source scan (ship now, warn only).** AST-walk `solver.py` for assignments binding a spec
  parameter name to a numeric constant. Catches `eps = 1e-2`; misses `eps = 1e-3 * 10`;
  false-positives on nondimensionalised solvers where the parameter never appears literally.
- **Contract echo (propose, do not ship here).** `solve_pde` returns `params_used`. Stronger, but it
  touches `solver-{pde,sde}.md`, three templates, `runner.py`'s extraction and every existing
  solver. Also self-reported — though a solver echoing `1e-3` while using `1e-2` is actively
  deceptive rather than sloppy, which the source scan then catches. Worth doing deliberately, in its
  own change.

### 5b. Gate versus warn **[C]**

| Finding source | Consequence |
|---|---|
| Output-derived (grid, domain, `t_final`, Dirichlet values, SDE `run_meta`) | **Hard gate.** Facts about the run. Score capped at 3; the review names the ledger entry by `id` and quote |
| Source-derived (parameter AST scan) | **Warning.** Surfaced to the evaluator and recorded as `ledger_warnings`. Never an automatic cap |

A false hard-fail on a correct solver is worse than a missed relaxation, because it is invisible as
a false positive — it looks exactly like a caught bug.

### 5c. Where it runs

`verifylib/audit.py`, called from `benchmark/verify.py` at certification (it already holds the run
output) and available as `python -m verifylib.audit <plan_dir>`. Not a hook — it needs a completed
run, not a written file.

---

## 6. Layer 3 — Answer-key leakage

| Guard | Source | Blind spot it covers |
|---|---|---|
| Harness `deny` rules, honoured under `--dangerously-skip-permissions` | **[A]** | Stops the read at the tool layer |
| Payload sanitizer — blocked keys + free-text scrub | **[B]** | Leakage through data the pipeline chose to pass along |
| **Text citation scan** over `problem_spec.json`, `evaluate.py`, `SOLUTION.md` | **[N]** | Answer-key knowledge arriving as *prose* and transcribed by hand |
| **Narrowed** AST scan for executable oracle access | **[B, C]** | A file opened at runtime |
| Enforced import policy in a sandboxed subprocess | **[A]** | A solver reaching for a forbidden library |

### 6a. The citation scan, and where the chokepoint actually is **[C]**

Measured against the archived leaks:

| Artifact | Citation scan | AST scan |
|---|---|---|
| `pde_heston_2d/problem_spec.json` | **caught** — `_heston_call`, `benchmark/`, `problems.py` | n/a (not Python) |
| `pde_heston_2d/.../evaluate.py` | **clean** | `.get()`, `open()` — **both false positives** |
| `sde_quintic.../evaluate.py` | **caught** — `benchmark/`, `verify.py`, `verify_sde_stability` | clean |
| `sde_quintic.../SOLUTION.md` | **caught** | n/a |

Rev 2 said the citation scan catches both leaks. It catches both *incidents*, but not every
artifact: the Heston `evaluate.py` carries the transcribed formula with **no citation at all**.
Nothing text-based can see it there.

That sharpens the design rather than weakening it. **The spec is the chokepoint.** The Heston leak
entered at `problem_spec.json` — where the citation scan does fire, and where §4a's
expression-not-program check fires independently — and propagated by transcription. Block it at the
spec and the downstream copy never happens. This is the strongest argument for putting the schema
guard on `problem_spec.json` specifically.

```python
CITATION_PATTERNS = (
    r"benchmark/", r"\bproblems\.py\b", r"\bverify\.py\b", r"\bmanifest\.json\b",
    r"\brunner\.py\b", r"\bvalidate_ground_truth\b", r"verify_(sde|pde)_\w+", r"_heston_call\b",
)
```

One legitimate occurrence must be allowlisted: `references/project_manual.md`'s "Never read
`benchmark/`" section names `problems.py`, `verify.py` and `manifest.json` — the exact vocabulary
that appeared in both leaks. A rule has to name what it protects; the scanner has to know that.

### 6b. The AST scan must be narrowed or it is unusable **[C]**

Rev 1 specified flagging `.get()`, `getattr`, `open()` and oracle-ish subscripts. Measured over
**137 pipeline `solver.py` and `evaluate.py` files: 124 flagged — 91%, every one a false positive.**
A guard that fires on nine files in ten is noise, and it would have to be disabled within a day.

Narrow it to two patterns:

- an `Import`/`ImportFrom` of a prohibited top-level module (`benchmark`, `problems`, `verify`,
  `manifest`, `runner`);
- a call to `open`/`Path`/`read_text`/`load`/`loadtxt`/`genfromtxt` whose argument is a **string
  literal containing `benchmark`**.

Measured: **0 of 137 flagged**, with both genuine patterns still caught. Bare `.get()` must go.

### 6c. Wiring **[C]**

Rev 1 said to call the scan in `benchmark/runner.py`. `runner.run_solver` is only reached when
`sandbox=True`, which per [verify.py:756](../benchmark/verify.py#L756) is the **one-shot baseline**
path. The pipeline path is `sandbox=False` and imports inline through
[verify.import_solver](../benchmark/verify.py#L140). As specified the scan would never see a
pipeline-produced solver. Call it from **both**.

### 6d. Deny-list gaps **[N]**

`pipeline-settings.json` covers `Read`, `Edit` and `python -c`. It does not cover
`Bash(cat|grep|sed|head|tail|awk benchmark/...)`, nor `Bash(uv run python benchmark/verify.py ...)`,
which prints ground truth. Confirm `Read(./benchmark/**)` also matches an absolute path, and that
the pipeline deny beats the `Bash(cat *)` and `Bash(uv run python *)` allows in
`.claude/settings.local.json`. One command to test; the "0 overclaims" claim rests on it.

---

## 7. Layer 4 — Self-grading

**In this scope Layer 4 is almost entirely already built.** Stating that plainly beats restating it
as new work.

| Guard | Source | Status |
|---|---|---|
| Separate evaluator; file permission matrix | **[A]** | Built, but **prose only** |
| Out-of-band `benchmark/verify.py` against independent ground truth | **[A]** | Built |
| Candidate-reported residual demoted to `legacy_*` | **[B]** | **Not applicable** — no such field here |
| The deterministic kernel | **[N]** | **Out of scope** — plan-no-closed-form §2 |

One new guard: **detect a permission-matrix violation instead of trusting it.** Hash `solver.py`
before dispatching the evaluator and compare after; a changed hash means the evaluator edited the
code it was grading. ~5 lines in the conductor cycle, and it converts a rule into a detection.

---

## 8. Layer 5 — Grader drift

**Already in place [A]** (§25 of `verification_manual.md`): never score on plausibility; capped at 2
if no test ran; a missing reference is not a solver crash; a check skipped for cost never counts
against a plan; a faulty MMS probe is skipped, not failed.

**From [B], to add to the evaluator prompts:** plan prose cannot establish correctness; execution
success alone is weak evidence; do not reward raw grid density, with a forced
`resolution_evidence: not_used_as_accuracy_evidence`.

### 8a. Numeric-claim binding **[N, C]**

Every number in the **"Numerical Accuracy"** section of a `<review>` must trace to a value the
pipeline computed. Three corrections to rev 1's version:

1. **Scope it to that section.** "Feedback for solver" is where the evaluator does arithmetic on
   purpose — [evaluator-pde.md:424](../agents/evaluator-pde.md#L424) *instructs* it to write
   `"dt=0.01, dx=0.1 gives dt/dx²=1.0 > 0.5 stability limit"`, four numbers, none of them metrics.
   Binding that section rejects the review the prompt asked for.
2. **Widen the backing set** beyond `<metrics>` to `problem_spec.json`'s `parameters`,
   `evaluation_thresholds`, and the grid sizes actually run.
3. **Warn and regenerate once**, then pass with the discrepancy recorded.

**Binding the section was still too broad; bind the *claim*. [C]** Measured over 206 archived
reviews, binding every number in the Numerical Accuracy section flags **28%** — all of them on
diagnostic context the pipeline genuinely computes but records nowhere: per-grid error history,
invariant drifts, the tolerance each invariant was judged against. What must trace is the **claim**
on each keyed measurement line (`error`, `observed_order`, `tolerance`, …): its *leading* token, not
every number after it. `"1.97e-04 at N=128 (PASS) — N=64: 3.15e-03"` claims 1.97e-04 and then
recalls the ladder it came from.

Two further calibrations, both measured:

- **Match at the precision the number was printed to**, not at a flat relative tolerance. A review
  writes `0.000026` for a value carried as `2.556e-05`, and the metrics block writes `1.968e-04` for
  one the review gives as `1.9675e-04`. Both are correct rounding, in opposite directions; whichever
  side rounded harder sets the slack. A flat rtol cannot separate that from drift at every magnitude.
- **A line whose value is not a number claims nothing.** `"observed_order: inf — both grid errors are
  below the 1e-9 floor"` claims `inf`, not 1e-9.

Result: **0 false positives across all 206 reviews**, while a headline that disagrees with the
metrics block above it is still caught.

**Be honest about what it establishes.** The same LLM writes `<metrics>` and `<review>`, so binding
one to the other catches *transcription drift* — the most common LLM-grader failure, worth catching
— but not fabrication. That property arrived with the kernel on 2026-09-05: metrics
are now kernel-computed, and the review's score is bound to them exactly (§14 C6).

Retained from **[B]**: output-schema validation — every candidate ranked exactly once, component
score limits, `evidence_used` naming only supplied sources, no-gap rank contract.

---

## 9. Deferred: Layers 6 and 7

**Layer 6** (machine-enforced provenance at report time, measured/estimated column separation,
Verification-gaps section, content-hash manifest) — deferred whole. One piece is pulled forward into
§13 for a different reason.

**Layer 7** (nine seeded canaries, grader-disagreement reporting) — deferred, with two exceptions:

- Rev 1's own phasing put canaries *before* the ledger re-audit — "build the test harness before
  adding the thing it tests." Cutting Layer 7 wholesale removes that in a repo with **no test
  infrastructure at all**. §12 restores the minimum.
- Seven canaries test the numerical kernel, which is not being built. **Two test this plan's layers**
  and are cheap: `hardcoded_exact.py` (leakage scan + order check) and `shortened_horizon.py`
  (ledger re-audit's `t_final` check). Keep those.

---

## 10. Worked failure scenarios

### A — A mis-recalled analytic solution

| Layer | Outcome |
|---|---|
| **1** §4a | **Caught, if it arrives as a program.** This is what happened: a 2063-char `def` in an expression field |
| **1** §4c | **Caught, if it arrives as an expression** — *once the Heston operator has a source term*. On today's fallback path Heston's `analytic_solution` is already `null`, so the live instance of this scenario is the general one: a wrong `heat_1d` kernel separates by 9 orders |
| **3** §6a | **Caught at the spec** by the citation scan — the actual Heston leak cited `benchmark/problems.py` |
| If it slipped through | `verify.py` → `OVERCLAIM`, **on the benchmark only**. A user's own problem has no such backstop, which is what §4 exists for |

### B — A quietly relaxed problem

*Convection–diffusion at `eps = 1e-3`; the solver drifts to `1e-2` and converges beautifully.*

| Layer | Outcome |
|---|---|
| **2** ledger | `R1` quotes `"eps = 1e-3"`, mapped to `parameters.eps`. Spec correct; nothing fires |
| **2** re-audit, output-derived | **Does not fire.** `eps` is not recoverable from the run output (§5a) |
| **2** re-audit, source scan | **Warns.** `solver.py` binds `eps = 1e-2`; surfaced as `ledger_warnings`, not a cap |
| Backstop | The eventual D1 gate uses the **spec's** `eps`. Out of scope here; noted for the kernel plan |

Rev 3 **weakens** rev 1's claim here, because rev 1's claim was not implementable. A warning is what
is deliverable without a contract change.

### C — A wrong closed form in a user's own `problem.md`

| Rule | Outcome |
|---|---|
| Quote-or-null (rev 1) | **Accepted at full Tier A** — the source-text check passes cleanly |
| Graded provenance (review proposal) | **Accepted** — `quoted` is the top tier |
| §4b (rev 3) | **Caught.** Validation runs on quoted formulas too; `failed` nulls the field regardless of provenance |

### D — A spec whose exact solution does not solve its own equation

| Layer | Outcome |
|---|---|
| **1** §4c/§4f | **Caught, `failed`.** Global (100% of the domain) and non-converging — the two signals that distinguish a wrong formula from a singularity or from under-resolution. Measured on injected corruptions: a dropped `pi**2` gives 8.99e-01, a doubled mode 7.50e-01, against ≤ 4.8e-09 for the correct form |
| Today | Invisible. The evaluator grades against it and every plan agrees |

**The example this scenario used to cite was a false positive.** Rev 3–5 named
`pde_cahn_hilliard_2d` here as a live instance. It is not one: the solution is manufactured and its
source is declared (§2). The scenario is still real — it is what §4c is for — but it is now
demonstrated on injected corruptions rather than on a spec that turned out to be correct. That
distinction is the whole point of §16's warning that a false hard-fail is invisible as a false
positive.

---

## 11. Implementation — file by file

| File | Change | Layer |
|---|---|---|
| `verifylib/__init__.py` **(new)** | Package marker. Deterministic guards only — **not** the numerical kernel | — |
| `verifylib/operator.py` **(new)** | Helper namespace + stencils + AST term-splitting + restricted eval. **Shared with plan-no-closed-form's D1** — §14 C1 | 1 |
| `verifylib/schema.py` **(new)** | Expression-not-program; evidence rule over `analytic_solution` + `verification`; ledger shape | 1, 2 |
| `verifylib/reference.py` **(new)** | The reference check: PDE residual + IC + BC; SDE moment-ODE consistency; median-gated classification; five outcomes | 1 |
| `verifylib/audit.py` **(new)** | Ledger re-audit — output-derived gate, source-derived warn | 2 |
| `verifylib/leakage.py` **(new)** | Citation scan + **narrowed** AST scan | 3 |
| `verifylib/review.py` **(new)** | Scoped numeric-claim binding, warn-once | 5 |
| `verifylib/cli.py` **(new)** | Single entry point, invoked by absolute path as `${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py <check>` so it works from any working directory | 1, 3, 5, 6 |
| `verifylib/hooks/` **(new)** | `PreToolUse` / `PostToolUse` / `Stop` handlers; in-hook path filtering; `Bash` redirect parsing; the 3-attempt bail-out | 3 |
| `hooks/hooks.json` **(new, plugin root)** | Ships the hooks with the plugin so they apply to a user's own problem, not only to benchmark runs. Measured to load via `--plugin-dir` | 3 |
| `verifylib/tests/` | **exists** — archived leak fixtures, hook-probe findings, 32 tests. Add the two retained canaries | 12 |
| `benchmark/pipeline-settings.json` | Close the `Bash` read gaps in `deny`. **No hooks here** — they ship with the plugin instead, or they would only ever fire on benchmark runs | 3 |
| `benchmark/runner.py` | Leakage scan in `_child_main` before executing | 3 |
| `benchmark/verify.py` | Leakage scan in `import_solver`; `verifylib.audit` at certification | 2, 3 |
| `benchmark/run.py` | Agent-file hashes per run in `results.json` (§13) | — |
| `agents/formulator.md` | The §4b rule; the five outcomes; `analytic_solution_source_text` on the quote route; **`verification.operator` required on every PDE spec, both paths**, with `null` + a reason where the operator is inexpressible | 1 |
| `templates/problem_spec-example-pde.json` **(edit)** | Add the scalar `verification.operator` shape | 1 |
| `templates/problem_spec-example-pde-system.json` **(edit)** | Add the multi-equation `operator` shape | 1 |
| `references/verification_manual.md` **(edit)** | Document `verification.operator` as spec surface, so the formulator authors it from the manual it already reads | 1 |
| `agents/evaluator-{pde,sde}.md` | Anti-plausibility rules from **[B]**; `resolution_evidence`; numeric-binding contract | 5 |
| `commands/conductor.md` | Out-of-band gate calls; `GUARDRAIL_UNSATISFIED` → `phase: blocked`; `solver.py` hash check | 1, 3, 4 |
| `references/project_manual.md` | The §4b rule, the five outcomes, the gate/warn split | 1, 2 |
| `pyproject.toml` | `pytest` in the `dev` extra | 12 |

**Deliberately not here:** the D1 gate, the ladder, Richardson, invariants,
the SDE numerical surrogates — all plan-no-closed-form. `verifylib/reference.py` shares that plan's
helper vocabulary but is far smaller: it differentiates a *formula*, not a solver output, so it needs
no discretization-family selection, no two-snapshot `dt(u)`, and no `override` hook.

---

## 12. Phasing

| Phase | Deliverable | State |
|---|---|---|
| **0** | `verifylib/` package, `pytest` dependency, `tests/` with the archived leak fixtures and the hook-probe findings | **done** |
| **1** | `verifylib/schema.py` — expression-not-program, the evidence rule, ledger shape, the `verification.operator` requirement | **done** — 0 false positives over 29 staged specs |
| **2** | `verifylib/leakage.py` wired into `verify.import_solver` **and** `runner._import_solver`; `deny` gaps closed in `pipeline-settings.json` | **done** — both paths tested |
| **3** | `verifylib/cli.py`; `hooks/hooks.json` shipping with the plugin: `PreToolUse` for leakage, `PostToolUse` for schema, `Bash` in the matcher, the `Stop` gate with its 3-attempt bail-out | **done** |
| **4** | `verifylib/reference.py` complete + the §4b rule in `formulator.md`, `verification.operator` in both templates and `verification_manual.md` | **done** — 21 of 25 Tier-A claims validated |
| **5** | `verifylib/audit.py` — ledger re-audit, output-derived gate / source-derived warn; wired at certification | **done** |
| **6** | `verifylib/review.py` — scoped numeric binding | **done** — 0 false positives over 206 reviews |
| — | Guards degrade to AST-only on an interpreter without numpy (§15 b) | **done** |
| — | §13 run provenance: agent-file hashes in `results.json` | **done** — 23 files |
| — | The two retained canaries (§9) | **done** — they could not be written until Phases 2 and 5 existed |

One deviation from the plan as written, forced by the language rather than by the design:
`verifylib/operator.py` shadows the stdlib `operator` module whenever `cli.py` runs as a script,
because Python puts the script's own directory first on `sys.path`. `collections` imports
`operator`, `functools` imports `collections`, `re` imports `functools`, `json` imports `re` — so
the failure surfaces as a circular-import error *inside stdlib json*, with nothing pointing at the
cause. `cli.py` drops its own directory from `sys.path` before importing anything but `sys` and
`os`. The module keeps its name, because §14 C1 freezes it as a cross-plan interface, and
`test_cli_runs_from_any_directory_as_a_script` runs the real subprocess so it cannot come back.

### What was actually in Phase 4

It is the largest phase by a wide margin, so it is itemised rather than left as one line. The seed in
`verifylib/` covers the first three rows; the rest is new work.

| Piece | State | Note |
|---|---|---|
| Term splitting, signed top-level operands | **done** | 19/19 real operator strings |
| Restricted eval; namespace passed as **globals** | **done** | several specs contain a lambda; locals fails |
| Stencils + helper namespace, both `lap_lap_u` and `lap(·)` spellings, `u_xy` | **done** | shared with plan-no-closed-form's D1, §14 C1 |
| Sub-term decomposition for single-term operators | **done** | Laplace is degenerate without it |
| Median-gated classification + refinement re-probe | **done** | five outcomes, §4f |
| SDE moment consistency, scalar **and vector** state | **done** | one path, by forward-integrating the declared ODE. 5 of 5 |
| Test 2 — formula at the initial time vs `initial_condition` | **done** | two candidate times, two thresholds (§4c) |
| Test 3 — formula on the boundary vs `boundary_conditions` | **done** | Dirichlet faces; moving/symbolic boundaries skipped, not guessed |
| **`verification.operator` schema + formulator instruction + templates** | **done** | Owned here, not by plan-no-closed-form (§4d, §14 C2) |
| Source-term path | **done** | via `source_term`, which already existed — Poisson / Helmholtz / Monge–Ampère / anisotropic |
| Multi-field systems path (`operator.equations`, `combine`) | **done** | validated on real Taylor–Green; sign-flipped pressure fails at 2.0 |
| Non-Cartesian axis names (`u_SS`, `u_Sv`, `u_vv`) | **done** | built from `spatial_variables`, so they come free |
| Capitalised coordinate aliases (`X`, `Y`) | **new, unplanned** | several specs write the meshgrid capitalised while `spatial_variables` stays lower-case |
| `formulator.md`: the §4b rule, five outcomes, operator required on every PDE spec | **done** | Step 3-0 |

---

## 13. Run provenance **[N]**

`agents/formulator.md` is a pipeline input, so editing it changes behaviour and results either side
of the edit are not strictly comparable. That is **accepted**: these three plans change what the
pipeline measures, and the point of changing it is that the new numbers are better grounded. The
requirement is not to avoid the discontinuity but to make it **legible**.

**Record agent-file hashes in `results.json` per run** — SHA-256 of every file under `agents/`,
`commands/`, `references/`, plus the `verifylib` version. This is the one piece of Layer 6 worth
pulling forward, it is what makes a result attributable to a prompt version a year later, and it is
what plan-blind-evaluator's offline experiments need to mean anything when they say "frozen
artifacts" (§14 C8).

Re-run the benchmark after Phase 4 and report both numbers, with the hash difference cited. A
provenance change that is recorded is a finding; one that is silent is a confound.

---

## 14. Compatibility with the other two plans

Both related plans are in flux, so this section lists **interfaces to hold fixed** and **conflicts to
resolve**, not a merge order.

| # | Conflict | Between | Resolution |
|---|---|---|---|
| **C1** | **Two implementations of the same stencils.** This plan's `reference.py` and plan-no-closed-form's `operator.py` both discretize operator expressions. Two implementations of `lap_u` can silently disagree — the reference check passes with one stencil set while D1 fails with the other | guardrails ↔ no-closed-form | **One shared `verifylib/operator.py`**, owned by whichever lands first (this plan, Phase 4) and consumed by D1. It must supply `u_xy` (needed by `monge_ampere_2d`) and both `lap_lap_u` and `lap(·)` spellings — plan-no-closed-form's helper list currently omits all three |
| **C2** | **Who defines `verification.operator`.** The reference check needs its `source` field (Poisson / Helmholtz / Monge–Ampère) and its per-equation systems form (MHD / Navier–Stokes). plan-no-closed-form §3 also specifies the field, for D1 | guardrails ↔ no-closed-form | **Resolved: this plan defines and requires it (§4d); plan-no-closed-form consumes it.** Same rule as C1 — whoever lands first owns it. Rev 3 had this the other way round, which meant a user's own source-bearing problem got no reference check until a different plan landed. A cross-reference is added to plan-no-closed-form §3 so the shape cannot be defined twice. **Interface to freeze:** `fields` / `terms` / `source`, or `fields` / `equations` / `combine` for systems |
| **C3** | **Metrics shape change.** plan-blind-evaluator §7 changes metrics values from scalars to `{"value": x, "oracle_derived": bool}`. This plan's numeric binding parses `<metrics>` | guardrails ↔ blind-evaluator | `verifylib/review.py` must accept both shapes from day one. Cheap if written that way; a rewrite if not |
| **C4** | **Provenance stops being totally ordered once Tier A splits.** plan-blind-evaluator ranks on `analytic > surrogate > manufactured > manufactured_partial > self_convergence > none`. This plan splits Tier A by evidence quality, so an **unvalidated quote would outrank a validated surrogate** | guardrails ↔ blind-evaluator | **Decided: fix it here, now.** This plan owns the split, so it owns the vocabulary. The total order becomes `analytic` > `analytic_unvalidated` > `surrogate` > `manufactured` > `manufactured_partial` > `self_convergence` > `none`, with `validated` and `validated_off_singularity` mapping to `analytic` and `quoted` / `unavailable` mapping to `analytic_unvalidated`. A one-line cross-reference is added to plan-blind-evaluator §4 pointing here, so the ranker cannot be built against the stale list |
| **C5** | **Two leakage mechanisms for the same concern.** plan-blind-evaluator §7 wants a leakage scan over the *evidence* fed to a blind ranker, with `oracle_derived` flags plus a `PROHIBITED_TOKENS` fallback. This plan owns the scanner | guardrails ↔ blind-evaluator | One implementation in `verifylib/leakage.py`, two call sites. `oracle_derived` is the structured version of the token blocklist — keep both, they fail differently, but do not write the scanner twice |
| **C6** | **`evaluate.py` shrinks to a `verifylib.run` call** under plan-no-closed-form §8b | guardrails ← no-closed-form | *Synergy, not conflict.* **Resolved 2026-09-05, and better than this row predicted:** `evaluate.py` does not shrink, it **disappears** — the kernel is reached through `cli.py evaluate <plan_dir>`, so the citation-scan surface loses a whole file rather than most of one. Layer 5's binding upgraded as described, and gained a stronger form: the kernel emits `score`, so `review.py` now requires the review's `Score: N/10` headline to equal `metrics.score` **exactly** rather than within a tolerance. `SOLVER_FILES` still lists `evaluate.py`, because a legacy plan directory may contain one and it must stay scanned |
| **C7** | **All three plans edit `commands/conductor.md`**: guardrails adds the out-of-band gate calls, `phase: blocked` on `GUARDRAIL_UNSATISFIED`, and the `solver.py` hash check; blind-evaluator rewrites the dispatch sort and Phase 3 ranking key; no-closed-form adds `manufactured_partial` parsing | all three | No logical conflict; a merge conflict. Sequence conductor edits, do not parallelise them |
| **C8** | **Guardrails increases the no-closed-form caseload.** Every spec demoted to `null` lands on the Tier B/C ladder, whose cost model (§6 there) assumed a smaller population. Conversely, plan-blind-evaluator's offline experiments assume "frozen artifacts" — which only means something with §13's agent-file hashes | guardrails → both | Re-run plan-no-closed-form's cost table once §4's outcome distribution is known. Land §13 before any offline rank study |
| **C9** | **`chaotic` specs have no closed form**, so Tier A never applies | guardrails ↔ no-closed-form | Compatible. The reference check should short-circuit on `chaotic: true` rather than reporting `unavailable`, so the metrics read correctly |

Two interfaces worth freezing now, before either plan moves:

- **`verifylib/operator.py`'s helper namespace** — the names, and the rule that variables are passed
  as *globals* (§4c). Both plans evaluate spec-authored expressions; if the namespaces diverge, a
  spec that validates will fail D1 for reasons that look numerical and are not.
- **The `<metrics>` key set**, including `oracle_derived` and the reference-check outcome. Three
  plans read or write that block.

---

## 15. Acceptance criteria

1. Every `analytic_solution.expression` and `analytic_moments.*_expression` in `workspace/` parses
   under `ast.parse(mode="eval")`. The archived `pde_heston_2d` spec does not.
2. The reference check reports one of `validated` / `validated_off_singularity` / `quoted` /
   `unavailable` / `failed` on every non-null closed form, and reproduces the §2 distribution over
   all 20 PDE Tier-A specs: **16 validated (9 clean + 7 off a singularity), 2 needing the systems
   operator form, 1 needing any operator string, 1 non-local.** ✅ *met*
3. Corrupting a closed form — dropping `pi**2`, flipping a sign, doubling the mode — yields `failed`
   with a **median** term-balanced residual at least 7 orders above the correct formula's.
4. `burgers_inviscid`, `stefan_1d_similarity`, `convection_diffusion_bl` and `black_scholes_call`
   report `validated_off_singularity` — median below tolerance, `max` above it — never via a
   formulator-set flag, and never via a thresholded singular-set fraction.
5. `advection_1d` and `fokker_planck_ou`, under-resolved at N = 129, **validate at N = 513** rather
   than failing.
6. ~~`cahn_hilliard_2d` reports `failed`, and the report names it as a spec defect.~~
   **Withdrawn — the premise was false.** It is a manufactured solution whose source is declared, and
   it validates at 1.2e-08 (§2). The criterion is replaced by its real content: *an analytic solution
   that genuinely does not solve its declared operator reports `failed`, demonstrated on injected
   corruptions rather than on a spec assumed to be broken*, and a manufactured solution with a
   declared source is **not** reported as a defect. Both are pinned by tests. ✅ *met*
7. `sde_ornstein_uhlenbeck` with a halved decay rate is caught; **all five** SDE Tier-A specs are
   consistent to < 1e-9, vector states included. ✅ *met* — 5 of 5 at ≤ 2.1e-10, every corruption at
   ≥ 2.5e-01.
8. The citation scan catches the Heston **spec** and both quintic artifacts. The narrowed AST scan
   flags **0 of 137** pipeline files while still catching `import problems` and
   `open("benchmark/...")`.
9. A solver run through the pipeline path (`sandbox=False`) is leakage-scanned — verified by a test.
10. A shortened time horizon (`T/2` reported as `T`) is a **hard gate**; a relaxed `eps` is a
    **warning**.
11. Every `<review>` passes scoped numeric binding, including those whose "Feedback for solver"
    sections contain CFL arithmetic.
12. A `Bash` heredoc write to `problem_spec.json` fires the schema hook.
13. An agent that abandons a file after 3 hook rejections leaves a `GUARDRAIL_UNSATISFIED` marker,
    and the conductor halts at `phase: blocked` — **and the gate catches the same file
    independently, with the write hooks disabled.**
14. **The whole battery runs on a problem outside `benchmark/`.** Take a `problem.md` in a scratch
    directory with no `benchmark/` anywhere, run `/conductor` on it with the plugin installed by
    `--plugin-dir`, and confirm: the hooks fire, `${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py` resolves,
    a deliberately corrupted `analytic_solution` is caught, and the `Stop` hook refuses to end the
    session until it is fixed. This is the criterion that says Autonumerics is a tool rather than a
    benchmark harness.
15. **The `Stop` gate terminates.** A deliberately unsatisfiable finding ends the session within 3
    firings with a recorded reason, and does **not** run to max-turns. Measured: without the
    bail-out it fired 8 times and the run died.
16. `results.json` records agent-file hashes for every run.
17. `uv run --extra dev python -m pytest verifylib` is part of the standard workflow and passes.
    (`--extra dev` is required; a plain `uv run` re-syncs to base deps and uninstalls pytest.)
    ✅ *met* — 132 tests.

### Status against the criteria

Criteria 1–13, 16 and 17 are met and pinned by tests. **Criteria 14 and 15 are met only in
part**, and the gap is worth stating rather than glossing:

- **15 (the `Stop` gate terminates)** is tested at the unit level —
  `test_stop_gate_blocks_then_gives_up_with_a_recorded_reason` drives the handler three times and
  asserts it blocks twice and then releases with `GUARDRAIL_UNSATISFIED` on disk. What is *not*
  re-measured is the original 8-firing live run that motivated the counter.
- **14 (the whole battery on a problem outside `benchmark/`)** has **not been run**. Every component
  it depends on is tested — the hooks are wired to `cli.py`, `cli.py` resolves from any working
  directory as a real subprocess, `active_problem` falls back to a bare directory holding a single
  `problem_spec.json`, and the `Stop` gate runs `check_workspace` over whatever directory it is
  given, with no `benchmark/` anywhere in the path. But component tests are not the criterion.
  **This is the one criterion that says Autonumerics is a tool rather than a benchmark harness, so
  the claim should not be made until it passes.**

  What it takes, concretely:

  ```bash
  mkdir -p /tmp/anum-c14/workspace/my_problem
  # write a problem.md by hand; no benchmark/ anywhere on the path
  cd /tmp/anum-c14
  claude --plugin-dir /path/to/Autonumerics -p "/conductor workspace/my_problem/problem.md"
  ```

  Four things have to hold, and each can fail independently:

  | | Check |
  |---|---|
  | a | The hooks fire at all — plugin-declared hooks load via `--plugin-dir`, measured once on `claude 2.1.226`, and nothing since has re-measured it |
  | b | ~~`cli.py` runs on whatever interpreter the hook resolves~~ — **measured and fixed**, see below |
  | c | A deliberately corrupted `analytic_solution` is caught — inject one and confirm the run does not sail past it |
  | d | The `Stop` hook refuses to end the session until it is fixed, then releases after 3 attempts |

  **(b) was a real defect, found by testing the risk rather than only noting it. [C]** A plugin
  hook's `command` is run by whatever `python3` the shell resolves, not by the repo's `.venv`, and
  nothing guarantees that interpreter has numpy — measured, with `PATH=/usr/bin:/bin` on macOS it
  does not, and `cli.py` died on `ModuleNotFoundError: numpy` before running a single check. It
  worked in development only because this machine's `PATH` happens to put a numpy-bearing
  interpreter first, which is exactly the kind of accident that survives testing and fails on
  someone else's machine.

  The fix is not a shebang. The guards **split cleanly by dependency**: expression-not-program, both
  leakage scans, the ledger shape and the operator declaration need nothing but `ast` and `re`; only
  the reference check and the re-audit need numpy. Those AST-only guards are the ones that catch
  both archived incidents. So `verifylib.operator` imports numpy optionally and `gate` imports
  `reference` lazily: on a numpy-less interpreter every AST guard still runs, exits 2 on the Heston
  leak, and the reference check reports `reference check skipped: No module named 'numpy'` rather
  than being silently absent. Verified end to end under `env -i PATH=/usr/bin:/bin`, and pinned by
  `test_ast_only_guards_survive_an_interpreter_without_numpy`.

  What is left for the live run is (a), (c) and (d) — the parts that need a real session.

---

## 16. Risks

| Risk | Mitigation |
|---|---|
| **A hook is not a gate** — the agent stops and escalates, leaving an invalid file, and headless runs have nobody to answer | Every guard also runs out-of-band at certification (§3c). Criterion 13 tests the out-of-band path *with hooks disabled* |
| The reference check false-fails a correct non-smooth solution | Median-gated classification (§4f), measured on all four real cases. Criterion 4 |
| ~~Coverage is lower still once the un-run specs are attempted~~ | **Resolved the other way.** Coverage rose to 16 of 20 PDE and 5 of 5 SDE, because two of the diagnosed blockers (E23, E24) were misdiagnoses rather than real ceilings. The remaining 4 are root-caused individually in §2 |
| **The reference check false-*accuses* a correct spec.** The mirror of the row above, and the one that actually happened: rev 3–5 named `cahn_hilliard_2d` as a spec defect on a residual measured against the wrong operator | The operator resolution consults every source the spec offers before concluding anything (§4d), and the correction is pinned by a regression test. This is the failure mode to keep watching: a false hard-fail looks exactly like a caught bug |
| The reference check false-*passes* because operator and solution are wrong the same way | Ledger entry on the operator declaration (§4g). Partial by construction; stated, not hidden |
| **The `Stop` gate loops and burns the turn budget** — measured, it fired 8 times and the run died at max-turns | The 3-attempt bail-out is mandatory, not defensive. Criterion 15 |
| A user's own problem has no `benchmark/verify.py` behind it | The `Stop` hook is the gate there, shipped with the plugin. Criterion 14 |
| `PreToolUse` on `Edit` sees only the diff, not the file | Use `PreToolUse` only where blocking matters (leakage on `solver.py`/`evaluate.py`, where `Write` carries the whole content); `PostToolUse` + out-of-band elsewhere |
| Source-derived ledger findings false-positive on nondimensionalised solvers | Warn, never gate (§5b) |
| Divergent stencil implementations across plans | One shared `verifylib/operator.py` (§14 C1) |
| `verifylib` becomes a new single point of failure | Deterministic, dependency-light, under test from Phase 0. Better than the status quo of no test at all |
| The citation scan false-positives on legitimate prose | One allowlisted file; patterns are narrow file/function names, not English words |

---

## 17. Provenance and errata

| Element | Source |
|---|---|
| Requirements ledger; conductor halt at `phase: blocked` | **[A]** |
| Harness `deny` rules honoured under skip-permissions | **[A]** |
| File permission matrix; evaluator imports but does not modify siblings | **[A]** |
| Out-of-band independent verifier | **[A]** |
| Anti-plausibility rules; skipped-check and faulty-probe handling | **[A]** §25 |
| Sandboxed execution with enforced import policy | **[A]** |
| `mms_probe.operator_check` as an evaluable operator string | **[A]**, repurposed |
| Source-text evidence rule; prose-as-formula blocklist | **[B]** |
| Payload sanitizer | **[B]** |
| AST oracle-access scan | **[B]**, narrowed |
| Judge-output schema validation; anti-grid-density rule | **[B]** |
| Time-horizon pinning | **[B]** |
| Content-hash manifest (agent files only) | **[B]**, pulled forward |
| Expression-not-program check | **[N]** |
| Reference check — PDE residual + IC + BC | **[N]** |
| Reference check — SDE moment-ODE consistency | **[N]** |
| Validate-or-quote-or-null, with `failed` overriding a quote | **[N]** |
| AST term-splitting; sub-term decomposition for single-term operators | **[N]** |
| Two-signal (concentration + refinement) classification | **[N]** |
| Moment-ODE forward integration for any state dimension | **[N]**, replacing **[N]** differentiation |
| Answer-key / guidance split on the expression check | **[C]** |
| Claim-level numeric binding, matched at printed precision | **[C]** |
| Text citation scan | **[N]** |
| Ledger re-audit, gate/warn split | **[N]**, generalizing **[B]** |
| Two-place enforcement: hook + out-of-band gate | **[N]** |
| Scoped numeric-claim binding | **[N]** |
| Agent-file hashes in `results.json` | **[N]** |

### Errata

| # | Said | Correction | Rev |
|---|---|---|---|
| E1 | Quote-or-null is the highest-consequence, smallest change | Grades origin, not truth; accepts a wrong user-supplied formula. Replaced (§4b) | 1→2 |
| E2 | Graded provenance is the alternative | Worse — a model's self-report about its own recall, produced by the same process as the failure | 1→2 |
| E3 | The AST scan catches the two observed leaks | False for both (§6a) | 1→2 |
| E4 | Run the AST scan in `benchmark/runner.py` | That path is `sandbox=True` only (§6c) | 1→2 |
| E5 | The ledger re-audit probes the run's actual `eps` | Not implementable (§5a) | 1→2 |
| E6 | Any re-audit finding is a hard gate | Output-derived gates; source-derived warns (§5b) | 1→2 |
| E7 | Deterministic gate with reason-fed retry, hard fail after 3 | No Python calls the formulator. Ported to hooks | 1→2 |
| E8 | Numeric binding over the whole review, hard raise | Rejects the feedback text the prompt mandates (§8a) | 1→2 |
| E9 | A fabricated MMS source is symmetric with a fabricated closed form | It is not — a wrong MMS source is detectable (§4g) | 1→2 |
| E10 | Layer 4 contributes new guards | Exactly one: the `solver.py` hash check (§7) | 1→2 |
| E11 | Canaries are Layer 7, therefore cut | Two of nine test *these* layers; retained (§9) | 1→2 |
| **E12** | **`PostToolUse` with a path matcher blocks a bad write** | **Neither half is true.** `matcher` is tool-name only, and `PostToolUse` cannot block. Measured (§3) | 2→3 |
| **E13** | **Hooks make the guards unskippable** | **They do not.** The agent stops and escalates, leaving the bad file. Every guard also runs out-of-band (§3c) | 2→3 |
| **E14** | **Singular sets = "residual grows under refinement"** | **Misclassifies three ways, including waiving a wrong formula.** Replaced; see E19 for the final rule (§4f) | 2→3 |
| **E15** | **17 of 20 PDE Tier-A specs are checkable** | **11 of 20** measured (10 before E19); root causes in §2 | 2→3 |
| **E16** | **SDE coverage is 5/5** | **3 of 5**; two have vector moment ODEs (§4e) | 2→3 |
| **E17** | **The term-balanced denominator is well-defined** | Degenerate for single-term operators; needs sub-term decomposition (§4c) | 2→3 |
| **E18** | The AST scan is usable as specified | Flags **124 of 137** pipeline files, all false positives. Narrowed (§6b) | 1→3 |
| **E22** | **`verification.operator` is plan-no-closed-form's deliverable, so 5 specs are hard-blocked** | Ownership inverted. **This plan defines and requires the field**; the other consumes it for D1 (§4d). The rev-3 ordering meant a user's own source-bearing problem got no reference check until an unrelated plan shipped, which contradicts the requirement that Autonumerics run on any problem | 4→4 |
| **E20** | **Hooks in `benchmark/pipeline-settings.json`** | That confines every guard to benchmark runs. Hooks ship with the **plugin** (`hooks/hooks.json`), measured to load via `--plugin-dir`, with `${CLAUDE_PLUGIN_ROOT}` expanding in the command (§3) | 3→3 |
| **E21** | **The bail-out counter is defensive, since no deadlock was observed** | True of `PostToolUse`; **false of `Stop`**, which looped 8 times and killed the run. The gate that actually gates is the one that needs the counter (§3d) | 3→3 |
| **E23** | **`pde_cahn_hilliard_2d` is a genuine spec defect — the plan's headline finding** | **It is not.** The 0.247 residual is against the *homogeneous* operator; the solution is manufactured and its source is declared in the top-level `source_term`. Bound, the median is **1.2e-08**. The plan would have published a false accusation of a spec defect — precisely the failure §16 calls invisible, because it looks exactly like a caught bug (§2) | 5→6 |
| **E24** | **Poisson / Helmholtz / Monge–Ampère are blocked: the source exists only as prose in `governing_equation`, and `residual_operator`'s `f` is never evaluable** | **Both halves false.** `source_term` is a real top-level field on 5 specs, and 11 specs' `residual_operator` reference it as `f`. Binding it validates all three plus `anisotropic_diffusion`, with no spec change (§4d) | 5→6 |
| **E25** | **SDE coverage is 3 of 5; vector moment shapes need per-shape handling** | **5 of 5, with less code.** Differentiating the claimed moments needs the state recovered from them and fails when a component appears in none (`p12`). Integrating the declared ODE forward needs nothing from them, so state dimension stops mattering (§4e) | 5→6 |
| **E26** | **PDE coverage is 11 of 20** | **16 of 20** — 9 clean, 7 off a singularity, following E23 and E24 (§2) | 5→6 |
| **E27** | **Apply expression-not-program to `drift_expression` and `diffusion_expression`** | Hard-fails 3 of 6 real SDE specs on *correct* content: a vector drift is legitimately prose describing a matrix action. Split into answer-key fields (must parse) and guidance fields (must not be a *program*), which is the sharper form of the same rule and still catches the Heston `def` anywhere (§4a) | 5→6 |
| **E28** | **Test 2 compares the formula at `t = 0` against `initial_condition`** | Two errors. A backward parabolic problem states its condition at maturity — reading `black_scholes_call` at 0 reports 3.5% on a correct spec. And exact agreement is the wrong bar: `advection_1d`'s non-periodic IC differs from its periodic solution by 1.2e-04, real and benign. Two candidate times, two thresholds (§4c) | 5→6 |
| **E29** | **`verifylib/operator.py` is a safe module name** | It shadows the stdlib `operator` whenever `cli.py` runs as a script, breaking `import json` inside the interpreter with a circular-import error that names nothing relevant. The name is kept (§14 C1 freezes it); `cli.py` drops its own directory from `sys.path` first (§12) | 5→6 |
| **E32** | **A plugin hook can rely on numpy being importable** | It cannot: the hook's `command` is run by whatever `python3` the shell resolves, and with `PATH=/usr/bin:/bin` on macOS that interpreter has no numpy — `cli.py` died before running any check. The AST-only guards (which catch both archived incidents) now survive it, and the reference check reports its own absence (§15) | 6→6 |
| **E33** | **`schema` may accept only `mms_probe.operator_check` as the legacy operator fallback** | `reference.resolve_operator` also consults `residual_operator`, so the gate errored on `pde_anisotropic_diffusion` while the check it gates for **validated it at 2.5e-10**. A gate contradicting its own check is invisible from either side alone (§2) | 6→6 |
| **E31** | **The `Stop` gate runs "over the workspace directory"** | Wrong scope. `workspace/` holds every problem from every previous run, so one stale spec blocks every future session and writes a marker the conductor misreads as a block on the current problem. Gate the **active** problem directory (§3) | 5→6 |
| **E30** | **Numeric binding over the Numerical Accuracy section** | Still too broad: flags 28% of 206 archived reviews, all on diagnostic context. Bind the *claim* on each keyed line — its leading token — and match at the precision each side printed to. 0 false positives (§8a) | 5→6 |
| **E19** | **Gate on `max` residual plus a "singular set < 2%" threshold** | Fragile — the three real singular cases sit at 0.8/1.2/1.6% against a 2% bar — and it cannot classify `black_scholes_call` at all. **Gate on the median** (`< 1e-6`, a seven-order gap); concentration demotes to a diagnostic. Coverage 10 → **11 of 20** (§4f) | 3→3 |
