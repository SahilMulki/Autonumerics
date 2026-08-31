# Hallucination Guardrails — Implementation Plan

**Status:** revised proposal (rev 3, 2026-08-30) — supersedes rev 2, which superseded rev 1
**Scope:** Layers 1–5. Layers 6 and 7 are **deferred** (§9); a small, load-bearing part of 7 is
retained.
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[plan-blind-evaluator.md](plan-blind-evaluator.md) — compatibility analysed in §14.

Provenance labels: **[A]** = already in this repo, **[B]** = adapted from the AutoNumerics
paper-release pipeline, **[N]** = new in this plan, **[C]** = corrected against measurement.

Every quantitative claim below was measured against this repo on 2026-08-30. Where a rev-2 claim did
not survive measurement it is marked **[C]** and the corrected number is given. Errata: §17.

---

## 0. What changed, and why

### Rev 1 → rev 2 (design review)

| Rev 1 claim | Outcome |
|---|---|
| Quote-or-null is the "highest consequence, smallest change" | **Wrong predicate.** It grades the formula's *origin*, not its *truth*: it accepts a wrong user-supplied formula and rejects a correct derived one. 0 of 29 `problem.md` state a closed form while 25 of 28 specs carry one, so it also nulls 25 specs |
| The AST scan catches the two observed leaks | **False for both.** One is transcribed numpy, one is a docstring — and the plan itself specified that comments are not access |
| The ledger re-audit probes the run's actual `eps` | **Not implementable.** A PDE solver returns `{numerical_solution, grid, t_final}`; no channel carries `eps` |

### Rev 2 → rev 3 (measurement)

Rev 2 was prototyped and hook behaviour was tested empirically. Four of its claims did not survive.

| Rev 2 claim | Measured outcome | Fixed in |
|---|---|---|
| `PostToolUse` hooks with a **path matcher** block a bad write | **Both halves false.** `matcher` is tool-name only — a `"**/target.txt"` matcher never fires. And `PostToolUse` exit 2 does **not** block: the file is on disk and is not reverted. Only `PreToolUse` blocks | §3 |
| Singular sets are found by "points where the residual grows under refinement" | **Misclassifies in three directions**, including waiving a wrong formula as "singular" — the exact failure it exists to catch | §4f |
| The reference check runs on 17 of 20 PDE Tier-A specs | **10 of 20** with a real implementation. An operator *string* existing is not the check *working* | §2, §4d |
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
| Specs carrying `verification.operator` | 0 — the field does not exist yet |
| PDE specs carrying `mms_probe.operator_check` | 19 of 22 |
| Operator strings that AST-split into signed top-level terms | **19 of 19** |
| **PDE Tier-A claims the reference check validates** | **10 of 20** — 7 clean, 3 off a localized singularity |
| **SDE Tier-A claims the reference check validates** | **3 of 5** — the other 2 have vector moment ODEs |
| Pipeline files flagged by the rev-1 AST leakage rule | **124 of 137 (91%)**, all false positives |
| Pipeline files flagged by the narrowed rule | **0 of 137**, with the genuine patterns still caught |

### Why 10 of 20, and what blocks the other 10 **[C]**

Rev 2 counted specs that *have* an operator string. Running the check is different. Root causes,
because they determine which are fixable and by whom:

| Root cause | Specs | Owner |
|---|---|---|
| **Validated** | `heat_1d`, `heat_2d`, `laplace_2d`, `wave_1d`, `wave_2d`, `advection_1d`, `fokker_planck_ou` | — |
| **Validated off a localized singularity** | `burgers_inviscid` (shock, 1.2% of domain), `stefan_1d_similarity` (kink, 1.6%), `convection_diffusion_bl` (layer, 0.8%) | — |
| **Source term missing from any evaluable field** | `poisson_2d`, `helmholtz_2d`, `monge_ampere_2d` | **plan-no-closed-form** (§14 C2) |
| **Multi-field system; needs the systems operator form** | `mhd_2d`, `navier_stokes_2d` | **plan-no-closed-form** (§14 C2) |
| **No operator string in the spec** | `anisotropic_diffusion`, `porous_medium_2d` | formulator |
| **Non-local operator — genuinely unavailable** | `fractional_diffusion` (Caputo derivative) | — |
| **Spec is internally inconsistent — a real finding** | `cahn_hilliard_2d` | see below |
| Already `null` | `heston_2d`, `kuramoto_sivashinsky` | — |

`pde_cahn_hilliard_2d` deserves its own line. Its claimed exact solution
`0.2*sin(2πx)*sin(2πy)*cos(t)` gives a median term-balanced residual of **0.247, flat across
N = 129, 257 and 513**. It does not converge, so it is not under-resolution — that profile is not a
solution of the declared Cahn–Hilliard operator. Either the spec's `analytic_solution` is wrong or
it is a manufactured profile whose source term was never recorded. **This is the first real defect
the guardrail found, and it was found before the guardrail was built.**

---

## 3. The enforcement model **[N, C]**

*Cross-cutting; every deterministic check depends on it, so it comes first.*

Rev 2 specified `PostToolUse` hooks with a path matcher, blocking on exit 2. Six controlled runs
against `claude 2.1.226`:

| Test | Result |
|---|---|
| `"matcher": "**/target.txt"` alongside `"matcher": "Write"` | **Only `Write` fired.** `matcher` is tool-name only; there is no path matcher |
| `PostToolUse` exit 2 on a `Write` | stderr reached the agent; **the file was on disk and not reverted** |
| `PreToolUse` exit 2 on a `Write` | **The file was never created.** Payload has `tool_input` but no `tool_response` |
| Reason-fed retry (satisfiable constraint) | `REJECT` → 3.2 s → `PASS`. The agent read the stderr and corrected |
| Hooks inside a Task-dispatched **subagent** | **Fire normally**, same REJECT→PASS cycle |
| `Bash` heredoc (`cat > target.json`) against a `Write\|Edit` matcher | **No hook fired.** The file landed unvalidated |
| Hook contradicting an explicit user instruction | The agent **stopped, left the bad file on disk, and escalated to the user** |

Three consequences, and the last one is the important one.

**(a) Path filtering happens inside the hook.** The payload carries `tool_input.file_path`, so this
costs three lines. But the matcher must be `Write|Edit|Bash` or a heredoc walks straight past it,
and the `Bash` branch has to inspect `tool_input.command` for a redirect target.

**(b) Blocking requires `PreToolUse`, which sees the wrong thing.** `PreToolUse` gets `tool_input`,
not the resulting file. For `Write` that is the whole content and is fine; for `Edit` it is
`old_string`/`new_string`, so a validator that must parse the finished `problem_spec.json` would
have to apply the edit itself to reconstruct it. Pay that cost only where blocking genuinely
matters.

**(c) A hook is feedback, not a gate.** The observed failure is not the deadlock rev 2 planned for.
It is the opposite: the agent stops, leaves the invalid file in place, and asks the user. In an
interactive session that is correct behaviour. In a headless `claude -p` benchmark run **nobody
answers** — the subagent returns, the file is invalid, and the conductor never learns. Therefore:

> **Every guard runs in two places: as a hook, for fast in-loop correction; and out-of-band at
> certification, where no agent's cooperation is required. The out-of-band call is the gate. The
> hook is an optimisation.**

### Assignment

| Guard | Hook | Authoritative gate |
|---|---|---|
| Spec schema + reference check (§4) | `PostToolUse` on `Write\|Edit\|Bash` → `problem_spec.json` | Conductor, before dispatching plan-creator — same halt path as the ledger |
| Leakage scan (§6) | `PreToolUse` on `Write\|Edit` → `evaluate.py`, `solver.py` (blocks; `tool_input.content` is the whole file) | `verify.import_solver` and `runner._child_main` |
| Ledger re-audit (§5) | — needs a completed run | `benchmark/verify.py` at certification |
| Numeric-claim binding (§8) | `PostToolUse` on `SOLUTION.md` | Conductor, when reading the score |

Keep the 3-attempt counter and the `GUARDRAIL_UNSATISFIED` → `phase: blocked` path. Its purpose is
no longer deadlock prevention (not observed) but **turning a silent stop into a recorded one**: when
an agent gives up against a hook, the marker is what tells the conductor the file is untrusted.

Hooks live in `benchmark/pipeline-settings.json`, passed per-invocation via `--settings`
([run.py:69](../benchmark/run.py#L69)), never in `.claude/settings.json` — otherwise they fire in
ordinary development sessions.

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
evaluate.py". This rejects it instantly. One line, zero judgment, no false negatives. Apply to
`analytic_moments.*_expression`, `drift_expression`, `diffusion_expression` and every
`verification.*` expression field.

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
| Formula at `t = 0` matches `initial_condition` | Right PDE, **wrong solution branch or mode** |
| Formula on the boundary matches `boundary_conditions.values` | Right PDE and IC, wrong BC selection |

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

### 4d. Where the operator comes from **[C]**

The check reads `verification.operator` when present and falls back to
`mms_probe.operator_check`, which **19 of 22 PDE specs already carry** in exactly the required form
(`"u_t - alpha*lap_u"`, `"-lap_u - k**2*u"`, `"u_tt - c**2*lap_u"`). That fallback is what lets
Layer 1 ship without waiting for [plan-no-closed-form.md](plan-no-closed-form.md).

But the fallback has a hard ceiling, and rev 2 missed it. `operator_check` is the **homogeneous
operator**; it has nowhere to record a source term. `pde_poisson_2d`'s source
(`f = 2*pi^2*sin(pi*x)*sin(pi*y)`) exists **only as prose inside `governing_equation`**. The
`verification.residual_operator` field references a bare `f` that is never defined anywhere
evaluable. So Poisson, Helmholtz and Monge–Ampère cannot be checked on the fallback path at all —
they need `verification.operator.source`, which is plan-no-closed-form's §3 deliverable. Likewise
the two multi-field systems need its per-equation form. **That is a hard dependency for 5 of the 20
specs, and it is documented in §14 as C2.**

Helper namespace required by the real strings: `u`, `u_t`, `u_tt`, `grad_u` (and `grad_u[i]`),
`lap_u`, `lap_lap_u`, `lap(·)` as a callable, `u_xx`/`u_yy`/`u_xy` and the non-Cartesian analogues
(`u_SS`, `u_Sv`, `u_vv`), plus `adv_u`, `grad_p`, `lorentz_u` for the systems. Two vocabularies for
the same thing already coexist — `lap_lap_u` in Cahn–Hilliard, `lap(lap_u)` in Kuramoto–Sivashinsky
— so supply both. `pde_monge_ampere_2d` needs the mixed derivative `u_xy`, which
plan-no-closed-form's helper list omits; that is §14 C1.

### 4e. The reference check — SDE **[N, C]**

`analytic_moments.mean_expression` and `variance_expression` are functions of `t`. Differentiate
them numerically in `t` and compare against the moment ODE the spec already declares:

```
d/dt m1_claimed(t)  ==  rhs[0](m1_claimed(t), m2_claimed(t))
d/dt m2_claimed(t)  ==  rhs[1](m1_claimed(t), m2_claimed(t))
```

plus `m1_claimed(0) == initial[0]` and `m2_claimed(0) == initial[1]`. Run it only where
`closes_exactly` is `true`; a closure approximation is not a reference.

Measured over 12 time points on the real specs:

| Spec | max relative mismatch | Verdict |
|---|---|---|
| `sde_bm_with_drift` | 2.35e-11 | consistent |
| `sde_cir_feller_violated` | 7.47e-11 | consistent |
| `sde_ornstein_uhlenbeck` | 1.55e-10 | consistent |
| `sde_ornstein_uhlenbeck`, decay rate halved | **5.10e-01** | **caught** |
| `sde_gbm_2d_high_corr` | — | vector state: 5 coupled moments |
| `sde_multichannel_stiff_m13` | — | vector state: 5 coupled moments |

**Coverage is 3 of 5, not 5 of 5. [C]** The two remaining specs have vector-valued moment ODEs —
`['m1x','m2x','m1y','m2y','mxy']` and `['m1','m2','p11','p12','p22']` — where the claimed moments
are lists and the ODE carries cross-moments. Supporting them is a shape-handling problem, not a new
idea, but it is real work and must be scoped as such.

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

**Two signals are needed, and neither works alone.**

- **Spatial concentration** separates a localized singularity from a global error. A wrong formula
  is wrong everywhere; a shock is wrong on O(1) points.
- **Refinement** separates under-resolution from a wrong formula. An under-resolved smooth feature
  converges; a wrong formula does not.

```
bad(N)  = fraction of interior points with term-balanced residual > tol
```

| `max` residual | `bad(N)` falls toward 0 under refinement? | `bad` concentrated (< 2%)? | Verdict |
|---|---|---|---|
| < tol | — | — | `validated` |
| ≥ tol | **yes** | — | `validated` — re-probe at the finer grid |
| ≥ tol | no | **yes** | `validated_off_singularity` — gap recorded |
| ≥ tol | no | no (global) | **`failed`** |

Measured, this classifies every case correctly:

| Case | max | `bad` at N=129 / 257 / 513 | Verdict |
|---|---|---|---|
| `heat_1d` correct | 3.1e-10 | 0% | `validated` |
| `heat_1d`, `pi**2` dropped | 8.99e-01 | 100% / 100% / 100% | **`failed`** |
| `heat_1d`, mode doubled | 1.01e+00 | 100% | **`failed`** |
| `heat_2d`, `laplace_2d`, `wave_1d` correct | ≤ 4.9e-09 | 0% | `validated` |
| `advection_1d` correct (narrow Gaussian) | 4.5e-01 | 58% → 33% → **1%** | `validated`, under-resolved at coarse N |
| `fokker_planck_ou` correct | 1.1e-04 | 67% → 45% → **1%** | `validated`, under-resolved at coarse N |
| `burgers_inviscid` correct (shock) | 1.00e+00 | **1.2%**, flat | `validated_off_singularity` |
| `stefan_1d_similarity` correct (kink) | 1.28e+00 | **1.6%**, flat | `validated_off_singularity` |
| `convection_diffusion_bl` correct (layer) | 5.7e-03 | **0.8%**, flat | `validated_off_singularity` |
| `cahn_hilliard_2d` as specified | 1.0e+00 | 100% / 100% / 100% | **`failed`** — see §2 |

Two things fall out that are better than rev 2 claimed. The non-smooth cases are not merely waived:
**trimming the worst 2% of points drops their residual to 0.0, 2.3e-10 and 9.4e-10 respectively**,
so they are genuinely *validated on the smooth complement* — a real Tier A with a recorded gap,
rather than an unchecked one. And an under-resolved probe grid self-corrects by refining, instead of
producing a false failure.

`np.where` / `np.maximum` / `np.minimum` / `np.sign` / `np.heaviside` in the expression is a
syntactic tell for a piecewise solution, detectable by AST. Use it to *predict* where a singular set
will appear, never to excuse one — a formulator-set waiver flag would reintroduce exactly the
self-report §4b exists to eliminate.

### 4g. Outcomes and honest limits

| Outcome | Meaning | Consequence |
|---|---|---|
| `validated` | Residual, IC and BC all pass | Tier A |
| `validated_off_singularity` | Passes on the smooth complement; singular set < 2% | Tier A, gap recorded in `<metrics>` and `REPORT.md` |
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

**Be honest about what it establishes.** The same LLM writes `<metrics>` and `<review>`, so binding
one to the other catches *transcription drift* — the most common LLM-grader failure, worth catching
— but not fabrication. That property arrives with the kernel (§14 C6).

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

*Not hypothetical:* `pde_cahn_hilliard_2d`, median residual 0.247 flat across three grids.

| Layer | Outcome |
|---|---|
| **1** §4c/§4f | **Caught, `failed`.** Global (100% of domain), non-converging — the two signals that distinguish a wrong formula from a singularity or under-resolution |
| Today | Invisible. The evaluator grades against it and every plan agrees |

---

## 11. Implementation — file by file

| File | Change | Layer |
|---|---|---|
| `verifylib/__init__.py` **(new)** | Package marker. Deterministic guards only — **not** the numerical kernel | — |
| `verifylib/operator.py` **(new)** | Helper namespace + stencils + AST term-splitting + restricted eval. **Shared with plan-no-closed-form's D1** — §14 C1 | 1 |
| `verifylib/schema.py` **(new)** | Expression-not-program; evidence rule over `analytic_solution` + `verification`; ledger shape | 1, 2 |
| `verifylib/reference.py` **(new)** | The reference check: PDE residual + IC + BC; SDE moment-ODE consistency; two-signal classification; five outcomes | 1 |
| `verifylib/audit.py` **(new)** | Ledger re-audit — output-derived gate, source-derived warn | 2 |
| `verifylib/leakage.py` **(new)** | Citation scan + **narrowed** AST scan | 3 |
| `verifylib/review.py` **(new)** | Scoped numeric-claim binding, warn-once | 5 |
| `verifylib/hooks/` **(new)** | `PreToolUse`/`PostToolUse` entry points; in-hook path filtering; `Bash` redirect parsing; 3-attempt marker | 3 |
| `verifylib/tests/` **(new)** | Unit tests per module + the two retained canaries | 12 |
| `benchmark/pipeline-settings.json` | Add `hooks`; close the `Bash` read gaps in `deny` | 3 |
| `benchmark/runner.py` | Leakage scan in `_child_main` before executing | 3 |
| `benchmark/verify.py` | Leakage scan in `import_solver`; `verifylib.audit` at certification | 2, 3 |
| `benchmark/run.py` | Agent-file hashes per run in `results.json` (§13) | — |
| `agents/formulator.md` | The §4b rule; the five outcomes; `analytic_solution_source_text` on the quote route; require an operator string on every PDE spec | 1 |
| `agents/evaluator-{pde,sde}.md` | Anti-plausibility rules from **[B]**; `resolution_evidence`; numeric-binding contract | 5 |
| `commands/conductor.md` | Out-of-band gate calls; `GUARDRAIL_UNSATISFIED` → `phase: blocked`; `solver.py` hash check | 1, 3, 4 |
| `references/project_manual.md` | The §4b rule, the five outcomes, the gate/warn split | 1, 2 |
| `pyproject.toml` | `pytest` in the `dev` extra | 12 |

**Deliberately not here:** `verification.operator`, the D1 gate, the ladder, Richardson, invariants,
the SDE numerical surrogates — all plan-no-closed-form. `verifylib/reference.py` shares that plan's
helper vocabulary but is far smaller: it differentiates a *formula*, not a solver output, so it needs
no discretization-family selection, no two-snapshot `dt(u)`, and no `override` hook.

---

## 12. Phasing

| Phase | Deliverable | Why here |
|---|---|---|
| **0** | `verifylib/` skeleton, `pytest`, tests dir with the two retained canaries | Rev 1's own rule: build the harness first. Cutting Layer 7 removed it |
| **1** | `verifylib/schema.py` — expression-not-program first | One line, zero judgment, catches the observed Heston leak |
| **2** | `verifylib/leakage.py` — citation + narrowed AST; wire into `verify.import_solver` and `runner._child_main`; close the `deny` gaps | Catches both incidents. No prompt changes, safe to land any time |
| **3** | Hook wiring: `PreToolUse` for leakage, `PostToolUse` for schema/review, `Bash` in the matcher, 3-attempt marker — **plus the out-of-band gate calls in the conductor** | Both halves, per §3(c). The hook alone is not a gate |
| **4** | `verifylib/operator.py` + `reference.py` + the §4b rule in `formulator.md` | Highest consequence; the only prompt-touching change. Gate on §13 |
| **5** | `verifylib/audit.py` | Closes the open loop in the strongest existing guardrail, honestly scoped |
| **6** | `verifylib/review.py` | Cheapest, weakest, depends on nothing |

Phases 1–3 touch no agent prompt. **Phase 4 is the cut line** — §13.

---

## 13. Run provenance and the sequencing constraint **[N]**

`agents/formulator.md` is a pipeline input. Editing it changes behaviour, 28 problems are already
scored in `benchmark/results/REPORT.md`, and an Opus campaign is queued per
`BASELINE_COMPARISON_PLAN.md`. **Results either side of a prompt edit are not comparable**, and
nothing records which prompt version produced which number.

1. **Record agent-file hashes in `results.json` per run** — SHA-256 of every file under `agents/`,
   `commands/`, `references/`, plus the `verifylib` version. This is the one piece of Layer 6 worth
   pulling forward, and it is what plan-blind-evaluator's offline experiments need to be
   reproducible over "frozen artifacts" (§14 C8).
2. **Land Phase 4 before the campaign, or after it — never during.** Phases 0–3 and 5–6 are
   prompt-neutral.

---

## 14. Compatibility with the other two plans

Both related plans are in flux, so this section lists **interfaces to hold fixed** and **conflicts to
resolve**, not a merge order.

| # | Conflict | Between | Resolution |
|---|---|---|---|
| **C1** | **Two implementations of the same stencils.** This plan's `reference.py` and plan-no-closed-form's `operator.py` both discretize operator expressions. Two implementations of `lap_u` can silently disagree — the reference check passes with one stencil set while D1 fails with the other | guardrails ↔ no-closed-form | **One shared `verifylib/operator.py`**, owned by whichever lands first (this plan, Phase 4) and consumed by D1. It must supply `u_xy` (needed by `monge_ampere_2d`) and both `lap_lap_u` and `lap(·)` spellings — plan-no-closed-form's helper list currently omits all three |
| **C2** | **The reference check is blocked on 5 of 20 specs** until `verification.operator` exists: Poisson / Helmholtz / Monge–Ampère need its `source` field (the source is prose in `governing_equation` today), and MHD / Navier–Stokes need its per-equation systems form | guardrails ← no-closed-form | Hard dependency. Until then those specs return `unavailable` and fall back to the quote route (which they will fail, since no `problem.md` quotes a solution) → `null` → they land on the Tier B/C ladder. **Acceptable, but it must be a deliberate choice, not a surprise** |
| **C3** | **Metrics shape change.** plan-blind-evaluator §7 changes metrics values from scalars to `{"value": x, "oracle_derived": bool}`. This plan's numeric binding parses `<metrics>` | guardrails ↔ blind-evaluator | `verifylib/review.py` must accept both shapes from day one. Cheap if written that way; a rewrite if not |
| **C4** | **Provenance vocabulary is not totally ordered once Tier A splits.** plan-blind-evaluator ranks on `analytic > surrogate > manufactured > manufactured_partial > self_convergence > none`. This plan introduces `validated` / `validated_off_singularity` / `quoted` / `unavailable` / `failed`. An **unvalidated quote would outrank a validated surrogate**, which is wrong | guardrails ↔ blind-evaluator | Add `analytic_unvalidated` to the tier list immediately below `analytic` and above `surrogate`, and map `quoted` → `analytic_unvalidated`. Keeps one total order, which the ranker needs |
| **C5** | **Two leakage mechanisms for the same concern.** plan-blind-evaluator §7 wants a leakage scan over the *evidence* fed to a blind ranker, with `oracle_derived` flags plus a `PROHIBITED_TOKENS` fallback. This plan owns the scanner | guardrails ↔ blind-evaluator | One implementation in `verifylib/leakage.py`, two call sites. `oracle_derived` is the structured version of the token blocklist — keep both, they fail differently, but do not write the scanner twice |
| **C6** | **`evaluate.py` shrinks to a `verifylib.run` call** under plan-no-closed-form §8b | guardrails ← no-closed-form | *Synergy, not conflict.* The citation-scan surface shrinks (the spec and `SOLUTION.md` remain), and Layer 5's binding upgrades from LLM-written metrics to **kernel-computed** metrics — which is when it starts catching fabrication rather than only transcription drift (§8a) |
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
   `unavailable` / `failed` on every non-null closed form, and reproduces the §2 distribution:
   **10 validated, 5 blocked on C2, 2 needing an operator string, 1 non-local, 1 failed.**
3. Corrupting a closed form — dropping `pi**2`, flipping a sign, doubling the mode — yields `failed`
   with a term-balanced residual **at least 8 orders** above the correct formula's.
4. `burgers_inviscid`, `stefan_1d_similarity` and `convection_diffusion_bl` report
   `validated_off_singularity` via the **two-signal** rule, with singular sets ≤ 2% of the domain —
   never via a formulator-set flag.
5. `advection_1d` and `fokker_planck_ou`, under-resolved at N = 129, **validate at N = 513** rather
   than failing.
6. `cahn_hilliard_2d` reports `failed`, and the report names it as a spec defect rather than a
   solver defect.
7. `sde_ornstein_uhlenbeck` with a halved decay rate is caught; the three scalar SDE specs are
   consistent to < 1e-9. Vector moment shapes are either supported or explicitly `unavailable`.
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
    and the conductor halts at `phase: blocked` — **and the out-of-band gate catches the same file
    independently, with hooks disabled.**
14. `results.json` records agent-file hashes for every run.
15. `uv run pytest verifylib` is part of the standard workflow and passes.

---

## 16. Risks

| Risk | Mitigation |
|---|---|
| **A hook is not a gate** — the agent stops and escalates, leaving an invalid file, and headless runs have nobody to answer | Every guard also runs out-of-band at certification (§3c). Criterion 13 tests the out-of-band path *with hooks disabled* |
| The reference check false-fails a correct non-smooth solution | Two-signal classification (§4f), measured on all four real cases. Criterion 4 |
| The reference check false-*passes* because operator and solution are wrong the same way | Ledger entry on the operator declaration (§4g). Partial by construction; stated, not hidden |
| Coverage is lower than hoped — 10 of 20 today | Root-caused in §2. 5 of the 10 unblock with plan-no-closed-form (C2); the rest are named individually |
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
| **E14** | **Singular sets = "residual grows under refinement"** | **Misclassifies three ways, including waiving a wrong formula.** Replaced by the two-signal rule (§4f) | 2→3 |
| **E15** | **17 of 20 PDE Tier-A specs are checkable** | **10 of 20** measured; root causes in §2 | 2→3 |
| **E16** | **SDE coverage is 5/5** | **3 of 5**; two have vector moment ODEs (§4e) | 2→3 |
| **E17** | **The term-balanced denominator is well-defined** | Degenerate for single-term operators; needs sub-term decomposition (§4c) | 2→3 |
| **E18** | The AST scan is usable as specified | Flags **124 of 137** pipeline files, all false positives. Narrowed (§6b) | 1→3 |
