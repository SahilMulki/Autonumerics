# agents

Seven agent definitions, one Markdown file each. An agent here is **a role, a set of file
permissions, and a workflow** — not code. Everything numeric lives in
[`verifylib/`](../verifylib/) under test, and that division is the point: the numerical judgement
stays reviewable as prose, the measurement stays reproducible without a model in the loop, and the
handoffs between the two are enforceable.

None of these files is dispatched by hand. [`commands/conductor.md`](../commands/conductor.md)
routes a problem through them; `docs/architecture.md` draws
[the whole loop](../docs/architecture.md#the-agent-loop) on one page.

## The seven roles

| File | Model | Argument | Writes | Lines |
|---|---|---|---|---|
| [formulator.md](formulator.md) | opus | `workspace/{slug}` | `problem_spec.json` | 624 |
| [plan-creator-pde.md](plan-creator-pde.md) | opus | `workspace/{slug}` | `plans/{id}-{slug}/SOLUTION.md` (new) | 155 |
| [plan-creator-sde.md](plan-creator-sde.md) | opus | `workspace/{slug}` | `plans/{id}-{slug}/SOLUTION.md` (new) | 139 |
| [solver-pde.md](solver-pde.md) | sonnet | a plan directory | `solver.py`, SOLUTION.md **Results** | 178 |
| [solver-sde.md](solver-sde.md) | sonnet | a plan directory | `solver.py`, SOLUTION.md **Results** | 131 |
| [evaluator-pde.md](evaluator-pde.md) | sonnet | a plan directory | SOLUTION.md `<metrics>` + `<review>` | 200 |
| [evaluator-sde.md](evaluator-sde.md) | sonnet | a plan directory | SOLUTION.md `<metrics>` + `<review>` | 168 |

The opus/sonnet split follows where the irreversible judgement sits. The formulator and the
plan-creators decide what the problem *is* and which schemes are worth trying, and a mistake there
is carried silently: `problem_spec.json`, not `problem.md`, is what the pipeline solves and scores
against, so a constraint the formulator fails to carry across does not merely go unrecorded — it
stops existing. The solver and evaluator work inside a cycle that corrects them: the solver's code
either runs or it does not, and the evaluator's numbers come from the kernel rather than from the
model.

**The formulator is 39% of the folder** because it owns the pipeline's ground truth. Its longest
step is the verification plan, and the rule there is that `null` is never a complete answer — a spec
with no closed form must still say how a solution is to be judged, or every plan on that problem
caps at 2.

## Why three roles are split by equation type and one is not

Classification happens *inside* the formulator (Step 1), so there can only be one of it; it is the
agent that decides which branch the rest of the run takes. From there the conductor reads
`equation_type` and dispatches the matching pair for every later step.

The split is not cosmetic — the two halves are graded on different things:

- **PDE** — error in space, so the evaluator reads a nested grid ladder (`N`, `2N−1`, `4N−3`), an
  observed convergence order against a floor, declared invariants, and hard structural gates
  (`div u = 0`, positivity) that fail a solution however accurate it is.
- **SDE** — Monte Carlo error sits *on top of* discretization error, so every comparison is made
  against a confidence interval and every check returns pass / fail / **inconclusive**.
  `resolved: false` scores 6 and the feedback is "raise `num_paths`", never "fix the scheme". An
  evaluator that reported MC-inconclusive as either a pass or a solver bug would be wrong twice.

## What a file contains

Every agent file has the same frontmatter — `name`, `description`, `argument-hint`, `model` — and no
`tools:` field, so each inherits the session's tool set. Confinement comes from the permission rules
in [`benchmark/pipeline-settings.json`](../benchmark/pipeline-settings.json) during a benchmark run,
not from the agent definition.

The body follows one shape: **Setup** (which manuals to read, stated as `${CLAUDE_PLUGIN_ROOT}/…`
paths) → **Workflow** → **Key Rules** → **File Permissions** → **Report Back**. The last section is
load-bearing rather than courtesy: it is what the conductor gets back, and it is the only channel
for a defect an agent found in someone else's file — a faulty MMS probe or an undeclared surrogate is
a *formulator* bug, and the evaluator's job is to say so rather than write solver feedback about a
bug that does not exist.

Which manuals each agent is required to read:

| | `project_manual` | `pde_manual` | `sde_manual` | `verification_manual` |
|---|---|---|---|---|
| formulator | ✓ | ✓ | ✓ | ✓ (authors the `verification` blocks) |
| plan-creator-pde | ✓ | ✓ | | |
| plan-creator-sde | ✓ | | ✓ | |
| solver-pde | ✓ | ✓ | | §7, §8, §26 |
| solver-sde | ✓ | | ✓ | §18, §20 |
| evaluator-pde | ✓ | ✓ | | §1, §25, §26, §27 |
| evaluator-sde | ✓ | | ✓ | Part II: §14, §15–17, §18–19, §20, §21 |

The evaluators read the verification manual to *understand* what each check establishes, so they can
diagnose a failure. They implement none of it. Note that neither plan-creator reads that manual even
though both declare fields the kernel consumes — those fields are documented inline in the
plan-creator files instead.

## The contracts these files define

The most consequential content here is not prose, it is interface. These files are where the
following are written down, and Python elsewhere enforces them:

| Contract | Declared in | Enforced by |
|---|---|---|
| `solve_pde(N, override=None) -> dict` — `numerical_solution` or `fields`, `grid`, `t_final`, optional `invariant_trace` and `snapshots` | solver-pde.md | the kernel's ladder driver; `benchmark/verify.py` re-runs the same signature |
| `solve_sde(num_paths, dt, T, seed, dW, observables) -> dict` — `terminal_paths`, `empirical_mean`, `empirical_variance`, optional `path_integrals` | solver-sde.md | the kernel's CRN order ladder and Dynkin check |
| `scheme_family` — and, on the PDE side, `spatial_order` — in SOLUTION.md frontmatter | plan-creator-pde.md, plan-creator-sde.md | PDE: the kernel differences the field with a stencil of *higher order and a different family* than the plan's, and picks the asymptotic guard (a spectral method's exponential convergence fails the algebraic test). SDE: it selects which of the spec's declared strong orders the plan is held to |
| the `requirements` ledger — one entry per constraint in `problem.md`, verbatim quote, `mapped`/`ambiguous`/`dropped` | formulator.md | `verifylib/schema.py` and `audit.py`; the conductor **halts** on any `dropped` entry |
| `verification.operator` / `verification.moment_ode` | formulator.md | `verifylib/reference.py` (the analytic check) and the D1 residual gate |
| `<metrics>` immediately above `<review score=X>` | evaluator-pde.md, evaluator-sde.md | `verifylib/review.py` rejects a review whose headline score differs from the metrics block |

Two of these exist specifically because a solver can satisfy its interface and still hand back a
measurement of the wrong thing. `override` must flow through the *same* solve path as a normal run —
a solver that accepts `override["params"]` and then reads a module-level constant runs cleanly,
returns `status: ok`, and gives back the unmodified answer. And `dW` must be used exactly as passed:
drawing its own noise breaks the shared Brownian path that makes the strong-order estimate mean
anything, which is what catches a Milstein plan whose correction term is silently wrong.

## The agent does not assign the score

This is the rule that shaped the two evaluator files, and it was a measurement before it was a
policy. The evaluator used to write a fresh `evaluate.py` every cycle — a numerical harness grading
its own solver, with nothing re-checking its arithmetic. On `workspace/pde_kuramoto_sivashinsky`,
three independently regenerated evaluators reported observed orders of **10.467, 10.925 and 0.003**
for the same problem, and all three scored 10.

So the measurement moved to a fixed, version-controlled kernel, and the evaluators are now forbidden
from writing numerical code of any kind — there is deliberately no `evaluate.py` in the permission
table, and a plan directory created after that change must not contain one. The evaluator runs
`verifylib/cli.py evaluate`, transcribes the `<metrics>` block it printed, and spends its actual
effort on the part that needs a language model: turning a metric into *what to change*. "Reduce
`dt`, do not change the stencil" is the diagnosis; `temporal_ok: false` is only the symptom.

Its one authority is **asymmetric**:

```bash
VERIFYLIB_EVAL_CONFIG='{"agent_cap": {"score": 3, "reason": "SOLUTION.md describes ETDRK4; solver.py integrates with forward Euler"}}' \
  "${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py" evaluate <plan_dir>
```

A cap is recorded with its reason and **can only lower a score, never raise one**; a cap above the
computed score is rejected and the rejection is recorded. An agent that cannot inflate cannot
launder a 4 into a 10, and one that can deflate can still stop a wrong 10 from shipping. It exists
for the three things a deterministic check cannot see: prose that describes a different scheme than
the code implements, an answer smuggled in as a table or fitted constant, and a plan that
"converges" by returning its input.

If a *correct* solver scores low, that is a rubric bug. It gets fixed in the kernel with a test,
once, visibly — not re-litigated by an agent every cycle.

## File permissions

No agent may write another's files. `STATE.md` is the durable ledger, so an interrupted run resumes
mid-cycle rather than restarting.

| File | conductor | formulator | plan-creator | solver | evaluator |
|---|---|---|---|---|---|
| `problem.md` | — | R | R | R | R |
| `problem_spec.json` | R *(3 fields)* | **W** | R | R | R |
| `plans/{id}/SOLUTION.md` | R | — | **W** | **W** *(Results)* | **W** *(metrics + review)* |
| `plans/{id}/solver.py` | — | — | — | **W** + run | R *(import only)* |
| `plans/*/solver.py` *(siblings)* | — | — | — | — | R *(import only)* |
| `STATE.md` | **W** | — | — | — | — |
| `REPORT.md` | **W** | — | — | — | — |

Nothing in this pipeline may read anything under `benchmark/` — that is where the ground truth it is
graded against is authored, and an agent that reads the answer key turns an independent check into a
self-check, silently. The conductor re-hashes `solver.py` around every evaluator dispatch; a changed
hash is recorded as a `grading_violation` and the score is discarded, because a self-graded result is
not evidence.

## Editing one of these

- **Check what else has to move.** A contract in the table above has consumers in `verifylib/`,
  `benchmark/verify.py`, or both, and `references/project_manual.md` carries the same handoff table.
  Changing the return dict of `solve_pde` in prose alone produces a solver the kernel cannot grade.
- **The manuals are shared; these files are not.** A rule that applies to every agent belongs in
  [`references/project_manual.md`](../references/project_manual.md); a rule about one role belongs
  here. Six of the seven files cite manual sections by number (all but `plan-creator-sde.md`), so
  renumbering a manual silently invalidates the citations in all of them.
- **Keep the numbers traceable.** The house rule throughout the repo is that a figure names the run
  that produced it. The measurements quoted in these files — the three KS evaluators, the two KS
  solvers that ignored `params["t_final"]`, the `K = 3` snapshot sufficiency, the 0.49/1.01 strong
  orders on GBM — are there to explain why a rule exists, and a rule with no measurement behind it
  reads as a preference and gets ignored.
- **Writes are hooked, not gated.** [`hooks/hooks.json`](../hooks/hooks.json) runs
  `verifylib/cli.py hook` on `Write`/`Edit`/`Bash` and on `Stop`, which gives an agent in-loop
  feedback on a malformed artifact. That is an optimisation: when a hook contradicts an instruction
  the agent stops and escalates, and in a headless run nobody answers. The conductor's out-of-band
  `check-spec` and `gate` calls are what actually gate.
