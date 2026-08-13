# AutoNumerics Baseline Comparison — Plan

The plan for proving that AutoNumerics' **structure** (not just its model) drives
performance, plus the two fairness upgrades that came with folding in hard PDEs
(leakage removal, PDE convergence-order scoring) and the roadmap for the
**systems / 3D** extension.

Status legend: **[built]** implemented + smoke-tested · **[pending]** designed, not
yet run · **[roadmap]** scoped, not yet built.

---

## 1. Objective

Show that the agentic abstraction — formulator → planner → parallel
solver↔evaluator cycles — beats weaker structures **on the same model**, so any
gap is attributable to structure, not to using a bigger model. Keep the benchmark
**unbiased**: identical inputs, identical grading, honest cost accounting.

## 2. The comparison ladder

| Rung | What it is | Isolates |
|---|---|---|
| **C0** naked one-shot | a single tool-less completion; given only `problem.md`, returns `solver.py` | "vs pasting it into a chat" |
| **C1** single agent + tools | one agent may write, run, and iterate on `solver.py` (Bash/Read/Write) up to a turn budget, but with **no** decomposition — no planner, no separate evaluator, no parallel plans | the value of the multi-agent structure *beyond* tool access + iteration |
| **C2** the pipeline | the full conductor pipeline (`run.py`) | — |

Each baseline is also run **naked** (problem.md only) and **manual** (+ the same
`sde_manual.md` / `pde_manual.md` the pipeline's agents read), to separate the
structure's contribution from the reference material's.

## 3. Model matrix & the two headline comparisons

- **Main:** C2a (pipeline, Opus+Sonnet — the product as-is) vs **C0@Opus** vs **C1@Opus**.
- **Cross-tier:** C2b (pipeline pinned to Sonnet, `run.py --model sonnet`) vs the
  *same* C0@Opus / C1@Opus. If C2b(Sonnet) beats C0/C1@Opus, that is "structure
  beats a bigger model" in one line.

C0@Opus and C1@Opus are computed **once** and feed both comparisons. The only
genuinely new expensive item over a normal pipeline run is the one C2b sweep.

## 4. Repetitions & the discriminator subset

LLM output is stochastic, so pass-rate is estimated over repetitions (pass@1 = mean
over reps). Reps are placed where they are cheap and where the thesis lives:

- **Full suite:** C0 at **R=5** (cheap — a single completion), C1 and C2 at **R=1**.
- **Discriminator subset:** C0/C1 at **R=5**, C2 at **R=3** (each C2 rep is a full
  pipeline run). Equal footing where it counts.

The **discriminator subset** = `problems.discriminator_problems()` — currently the 7
hard-literature SDEs. Flag incoming hard PDEs with `"discriminator": True` to fold
them in (one line per problem; the subset lives in exactly one place).

## 5. What is built

**[built]** the whole apparatus is driven off `problems.py` / `manifest.json`, so
not-yet-final problems flow in automatically.

| File | Role |
|---|---|
| `runner.py` | sandboxed subprocess solver execution: wall-clock timeout (the hard guard), best-effort mem cap, and a **type-aware import policy** matching the pipeline (SDE: numpy+stdlib; PDE: +scipy). Structured crash reasons; non-finite output passes through (a stability blow-up signal). |
| `verify.py` | `sandbox=` seam so the sandboxed baseline path and inline pipeline path share **byte-identical scoring**; SDE (exact/reference/stability) + PDE (single-grid **and** 2-grid order check). |
| `oneshot.py` | the C0/C1 driver: prompt build, `claude -p` invocation, code extraction, sandbox grading, cost/token capture, crash-safe `oneshot_results.json`. |
| `report.py` | `compute_oneshot_verdict` (PASS/FAIL/CRASHED/STABLE_PASS/BLOWUP/NO_GT — a crash is a **FAIL**, never UNVERIFIED). |
| `compare.py` | head-to-head `COMPARISON.md`: pass-rate by type/tier/discriminators, accuracy-vs-cost, per-problem tables; robust to partial data. |

## 6. Fairness mechanics

- **Identical input:** both sides read the exact `problem.md` (contract footer and all).
- **Identical grading:** the same `verify.py` + `problems.py` ground truth at the same
  fixed configuration. "Pass" = an *independently confirmed* pass on both sides
  (pipeline: VERIFIED_PASS/STABLE_PASS; baseline: PASS/STABLE_PASS).
- **Import parity (sandbox):** the pipeline's own rule is asymmetric — SDE solvers are
  numpy+stdlib (`solver-sde.md`), PDE solvers get `scipy.sparse` (`solver-pde.md`, 38/55
  pipeline PDE solvers use it). The sandbox mirrors this per type, so neither side has
  a library the other lacks. A blocked import is a FAIL.
- **Honest cost:** every baseline unit records tokens / cost / turns / wall time;
  `compare.py` shows accuracy *beside* cost so the pipeline's higher accuracy is always
  read against its higher cost.
- **[built] Leakage removed:** no PDE `problem.md` states its solution anymore — MMS
  problems ship the explicit **source term** instead (still fully specified), so a
  solver cannot pass by transcribing a printed formula. (The hard SDEs never leaked —
  reference/stability ground truth has no elementary closed form.)
- **[built] PDE convergence-order scoring:** see §7 — a solver must actually *converge*,
  which is the second, independent defense against single-grid hard-coding and the real
  discriminator HardNumerics was designed around.

## 7. The PDE contract change: `solve_pde(N)` + 2-grid order check  **[built]**

To measure convergence *order*, the harness must control resolution, so the PDE
contract changed from `solve_pde()` to **`solve_pde(N)`** everywhere (per the "change
it everywhere" decision):

- The harness runs the solver at `N` and `2N`, computes relative L2 error at each, and
  the observed order `p = log2(err_N / err_2N)`.
- **Pass = converged (err at 2N < 1%) AND order ≥ floor** (`min_order`, default 1.0;
  set higher per problem for singularity problems). A scheme accurate on one grid but
  not converging **fails** — single-grid L2 alone would have passed it.
- **Super-convergence guard:** if the fine-grid error is already at the noise floor
  (< 1e-9), the order is treated as satisfied (a spectral method hitting machine
  precision should pass).
- **Backward compatible:** a legacy no-argument `solve_pde()` is detected by signature
  and verified single-grid, so existing `results.json` and workspace solvers still work.
- **Shock waiver:** `pde_order_check=False` (e.g. `burgers_inviscid`) scores single-grid
  only, because the L2 order at a discontinuity is inherently fractional.

Propagated through the pipeline so the pipeline is graded fairly (its evaluator now
rewards convergence, matching what `problem.md` tells the baseline): `solver-pde.md`,
`evaluator-pde.md`, `plan-creator-pde.md`, `formulator.md`, `pde_manual.md`, and the
PDE `problem_spec` template. Per-problem knobs: `grid_N` (base N), `min_order`,
`pde_order_check`, surfaced in `manifest.json`.

> **Validation to do before the real run:** confirm a *correct* reference scheme clears
> the order floor at each PDE's `grid_N` (analogous to the SDE "solvable at verifier dt"
> guard) so thresholds don't false-fail a good solver. Defaults are deliberately lenient
> (floor 1.0); tighten per problem (e.g. imported problems can use HardNumerics'
> published `minimum_spatial_order`).

## 8. How to run it

This is a multi-hour, real-money run. Read §8.1 (survival) and §8.4 (what each
command does) before starting — two mistakes here (clobbering C2a, double-paying
for overlapping units) are easy to make and expensive.

### 8.1 Before you start — keep the run alive

Each command below is a plain OS process in *your* terminal; it does **not** depend
on any Claude Code / editor session, so closing those is harmless. What kills a long
run is the terminal or the machine:

- **Terminal closing / SSH drop / logout** → SIGHUP kills the process. Run inside
  **`tmux`** (or `screen`, or `nohup … &`) so it survives a detached terminal.
- **Sleep** suspends the process; an in-flight API call then drops on wake, failing
  the single unit in progress (everything finished is already saved; `--resume`
  redoes just that one). `caffeinate -is` prevents *idle* sleep.
- **Closing the laptop lid** puts the Mac to sleep even under `caffeinate` — clamshell
  sleep is separate from idle sleep. To run unattended either **keep the lid open**
  (with `caffeinate`), use **clamshell mode** (lid shut but on AC power + an external
  display/keyboard), or run on an **always-on machine**.

Recommended launch wrapper (inside a `tmux` session, lid open):

```bash
caffeinate -is uv run python benchmark/oneshot.py … --resume \
    2>&1 | tee -a benchmark/results/logs/<name>.log
```

If anything dies, re-run the **identical** command — `--resume` skips every unit
already recorded and continues. Usage limits stop the driver cleanly on their own;
after the reset (~5 h), re-run with `--resume`.

### 8.2 The commands (run in order)

Run ALL of these inside a **`tmux` session with the laptop lid OPEN** (§8.1) so a
detached terminal or sleep can't kill a multi-hour run:

```bash
tmux new -s baseline        # detach: Ctrl-b d   ·   reattach: tmux attach -t baseline
```

Every command below is wrapped identically: **`caffeinate -is`** (no idle/system
sleep — does NOT cover lid-close, hence "lid open") **+ `tee -a`** (a durable,
appendable log under `benchmark/results/logs/`, alongside the console output). If a
command dies, re-run the **identical** line — `--resume` continues and `tee -a`
appends to the same log.

```bash
# 0. ALWAYS smoke-test the plumbing cheaply first (haiku, 2 problems, ~1-2 min)
caffeinate -is uv run python benchmark/oneshot.py --only sde_gbm,pde_heat_1d --condition c0 --model haiku \
    2>&1 | tee -a benchmark/results/logs/smoke.log

# 1. full-suite baselines @ Opus.  C0 at R=5 (cheap: one completion each),
#    C1 at R=1 (each is an agent loop).  This also covers the discriminator
#    problems at c0/R5 and c1/R1, so run it FIRST.  Let C0 finish before C1.
caffeinate -is uv run python benchmark/oneshot.py --condition c0 --manuals both --model opus --reps 5 --resume \
    2>&1 | tee -a benchmark/results/logs/c0_full.log
caffeinate -is uv run python benchmark/oneshot.py --condition c1 --manuals both --model opus --reps 1 --resume \
    2>&1 | tee -a benchmark/results/logs/c1_full.log

# 1b. discriminator subset @ Opus, R=5 on BOTH conditions.  With --resume this
#     only adds the still-missing units (c1 reps 2-5 for the discriminators);
#     the c0/R5 and c1/rep1 discriminator units are already done from step 1.
caffeinate -is uv run python benchmark/oneshot.py --discriminators --condition c0,c1 --manuals both --model opus --reps 5 --resume \
    2>&1 | tee -a benchmark/results/logs/disc_c0c1.log

# 2. the pipeline runs.
#    2a. C2a — the product as-is (Opus+Sonnet), the pipeline baseline.  Run FRESH
#        over all 50 problems: the checked-in results/results.json is stale (37
#        problems, pre-HardNumerics, scored under an older verify.py + older agents),
#        and C2a must share ONE verify.py + agent version with C0/C1/C2b or the
#        comparison is not apples-to-apples (§6).  Move the stale file aside first so
#        --resume (for continuation, below) can't mistake old records for fresh ones.
mv benchmark/results/results.json benchmark/results/results.stale-jul16.json
caffeinate -is uv run python benchmark/run.py --fresh \
    2>&1 | tee -a benchmark/results/logs/c2a_fresh.log
#        If 2a is interrupted, CONTINUE it (do not restart) with:
#            caffeinate -is uv run python benchmark/run.py --fresh --resume \
#                2>&1 | tee -a benchmark/results/logs/c2a_fresh.log
#        (results.json now holds only freshly-scored records, so --resume correctly
#        skips just the problems already redone this run.)
#
#    2b. C2b — pipeline pinned to Sonnet, written to ITS OWN file so it does NOT
#        overwrite C2a.  (--results-json is why the old "copy results.json aside"
#        step is gone; without it this clobbers results/results.json = C2a.)
#        --model actually pins the WHOLE pipeline: run.py rewrites every agent's
#        `model:` frontmatter to sonnet for the run (Claude Code otherwise honors
#        each subagent's own model, so formulator/plan-creators would stay Opus and
#        C2b would silently == C2a on its planners) and restores them on exit.
#        NO --fresh here: the one-time Opus-planner-plan regeneration is already done,
#        and with a ~1-problem-per-session usage budget --fresh would re-wipe the
#        in-flight problem's STATE on every resume — restarting it from the formulator
#        and losing a whole first round (a slow problem could then never finish).  Plain
#        --resume skips completed problems and CONTINUES the interrupted one from its
#        saved STATE (Phase pre-2).  Interrupted? re-run the identical line.
caffeinate -is uv run python benchmark/run.py --model sonnet --results-json benchmark/results/pipeline_sonnet.json --resume \
    2>&1 | tee -a benchmark/results/logs/c2b_sonnet.log

# 3. the head-to-head (quick render; no sleep-guard needed, but logged for the record).
#    Paths are benchmark/results/... (relative to the repo root, where you run these):
#    run.py --results-json and compare.py --pipeline both resolve paths against your
#    cwd, and compare.py SILENTLY SKIPS a --pipeline file it can't find (stderr warning
#    only) — a bare "results/..." would land in a stray top-level results/ dir and drop
#    the column from the comparison with no hard error.
uv run python benchmark/compare.py \
    --pipeline benchmark/results/results.json=C2a:opus+sonnet \
    --pipeline benchmark/results/pipeline_sonnet.json=C2b:sonnet \
    2>&1 | tee -a benchmark/results/logs/compare.log
```

### 8.3 Why `--resume` on every command after the first

The one-shot unit key is `slug|condition|manuals|model|rep`, so the same unit run
twice is paid for twice. The full-suite runs (step 1) and the discriminator run
(step 1b) **overlap** — the discriminator problems' c0/R5 and c1/rep1 units are
identical in both. `--resume` skips any unit already in `oneshot_results.json`, so
the overlap costs nothing and command order past step 1 stops mattering. It also
makes every command safe to just re-run after an interruption. Both drivers write
their results file **atomically after each unit**, so a resume never sees a
half-written file.

### 8.4 Where results and progress land

| Command | Live results file (written per unit) | Per-unit artifacts / logs |
|---|---|---|
| `oneshot.py` (C0/C1) | `benchmark/results/oneshot_results.json` | `workspace_oneshot/<model>/<slug>/<cond>-<manuals>/rep<N>/` — prompt.txt, response.json, verify.json, solver.py |
| `run.py` C2a | `benchmark/results/results.json` | `benchmark/results/logs/` + workspace plan dirs |
| `run.py` C2b | `benchmark/results/pipeline_sonnet.json` (via `--results-json`) | same, alongside |
| `compare.py` | `benchmark/results/COMPARISON.md` | — |

The live JSON reflects every unit finished so far; `tee`ing to a log (§8.1) plus the
console `[i/N]` progress lines is the easiest way to watch a run. Add `--list` to any
`oneshot.py`/`run.py` command to print the unit plan and exit without spending.

## 9. Open items before the real baseline

- **[pending]** Run the real Opus baseline once the benchmark (incoming hard PDEs) is final.
- **[done]** `run.py --results-json PATH` (+ `--report PATH`) so the C2b Sonnet sweep
  writes to its own file instead of needing a manual copy-aside — used in §8.2 step 2b.
- **[done]** PDE order-threshold validation pass (§7) — 11/13 imported floors confirmed
  with real reference solvers; elasticity/MHD via an order-2 error model with their
  structure gates exercised; harness false-fails fixed (coarse-grid order waiver,
  zero-field norm, acoustic wavenumber).
- **[pending]** Finalize the discriminator subset once the hard PDEs land (flag them).
  *Current decision: keep all 20 hard problems flagged `discriminator: True`.*

---

## 10. Systems / 3D extension — scoping  **[roadmap]**

High-dimensional / complex problems were always part of the AutoNumerics goal, and the
HardNumerics-LowDim set is mostly 2D systems (velocity+pressure, div-free fields, etc.)
and a few 3D problems. Here is exactly what folding them in requires.

### 10.1 What the current harness assumes (the blockers)

The PDE path is built around a **single scalar field on a 1-D/2-D rectangular grid**:

1. `verify.py` handles `dims ∈ {1, 2}` only — `_extract_coords` pulls ≤ 2 axes, and the
   exact solution is called as `analytic(t, x)` or `analytic(t, X, Y)`.
2. `numerical_solution` is one array, shape `(N,)` / `(N, N)`; `grid` is `{x}` / `{x, y}`.
3. Scoring (`rel_l2` and the order check) operates on that one scalar array.

Systems (multiple coupled fields) and 3D each break a different assumption, so they are
two separable pieces of work.

### 10.2 Phase 1 — 3D scalar  *(smallest; enables `fichera_3d`, `acoustic_3d`)*

Scalar field, cubic grid. Changes:
- `verify.py`: extend `_extract_coords` and `_pde_eval_run` to `dims == 3`
  (`X, Y, Z = np.meshgrid(x, y, z, indexing="ij")`, `analytic(t, X, Y, Z)`,
  `numerical_solution` shape `(N, N, N)`). The order check works unchanged.
- `setup.py`: 3-D schema in the contract footer (`(N, N, N)`, `grid={x,y,z}`).
- Cost: `2N` in 3D is **8×** the unknowns — default `grid_N` must be small (≈24→48), and
  the sandbox mem cap / timeout matter. No new metric types.
- **Effort: small** — a few functions, no contract shape change (still one array).

### 10.3 Phase 2 — 2D systems  *(the main effort; the "structure-preserving" stars)*

Navier–Stokes (Taylor–Green), Euler vortex, shallow water (well-balanced), elasticity
(locking), Keller–Segel / Gray–Scott (2-species), MHD (div-free). These need a
**multi-field contract** and **structure diagnostics**:

- **Contract:** `solve_pde(N)` returns multiple named fields, e.g.
  `numerical_solution = {"u": ..., "v": ..., "p": ...}` (or a `fields` dict) + `grid`.
  This is a schema change to `setup.py`, `runner.py` (`_extract_pde` must carry a dict of
  arrays), and `verify.py` (score per field).
- **Ground truth:** `problems.py` `analytic` returns the per-field exact solution (a dict
  of callables / arrays), authored and cross-checked here as usual.
- **Metrics:** per-field relative L2 (`velocity_L2`, `pressure_L2`, `global_L2`) **plus**
  the structure diagnostics that make these problems hard and that a naive scheme gets
  wrong — each an independent checker in `problems.py`:
  - divergence constraint norm (`div_B_norm`, `spurious_velocity_norm`) for MHD/Maxwell/NS,
  - well-balancedness (`lake_at_rest` residual) for shallow water,
  - volumetric-locking indicator for nearly-incompressible elasticity,
  - positivity / max-principle violation, mass conservation for Keller–Segel / Gray–Scott.
  Verdicts likely need a small vocabulary beyond PASS/FAIL (e.g. a `CONSTRAINT_VIOLATION`
  akin to the SDE `BLOWUP`).
- **Pipeline agents:** `solver-pde.md` / `evaluator-pde.md` / `formulator.md` extend to
  multi-field I/O and per-field + diagnostic scoring. (`solver-pde.md` already hints at
  multi-field returns for Navier–Stokes; the evaluator does not yet score them.)
- **Effort: medium–large** — the contract + scoring generalization is the bulk; each
  problem then needs its exact fields + its diagnostic checker.

### 10.4 Phase 3 — 3D systems  *(largest; `maxwell_3d`, `stokes_3d`)*

Both extensions at once: multi-field on a 3-D grid, with structure preservation
(div-free E/B, inf-sup for Stokes). Do last, after 1 and 2 are proven.

### 10.5 Recommended phasing & open questions

- **Phasing:** Phase 1 (3D scalar) is a cheap, high-signal win — do it first. Phase 2
  (2D systems) is the substantive extension and where the "complex problems" goal really
  lands. Phase 3 last.
- **Open design questions for when we start:**
  1. Multi-field return schema — a `fields` dict vs. `numerical_solution` becoming a dict?
  2. Which structure diagnostics become hard **pass/fail gates** vs. reported-only?
  3. Do we adopt any of HardNumerics' resource budget (60 s / 2 GB / 1 thread) as a gate,
     or only report wall/memory? (Their scoring leans on it; ours currently does not.)
  4. `grid_N` defaults per 3D problem given the 8× cost of `2N`.
