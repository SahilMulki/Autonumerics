# No-Closed-Form Pipeline — Findings and Proposed Fixes

**Date:** 2026-09-19
**Inputs:** the four-problem run in `benchmark/results/no_closed_form.json` /
`NO_CLOSED_FORM_REPORT.md` (logs under `benchmark/results/logs/`), the staged workspaces
under `workspace/`, and the test suite landed the same day (`verifylib/tests/test_kernel_contract.py`,
`test_kernel_sde_path_b.py`, `benchmark/tests/`; 415 passing, 14 strict xfails).
**Related:** [plan-no-closed-form.md](plan-no-closed-form.md),
[NO_CLOSED_FORM_CANDIDATES.md](../benchmark/NO_CLOSED_FORM_CANDIDATES.md).

Every number below was measured, not inferred; the command or test that produced it is named so
it can be re-run.

---

## 0. Summary

| Problem | Route | Pipeline | Kernel provenance | Harness | Verdict |
|---|---|---|---|---|---|
| `pde_kuramoto_sivashinsky` | R | 10 (`3-fd4-imex-cnab2`) | `manufactured` | relL2 **7.2e-02** at N=128 | **OVERCLAIM** |
| `pde_burgers_viscous_1d` | S | 10 (`2-fd4-central-rk4`) | `analytic` (!) | 3.6e-04 | PASS |
| `pde_cahn_hilliard_2d_coarsening` | R+I | 10 (`3-fd4-imex-bdf2`) | `manufactured` | 6.6e-05 | PASS |
| `pde_schrodinger_eigen_2d` | S+F | 10 (`1-fd5-shift-invert`) | `manufactured` | 8.4e-05, λ₁ 2.9e-04 | PASS |

The suite did what it was built to do: it produced an **OVERCLAIM on the one problem whose difficulty
only appears at the reporting horizon**, and in doing so exposed that the pipeline ranked the one
wrong plan out of three above two correct ones. The harness side held up — every route to truth
graded correctly, both gates on Cahn–Hilliard bit on synthetic defects, and the memorization trap on
the eigenproblem is closed (tested). The fixes below are almost all on the **pipeline** side:
the chaotic waiver, the winner early-exit, D1's blindness to spectral solvers, and the certification
grid. Section 6 lists the kernel-contract gaps the tests recorded as strict xfails.

Recommended order of work: **F1 → F2 → F4 → F5 → F6**, then the xfail sweep (§6). F1 alone turns the
KS run from OVERCLAIM into a correct ranking.

---

## 1. F1 — The chaotic waiver certifies accuracy at a horizon the problem does not ask about

**Severity: the cause of the OVERCLAIM.**

### What happened

`chaotic: true, chaotic_T_ref: 5.0` moved Tier C to `t = 5` (plan §8a, errata I14). The kernel's
ladder `N = 64/127/253` at `t = 5` gave `d10 = 1.8e-3, d21 = 1.3e-4`, order 3.84, GCI **1.47e-05**,
and the plan was certified 10 `manufactured` with `error_horizon: 5.0` in the metrics block. The
conductor's own summary said, verbatim: *"nothing in this run verifies accuracy at that horizon"* —
and declared the plan the winner anyway.

Measured on the same solver at the same grids (`uv run python` on
`workspace/pde_kuramoto_sivashinsky/plans/3-fd4-imex-cnab2/solver.py`, reference from `problems.py`):

| Grid | rel. error vs reference at `t = 50` | ladder difference at `t = 5` | at `t = 50` |
|---|---|---|---|
| 64 | 1.94 | — | — |
| 127 | **7.4e-02** | 2.2e-03 | 1.90 |
| 253 | 4.6e-03 | 1.6e-04 | **7.0e-02** |
| 505 | 2.8e-04 | 1.0e-05 | 4.3e-03 |

At `t = 5` the initial condition (two long modes, `k = 1/16, 1/8`) is still smooth and a 4th-order
stencil on `dx = 0.8` resolves it. By `t = 50` the KS instability has filled the spectrum out to
`k ≈ 1` and the same grid is 7% wrong. **The waiver's premise — "trajectories separate past the
Lyapunov time, so full-`T` convergence is meaningless" — is true of the *order estimate* and false
of the *accuracy statistic*.** Chaos amplifies differences that exist; a pair of grids that both
resolve the developed spectrum has none to amplify. Measured, for all three KS plans, the two
finest full-`T` solves (127 vs 253):

| Plan | full-`T` `d21` | harness at N=128 |
|---|---|---|
| `1-spectral-etdrk4` | 2.08e-03 | 2.08e-03, PASS |
| `2-spectral-imex-sbdf3` | 2.08e-03 | 2.08e-03, PASS |
| `3-fd4-imex-cnab2` | **6.99e-02** | 7.22e-02, **FAIL** |

The full-`T` pair difference reproduces the harness verdict on every plan, without a reference.

### Proposed change

**The chaotic waiver waives the order, not the accuracy.** In `driver._path_b_richardson`
(`verifylib/kernel/driver.py`), when the ladder is re-solved at `chaotic_T_ref`, additionally form
`d21_T = rel_err(restrict(u_T^{N2}), u_T^{N1})` from the full-`T` runs already in memory (they are
the runs D1 and D2 read), and require `d21_T < tol` for `converged`. Record it in the metrics as
`estimated_rel_error_T` beside the existing `estimated_rel_error` / `error_horizon`, and render both
in the block so REPORT.md quotes the one at the requested horizon.

Score consequence on this run: plan 3 → `converged: False` → rubric row "converging, not accurate
enough" → **7**; plans 1 and 2 stay at 9. The conductor's ranking (score, then provenance, then
error) then picks a spectral plan, which the harness passes.

### Why this change, and not the alternatives

- *Scale the `T_ref` estimate by a Lyapunov factor `e^{λ(T−T_ref)}`?* It answers the wrong question.
  With `λ ≈ 0.05–0.1` the factor is 10–90×, taking 1.5e-5 to at most 1.3e-3 — still "pass". The
  failure is not trajectory divergence; it is that the resolution requirement is set by the
  spectrum at `T`, which does not exist yet at `T_ref`.
- *Drop the waiver and run Tier C at full `T`?* Errata I14 measured why not: the coarse level is
  garbage (`d10 = 190%` here), so no order can be read. The order study stays at `T_ref`; only the
  accuracy statistic moves to `T`.
- *Require a harness-owned reference for every chaotic problem?* That is the benchmark's remedy
  (§8c) and cannot apply to a user-brought problem. The pair difference is available on every
  problem the kernel evaluates.
- The rule is not circular: identical fields at both grids (a solver that ignores `N`) give
  `d21_T = 0`, which is the separate defect §6.1 already covers.

### Verify

Add a canary: the honest reaction–diffusion spec with `chaotic: true, chaotic_T_ref` and a seeded
solver whose error grows with `t` and with `dx` (accurate at `T_ref`, wrong at `T`); assert it does
not certify. On frozen artifacts: `cli.py evaluate` on the three KS plans must give 7 / 9 / 9.

---

## 2. F2 — The winner early-exit discards the only independent evidence Path B has

**Severity: turned F1 into a wrong ranking; would have caught it on its own.**

### What happened

In all four runs the conductor stopped the moment one plan reached 10:

| Problem | Evaluated | Left unscored |
|---|---|---|
| KS | plans 3 (10), 1 (9) | plan 2 *solved, never evaluated*; plan 4 never dispatched |
| Burgers | plans 1–3 | plan 4 never started |
| Cahn–Hilliard | plans 3, 1, 2 | plan 4's evaluator *killed mid-kernel-run* |
| Schrödinger | plan 1 | plan 2 never implemented; plan 3 solved, never evaluated |

On a Path-A problem this is a scheduling economy: the 10 was measured against a closed form. On
Path B the 10 rests on the kernel's *estimate*, and the strongest cheap evidence about the *answer*
is whether an independently written plan of a different method family agrees with it — the very
standard the harness uses to certify a Route-R reference ("two independent method families agreeing
far below the problem's tolerance", candidates §2.3). The run had that evidence and threw it away:
at `t = 50`, plan 3 vs plan 1 differ by **7.6e-02 at N=127** and 4.6e-03 at N=253 (measured); plans
1 and 2 agree to ~1e-5. A 10 that disagrees with a 9 by 7.6% at the grid the problem names should not
be a winner.

### Proposed change

In `commands/conductor.md`:

1. **On Path B (`analytic_solution: null`), the winner early-exit does not fire until at least two
   plans of different `scheme_family` have been evaluated.** Cost: one more evaluator pass, which
   the CH log measured at seconds of kernel time.
2. **A cross-plan agreement gate.** When two evaluated plans both score ≥ 9, the conductor asks the
   kernel to compare their finest full-`T` fields (a small `cli.py compare <planA> <planB>`
   subcommand: restrict to the common grid, report the relative difference). If they disagree by
   more than `rel_l2_err_max`, neither may be declared a winner; the conductor records the
   disagreement as an `agent_cap` on the *higher*-scoring plan with the measured number as the
   reason, and dispatches another cycle. This is the asymmetric authority §2c already grants the
   agent, used for exactly what it was meant for.
3. **Agreement can count as the second circularity break** (see F3): two plans of different
   families agreeing at full `T` within `tol` is the same evidence class as a harness reference,
   and Path B currently has no other route to a 10 for a spectral solver.

### Why

`plan-blind-evaluator.md` already owns cross-plan ranking, and rev 2 of the plan moved consensus
"out of the ladder entirely" (§4). That was the right home for *ranking*; it is the wrong place for
a *gate*, because the conductor exits before ranking happens. The gate has to be on the exit
condition.

### Verify

A conductor-level test is out of reach of pytest; the check is the kernel `compare` subcommand plus
a `benchmark/tests` assertion that no `closed_form: False` record in `results.json` has
`n_plans - evaluated < 1` with a 10 winner. Re-run KS: with F1 and F2, the expected outcome is plans
1 and 2 at 9 agreeing to 1e-5 and plan 3 capped.

---

## 3. F3 — D1 is structurally unavailable to spectral solvers, and that decides the ranking

**Severity: systematic bias; picked the less accurate plan in both Route-R runs.**

### What happened

| Problem | Spectral plans | FD plan |
|---|---|---|
| KS | `1-spectral-etdrk4`: 9, est. err 1.4e-10, `d1_outcome: unresolved` | `3-fd4`: **10**, est. err 1.5e-5, D1 clean |
| Cahn–Hilliard | plans 1, 2: 9, est. err 1.4e-15, D1 `slow` (p_res 2.915 vs 3.0 needed) / `unresolved` | `3-fd4`: **10**, est. err 5.8e-6, D1 clean |

The CH conductor summary put it plainly: *"FD4 won on strength of evidence, being the only plan to
earn the second circularity break"* — with error estimates nine orders larger. The mechanism is not
the §5b stencil pair alone. From `cli.py evaluate --json` on the KS spectral plan, the residual
medians were **9.2e-02 at N=127 and 1.5e-04 at N=253** for a field the harness measures at 2e-3 and
~1e-5: the residual on a near-Nyquist chaotic field is dominated by the kernel's own
`O(dt²)` time term (`K = 3` snapshots) and by product aliasing, not by the solver. On FD4 the
same residual is 1.6e-2 → 7.3e-4 — comparable magnitude, but it *falls at the FD4 rate*, so the
slope test is satisfied on a field that is 7% wrong at N=127. D1 measures whether the residual
converges, which an FD scheme's does by construction and a resolved spectral scheme's cannot
(there is nothing left to converge).

### Proposed change

Three parts, cheapest first:

1. **`snapshots` contract: ask for `K ≥ 5`** in `agents/solver-pde.md`. The Fornberg time term is
   then 4th order and the spectral plans' residual floor drops by orders. The CH log identified this
   as "a one-line path to a 10 that was never tested".
2. **Do not let a check-limited D1 forfeit the break by default.** `unresolved` is defined as "the
   check, not the solver, is the limit" and "never counts against the plan" — but forfeiting the
   10 is counting against it whenever an FD plan is in the same pool. Either treat `unresolved` with
   a validated operator as neutral for the *ceiling* (cap at 9 only when D1 ran and was `slow`), or
   accept F2(3): cross-plan agreement as the substitute second break on Path B.
3. **Longer term:** for a declared `spectral` plan on a periodic grid, use the spectral derivative
   as the primary residual stencil and read the FD8 disagreement as an *under-resolution detector*
   rather than as `unresolved`. Rule 2 of §3b ("different family") exists to keep truncation error
   from being invisible; on a resolved spectral field there is no truncation to hide, and the time
   term and the nonlinear product are still formed by the kernel.

### Why

Without this, the pipeline's plan pool has a hidden preference for finite differences on every
periodic problem, and "no closed form" problems are exactly where spectral methods are the right
answer (KS, CH, the second-wave NLS/KdV candidates). The benchmark would keep measuring that
preference as OVERCLAIMs.

---

## 4. F4 — The kernel certifies at a grid the problem does not name

**Severity: moderate; a second, independent cause of the same OVERCLAIM.**

### What happened

`problem.md` (written by `benchmark/setup.py`) says: *"The harness ... calls it at two resolutions,
N = 64 and N = 128 ... accuracy — relative L2 error < 1% at the finer grid (N = 128)."* The
formulator set `grid_N: 64, refinement_levels: 3`, so the kernel's ladder was 64/127/253 and the
certification — GCI on the finest grid — is a statement about **N = 253**. The FD4 plan is within
1% there (4.6e-3) and 7% off at N=128. The requirements ledger flagged the deviation as an
*ambiguity* (KS R14, Schrödinger R9: "grid ladder 64/127/253 vs the stated 64/128") and proceeded.

### Proposed change

- The kernel already has `u_star`; report the Richardson-estimated error of the **middle** grid
  (`rel(u_1 − u_star)`) as `estimated_rel_error_at_N2`, and when the spec carries a
  `graded_N` (new optional threshold the formulator copies from `problem.md`), make `converged`
  require the estimate **at that grid** to clear `tol`. The top level stays the order-measuring
  level; it stops being the certified one.
- `formulator.md`: a stated grading resolution is a requirement (`kind: evaluation`), not an
  ambiguity; the ledger must map it to `evaluation_thresholds.graded_N`.

### Why

"Honor the `N` you are given" is in the statement; the pipeline honoured it in the solver and
ignored it in the certification. On this run the fix would have reported ~2e-4 at N=127 at
`t = 5` — still a pass, which is why F1 is the primary fix — but on any problem sitting at the
resolution edge it is the difference between a pass and an OVERCLAIM, and it costs nothing.

---

## 5. F5 — A quadrature program passed as a closed form (Route S leak)

**Severity: undermines the controlled experiment for Route-S problems.**

### What happened

`workspace/pde_burgers_viscous_1d/problem_spec.json` carries a 2,010-character
`analytic_solution.expression`:

```
(np.sin(x)*np.ones_like(x) if t <= 0.0 else (lambda y, X: (lambda G: (lambda w:
(np.sum(((X - y[:, None])/t)*w, axis=0)/np.sum(w, axis=0)) ...)(np.linspace(-2.2, 2.0*np.pi + 2.2, 4001), ...))
```

— a log-sum-exp Cole–Hopf quadrature on 4001 nodes, written as nested lambdas. It passed
`schema.py`'s expression gate, evaluated in the restricted namespace, and put the problem on
**Path A** with provenance `analytic`. The candidates file (§2.4) claimed "none of these can be
written in the restricted expression namespace"; that claim is false, and
`benchmark/tests/test_leakage_ncf.py::test_staged_workspace_specs_take_the_route_the_problem_allows`
now reports it. Two further consequences:

- `reference_outcome: validated_off_singularity` — the closed form's residual has a passing
  median (7e-9) but a max of 2.2 over **14.9% of the domain at `x = π`**, i.e. the check was
  waived exactly at the layer the problem exists to test.
- The kernel spent nothing on Tier B or D1 for the no-closed-form suite's Route-S member; the
  suite ran only two genuine Path-B PDEs.

### Proposed change

1. `schema.py`: reject `ast.Lambda`, comprehensions and `np.linspace`/`np.sum`-style reductions
   inside `analytic_solution` (the operator module's own docstring says a lambda "opens a new
   scope" — it is a program, and the standing rule is "prose is allowed, a program is not"). Add an
   expression-length ceiling (e.g. 400 characters) so a formula stays a formula.
2. `reference.py`: `validated_off_singularity` should become `analytic_unvalidated` (A⁻, 9) when
   the excluded fraction exceeds a few percent **or** overlaps a declared `structural_facts`
   feature (`layer`, `shock`). Accepting a closed form everywhere except where the problem is hard
   is A⁻ evidence, not A.
3. Decide whether Route-S problems belong in the "Path B" experiment at all. The honest position
   (already in the candidates file) is that they sit on the boundary; if they stay, the
   `closed_form: False` flag should not imply Path B in any report.

### Why

The matched-pair design isolates one variable — the formula — only if the pipeline cannot
reconstruct it. Here it reconstructed it as a program, which is a legitimate numerical method and
an illegitimate *expression*.

---

## 6. F6 — Solver drift after an early-exit; the harness grades unscored code

**Severity: low today, silent when it bites.**

The CH log: *"Plan 1's on-disk `solver.py` no longer matches its scored version — a second cycle was
cut short mid-edit. The 9 belongs to the earlier file."* The harness re-imports whatever is on disk
for the **best** plan; had plan 1 been the winner, the benchmark would have graded code the kernel
never scored, and neither layer would know.

**Proposed change:** the kernel records `solver_sha256` in the metrics; `benchmark/verify.py`
recomputes it before importing and reports `status: error, "solver.py changed after scoring"`
(rendered `UNVERIFIED` with that reason) on a mismatch; the conductor restores the scored file when
it kills a solver mid-cycle. The guardrails review already hashes the file across an evaluation —
this extends the same check across the layer boundary.

---

## 7. Kernel-contract gaps recorded by the test suite (14 strict xfails)

All in `verifylib/tests/test_kernel_contract.py`, `test_kernel_score.py`,
`test_kernel_sde_path_b.py`, `benchmark/tests/test_truth_routes.py`. Each `xfail(strict=True)`
names the gap; fixing it means deleting the marker, never rewriting the assertion.

### 7.1 The ladder misreads "no difference" as "converged to round-off" — **fix first**

`richardson.from_ladder`: identical fields at every level give `d10 = d21 = 0`, and
`not_at_roundoff` reads that as machine-precision convergence (`order: inf, converged: True`).
Measured: a solver that ignores `N` and returns no snapshots scores **9 `manufactured_partial`**;
a field with a NaN in it gets the same "round-off" verdict (and is sunk only by the positivity
gate noticing the NaN). Fix: in `driver.evaluate_pde`, before the ladder, require the returned
grid to change with `N` (`len(axes) == N` up to the endpoint convention) and the fields to be
finite; report `crashed: bad_schema` / a named `converged: False` otherwise. The round-off branch
should require `d10 > 0` *or* an explicit floor relative to `rms(u)`, not exactly zero.

### 7.2 Snapshot contract validation

- `sandbox._dump_pde` checks `"fields" in snap` but reads `snap["t"]` unguarded → a missing time is
  reported as `exception`, not `bad_schema`.
- Duplicate snapshot times reach `fornberg_weights` outside any `try` → NaN weights → D1
  `unresolved (nan vs nan)` and a 9. Validate strictly increasing `t` and make D1 `unavailable`
  with a reason.
- Snapshots whose last `t ≠ t_final` are used as the final state silently (score 10 with a 0.1
  shift). Check `abs(t_last − t_final) < tol` and report.

### 7.3 `ic_consistency` is defined and never called

Plan §5f lists three gating trivial-attractor guards; `residual.ic_consistency` exists,
`driver._run_d1` calls only `trivial_guards`. A solver starting from `2·sin(πx)` scores 10 (linear
problem: the residual is satisfied by construction, MMS supplies its own IC). Wire it: one solve at
`t_final = 0` (or read the first snapshot when `K` covers `t = 0`), compare to
`spec.initial_condition`, gate.

### 7.4 Domain mask: wrong grid, silent fallback

`driver.evaluate_pde` builds the mask on the finest grid; `_run_d1` differences the top *two*
grids, so every masked problem gets `ValueError: shapes (65,) (129,)` → D1 `unavailable`. And an
unevaluable mask expression returns `None`, so D1 runs unmasked across the hole (§5e: "never
quietly difference across a hole"). Build the mask per level; make an unevaluable mask
`unavailable` with a reason.

### 7.5 Scoring details

- `certify` applies an `agent_cap` with no reason (§2c requires one) — record it as
  `agent_cap_rejected` instead.
- `driver._reference_outcome` returns `"unavailable (ValueError)"`, which `tier_of` does not match,
  so a Path-A plan whose reference check raised falls to **8** `self_convergence` instead of 9
  `analytic_unvalidated`. Return the bare token and carry the exception in `detail`.

### 7.6 SDE Path B

- `rubric_sde` tests `failed(resolved)` before `dynkin_ok`: a sign-flipped drift (Dynkin z ≈ 1500)
  is scored **6 "MC-inconclusive — not a solver bug"**, and the feedback it drives (raise
  `num_paths`) is wrong. Order the rows: gate → Dynkin/orders failure → inconclusive.
- The SDE driver hardcodes `operator_validated: False`, so a clean Dynkin never counts as a
  circularity break and manual §23's "orders + Dynkin + constraints → 9" row is unreachable
  (honest solver, no surrogate: 8). Dynkin is built from the spec's own drift/diffusion; treat a
  clean Dynkin as the SDE's D1 break, or say in §23 that 8 is the ceiling.
- A solver that ignores the CRN increments (strong order 0.001, `orders_ok: False`) still certifies
  **10** through the surrogate row, which never consults `orders_ok`. Decide whether §7's rule for
  `override` ("a solver that ignores its inputs cannot be certified") extends to §18's `dW`. The
  test records the current behaviour either way.

### 7.7 Harness

`verify._functional_errs`'s docstring says a solver returning no `functionals` on a
`functional_truth` problem "has violated the contract, which is a failure of that solver, not an
inconclusive"; the code returns `inconclusive` → `UNVERIFIED`. The docstring is the design: a
solver told to return `lambda_1` and not doing so has failed.

---

## 8. Smaller observations from the run

- **Endpoint convention on periodic domains.** `problem.md` asks for `np.linspace` "including the
  boundary points" on a periodic domain; every formulator logged an ambiguity (KS R14, Burgers R8)
  and chose `N−1` distinct nodes. The harness's trig interpolation accepts either convention
  (tested), so state that in `setup.write_problem_md` for periodic problems and stop generating the
  ambiguity.
- **Steady problems.** Schrödinger's ledger carried "R12 `t_final` on a steady problem"; the
  template's "Report the solution at t = ..." line needs a steady-state variant.
- **`phase_amplitude` was `not_reported` on every CH plan** — an ungated free-form invariant the
  formulator declared and nothing evaluates. Harmless (no cap), but a coarsening problem's natural
  structural check (interface count / energy decay) could be a registry entry.
- **Plan-creator sign error** (CH plans 3 and 4: implicit-symbol formula not negative-semidefinite),
  caught and corrected by both solvers. Worth a plan-creator note, not a pipeline change.
- **Usage limit hit mid-run** on Schrödinger (`limit_hit: true`); the run completed on the single
  evaluated plan. The runner's whole-word limit detection (F2's "degenerate limit" regression) is
  now tested.
- Mean wall time 22.6 min per problem, all four completed, no `SPEC_BLOCKED`, no leakage findings.

---

## 9. What worked and should be left alone

- **Route R end to end.** The KS reference caught a 7% error the pipeline certified; the
  tolerance-widening arithmetic, artifact/`problems.py` agreement, periodic interpolation under both
  endpoint conventions and non-periodic nesting refusal are all tested
  (`benchmark/tests/test_truth_routes.py`).
- **Route F and the memorization trap.** A recalled `λ₁` with a random field fails on the field;
  the right mode with `λ₁` off by 2e-3 fails on the functional; sign and scale are gauged away;
  an excited state trips `ground_state_nodeless` (all tested).
- **Route I paired with R.** Cahn–Hilliard's `mass_error` and `nontrivial` gates bite on synthetic
  defects; the pipeline's winner cleared both with mass error 5e-7.
- **Isolation.** The uncommitted `run.isolated_settings` denies every sibling workspace per run
  (tested); the suite split keeps the 49-problem baseline denominators fixed.
- **Kernel–harness agreement on the honest canary and on four seeded defects** — none of the
  seeded defects is an OVERCLAIM, which is the calibration point for the word.

---

## 10. Suggested sequence

| Step | Change | Flips |
|---|---|---|
| 1 | F1: full-`T` `d21` in `converged` on chaotic problems | KS 10 → 7; new canary |
| 2 | 7.1: grid-changes-with-`N` and finiteness checks before the ladder | 2 xfails |
| 3 | F2: Path-B early-exit rule + `cli.py compare` + agreement gate | conductor.md; re-run KS |
| 4 | F4: `graded_N` and the middle-grid estimate | new metrics key |
| 5 | F5: lambda/reduction ban + length ceiling in `schema.py`; `validated_off_singularity` → A⁻ over a declared feature | schema test; Burgers re-run on Path B |
| 6 | 7.2–7.5: snapshot validation, `ic_consistency`, per-level mask, cap reason, reference token | 8 xfails |
| 7 | 7.6: SDE rubric order, Dynkin as a break, CRN-contract decision | 3 xfails |
| 8 | F3 (1)–(2): `K ≥ 5` snapshots; `unresolved` not forfeiting the break | CH/KS spectral plans → 10 |
| 9 | F6 + 7.7: solver hash across layers; functional-missing → failure | 1 xfail |

After steps 1–3, re-run `benchmark/run.py --no-closed-form --fresh`. The expected report: KS
VERIFIED_PASS on a spectral plan, four of four, zero OVERCLAIMs — and the tests that got there
are the ones to keep.
