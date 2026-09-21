# No-Closed-Form Pipeline — Findings and Proposed Fixes

**Date:** 2026-09-19; reviewed and extended 2026-09-20 (the review re-graded every plan of KS and
Cahn–Hilliard against the harness reference — §11.1 — and adjusted F1–F5 where the proposal
outran the evidence; each adjustment is marked **[rev]**). A second pass the same day re-measured
the review's claims and folded in four corrections, marked **[rev2]**; one item (§11.2) is left as
a recorded disagreement.
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
grid. Section 7 lists the kernel-contract gaps the tests recorded as strict xfails.

**[rev]** The verdict column above grades only the winner. Grading every evaluated plan (§11.1)
shows the ranking defect is not confined to KS: on **both** Route-R problems the kernel ranked the
least accurate plan first, by three orders of magnitude on Cahn–Hilliard (spectral 4.5e-08 at 9,
FD4 6.6e-05 at 10). The KS OVERCLAIM is the one case where that bias crossed the pass line.

Recommended order of work: **F1 (with F4 folded in) → 7.1 → F2 → F3(1) → F5 → F6**, then the rest
of the xfail sweep (§7), then the harness-side additions in §11. F1 alone turns the KS run from
OVERCLAIM into a correct ranking; F3(1) is what stops the same bias from producing the next one.

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
  `d21_T = 0`, which is the separate defect §7.1 already covers.

### [rev] An `unshadowable` outcome, so F1 does not blame the scheme for the horizon

`d21_T < tol` is the right test when resolved grids agree at `T`. It is the wrong *diagnosis* when
they cannot: on a horizon past what any resolution shadows (KS at `T = 500`, say — amplification
`e^{λT}` of round-off alone is O(1)), `d21_T` is O(1) for every plan, correct ones included, and F1
as written reports "converging, not accurate enough → 7", which sends the solver off to refine a
grid that is not the problem. The kernel can tell the two cases apart from data it already has:

- `T_ref` ladder converges cleanly **and** `d10_T ≈ d21_T ≈ O(1)` (the full-`T` pair difference does
  not shrink under refinement) → the horizon is **unshadowable**: pointwise accuracy at `T` is not
  attainable at any resolution and the honest deliverable is a statistic, not a field. Name it
  (`converged: "unshadowable"` in the skip vocabulary), cap the score as `self_convergence` is
  capped, and make the rendered reason say so, so the conductor's feedback is not "refine".
- `T_ref` ladder converges **and** `d21_T` shrinks under refinement but is above `tol` → the F1
  case: under-resolved at `T`. Rubric row "converging, not accurate enough", 7.

This is the same standard the harness applies to itself: a Route-R reference that two schemes cannot
reproduce at `T` does not ship (candidates §2.3), and the problem is reformulated around a
statistic. The kernel should reach the same conclusion from the inside rather than failing the
solver.

**[rev2] The outcome is right; inferring it from the ladder is not.** "The full-`T` differences do
not shrink at these grids" is not the same fact as "cannot shrink at any grid". On a 32/64/128
ladder the under-resolved FD4 plan above shows exactly the proposed signature — `T_ref` ladder
clean, `d10_T ≈ d21_T ≈ O(1)` — and would be labelled unshadowable, with feedback that says *do not
refine* on the one problem where refinement is the whole answer. This is the shape plan §8a already
names for `chaotic: true`: a self-relaxation vector, so it is **declared, not inferred**. Add
`unshadowable: true` to the spec beside `chaotic`, with the same ledger-quote requirement and a
`chaotic_T_ref`; the kernel then reports `converged: "unshadowable"` and caps as above. Without the
declaration the kernel keeps the F1 verdict (7) and *reports* the non-shrinking pattern as a hint
in the notes, so a formulator can add the flag on the next cycle — an inference may loosen a test
only when a person has signed it. (Arithmetic note: `e^{λT}` of round-off at `T = 500` is O(1) for
`λ ≈ 0.1` and ~1e-5 for `λ ≈ 0.05`; the example needs the larger exponent.)

### [rev] The certified grid (F4, folded in here)

The same rule fixes F4. `d21_T` between the two finest full-`T` runs *is* the error at the **middle**
grid, and on a `grid_N / 2·grid_N / 4·grid_N` ladder the middle grid is the one `problem.md` names.
So "certify at the graded grid" is not a second mechanism: it is F1's statistic, applied at both
horizons. What F4 adds is only the spec key — `evaluation_thresholds.graded_N`, which the formulator
copies from the statement as a requirement (not an ambiguity) — so the kernel knows which ladder
level is being certified when the ladder does not happen to be built from it. Report both the
top-level GCI (the order-measuring level) and the graded-level estimate, and make `converged` read
the graded one.

### Verify

Add two canaries on the honest reaction–diffusion spec with `chaotic: true, chaotic_T_ref`: a seeded
solver whose error grows with `t` and with `dx` (accurate at `T_ref`, wrong at `T`) must not
certify; the same solver on a spec that also declares `unshadowable: true` must come back
`unshadowable`, not 7, and on a spec without the flag must come back 7 with the pattern noted
**[rev2]**. On frozen artifacts: `cli.py evaluate` on the three KS plans must give 7 / 9 / 9.

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
   more than `rel_l2_err_max`, neither may be declared a winner. **[rev]** Disagreement does not
   say *which* plan is wrong, so the cap goes on **both** (8, with the measured difference as the
   reason), not on the higher-scoring one — capping the higher score is a guess dressed as a rule.
   A third family, or the next cycle, breaks the tie. This is the asymmetric authority §2c already
   grants the agent, used for exactly what it was meant for.

   **[rev]** With F1 in place most disagreements will already surface as one plan's `d21_T`
   failing. The gate earns its keep in the remaining case — both plans converge individually, to
   *different* answers — which is a wrong coefficient or a sign error in one solver, and is a
   stronger check than the ranking framing suggests.
3. **Agreement can count as the second circularity break** (see F3): two plans of different
   families agreeing at full `T` within `tol` is the same evidence class as a harness reference,
   and Path B currently has no other route to a 10 for a spectral solver. **[rev]** State the
   circularity it does *not* break: both plans read the same formulator's operator, so a
   mis-transcribed equation passes unanimously. MMS, the degenerate limit and D1 are built from the
   same spec and share the limitation, so agreement is no weaker than the break it substitutes for
   — but the harness is the only spec-independent check, and REPORT.md must not imply otherwise.

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
earn the second circularity break"* — with error estimates nine orders larger. **[rev]** The
harness confirms the bias is not a scoring artefact but a wrong ranking of *accuracy*: CH spectral
plans 1 and 2 grade at **4.5e-08** and 1.9e-07 against the reference, the FD4 winner at 6.6e-05
(§11.1). The mechanism is not the §5b stencil pair alone. From `cli.py evaluate --json` on the KS
spectral plan, the residual medians were **9.2e-02 at N=127 and 1.5e-04 at N=253** for a field the
harness measures at 2e-3 and ~1e-5; the FD8 and FD6 stencils then disagree by 2.5× at N=253, which
is the `unresolved` verdict. On FD4 the same residual is 1.6e-2 → 7.3e-4 — comparable magnitude,
but it *falls at the FD4 rate*, so the slope test is satisfied on a field that is 7% wrong at N=127.
D1 measures whether the residual converges, which an FD scheme's does by construction and a
resolved spectral scheme's cannot (there is nothing left to converge).

**[rev] What is measured and what is hypothesis.** That the spectral residual is dominated by the
kernel's own `O(dt²)` time term (`K = 3` snapshots) is a *hypothesis*, and the per-level data cuts
against it as the main cause: the residual falls **600×** from N=127 to N=253, and a time-term floor
would not move with `N`. That drop looks like FD8 truncation on near-Nyquist modes, which more
snapshots do not touch. So part 1 below is cheap and probably harmless, but it is not known to be
the fix, and it cannot be tested on frozen artifacts (snapshots come from the solver). Part 2 is the
fix that follows from the measurement.

**[rev2] Confirmed, and the cause is upstream of the stencil pair.** `cli.py evaluate --json` on
the KS spectral plan records `endpoint_exclusive: [False]`, `stencil_pair:
at-or-above-kernel-order`, and `stats.stencil == '8'` at **both** levels. `residual.stencil_pair`
selects the spectral derivative only when every axis is endpoint-*exclusive*, and `problem.md`
mandates an endpoint-inclusive `np.linspace` — so on every pipeline plan the "spectral" pair
degrades to **FD8 + FD6**, and the residual on a resolved spectral field is the kernel's FD8
truncation on near-Nyquist modes (9.2e-2 at N=127, where the harness measures the field at 2e-3).
The 2.5× disagreement is FD8 against FD6, as the review says. Two consequences: part 3 below is not
"deferred", it is currently **unreachable**; and §8's endpoint-convention bullet is not cosmetic —
errata I5 taught D2 to trim the duplicated wrap node (`invariants.duplicated_endpoint`) and D1
never learned it. That trim is the first fix, and it is measurable on the frozen KS and CH
artifacts.

### Proposed change

Four parts, in the order they are known to matter **[rev2: reordered]**:

0. **Trim the duplicated wrap node in D1 before choosing the stencil pair.** `driver._run_d1`
   passes the raw `exclusive` flags to `stencil_pair`; it should pass the BC's periodicity after
   dropping the duplicated endpoint, exactly as `invariants.PdeContext` does via
   `duplicated_endpoint`. This alone makes the spectral stencil selectable on every pipeline plan
   and turns the KS/CH `unresolved` verdicts into a measurement. Verify on the frozen artifacts:
   the KS spectral plan's per-level `stats.stencil` must read `spectral`, and its N=127 residual
   must fall from 9.2e-2 to the field's own accuracy scale.
1. **Do not let a check-limited D1 forfeit the break.** `unresolved` is defined as "the check, not
   the solver, is the limit" and "never counts against the plan" — but forfeiting the 10 is
   counting against it whenever an FD plan is in the same pool. Treat `unresolved` with a validated
   operator as **neutral for the ceiling**: cap at 9 only when D1 ran and was `slow`/`stalled`. On
   Path B the second break then comes from F2(3), cross-plan agreement, which a spectral plan can
   actually earn.
2. **`snapshots` contract: ask for `K ≥ 5`** in `agents/solver-pde.md`. The Fornberg time term is
   then 4th order. The CH log identified this as "a one-line path to a 10 that was never tested";
   **[rev]** measure it on one re-run spectral solver before writing it into the contract, and
   record whether the residual floor moved.
3. **After part 0:** for a declared `spectral` plan on a periodic grid, use the spectral derivative
   as the primary residual stencil and read the FD8 disagreement as an *under-resolution detector*
   rather than as `unresolved`. Rule 2 of §3b ("different family") exists to keep truncation error
   from being invisible; on a resolved spectral field there is no truncation to hide, and the time
   term and the nonlinear product are still formed by the kernel. **[rev]** Needs its own canary
   — an under-resolved spectral field with aliased energy must still show a residual — before it
   is trusted; without one this weakens D1's independence for the family it is meant to help.

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

### Proposed change — [rev] folded into F1

This is not a second mechanism. F1's statistic — the difference between the two finest runs — *is*
the error at the middle grid, which on a `grid_N / 2·grid_N / 4·grid_N` ladder is the grid the
statement names. One rule, applied at both horizons; see "The certified grid" under F1. What
survives here as its own item:

- `formulator.md`: a stated grading resolution is a requirement (`kind: evaluation`), not an
  ambiguity; the ledger must map it to `evaluation_thresholds.graded_N`, and the kernel reads that
  key to know which ladder level it is certifying when the ladder is not built from it.

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

### Proposed change — [rev] rewritten

The first draft proposed banning `ast.Lambda`, comprehensions and `linspace`/`sum` reductions in
`analytic_solution`, with a 400-character ceiling. **Withdrawn.** The formulator did something
genuinely strong here — derived Cole–Hopf, implemented it robustly, and wrote a correct note on why
the Bessel form fails in float64 — and the stated goal of the project is that the pipeline solves
any PDE; making it weaker to make an experiment cleaner inverts that priority. A length ceiling
would also reject legitimate closed forms the harness itself uses (`_heston_call`'s
characteristic-function integrand, Mittag-Leffler series). What is actually wrong is narrower:
the kernel treated a 4001-node quadrature as **exact**, when it has a discretization error the
kernel never measured.

1. **`schema.py` / `reference.py`: a discretized closed form is a numerical closed form.** An
   `analytic_solution` whose expression contains a discretization — `np.linspace`, `np.sum`/`mean`
   over a constructed axis, a `lambda` — is flagged `analytic_numerical` at parse time, and the
   metrics name the node count so a reader can see what "exact" rested on.
   **[rev2] Flag, no ceiling.** The review's premise — "a 4001-node quadrature has a
   discretization error the kernel never measured" — is true in general and false here: the
   formulator's expression evaluated against the harness's own `_burgers_viscous_1d` on 513 nodes
   at `t = 1.5` agrees to **max |diff| 2.2e-14** (rel L2 2.7e-15). A syntactic A⁻ ceiling would
   demote a formula that is exact to round-off, and the kernel cannot measure the quadrature's own
   convergence from inside an opaque expression. What *was* weak is the layer coverage, and item 2
   handles that on its own. So: record the flag, leave the ceiling to the residual check.
2. `reference.py`: `validated_off_singularity` should become `analytic_unvalidated` (A⁻, 9) when
   the excluded fraction exceeds a few percent **or** overlaps a declared `structural_facts`
   feature (`layer`, `shock`). Accepting a closed form everywhere except where the problem is hard
   is A⁻ evidence, not A. (Unchanged; independent of 1.)
3. **The experimental fix is on the benchmark side, not the pipeline side.** De-integrate Burgers
   (candidates §2.2): `ν = ν(x)` kills Cole–Hopf outright while keeping the operator, the internal
   layer and the matched pair with P11. The harness truth then comes from Route R (two schemes) or
   from a Cole–Hopf-free semi-analytic route, and the problem lands on Path B for real. Correct the
   candidates file's §2.4 claim that these forms "cannot be written in the restricted namespace" —
   it was false, and the test that reports it should stay.
4. Decide whether Route-S problems belong in the "Path B" experiment at all. The honest position
   (already in the candidates file) is that they sit on the boundary; if they stay, the
   `closed_form: False` flag should not imply Path B in any report.

### Why

The matched-pair design isolates one variable — the formula — only if the pipeline cannot
reconstruct it. Here it reconstructed it as a program. That is a *capability* of the pipeline to be
kept and a *flaw in the problem* to be fixed; the only pipeline-side defect is the one item 2
names — a closed form accepted everywhere except at the layer **[rev2]**.

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
  ambiguity. **[rev2]** Not cosmetic: the inclusive convention is what makes D1 fall back to FD
  stencils on every spectral plan — see F3, part 0.
- **Steady problems.** Schrödinger's ledger carried "R12 `t_final` on a steady problem"; the
  template's "Report the solution at t = ..." line needs a steady-state variant.
- **`phase_amplitude` was `not_reported` on every CH plan** — an ungated free-form invariant the
  formulator declared and nothing evaluates. Harmless (no cap), but a coarsening problem's natural
  structural check (interface count / energy decay) could be a registry entry.
- **Plan-creator sign error** (CH plans 3 and 4: implicit-symbol formula not negative-semidefinite),
  caught and corrected by both solvers. Worth a plan-creator note, not a pipeline change.
- **[rev] No usage limit was hit on Schrödinger.** The record's `limit_hit: true` is the
  `"degene·rate limit"` substring false positive the runner has since been fixed for (whole-word
  match, tested); the flag in the stored record is stale. Schrödinger stopped after one evaluated
  plan because of the winner early-exit (F2), not a limit. A `--skip-run` re-parse clears the flag.
- **[rev] Schrödinger's functional gate is the only binding one, and it is tight.** Every
  second-order scheme clears the 5e-4 `λ₁` gate by **1.7×** at N=128 (FD5 and Q1-FEM both at
  2.93e-04, from opposite sides) and fails it at N=64 (1.19e-03), while the 1% field gate is cleared
  by 120×. Not broken — `validate_ground_truth` confirms a correct scheme passes at the stated N —
  but a correct scheme with a worse error constant would be a near-miss, and a future fail on this
  problem should be read against that margin before it is called an OVERCLAIM. (The two schemes'
  errors are equal and opposite: their mean is within 5.5e-08 of the truth, which is the bracket
  F2 would have used.)
- **[rev] `run.py` should record wall-clock time beside monotonic.** The 2026-09-19 Cahn–Hilliard
  crash (three `Connection closed mid-response` in one run) was a laptop sleeping mid-run; it was
  only detectable by diffing the log's mtime (36.7 min) against `wall_seconds` (19.4 min, monotonic
  does not advance during sleep). One `time.time()` field makes it visible in the record.
  Operationally: wrap runs in `caffeinate -i`.
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
- **Isolation.** `run.isolated_settings` denies every sibling workspace per run (tested); the
  suite split keeps the 49-problem baseline denominators fixed. **[rev]** For the record, the need
  was found in a transcript: before the fix, the Cahn–Hilliard formulator `cat`ed its MMS twin's
  `problem_spec.json` and used it as a template, in two separate attempts. The completed CH run
  in the results file ran *with* isolation on.
- **Kernel–harness agreement on the honest canary and on four seeded defects** — none of the
  seeded defects is an OVERCLAIM, which is the calibration point for the word.

---

## 10. Suggested sequence

| Step | Change | Flips |
|---|---|---|
| 1 | F1 (+F4): full-`T` `d21` in `converged`, declared `unshadowable` outcome **[rev2]**, `graded_N` | KS 10 → 7; two new canaries |
| 2 | 7.1: grid-changes-with-`N` and finiteness checks before the ladder | 2 xfails |
| 3 | 11.1: harness grades every plan; per-plan block in the record | free; needed to measure steps 3–8 |
| 4 | F2: Path-B early-exit rule + `cli.py compare` + agreement gate (both plans capped) | conductor.md; re-run KS |
| 5 | F3 (0)–(1): trim the duplicated wrap node in D1, then `unresolved` neutral for the ceiling **[rev2]** | KS spectral N=127 residual 9.2e-2 → field scale; CH/KS spectral plans → 10 |
| 6 | F5 (1)–(2): `analytic_numerical` flag (no ceiling **[rev2]**); `validated_off_singularity` → A⁻ over a declared feature | schema test |
| 7 | 7.2–7.5: snapshot validation, `ic_consistency`, per-level mask, cap reason, reference token | 8 xfails |
| 8 | 7.6: SDE rubric order, Dynkin as a break, CRN-contract decision | 3 xfails |
| 9 | F6 + 7.7: solver hash across layers; functional-missing → failure | 1 xfail |
| 10 | F3 (2): `K ≥ 5` snapshots, measured on one re-run before it enters the contract | — |
| 11 | 11.2 (disputed, see note) –11.3: KS order gate; Burgers `ν(x)`; then the held-out set | benchmark problems |

**[rev] On the acceptance run.** The first draft ended: re-run the four, expect four of four and
zero OVERCLAIMs. That is the wrong acceptance test. Every fix above was designed on KS and CH and
would be confirmed on KS and CH — a test-set leak. The four problems are the *development* set;
they should re-run (steps 1–5 should make KS pass on a spectral plan, and §11.1 should show the
kernel ranking agreeing with the harness ranking), but the fixes are **accepted on held-out
problems**: the second-wave PDE candidates and, in particular, the SDE Route-S set, where
`kernel/sde/` has had no no-closed-form coverage at all and §7.6 already found three gaps by unit
test alone. The tests that got there are the ones to keep; the problems that got there are not the
ones to certify on.

---

## 11. Additions from the 2026-09-20 review

### 11.1 The harness should grade every plan, not only the winner

The single most informative measurement in this document was made by hand, three times: running
`benchmark/verify.py --plan-dir` on the plans the conductor did *not* pick. Each time it produced
the finding (the Schrödinger bracket, KS plans 1 and 2 passing, CH's spectral plans at the reference
noise floor). Measured on the completed CH run, `uv run python benchmark/verify.py --slug
pde_cahn_hilliard_2d_coarsening --plan-dir workspace/pde_cahn_hilliard_2d_coarsening/plans/<p>`:

| CH plan | kernel score | harness relL2 (48/96) | order | mass gate |
|---|---|---|---|---|
| `1-spectral-etdrk4` \* | 9 | **4.465e-08** | 7.12 | 1.6e-10 |
| `2-spectral-imex-bdf2` | 9 | 1.945e-07 | 5.00 | 1.7e-09 |
| `3-fd4-imex-bdf2` — *winner* | **10** | 6.628e-05 | 3.97 | 5.0e-07 |

\* **[rev2]** Graded on the *drifted* file: the CH log records that plan 1's on-disk `solver.py`
no longer matches the version that scored 9 (F6). The row is what the harness sees, not what the
kernel scored; plan 2, whose file is intact, carries the claim on its own (1.9e-07 vs 6.6e-05).
Re-measured 2026-09-20; all three rows reproduce.

The plan the kernel ranked first is the least accurate by three orders of magnitude, and the plan it
ranked below is at the reference's own error (9.3e-08). With KS this is 2-for-2 on Route R.

**Change (harness, `benchmark/run.py`):** after grading the winner, grade every plan directory
that has a `solver.py`, and store a `harness_by_plan` block in the record: per plan, the harness
verdict, `rel_l2_err`, order, and the kernel's score/provenance for that plan (from the plan's
`SOLUTION.md` metrics block, or `cli.py evaluate` when `--skip-kernel` is off). Then:

- **`wrong_plan_won`** becomes a first-class statistic: the winner's harness error exceeds the best
  evaluated plan's by more than the reference error. It is the number F2 and F3 are trying to
  drive to zero, and it is invisible in the winner-only record.
- The provenance × verdict cross-tab in `report.py` gets ~12 rows from this run instead of 4, which
  is what it needs to say anything about whether `manufactured` correlates with being right.
- Cost: one extra solve per plan, no pipeline call. Respect F6 — record the solver hash per plan so
  a drifted file is graded as `UNVERIFIED`, not as the plan.

### 11.2 KS `grid_N: 64` is a benchmark design flaw

At the 64/128 ladder the coarse grid is garbage for **every** scheme — spectral relL2 at N=64 is
2.10, FD4's is 1.94 — so the harness "observed order 9.98" on the spectral plans is measuring
garbage→resolved, not convergence, and the order gate cannot reject anything (a scheme that is
right at one grid and wrong at the other always shows `p ≥ 1`). Re-graded at 128/256
(`verify_pde` with `grid_N` overridden, same solvers):

| plan | 64/128 | 128/256 |
|---|---|---|
| `1-spectral-etdrk4` | 2.10 → 2.08e-03, p = 9.98, PASS | 2.08e-03 → ~0, p = 12.8, PASS |
| `3-fd4-imex-cnab2` | 1.94 → 7.22e-02, p = 4.75, **FAIL** | 7.22e-02 → **4.35e-03, p = 4.05, PASS** |

**Change (`benchmark/problems.py`):** KS `grid_N: 128`. Both grids are then resolved, the order
check is a real measurement, and a genuine 4th-order scheme passes a 1% gate as it should. This
does not unmake the finding — under the problem *as stated* the pipeline claimed 10/10 with a 7%
error, and F1/F3 are true at any ladder — but it changes the expected report: after the fix KS
passes on FD4 too, and the §10 acceptance criterion must not be "KS passes" but "the kernel ranks
the plans the way the harness does" (11.1). Keep the 2026-09-19 record as is; note the change in
`NO_CLOSED_FORM_CANDIDATES.md` §4.

**[rev2] Disputed — recorded, not resolved.** The diagnosis stands: at 64/128 the coarse grid is
garbage for every scheme and the order gate cannot reject anything. The remedy is the wrong one.
`problems.py` set N=64 deliberately so that "only a scheme that resolves the fourth-order operator"
clears 1% at N=128, and that discrimination is what exposed the plan-selection defect (F2/F3): at
128/256 the FD4 plan passes at 4.35e-3, the wrong ranking becomes invisible in the winner-only
verdict, and the acceptance criterion has to be rescued by §11.1. Keep the accuracy gate at the
grid that did its job and fix the vacuous order check on its own: either `pde_order_check: False`
for KS — the precedent is `pde_burgers_inviscid`, whose shock makes the L2 order equally
meaningless — or a 128/256 order ladder with a per-problem `accuracy_at: coarse` so the 1% gate
stays at N=128. Whichever is chosen, the number to watch is §11.1's `wrong_plan_won`, which is
non-zero on this run under either grid.

### 11.3 The matched-pair measurement is still undone

The suite ran the no-closed-form halves of three pairs (Burgers/P11, CH-coarsening/P20,
Schrödinger/P19) and nobody has run the closed-form twins under the same conditions. That
comparison is the experiment the pair design exists for — *what does losing the formula cost the
pipeline?* — and until 11.4 it was contaminated (the CH formulator copied P20's spec). With
isolation on, run P11, P20 and P19 with the same runner, same timeout, same model, into a separate
results file, and report per pair: pipeline score, kernel provenance, harness error, wall time,
iterations. Expected: the closed-form twins reach `analytic` faster; the interesting number is
whether the no-closed-form halves reach the *same harness error* at a different provenance, or a
worse one.

### 11.4 Cross-workspace contamination (fixed; recorded here for the pairs)

Found in the 2026-09-19 transcripts: the Cahn–Hilliard formulator ran
`cat workspace/pde_cahn_hilliard_2d/problem_spec.json` — the MMS twin's operator, invariants and
thresholds — and used it as a template, in both attempts; the overnight attempt also read the
just-finished KS, Burgers and Schrödinger workspaces. `pipeline-settings.json` denied `benchmark/`
and nothing else. `run.isolated_settings` now writes a per-run settings file next to the transcript
(`benchmark/results/logs/<slug>.settings.json`) that denies `Read`/`Edit` and the reader commands on
every sibling workspace, matched anywhere in the command so absolute paths are caught. Any pair
comparison (11.3) that predates it is not a controlled comparison.

---

## 12. Implementation record (2026-09-20)

Everything in §10 steps 1–10 landed in one pass; step 11's benchmark redesign and the pipeline
re-runs (§11.3) did not, by decision. Every number below was measured on the frozen artifacts with
the code as committed; the test that pins it is named.

| Item | Status | Measured / decided |
|---|---|---|
| F1 | **done** | `driver._horizon_accuracy`: `d21_T` between the two finest full-`T` runs; `converged` reads it; `estimated_rel_error_T` in the block. Frozen KS: FD4 `d21_T` **6.90e-02 → 7**, spectral 2.08e-03. `unshadowable: true` declared beside `chaotic` (ledger quote required, `schema.check_unshadowable`) → `converged: unshadowable`, capped at 8, feedback "do not refine"; without it the non-shrinking pattern is a note. Canary `grows_with_t_and_dx.py`: 7 / 8 / honest 10 (`test_kernel_horizon.py`) |
| F4 | **done** | `evaluation_thresholds.graded_N` (schema-checked); the pair difference at the graded level is `estimated_rel_error_graded` and `converged` reads it; formulator told it is a requirement, not an ambiguity |
| F2 | **done** | `cli.py compare A B [--N=] [--no-record]` → `<workspace>/agreements.json` with both solver hashes; the kernel reads it back against current hashes (`kernel/agreement.py`): different-family agreement is a circularity break (`score.agreement_is_break`), disagreement caps **both** at 8 (`cross_plan_disagreement`). Conductor: Path-B exit waits for two families; agreement gate at ≥ 9; re-dispatch the evaluator to price agreement in (no upward agent path). Frozen KS reproduces §2: 1 vs 3 **7.51e-02** at N=127, 1 vs 2 3.2e-06, 2 vs 3 4.56e-03 at N=253 |
| F3 (0) | **done** | `_run_d1` trims the duplicated wrap node (`invariants.duplicated_endpoint`) before `stencil_pair`. KS plan 2: pair `fd8+spectral`, N=253 medians 1.55e-04 vs 1.25e-04 (agree), D1 **clean → 10**. Plan 1 stays 9: its snapshots are on the `N−1` internal nodes while `fields` is the `N`-point array — a snapshot-shape contract violation the trim then doubled; now named `unavailable` with the shape (new §7.2 check), and `solver-pde.md` says snapshots must be on the returned grid |
| F3 (1) | **done** | `unresolved` no longer forfeits the 10 when agreement supplies the second break; a D1 that ran `slow`/`stalled` still caps |
| F3 (2) | **measured, not adopted** | K = 3 / 5 / 7 on the re-run KS spectral plan: N=253 residual 1.545e-04 / 1.547e-04 / 1.547e-04, N=127 9.233e-02 / 9.234e-02 / 9.234e-02, `p_res` 9.223 / 9.221 / 9.221. The floor does not move; the time term is not the limit. Contract stays `K ≥ 3`, with the measurement recorded in `solver-pde.md` |
| F3 (3) | not done | Needs the aliased-energy canary first; unchanged |
| F5 (1) | **done** | `reference.numerical_closed_form`: a `lambda`, or a constructed axis reduced over, flags `analytic_numerical` with node count (`linspace(4001)` on Burgers); a warning and a block line, no ceiling. Candidates §2.4 claim corrected |
| F5 (2) | **done, recalibrated** | `validated_off_singularity` → `unvalidated_at_feature` (A⁻, 9, operator still validated) on a declared `layer`/`shock` or a gross fraction. The fraction is measured **at the finest probe**, because it shrinks with the probe on *both* staged cases — BS 23% → 12% → 8.5%, Burgers 14.9% → 7.3% → 5.4% at N = 257/513/1025 — so a fixed-level "few percent" cannot separate an exact formula at a kink from one at a layer. Threshold 10% (keeps BS at A); the Burgers case is carried by the declared-feature rule, and `formulator.md` §3c-4 now asks for `structural_facts` |
| F5 (3)–(4) | not done | Burgers `ν(x)` and the Route-S/Path-B decision are recorded in the candidates file as open |
| F6 | **done** | `solver_sha256` in the metrics/block (PDE and SDE); `verify.solver_drift` refuses a changed file (`status: error` → UNVERIFIED); per-plan grading respects it; conductor keeps `solver.scored.py` and restores on a kill |
| §7.1 | **done** | `_check_returned_grid`: axis length within one of `N`, fields finite → `crashed: bad_schema` naming N; `richardson.from_ladder` no longer reads NaN, or `d10 = d21 = 0` on a grid that did not change size, as round-off (exactly-zero differences on grids that *did* change stay the manual's §4 machine-precision case) |
| §7.2 | **done** | Sandbox: a snapshot without `t` is `bad_schema`; driver: non-increasing times, last `t ≠ t_final`, or a shape that differs from `fields` → D1 `unavailable` with the reason |
| §7.3 | **done** | `_ic_consistency`: the `t0` snapshot, else one solve at `t_final = t0`; gates at 3 via `gates_clean`. Tolerance 5e-2 (a *wrong* state is O(1) off; FV cell averages and mollified steps are not the defect) |
| §7.4 | **done** | Mask per ladder level; an unevaluable mask makes D1 `unavailable` and is noted |
| §7.5 | **done** | Reason-less `agent_cap` → `agent_cap_rejected`; `_reference_outcome` returns the bare token with the exception in `detail` (and `tier_of` tolerates the old annotated string) |
| §7.6 | **done** | Rubric order gate → Dynkin/orders failure → inconclusive; a clean Dynkin is the SDE's D1 break (`operator_validated`, route `spec_generator`) → the 9 row is reachable; **decided:** a solver that ignores `dW` is capped at 8 through the surrogate row |
| §7.7 | **done** | Missing `functionals` / a missing name → `status: contract`, `passed: False` (FAIL, never UNVERIFIED) |
| §8 | **done** except the `phase_amplitude` registry entry | Periodic endpoint convention stated in `problem.md`; steady `t_final` variant; plan-creator symbol-sign note; `run.py` records `wall_clock_seconds` / `started_at` / `ended_at` |
| §11.1 | **done** | `run.grade_every_plan` → `harness_by_plan` (per plan: harness verdict, error, order, kernel block score/provenance/hash, verdict) and `wrong_plan_won`; `report.py` renders the section, the per-plan provenance × verdict cross-tab, and the summary count. `--skip-plans` to opt out |
| §11.2 | **decided** | `pde_order_check: False` with `grid_N: 128` for KS: the accuracy gate stays at the grid that did its job, the vacuous order check goes. Note in the candidates file §4 |
| §11.3 | not run | Pipeline re-runs need Claude usage; out of scope for this pass |

Test suite after the pass: the 14 strict xfails are all removed (11 kernel, 3 SDE/harness — every one
flipped by its fix, none by rewriting an assertion beyond a `.get` on a crashed record); new files
`test_kernel_horizon.py`, `test_reference_numerical.py`, `benchmark/tests/test_every_plan.py`. The
stale "four KS evaluators reconcile onto one number" test (written for the pre-waiver artifacts)
is replaced by the per-plan 9 / 10 / 7 check.
