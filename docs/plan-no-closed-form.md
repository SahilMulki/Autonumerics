# Verification Without a Closed Form — Implementation Plan

**Status:** proposal
**Scope:** how a solution is scored when `analytic_solution` is `null` (PDE) or
`analytic_moments.has_analytic_solution` is `false` (SDE)
**Related:** [plan-hallucination-guardrails.md](plan-hallucination-guardrails.md),
[plan-blind-evaluator.md](plan-blind-evaluator.md)

Provenance labels used throughout: **[A]** = already in this repo, **[B]** = adapted from the
AutoNumerics paper-release pipeline, **[N]** = new in this plan.

---

## 1. Why this exists

Two failure modes, one in each of the systems this design draws from.

**This repo.** The verification battery in `references/verification_manual.md` is broad and
principled — it covers any problem the formulator can write a `verification` block for. But the
evaluator agent *writes `evaluate.py` fresh every cycle*: nested ladders, Richardson, GCI guards,
asymptotic tests, MMS probe validation. That is a nontrivial numerical program regenerated on every
pass, and nothing re-checks its arithmetic. `benchmark/verify.py` re-checks the **solver's final
answer** against independent ground truth — so on a problem with no ground truth (Kuramoto–Sivashinsky
today), an evaluator that miscomputed the GCI is invisible.

**The paper-release pipeline.** Its evaluator-side term-balanced residual is fixed Python and cannot
be hallucinated. But it is a hard-coded allowlist of **8 problem IDs** (3 in Original-24, 5 CodePDE);
anything else returns `residual_metric_status: "unsupported"` and gets no diagnostic at all. It also
never checks convergence order, and it differences the field with the solver's own stencil family,
which makes agreement a tautology rather than evidence.

The two flaws are mirror images: general-but-regenerated versus deterministic-but-hard-coded. This
plan takes the coverage of the first and the trust model of the second.

---

## 2. Architecture — split the evaluator in two **[N]**

| Layer | Implementation | Responsibility |
|---|---|---|
| **Kernel** (`verifylib/`) | Fixed, human-written, version-controlled, under test. **Never regenerated.** | Every numerical check: ladders, Richardson/GCI, operator residual, invariants, MMS driver, temporal isolation, Dynkin, CRN. Emits one machine-readable metrics JSON. |
| **Evaluator agent** | `agents/evaluator-{pde,sde}.md` | *Selects and configures* checks, then **diagnoses** — turns kernel output into specific, actionable solver feedback. Writes no arithmetic. |

The evaluator's workflow becomes:

```python
metrics = verifylib.run(spec, solver_module, config)   # fixed code
# agent reads metrics, assigns score per rubric, writes feedback
```

Everything in `verification_manual.md` survives. The surface on which it can be miscomputed does not.
Part I and Part II of the manual stop being instructions to an LLM and become the **specification for
`verifylib`** — which is a better use for them, since they already read like one.

---

## 3. The operator declaration **[N, generalizing B]**

The paper-release residual needs a discretized PDE operator, which is why it has hand-written
`_evaluate_burgers`, `_evaluate_gray_scott`, and six siblings. Have the formulator declare it instead,
**term-decomposed** so the balanced denominator is computable.

This is a small step from what this repo already does: `verification.mms_probe.operator_check` is
*already* an evaluable operator expression (`"u_t + adv_u + grad_p - nu*lap_u"`). It is currently
used only to validate an MMS source term. Promote it.

> **Ownership note (guardrails plan §4d / §14 C2).** The `verification.operator` field is **defined
> and required by** [plan-hallucination-guardrails.md](plan-hallucination-guardrails.md), whose
> analytic-reference check needs it first — a user's own source-bearing problem gets no reference
> check without it. This plan **consumes** the declaration for D1 rather than defining it, the same
> way it consumes `verifylib.operator` for stencils. Take the shape from that plan and do not
> re-specify it here; the two must not diverge. The shape below is retained as the reference
> description of what D1 does with it.

### Schema addition — `problem_spec.json → verification.operator`

Scalar:

```json
"operator": {
  "fields": ["u"],
  "terms": {
    "u_t":       "dt(u)",
    "advection": "u*dx(u)",
    "diffusion": "-nu*dxx(u)"
  },
  "source": "0"
}
```

System (one entry per equation):

```json
"operator": {
  "fields": ["u", "v"],
  "equations": {
    "u": {"terms": {"u_t": "dt(u)", "diff": "-D_u*lap(u)",
                    "react": "u*v**2", "feed": "-F*(1-u)"}, "source": "0"},
    "v": {"terms": {"v_t": "dt(v)", "diff": "-D_v*lap(v)",
                    "react": "-u*v**2", "kill": "(F+k)*v"}, "source": "0"}
  },
  "combine": "rms"
}
```

The residual is `sum(terms) - source` per equation; systems combine by RMS across equations.

### Kernel-provided helpers

`dt`, `dx`, `dy`, `dz`, `dxx`, `lap`, `grad`, `div`, `adv`, plus `np` and the spec's `parameters`.
**Also required**, measured against the real specs (guardrails plan §4d): the mixed derivative
`u_xy` (`pde_monge_ampere_2d`), both the `lap_lap_u` and `lap(lap_u)` spellings (Cahn-Hilliard and
Kuramoto-Sivashinsky each use one), and non-Cartesian axis names such as `u_SS` / `u_Sv` / `u_vv`.
These live in the shared `verifylib.operator` (§14 C1), not in a second implementation here.
Three rules govern how the kernel discretizes them:

1. **Higher order than the solver used.** **[A]** — §9 already states this for the steady-state
   residual. Apply it to the time-dependent case too.
2. **A different discretization *family* than the solver's.** **[N]** — read
   `plan.spatial_discretization.scheme`; if the solver was spectral, difference with high-order finite
   differences, and vice versa. This is what prevents the paper-release pipeline's `5.03e-15` viscous-
   Burgers artifact: when the evaluator reuses the solver's own stencil, agreement is a tautology.
3. **Two-snapshot time term.** **[B]** — `dt(u)` is formed from `u(T)` and `u(T-dt)`, requested through
   the existing `override` hook. This is what lets the residual apply to time-dependent problems at
   all; §9 currently restricts residuals to steady-state for exactly the reason this solves.

### SDE side — derived, not authored

No new spec surface. The generator `Lf = f·phi' + 0.5·g^2·phi''` is built from the existing
`drift_expression` and `diffusion_expression`, and drives Dynkin (§20). The formulator supplies only
`dynkin_test_functions`, which it already does.

---

## 4. The ladder

Two structural changes from the current §1 table: **D1 and D2 move out of the scoring tiers into a
gate layer**, and **cross-plan consensus leaves the ladder entirely** (it is a cross-plan comparison,
so it belongs to ranking — see `plan-blind-evaluator.md`).

### 4a. Scoring tiers — what a score is allowed to rest on

| Tier | PDE evidence | SDE evidence | Certifies | Tag |
|---|---|---|---|---|
| **A** | Closed form **quoted from the problem statement** | Exact moments | The answer is right | `analytic` |
| **A'** | — | Moment ODE (`closes_exactly: true`), Kolmogorov solve (truncation guard), stationary density — §15–17 | Right to a controlled tolerance | `surrogate` |
| **B** | MMS (probe numerically validated) or degenerate limit — §5–6 | Degenerate limit (sigma→0 deterministic ODE; linear-coefficient limit with exact moments) | The **scheme and code** are right | `manufactured` |
| **C** | Richardson + GCI + asymptotic guards — §3–4 | CRN ladder + strong/weak order — §18–19 | It converges, and by how much it is off | `self_convergence` |
| — | Nothing runnable | Nothing runnable | — | `none` |

### 4b. Gate layer — runs every cycle, never raises a score, can always sink one **[N]**

| Gate | PDE | SDE |
|---|---|---|
| **D1** — operator residual | Term-balanced residual from `verification.operator`, plus the three guards in §5 | Dynkin identity, Richardson-extrapolated in `dt`, tested against the MC confidence interval of zero — §20 |
| **D2** — structure | Declared invariants with `gate: true`, BC/IC enforcement, positivity, symmetry, divergence — §8 | Positivity, support, martingale, finiteness, estimator stability — §21; MC-resolution — §14 |

### 4c. Certification rule **[A, tightened]**

> **A 10 without a closed form requires two independent circularity breaks.**

B and D1 break the circle in different directions. MMS checks the *discretization* against a known
solution of a *modified* problem; D1 checks the *produced field* against the *stated* operator. A bug
that fools one rarely fools the other, and D1 is nearly free — so requiring both is strictly stronger
than requiring MMS alone at almost no added cost.

| Condition | Max score | Tag |
|---|---|---|
| Tier A or A' clean, gates clean | **10** | `analytic` / `surrogate` |
| Tier B **and** D1 clean, C clean, gates clean | **10** | `manufactured` |
| Tier B **or** D1 alone, C clean, gates clean | **9** | `manufactured_partial` |
| C + gates only | **8** | `self_convergence` |
| Any `gate: true` D2 violation, or D1 failure | **3** | (regardless of everything else) |
| No test runnable | **2** | `none` |

Preserved unchanged from the current manual: a check skipped for cost never counts against a plan; a
faulty MMS probe is skipped, not failed; a missing reference is not a solver crash; never score on
plausibility.

---

## 5. The three residual guards, and the hole they close **[N]**

A residual-only metric has a defect that is live in the paper-release pipeline and worth stating
plainly:

> **A residual rewards any field sitting on a stable steady state of the operator, whether or not it
> is the state the initial condition actually evolves to.**

Gray–Scott is the sharpest case: `u ≡ 1, v ≡ 0` is an exact homogeneous solution. A solver that
over-diffuses and destroys the pattern satisfies every term to near machine precision and earns an
excellent term-balanced residual, having eliminated the only phenomenon the problem is about.

The degenerate case is worse: if `u → 0`, numerator and denominator both vanish and the `1e-12`
epsilon leaves you dividing zero by epsilon — a perfect score for solving nothing.

Guards, all cheap, all computed by the kernel:

```python
nontrivial    = rms(u_T) > triv_tol * rms(u_0)                 # not collapsed
ic_consistent = rel_err(solve_to(t=0), u_0) < 1e-10            # the run started where it should
structural    = all(gate_invariants_pass)                       # D2 — see below
```

**D1 gates but never certifies alone.** The residual tests a *point*; Richardson tests a *trajectory*;
invariants test *structure*. They fail independently, which is why the ladder needs all three.

### Why not residual-only, in one table

Each rung is the unique answer to a specific direction in which a residual is blind. The blindness is
structural — a consequence of testing the PDE rather than the initial-boundary value problem, on a
fixed grid, at a single instant.

| Residual blind spot | Why | Covered by |
|---|---|---|
| No error bound | `‖e‖ ≤ ‖L_h⁻¹‖·‖r‖`; the stability constant is ~`1/eps` for convection-dominated, unbounded near a Helmholtz resonance, growing with `T` for hyperbolic. Estimating it requires refinement. | **C** |
| No calibrated threshold | Residual magnitude depends on operator conditioning, grid, and stencil order. There is no universal passing value. | **C** (require it to *fall* at the design rate) |
| Amplitude / phase error | For any linear homogeneous PDE, `c·u` is also a solution; for translation-invariant problems, so is `u(x-s)`. Residual is **identically zero** on both. Phase lag is the dominant error of every advection scheme. | **D2** invariants (energy), **C** at full `T` |
| Boundary conditions | A stencil needs ghost points, so the residual is interior-only. BC violation lives exactly where it cannot be computed. | **D2** BC check |
| Under-resolution | An upwind scheme solves `u_t + a·u_x = (eps + eps_num)·u_xx`. On a coarse grid the field is smooth, every derivative small and mutually consistent, and the residual small — while the answer is 30% wrong. | **C** |
| Wrong operator in the spec | Solver and residual come from the same declaration; both solve the wrong problem, self-consistently. | **A**, and the requirements ledger — see the guardrails plan |
| SDEs | No pointwise residual exists for a Monte Carlo path solver. | §18–20 |

**Note on short-time Richardson.** An earlier draft of this design offered it as a general guard. It
is not one — shortening the horizon removes precisely the accumulated error you are hunting. It is
correct *only* on a chaotic problem where full-`T` convergence is meaningless (§7 below).

---

## 6. Cost — the merged ladder is cheaper than the current one

Using the existing §24 cost model (each ladder level costs `2^(d+2)`× the one below for an explicit
scheme):

| Path B evaluation | 1-D | 3-D |
|---|---|---|
| **Today:** 3-level ladder + MMS at certification | 73 + 9 = **82** | 1057 + 33 = **1090** |
| **Merged:** 2-level ladder + MMS *supplying* the order | 9 + 9 = **18** | 33 + 33 = **66** |
| Saving | **4.6×** | **16.5×** |

### The Tier-B-measured-order trick **[N]**

Richardson genuinely needs **three** numerical solutions to form two differences; you cannot shorten
it and still *measure* an order. But when a Tier-B probe has already measured the order at two grids,
the main ladder can run **2 levels** and plug the probe-measured order into the GCI:

```python
e_est = d10 / (2**p_probe - 1)      # p from the MMS probe, not from a third grid
```

You still get an error estimate; you have sourced the order from a check you were running anyway.
This removes the top level, which is 87% of a 3-level ladder's cost in 1-D and 97% in 3-D. With no
Tier-B probe available, Path B stays at 3 levels.

**D1 is free.** It needs `u(T)` and `u(T-dt)` on the finest grid already computed — one stored array
and a handful of stencil applications, roughly `1e-3` of a solve.

**Path A is unchanged** (already 2 levels; D1 adds nothing measurable).

---

## 7. Chaotic problems **[N]**

Add a top-level spec flag `"chaotic": true`. When set:

- Tier **C** at full `T` is **waived, not failed** — trajectories separate past the Lyapunov time, so
  the asymptotic guards cannot pass and their failure carries no information about the scheme.
- Tier **C** runs instead at a declared `T_ref` inside the predictable window.
- Tiers **B**, **D1**, **D2** run unchanged at full `T`.
- The waiver is recorded in `<metrics>` as `order_check: waived_chaotic` so it never reads as a pass.

### Worked example — Kuramoto–Sivashinsky

Currently `SELF_ONLY` in `benchmark/results/REPORT.md`. Under this design it certifies.

Formulator emits:

```json
"chaotic": true,
"verification": {
  "operator": {
    "fields": ["u"],
    "terms": {"u_t": "dt(u)", "u_xx": "dxx(u)", "u_xxxx": "dxxxx(u)", "burgers": "u*dx(u)"},
    "source": "0"
  },
  "degenerate_limit": {
    "params": {"nonlinear": 0.0},
    "exact_spectral": "uhat_k(0)*np.exp((k**2 - k**4)*t)",
    "note": "dropping u*u_x leaves a linear PDE, exactly solvable mode by mode"
  },
  "invariants": [
    {"name": "mean_conservation", "gate": true, "tol": 1e-10,
     "note": "every term is a derivative; the periodic mean is exactly conserved"},
    {"name": "energy_balance", "gate": false, "requires_trace": true,
     "identity": "d/dt(0.5*int(u**2)) == int(u_x**2 - u_xx**2)"}
  ]
}
```

Kernel result:

| Tier | Outcome |
|---|---|
| C at full `T` | Guards fail; `chaotic: true` **waives** rather than fails |
| C at `T_ref` | Order 4.0 against the floor, in asymptotic range |
| D1 | Solver is Fourier pseudospectral → kernel differences with 8th-order FD. Residual `~3e-06`, non-trivial, IC-consistent |
| B | Degenerate limit: kill the nonlinearity, compare per-mode against `exp((k²-k⁴)t)` → `~2e-12`. **Tier B achieved** |
| D2 | Mean drift `~4e-15`; energy-balance identity holds across the trace |

**Verdict: 10/10, provenance `manufactured`.** Two independent circularity breaks. The degenerate
limit is three lines of formulator work.

---

## 8. Implementation — file by file

### 8a. New: `verifylib/`

```
verifylib/
├── __init__.py          run(spec, solver, config) -> metrics dict
├── ladder.py            nested grids, refine/detect_periodic/build_ladder/restrict   [A §3]
├── richardson.py        GCI, asymptotic guards, probe-order variant                  [A §4] [N]
├── operator.py          term-decomposed residual, helper stencils, family selection  [B] [N]
├── mms.py               probe driver + numeric probe validation                      [A §5]
├── degenerate.py        degenerate-limit driver                                      [A §6]
├── invariants.py        the §8 catalogue, gate/non-gate                              [A §8]
├── temporal.py          dt_factor isolation                                          [A §10]
├── sde/
│   ├── surrogates.py    moment ODE, Kolmogorov + truncation guard, stationary        [A §15-17]
│   ├── crn.py           common random numbers, strong/weak order                     [A §18-19]
│   ├── dynkin.py        generator construction, dt-Richardson, CI test               [A §20]
│   └── constraints.py   positivity, support, martingale, estimator stability         [A §21]
├── metrics.py           the emitted JSON schema, incl. oracle_derived flags          [N]
└── tests/
    ├── test_analytic_cases.py     known-answer tests for every check
    └── test_canaries.py           seeded-defect suite (see guardrails plan §7)
```

Almost all of this is transcription: `verification_manual.md` already contains the reference
implementations. The work is packaging, not derivation.

### 8b. Modified files

| File | Change |
|---|---|
| `agents/formulator.md` | Add `verification.operator` to Step 3 as **required on every spec**, both paths. Add `chaotic` flag. Require the degenerate limit whenever a parameter setting collapses the problem to something closed-form. |
| `agents/evaluator-pde.md` | Replace "write `evaluate.py`" with "call `verifylib.run`, then diagnose". Keep the whole rubric, feedback-by-failure-mode table, and Key Rules. Delete the inline numerical code. |
| `agents/evaluator-sde.md` | Same. Path A/A'/B logic stays; the arithmetic moves to `verifylib.sde`. |
| `agents/solver-{pde,sde}.md` | Document that `override` must support returning `u(T-dt)` for D1. |
| `references/verification_manual.md` | Reframe as the **specification for `verifylib`**. Add §26 (operator declaration), §27 (D1 guards and the trivial-attractor hole), §28 (chaotic waiver). Move §11/§22 consensus out to the ranking plan. |
| `references/project_manual.md` | Update the score convention with the new certification table; add `manufactured_partial` to the provenance list. |
| `templates/problem_spec-example-pde.json` | Add `verification.operator`. |
| `templates/problem_spec-example-pde-system.json` | Add the multi-equation `operator` form. |
| `templates/problem_spec-example-sde.json` | Note that the generator is derived, not authored. |
| `benchmark/verify.py` | For `has_ground_truth=False` problems, run `verifylib` D1 + D2 instead of returning `SELF_ONLY` outright. New verdict: `STRUCTURALLY_VERIFIED`. |
| `commands/conductor.md` | Parse `provenance: manufactured_partial`; rank it below `manufactured`. |

### 8c. Phasing

| Phase | Deliverable | Gate to proceed |
|---|---|---|
| **1** | `verifylib` ladder + richardson + invariants, PDE only. Evaluator calls it. | Existing 9 PDE problems reproduce their current scores within tolerance |
| **2** | `operator.py` + D1 with all three guards. `verification.operator` required on new specs. | Gray–Scott-style trivial-collapse canary is caught |
| **3** | Tier-B-measured-order trick; ladder drops to 2 levels where a probe exists | Wall time on Path B problems falls measurably |
| **4** | `verifylib.sde` — surrogates, CRN, Dynkin, constraints | Existing 5 SDE problems reproduce their scores |
| **5** | `chaotic` flag; KS moves from `SELF_ONLY` to a real provenance | KS certifies at 10 with `manufactured` |

Phases 1–2 deliver most of the value. Phase 4 is the largest single chunk of work.

---

## 9. Acceptance criteria

The design is working when all of the following hold:

1. No `evaluate.py` in any plan directory contains numerical logic — only a `verifylib` call and
   result interpretation.
2. Every score in `STATE.md` carries a provenance tag, and `manufactured_partial` appears where only
   one circularity break was available.
3. The seeded-defect canaries (guardrails plan §7) are all caught, including trivial collapse and
   phase lag.
4. Kuramoto–Sivashinsky reports a provenance other than `none`.
5. Path B wall time is **lower** than the current 3-level default on problems with a Tier-B probe.
6. `benchmark/verify.py` returns something other than `SELF_ONLY` for at least one no-ground-truth
   problem.

---

## 10. Risks and open questions

| Risk | Mitigation |
|---|---|
| The formulator writes a wrong `verification.operator`, making solver and residual self-consistently wrong | Validate the operator against the MMS probe: applying it to `u_mms` must reproduce `source`. This reuses the existing `operator_check` machinery and catches an inconsistent declaration. Does **not** catch a spec that is wrong in the same way as the problem statement was misread — that is the requirements ledger's job. |
| `verifylib` becomes the new single point of failure | It is fixed code under test with a canary suite. That is a strictly better failure mode than regenerated code with no test at all. |
| "Different discretization family" is ambiguous for hybrid schemes (IMEX spectral-FD) | Default to high-order FD unless the solver is *purely* FD, in which case use spectral differentiation on a periodic grid or a higher-order compact stencil otherwise. Record the choice in the metrics JSON. |
| Degenerate limits do not exist for every problem | Then Tier B is unavailable and the plan tops out at 9 with `manufactured_partial` or 8 with `self_convergence`. That is the honest outcome, not a defect. |
| Extra spec surface raises formulator burden | `operator` is mechanical once `governing_equation` exists, and it replaces prose in `implementation_notes` that was carrying the same information unusably. |

---

## 11. Provenance summary

| Element | Source |
|---|---|
| Ladder of evidence, tiers A/A'/B/C, provenance tags, score ceilings | **[A]** §1, §12, §23 |
| MMS with numeric probe validation; faulty probe skipped not failed | **[A]** §5 |
| Degenerate limit as Tier-B evidence | **[A]** §6 |
| Nested ladder, Richardson, GCI, asymptotic guards | **[A]** §3–4 |
| Invariant catalogue, gate/non-gate distinction | **[A]** §8 |
| Temporal-error isolation | **[A]** §10 |
| SDE surrogates, CRN, weak/strong order, Dynkin, constraints | **[A]** §15–21 |
| Cost discipline, conditional Stage 2 gates | **[A]** §24 |
| Term-balanced residual and its normalization | **[B]** |
| Candidate-reported residual structurally demoted | **[B]** |
| Two-snapshot `dt(u)` for time-dependent residuals | **[B]** |
| Deterministic kernel / agent split | **[N]** |
| `verification.operator` term-decomposed declaration | **[N]**, generalizing **[B]** |
| Different-discretization-family rule | **[N]** |
| Trivial-attractor hole and the three D1 guards | **[N]** |
| D1 as a gate rather than a report | **[N]** |
| Two-circularity-breaks certification rule | **[N]**, tightening **[A]** |
| Tier-B-measured-order trick (2-level Path B ladder) | **[N]** |
| `chaotic` flag and the C waiver | **[N]** |
