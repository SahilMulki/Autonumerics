# STATE.md Example

The actual file contains only a YAML frontmatter block. Fields grow as the pipeline progresses.

## init

```yaml
phase: init
```

## blocked — the requirements gate stopped the run (terminal)

Written when the formulator's `requirements` ledger is missing/empty, has a `dropped`
entry, or has a `mapped` entry with no `spec_path`. No plans are created.

`blocked_reason` is a single line (the harness parses it): name the failing entry
ids, quote them, and say what was missing.

```yaml
phase: blocked
problem_spec: workspace/{problem_slug}/problem_spec.json
blocked_reason: R7 dropped — "the independent check confirms only that the terminal states stay finite (no blow-up)" — no field in evaluation_thresholds carries a stability criterion, so the spec has no scoreable pass condition.
```

## running — SDE example (after formulator + plan-creator-sde return)

`ambiguities` is present only when the formulator marked one or more requirements
`ambiguous`; the conductor carries it into REPORT.md.

```yaml
phase: running
problem_type: sde
problem_spec: workspace/{problem_slug}/problem_spec.json
ambiguities:
  - R3: 'problem.md states no condition at v = v_max; used the standard linear
      boundary V_vv = 0 (In ''t Hout & Foulon 2010)'
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama with dt=0.01, 50k paths.
    iter: 2
    score: 5
    provenance: analytic
    est_err: 1.4e-01
    state: await_solver
  2-milstein:
    one-sentence: Milstein with dt=0.01, 50k paths; eligible because noise is multiplicative scalar.
    iter: 1
    score: 7
    provenance: analytic
    est_err: 8.2e-02
    state: await_evaluator
```

## running — PDE example (after formulator + plan-creator-pde return)

```yaml
phase: running
problem_type: pde
problem_spec: workspace/{problem_slug}/problem_spec.json
plans:
  1-fd-explicit:
    one-sentence: FTCS finite difference, dt satisfying CFL dt<=dx^2/(2*alpha).
    iter: 1
    score: 8
    provenance: analytic
    est_err: 3.1e-02
    state: await_solver
  2-crank-nicolson:
    one-sentence: Crank-Nicolson implicit, dt tied to dx, unconditionally stable.
    iter: 0
    score: 0
    provenance: null
    est_err: null
    state: await_solver
  3-spectral:
    one-sentence: Spectral method via FFT for periodic extension of sine IC.
    iter: 0
    score: 0
    provenance: null
    est_err: null
    state: await_solver
```

`provenance` and `est_err` come from the `<metrics>` block the evaluator writes. They are `null`
until a plan has been evaluated at least once. The conductor uses `est_err` to break score ties in
Phase 3, and carries `provenance` into REPORT.md.

## running — Path B (no closed form) additions

`path: B` is recorded in Phase 1 when the spec's closed-form field is null. The winner early-exit
then waits for two evaluated plans of different `scheme_family`, and the agreement gate
(`cli.py compare`) records its outcome per plan: `agreement_cap: 8` on both plans of a pair that
disagree at full `T` (the conductor ranks on `min(score, agreement_cap)` until a refine cycle or a
third family breaks the tie), `agreed_with: <plan>` on a pair of different families that agree.
`single_family_pool: true` records that the second break could not be earned. In the example
below the spectral and FD4 plans disagreed by 7.5% at the graded grid, so the FD4 plan's 10 does
not make it the winner: both carry `agreement_cap: 8` until a third family or a refine cycle
settles which one is wrong.

```yaml
phase: running
problem_type: pde
path: B
plans:
  1-spectral-etdrk4:
    one-sentence: Fourier pseudospectral, ETDRK4 in time.
    iter: 1
    score: 9
    provenance: manufactured_partial
    est_err: 1.4e-10
    agreement_cap: 8
    state: await_solver
  3-fd4-imex-cnab2:
    one-sentence: 4th-order central FD, IMEX CNAB2 in time.
    iter: 1
    score: 10
    provenance: manufactured
    est_err: 1.5e-05
    agreement_cap: 8
    state: await_solver
```

Plan state values:
- `await_evaluator` — solver has written and run code, waiting for evaluator review
- `await_solver`    — evaluator has returned a score < 10, waiting for solver to refine
- `stopped`         — terminal: the plan plateaued (completed >= 2 cycles and its most recent score did not improve on the previous one); it receives no further cycles

## done

```yaml
phase: done
problem_type: sde
problem_spec: workspace/{problem_slug}/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama with dt=0.01, 50k paths.
    iter: 4
    score: 7
  2-milstein:
    one-sentence: Milstein with dt=0.01, 50k paths.
    iter: 2
    score: 10
best_plan: 2-milstein
```
