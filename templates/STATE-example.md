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
    state: await_solver
  2-milstein:
    one-sentence: Milstein with dt=0.01, 50k paths; eligible because noise is multiplicative scalar.
    iter: 1
    score: 7
    state: await_evaluator
```

## running — PDE example (after formulator + plan-creator-pde return)

```yaml
phase: running
problem_type: pde
problem_spec: workspace/{problem_slug}/problem_spec.json
plans:
  1-fd-explicit:
    one-sentence: FTCS finite difference, Nx=64, dt satisfying CFL dt<=dx^2/(2*alpha).
    iter: 1
    score: 8
    state: await_solver
  2-crank-nicolson:
    one-sentence: Crank-Nicolson implicit, Nx=64, dt=0.005, unconditionally stable.
    iter: 0
    score: 0
    state: await_solver
  3-spectral:
    one-sentence: Spectral method via FFT for periodic extension of sine IC.
    iter: 0
    score: 0
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
