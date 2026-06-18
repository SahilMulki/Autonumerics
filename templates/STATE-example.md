# STATE.md Example

The actual file contains only a YAML frontmatter block. Fields grow as the pipeline progresses.

## init

```yaml
phase: init
```

## running — SDE example (after formulator + plan-creator-sde return)

```yaml
phase: running
problem_type: sde
problem_spec: workspace/{problem_slug}/problem_spec.json
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
