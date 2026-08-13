```yaml
phase: running
problem_type: sde
problem_spec: workspace/sde_fene_blowup/problem_spec.json
plans:
  1-adaptive-euler-maruyama:
    one-sentence: Explicit Euler-Maruyama with boundary-aware adaptive substepping (wall condition shrinks dt near |X|=1); macro dt=0.01, 50000 paths.
    iter: 0
    score: 0
    state: await_solver
  2-drift-implicit-euler-maruyama:
    one-sentence: Drift-implicit (semi-implicit) EM solving a cubic per step via vectorized Newton with bisection fallback; root selection guarantees |X|<1 unconditionally, dt=0.01, 50000 paths.
    iter: 1
    score: 10
    state: await_solver
  3-tamed-euler-maruyama:
    one-sentence: Explicit tamed EM (f/(1+dt|f|)) with specular reflection at ±1 for residual overshoot; dt=0.005, 50000 paths.
    iter: 1
    score: 10
    state: await_solver
```
