```yaml
phase: done
problem_type: sde
problem_spec: workspace/sde_ou_stiff/problem_spec.json
plans:
  1-euler-maruyama:
    one-sentence: Exact OU transition kernel (zero discretization bias), 50k paths, seed=42, T=1.0; mean check skipped (exact mean ~0), variance rel err 0.90%.
    iter: 2
    score: 10
best_plan: 1-euler-maruyama
```
