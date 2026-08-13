---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Tamed Euler-Maruyama — divide the drift increment by (1 + dt*|drift|) so the superlinear -X^5 drift cannot blow up, keeping terminal states finite.
---

## SDE Reference

Ito SDE (scalar, multiplicative noise):

    dX(t) = -X(t)**5 * dt + X(t) * dW(t),    X(0) = 1.0

over t in [0, 1].

- **Drift**:     f(X) = -X**5
- **Diffusion**: g(X) = X
- **Parameters**: X_0 = 1.0
- **Time interval**: T0 = 0.0, T = 1.0

No closed-form solution exists. This is a **stability** benchmark: the independent
check confirms only that terminal states X(T) remain finite (no Inf/NaN).

## Numerical Scheme

- **Method**: Euler-Maruyama with **drift taming** (mandatory here)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42

**Tamed update** (X scalar, one path element shown; vectorize over all paths):
```
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
drift   = -X**5
tamed   = drift * dt / (1.0 + dt * abs(drift))     # taming: bounds the drift step
X_next  = X + tamed + X * dW
```

The taming factor `1 / (1 + dt*|drift|)` caps the per-step drift increment at
magnitude ~1, preventing the moment explosion that plain (untamed) Euler-Maruyama
suffers under the degree-5 superlinear drift.

## Implementation Notes

- **CRITICAL — do NOT use plain Euler-Maruyama.** By the Hutzenthaler-Jentzen-Kloeden
  divergence theorem, untamed EM diverges for this superlinear drift: moments blow up
  and floating point returns Inf/NaN. Taming the drift increment is required for the
  run to stay finite.
- Only the **drift** needs taming — that is the actual failure mode. The diffusion
  term `X * dW` is left as-is.
- No positivity guard is needed (g(X) = X has no sqrt/log); X may go negative and that
  is fine.
- After the run, sanity-check that `np.all(np.isfinite(terminal_paths))` is True. If any
  Inf/NaN appears, taming was not applied correctly or dt is too large.
- Report `terminal_paths` (shape (num_paths,)), `empirical_mean`, `empirical_variance`
  of X(T) at T = 1.

## Results

Ran tamed Euler-Maruyama with num_paths=50000, dt=0.01, T=1.0, seed=42:

- **Empirical mean**:     0.463570
- **Empirical variance**: 0.089653
- **All terminal states finite**: True (no Inf/NaN observed)

This confirms the taming factor successfully prevents the moment blow-up that
plain Euler-Maruyama would suffer under the degree-5 superlinear drift.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

### Numerical Accuracy

This problem has `has_analytic_solution: false` (quintic superlinear-drift
stability benchmark) — there is no closed-form mean/variance to compute a
relative error against. Per `problem_spec.json → implementation_notes`, the
independent check instead verifies finiteness and stability of the tamed
scheme. `evaluate.py` ran `solve_sde` three times (num_paths=50000, T=1.0,
seed=42) at dt=0.01 (SOLUTION.md's stated hyperparameters), a coarser stress
dt=0.05, and a finer dt=0.002:

| dt    | all finite | mean     | variance |
|-------|-----------|----------|----------|
| 0.01  | True      | 0.463570 | 0.089653 |
| 0.05  | True      | 0.443759 | 0.098622 |
| 0.002 | True      | 0.465258 | 0.088267 |

- Finiteness: PASS — no Inf/NaN at any tested dt, including the 5x coarser
  stress test (a scheme that only avoided blow-up by luck at dt=0.01 would
  likely fail at dt=0.05).
- Sanity bound (|mean| < 50, var < 2500): PASS — moments are small and
  physically reasonable, not "finite but enormous" near-blow-up values.
- Convergence consistency (dt=0.01 vs dt=0.002): PASS — mean_diff=0.0017,
  var_diff=0.0014, both well under the 0.1 tolerance, consistent with the
  tamed EM scheme's expected weak convergence as dt shrinks.
- Overall: PASS

### Feedback for solver

None.

</review>
