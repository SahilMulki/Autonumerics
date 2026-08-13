---
id: 2
plan_slug: milstein
scheme: milstein
strategy: Tamed Milstein — tame the superlinear -X^5 drift for finiteness while adding the Milstein diffusion correction to reach strong order 1.0 on the multiplicative X dW term.
---

## SDE Reference

Ito SDE (scalar, multiplicative noise):

    dX(t) = -X(t)**5 * dt + X(t) * dW(t),    X(0) = 1.0

over t in [0, 1].

- **Drift**:     f(X) = -X**5
- **Diffusion**: g(X) = X,   dg/dX = 1
- **Parameters**: X_0 = 1.0
- **Time interval**: T0 = 0.0, T = 1.0

No closed-form solution exists. This is a **stability** benchmark: the independent
check confirms only that terminal states X(T) remain finite (no Inf/NaN).

## Numerical Scheme

- **Method**: Milstein with **drift taming** (mandatory here)
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: `0.5 * g(X) * g'(X) * (dW**2 - dt) = 0.5 * X * (dW**2 - dt)`
  (since g(X) = X and dg/dX = 1)

**Tamed update** (X scalar, vectorize over all paths):
```
dW      = sqrt(dt) * Z,   Z ~ N(0,1)
drift   = -X**5
tamed   = drift * dt / (1.0 + dt * abs(drift))     # taming: bounds the drift step
X_next  = X + tamed + X * dW + 0.5 * X * (dW**2 - dt)
```

The taming factor `1 / (1 + dt*|drift|)` keeps the run finite; the Milstein term
`0.5 * X * (dW**2 - dt)` improves the strong order of the multiplicative diffusion.

## Implementation Notes

- **CRITICAL — the Milstein correction does NOT prevent divergence.** Taming addresses
  the drift blow-up, which is the actual failure mode (Hutzenthaler-Jentzen-Kloeden).
  The Milstein term only refines the diffusion (multiplicative-noise) order. You MUST
  still tame the drift increment, exactly as in the Euler-Maruyama plan.
- Only the **drift** is tamed; the diffusion and Milstein terms are left as-is.
- No positivity guard is needed (g(X) = X has no sqrt/log); X may go negative.
- After the run, sanity-check that `np.all(np.isfinite(terminal_paths))` is True.
- Report `terminal_paths` (shape (num_paths,)), `empirical_mean`, `empirical_variance`
  of X(T) at T = 1.

## Results

Ran `solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)`:

- All finite: True (`np.all(np.isfinite(terminal_paths))` confirmed)
- Empirical mean:     0.464741
- Empirical variance: 0.088352
- num_paths: 50000
- dt: 0.01
- T: 1.0

Stability confirmed: the tamed drift increment `f(X)*dt / (1 + dt*|f(X)|)`
kept all 50000 paths finite over the full run, and the Milstein correction
`0.5 * X * (dW**2 - dt)` was applied on top of it for the multiplicative
diffusion term. Since this problem has no closed-form solution
(`has_analytic_solution: false`), no mean/variance error check against an
analytic reference applies — only the finiteness check is meaningful.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy
- This problem has `has_analytic_solution: false` in problem_spec.json (no closed-form
  mean/variance) — it is a **stability** benchmark, not a moment-matching benchmark.
  The standard mean_rel_err / var_rel_err thresholds do not apply; pass/fail is
  determined by finiteness of the terminal states (mirrors
  `benchmark/verify.py::verify_sde_stability` with `stability_check = {"type": "finite"}`
  for this exact problem slug).
- All 50000 terminal paths finite: True (fraction finite = 1.000000).
- Reported empirical mean/variance finite: True.
- Empirical mean:     0.464741
- Empirical variance: 0.088352
- Diagnostic refinement check (re-ran at dt/2 = 0.005, same seed): mean = 0.462555,
  variance = 0.087088 — both remain finite and close to the dt=0.01 values, confirming
  the tamed-drift scheme is stable under time-step refinement (not just "finite by
  luck" at this particular dt).
- Cross-check against plan 1-euler-maruyama (also tamed, no Milstein correction):
  mean=0.463570, variance=0.089653 at the same dt/num_paths/seed — closely consistent
  with this plan's mean=0.464741, variance=0.088352, as expected since both use the
  same drift-taming approach and Milstein only refines the diffusion order.
- Overall: PASS ✓

### Feedback for solver
None.

</review>
