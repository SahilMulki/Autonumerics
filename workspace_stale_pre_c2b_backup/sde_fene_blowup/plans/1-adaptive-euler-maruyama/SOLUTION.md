---
id: 1
plan_slug: adaptive-euler-maruyama
scheme: euler-maruyama
strategy: Explicit Euler-Maruyama with boundary-aware adaptive substepping — refine the time step near |X|=1 so the drift displacement never overshoots the wall.
---

## SDE Reference

FENE (finitely-extensible nonlinear elastic) Ito SDE on the open interval (-1, 1):

```
dX(t) = -X(t) / (1 - X(t)**2) * dt + dW(t),    X(0) = 0.0
```

- Drift:      f(X) = -X / (1 - X**2)   (diverges as |X| -> 1, confines X to (-1, 1))
- Diffusion:  g(X) = 1                 (additive unit noise)
- Domain:     (-1, 1), open interval; |X(t)| < 1 for all t
- **Parameters**: X_0 = 0.0, domain_lower = -1.0, domain_upper = 1.0
- **Time interval**: T0 = 0.0, T = 1.0

Additive noise => dg/dX = 0, so Milstein reduces to Euler-Maruyama. No closed-form solution.

## Numerical Scheme

- **Method**: Euler-Maruyama with boundary-aware adaptive substepping
- **dt** (macro step): 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: none (additive noise, dg/dX = 0)

**Idea.** A fixed-step EM near the boundary takes a step `f(X)*dt` whose magnitude explodes as
`|X| -> 1`; the update overshoots past +-1, the drift flips sign, and the path diverges to
+-Inf / NaN. The fix is to shrink the step whenever a path gets close to the wall so that the
deterministic drift displacement stays a small fraction of the remaining distance to the boundary.

**Per macro-step (size dt) procedure, vectorized over all paths:**

1. Draw the macro Brownian increment `dW = sqrt(dt) * Z`, `Z ~ N(0, I)` for the population.
2. Integrate the macro interval with `k` equal substeps of size `h = dt / k`. Split the increment
   consistently: draw `k` sub-increments `dW_j = sqrt(h) * Z_j` with `Z_j ~ N(0, I)` (each substep
   gets a fresh normal; the sum over a macro interval is still `N(0, dt)`, so the Brownian statistics
   are preserved). Advance every substep with plain EM: `X <- X + f(X)*h + dW_j`.
3. Choose the substep count adaptively from a CFL-like wall condition. Before each substep compute
   the safe local step from the distance to the wall:
   ```
   dist   = 1 - |X|                       # distance to nearest boundary
   h_safe = c * dist**2 / |X|             # since |f| = |X| / (1 - |X|)(1 + |X|) ~ |X| / (2*dist)
   ```
   With a safety factor `c` (e.g. c = 0.1) this keeps `|f(X)| * h << dist`, so no substep can cross
   the wall. Use `k = ceil(dt / min_over_paths(h_safe))` substeps for the whole population (simple,
   vectorized), or bin paths by required refinement for efficiency. Cap `k` (e.g. k_max = 4096) and
   floor `h` to avoid pathological blow-up of the substep count.
4. After each substep, apply a hard safety clamp `X = clip(X, -1 + eps, 1 - eps)` with
   `eps = 1e-12` as a last-resort guard against floating-point overshoot; with correct adaptive
   `h` this clamp should almost never activate.

## Implementation Notes

- **Stability benchmark, not accuracy.** There is no analytic mean/variance
  (`analytic_moments.has_analytic_solution = false`). Success criterion: every terminal state
  satisfies `|X(T)| < 1` and is finite (no NaN/Inf). Report empirical mean and variance of X(T)
  at T = 1 for reference, but the pass/fail is the domain/finiteness check.
- **Vectorization.** Keep `X` as shape `(num_paths,)`. The substep loop runs over time; all paths
  advance together with NumPy broadcasting. A global (population-wide) `k` per macro step is the
  simplest correct implementation; a per-path adaptive `k` is more efficient but harder to
  vectorize — start global.
- **Consistency of the split increment.** Re-drawing `k` sub-increments of variance `h` each is
  statistically exact for the Brownian motion (independent increments summing to variance `dt`).
  Do NOT reuse the macro `dW` scaled down — draw fresh sub-normals.
- **Divergence detection.** `f(X)` divides by `1 - X**2`; guard with the clamp above so the
  denominator never hits zero. Watch for any `X` reaching `|X| >= 1` mid-run; if the adaptive
  `k` is chosen correctly this cannot happen.
- **Cost.** Per-step cost scales with `k`; near the wall `k` grows, so this is the most expensive
  of the three plans, but it stays fully explicit and needs no nonlinear solve.

## Results

(Filled by solver after running)

## Evaluation

(Filled by evaluator)

<review score=0>

Awaiting solver.

</review>
