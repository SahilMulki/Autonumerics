---
id: 2
plan_slug: drift-implicit-euler-maruyama
scheme: euler-maruyama
strategy: Semi-implicit (drift-implicit) Euler-Maruyama — treat the singular drift implicitly, reducing each step to a cubic whose in-domain root is guaranteed to lie in (-1, 1), so paths can never blow up.
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

- **Method**: Drift-implicit (semi-implicit) Euler-Maruyama. The drift is evaluated at the new
  point; the additive noise stays explicit (this is the standard/optimal treatment because
  additive noise has no Ito correction).
- **dt**: 0.01
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: none (additive noise, dg/dX = 0)

**Update rule.** With the drift implicit:

```
X_{n+1} = X_n + f(X_{n+1}) * dt + dW_n,     f(Y) = -Y / (1 - Y**2)
```

Let `b = X_n + dW_n` (all known at step n) and `Y = X_{n+1}`. Then

```
Y - dt * Y / (1 - Y**2) = b
```

Multiply through by `(1 - Y**2)` and collect terms:

```
Y**3 - b*Y**2 - (1 + dt)*Y + b = 0            (cubic in Y)
```

**Why this is unconditionally in-domain.** Define `P(Y) = Y**3 - b*Y**2 - (1+dt)*Y + b`. Then

```
P(-1) = -1 - b + (1 + dt) + b = +dt  > 0
P(+1) =  1 - b - (1 + dt) + b = -dt  < 0
```

so for every real `b` and every `dt > 0` there is a root strictly inside (-1, 1). Selecting that
root gives `X_{n+1}` in the open domain by construction — the scheme **cannot** overshoot the wall,
regardless of how large `dW_n` is or how close `X_n` sits to +-1. This is the classic robust
treatment for FENE dumbbell / nonlinear-spring models.

**Solving the cubic (vectorized over paths):**

- Preferred: closed-form / trigonometric (Cardano) root, then select the unique root in (-1, 1).
- Or: a few Newton iterations `Y <- Y - P(Y)/P'(Y)` with `P'(Y) = 3*Y**2 - 2*b*Y - (1+dt)`,
  seeded at `Y0 = X_n` (or `Y0 = b` clipped into (-1+eps, 1-eps)). Newton converges quadratically
  here; 3-5 iterations at `dt = 0.01` is ample. Because the target root is bracketed by
  `P(-1) > 0 > P(+1)`, fall back to a vectorized bisection on `[-1+eps, 1-eps]` for any path whose
  Newton step leaves the bracket, to guarantee convergence to the in-domain root.
- Clamp the final `Y` to `[-1 + eps, 1 - eps]`, `eps = 1e-12`, purely as a floating-point guard.

## Implementation Notes

- **Stability benchmark, not accuracy.** No analytic mean/variance
  (`analytic_moments.has_analytic_solution = false`). Success = every terminal `|X(T)| < 1` and
  finite. Report empirical mean and variance of X(T) at T = 1 for reference only.
- **Root selection is critical.** The cubic can have up to three real roots; pick the one in
  (-1, 1). With Newton seeded at `X_n` (which is already in-domain and close to the answer for
  small `dt`) you land on the correct root directly. Verify the bracket `P(-1) > 0 > P(+1)` holds
  numerically before trusting a root.
- **Vectorization.** `X` has shape `(num_paths,)`; solve all paths' cubics simultaneously with
  NumPy array ops. Newton + a masked bisection fallback both vectorize cleanly.
- **Largest stable dt.** Because the scheme is unconditionally in-domain, `dt = 0.01` is safe and
  could even be coarsened; keep 0.01 to match the population size / accuracy expectations. The
  solver may increase dt if the evaluator is satisfied and speed matters.
- **Cost.** A handful of vectorized Newton iterations per step — cheaper than deep adaptive
  substepping (Plan 1) near the wall, at the price of the nonlinear solve each step.

## Results

Ran `solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)` using the drift-implicit
Euler-Maruyama scheme (vectorized bisection on the bracket `[-1+eps, 1-eps]`, 60
iterations per step, to solve the cubic `Y**3 - b*Y**2 - (1+dt)*Y + b = 0` for the
in-domain root each step).

- **Empirical mean**: 0.003699
- **Empirical variance**: 0.203649
- **All terminal states finite**: True
- **All terminal states satisfy |X(T)| < 1**: True (max |X(T)| = 0.981009644159)
- **Nt (steps)**: 100, **dt**: 0.01, **runtime**: ~4.3s for 50000 paths

This is a stability benchmark (no analytic solution): the scheme is unconditionally
in-domain by construction (P(-1) = +dt > 0, P(+1) = -dt < 0 for every step), so every
path stays strictly inside (-1, 1) regardless of step size or noise realization. The
empirical mean is close to 0 (consistent with the symmetric drift and X_0 = 0), and the
empirical variance (~0.20) is below the free-diffusion value of t=1.0, reflecting the
confining effect of the drift near the boundaries.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Stability Benchmark Results
This problem has no analytic moments (`analytic_moments.has_analytic_solution = false`);
the pass criterion from `problem_spec.json -> evaluation_thresholds.stability_check` is
finiteness and `|X(T)| < 1` for every terminal path, evaluated with
`solve_sde(num_paths=50000, dt=0.01, T=1.0, seed=42)`.

- All terminal states finite: True (0 / 50000 non-finite)
- All terminal states satisfy |X(T)| < 1: True (max |X(T)| = 0.981009644159)
- Overall: PASS

### Reference-only moments (no exact value to compare against)
- Empirical mean:     0.003699
- Empirical variance: 0.203649

### Feedback for solver
None. The drift-implicit bisection scheme is unconditionally in-domain by construction
(`P(-1) = +dt > 0`, `P(+1) = -dt < 0` for every step and every real `b`), which the
evaluation run confirms empirically: every one of the 50000 paths stayed strictly inside
(-1, 1) with comfortable margin (max |X(T)| ≈ 0.981, well clear of the 1.0 boundary and
clamp eps). This matches plan 3 (tamed EM), which also passes the stability check.

</review>
