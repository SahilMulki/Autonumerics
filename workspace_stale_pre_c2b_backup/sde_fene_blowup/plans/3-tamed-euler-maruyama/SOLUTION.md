---
id: 3
plan_slug: tamed-euler-maruyama
scheme: euler-maruyama
strategy: Explicit tamed Euler-Maruyama with reflection — bound the singular drift displacement per step and reflect any residual overshoot back into (-1, 1), giving a cheap fully-explicit vectorized scheme.
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

- **Method**: Tamed Euler-Maruyama (explicit) with boundary reflection
- **dt**: 0.005
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: none (additive noise, dg/dX = 0)

**Tamed drift.** The blow-up in a naive EM step comes from the unbounded drift
`f(X) = -X/(1 - X**2)`. Taming caps the per-step drift displacement so it can never dominate:

```
f_tamed(X) = f(X) / (1 + dt * |f(X)|)
```

so `|f_tamed(X)| * dt < 1` always. The update is fully explicit:

```
X* = X_n + f_tamed(X_n) * dt + dW_n,     dW_n = sqrt(dt) * Z_n,  Z_n ~ N(0, 1)
```

Taming is the standard convergent explicit method for SDEs with non-globally-Lipschitz
(here, singular) drift: it preserves strong/weak convergence while preventing the numerical
explosion that plain EM suffers.

**Reflection guard.** Taming bounds the drift, but a large additive `dW_n` can still push `X*`
outside (-1, 1). Reflect any overshoot back into the domain (specular reflection about the wall):

```
# reflect across +1:  X = 2 - X   when X > 1
# reflect across -1:  X = -2 - X  when X < -1
X = where(X* >  1 - eps,  2*(1 - eps) - X*,  X*)
X = where(X  < -1 + eps, -2*(1 - eps) - X,   X)
```

Apply reflection iteratively (a small `while any(|X| >= 1)` loop, typically 1 pass) so a single
huge increment that reflects past the opposite wall is folded back in. As a hard backstop, clamp
to `[-1 + eps, 1 - eps]`, `eps = 1e-12`. A slightly smaller `dt = 0.005` reduces how often
reflection fires and keeps the tamed drift close to the true drift away from the wall.

## Implementation Notes

- **Stability benchmark, not accuracy.** No analytic mean/variance
  (`analytic_moments.has_analytic_solution = false`). Success = every terminal `|X(T)| < 1` and
  finite. Report empirical mean and variance of X(T) at T = 1 for reference only.
- **Fully explicit and cheapest per step.** No substep refinement (Plan 1) and no nonlinear
  solve (Plan 2) — just a vectorized tamed update plus a reflection mask. This is the fastest
  scheme but the most approximate near the wall, so it depends on `dt` being small enough and on
  the reflection guard to enforce the domain.
- **Reflection is a modeling approximation.** Specular reflection at +-1 imposes an effective
  reflecting boundary, whereas the true FENE process is confined purely by the diverging drift.
  For this stability benchmark (domain + finiteness check, no analytic moments) that is acceptable;
  keeping `dt` small limits how much reflection perturbs the interior statistics.
- **Vectorization.** `X` has shape `(num_paths,)`; the tamed update, the reflection `where`
  masks, and the final clamp all broadcast across paths. The reflection `while` loop runs at most
  a couple of iterations.
- **Divergence detection.** `f(X)` divides by `1 - X**2`; because every step ends inside
  `[-1+eps, 1-eps]`, the denominator at the start of the next step is bounded away from zero.

## Results

Solver: tamed Euler-Maruyama with iterative specular reflection and hard clamp, run with
`num_paths=50000`, `dt=0.005`, `T=1.0`, `seed=42`.

```
Empirical mean:     -0.002022
Empirical variance:  0.192640
dt used:              0.005000
num_paths:            50000
All finite:           True
Max |X(T)|:           0.999952142314
All |X(T)| < 1:       True
```

Every terminal path is finite and strictly inside (-1, 1) (max |X(T)| ≈ 0.99995, comfortably
below the domain boundary and the eps=1e-12 clamp). No analytic solution exists for this
family, so mean/variance are reported for reference only — this is a stability benchmark
(finiteness + domain containment), which the solver passes for all 50000 paths.

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Stability Benchmark (no analytic moments — problem_spec.json marks `has_analytic_solution: false`)
- All finite:            True   (PASS)
- Max |X(T)|:             0.999952142314
- All |X(T)| < 1:        True   (PASS) — 0 / 50000 domain violations
- Overall: PASS ✓

### Reference-only moments (no exact values to compare against)
- Empirical mean:     -0.002022
- Empirical variance:  0.192640

### Feedback for solver
- None. `evaluate.py` re-ran `solve_sde(num_paths=50000, dt=0.005, T=1.0, seed=42)` independently
  and confirms every one of the 50000 terminal paths is finite and strictly inside (-1, 1)
  (max |X(T)| ≈ 0.999952, closest approach to the wall is ~4.8e-5 below the boundary). The
  tamed drift + iterative specular reflection + hard clamp fully satisfies the stability
  criterion given in problem_spec.json (`stability_check`: all terminal states finite and
  |X(T)| < 1). This is the terminal condition for this stability benchmark.

</review>
