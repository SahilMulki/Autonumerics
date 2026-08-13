---
id: 2
plan_slug: euler-maruyama-fine
scheme: euler-maruyama
strategy: Fine-step vector Euler-Maruyama with dt=0.001 to suppress the O(dt) multiplicative-noise variance bias in this stiff, strong-noise system.
---

## SDE Reference

2D linear Ito system driven by m = 13 independent Brownian motions (multiplicative noise):

    dX = F @ X * dt + sum_{r=1}^{13} G_r @ X * dW_r,   X(0) = (1, 1)^T

Parameters:
- X_0 = (1.0, 1.0)^T
- F = [[-2.0, 3.0], [-3.0, -2.0]]   (eigenvalues -2 +/- 3i, spiral decay, |lambda| ~ 3.6)
- g_scale = 0.6
- M1 = [[0, 1], [0, 0]],  M2 = [[0, 0], [1, 0]]   ([M1, M2] != 0, non-commuting)
- G_r = 0.6*M1 for r odd (7 channels),  G_r = 0.6*M2 for r even (6 channels)
- T = 1.0

Exact moments at T=1 (reference):
- mean_X = -0.1148824,  mean_Y = -0.1530794
- variance_X = 0.1715989,  variance_Y = 0.1627163
- (both |mean| > 0.01, so the mean check applies to both components)

Exact mean: `mean = expm(F*T) @ X0`.
Exact variance: second-moment vector `p = (P11, P12, P22) = expm(A*T) @ (1,1,1)` with
`A = [[-4, 6, 2.52], [-3, -4, 3], [2.16, -6, -4]]`, then `Var_X = P11 - mean_X^2`, `Var_Y = P22 - mean_Y^2`.

## Numerical Scheme

- **Method**: Euler-Maruyama (vector form)
- **dt**: 0.001
- **num_paths**: 50000
- **T**: 1.0
- **seed**: 42
- **Milstein correction**: N/A. Multi-dimensional Milstein requires the Levy areas
  (iterated integrals int int dW_i dW_j) because the diffusion generators do not commute
  ([M1, M2] != 0). Milstein is not applicable here; use Euler-Maruyama.

Per-path EM step (state X shape (num_paths, 2)):
```
dX_drift = (X @ F.T) * dt
dX_noise = sum_r 0.6 * (X @ M_r.T) * dW_r
X = X + dX_drift + dX_noise
```

## Implementation Notes

- **Channel aggregation (recommended, exact in law and much cheaper)**: the 7 odd channels
  are identical (G = 0.6*M1) and the 6 even channels are identical (G = 0.6*M2). Draw two
  independent standard normals per step per path, Z_odd and Z_even, and form the aggregate
  noise as
  `0.6 * (M1 @ X) * sqrt(7) * sqrt(dt) * Z_odd + 0.6 * (M2 @ X) * sqrt(6) * sqrt(dt) * Z_even`.
  This is exactly equivalent to summing the 13 independent channels (sum of k i.i.d.
  N(0,dt) increments = N(0, k*dt)). Drawing all 13 explicitly gives the same law.
- With M1 = [[0,1],[0,0]]: `M1 @ X = [X[:,1], 0]` (odd channels feed Y into X-component).
  With M2 = [[0,0],[1,0]]: `M2 @ X = [0, X[:,0]]` (even channels feed X into Y-component).
- **State layout**: `terminal_paths` shape (num_paths, 2), column 0 = X, column 1 = Y.
- **STIFFNESS / STABILITY RISK**: This problem is flagged stiff. Implicit / semi-implicit
  schemes are the theoretically preferred tool for stiff SDEs but are NOT yet supported in
  this pipeline; we use explicit Euler-Maruyama. The distinguishing choice of this plan is a
  fine step dt=0.001 (5x finer than plan 1). The Euler-Maruyama weak/variance bias for
  multiplicative noise is O(dt), so halving/refining dt roughly proportionally reduces the
  terminal-variance error. dt=0.001 gives dt*|lambda| ~ 0.0036 (deep in the stable region)
  and is the recommended starting step for stiff problems. Cost is 1000 steps x 50000 paths;
  the two-normal aggregation keeps this cheap. If the variance error is already well under
  10% at a coarser step, this plan documents the converged reference; if not, it is the
  accuracy-first fallback.
- Use `rng = np.random.default_rng(seed)` for reproducibility.

## Results

Ran `solve_sde(num_paths=50000, dt=0.001, T=1.0, seed=42)` using the two-normal
channel-aggregation scheme (7 odd channels aggregated via `sqrt(7)*Z_odd`, 6 even
channels aggregated via `sqrt(6)*Z_even`), which is exact in law versus drawing all
13 channels explicitly.

Empirical results (Nt = 1000 steps):
- empirical_mean_X    = -0.117573   (exact -0.1148824, rel. err ≈ 2.35%)
- empirical_mean_Y    = -0.151942   (exact -0.1530794, rel. err ≈ 0.75%)
- empirical_variance_X = 0.173102   (exact 0.1715989, rel. err ≈ 0.88%)
- empirical_variance_Y = 0.173268   (exact 0.1627163, rel. err ≈ 6.49%)

All four checks are within the pipeline thresholds (variance rel. err < 10%,
mean rel. err < 5%, both means have |exact_mean| > 0.01 so the mean check applies).

## Evaluation

(Filled by evaluator)

<review score=10>

Score: 10/10 — Done

**Score: 10/10**

### Numerical Accuracy

Component X:
- mean_relative_error:     0.0234  (PASS)
- variance_relative_error: 0.0088  (PASS)

Component Y:
- mean_relative_error:     0.0074  (PASS)
- variance_relative_error: 0.0648  (PASS)

- Overall: PASS ✓

Both mean components have |exact_mean| > 0.01 (-0.11488, -0.15308), so the mean check
was applied (not skipped) to both, and both pass comfortably under the 5% threshold.
Both variance components pass under the 10% threshold, with Y's variance error (6.48%)
the tightest margin — consistent with the O(dt) multiplicative-noise variance bias
noted in the plan, here suppressed by the fine dt=0.001 step.

### Feedback for solver

None.

</review>
