---
id: 1
plan_slug: euler-maruyama
scheme: euler-maruyama
strategy: Vector Euler-Maruyama with dt=0.005, exploiting the odd/even channel aggregation to draw only 2 correlated normals per step.
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
- **dt**: 0.005
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
  are identical (G = 0.6*M1) and the 6 even channels are identical (G = 0.6*M2). Instead of
  drawing 13 independent increments, draw two independent standard normals per step per path,
  Z_odd and Z_even, and form the aggregate noise as
  `0.6 * (M1 @ X) * sqrt(7) * sqrt(dt) * Z_odd + 0.6 * (M2 @ X) * sqrt(dt) * sqrt(6) * Z_even`.
  This is exactly equivalent to summing the 13 independent channels (sum of k i.i.d.
  N(0,dt) increments = N(0, k*dt) = sqrt(k)*sqrt(dt)*Z). Drawing all 13 explicitly is also
  fine and gives the same result in law.
- With M1 = [[0,1],[0,0]]: `M1 @ X = [X[:,1], 0]` (odd channels feed Y into X-component).
  With M2 = [[0,0],[1,0]]: `M2 @ X = [0, X[:,0]]` (even channels feed X into Y-component).
- **State layout**: `terminal_paths` shape (num_paths, 2), column 0 = X, column 1 = Y.
- **STIFFNESS / STABILITY RISK**: This problem is flagged stiff. F has eigenvalues
  -2 +/- 3i (spiral decay) and the multiplicative noise is strong, so the terminal variance
  is sensitive to dt. Implicit / semi-implicit schemes (which would treat the drift implicitly
  for unconditional stability) are the theoretically preferred tool for stiff SDEs but are
  NOT yet supported in this pipeline. We therefore use explicit Euler-Maruyama with a small
  fixed step. dt=0.005 gives dt*|lambda| ~ 0.018, comfortably inside the explicit stability
  region for the drift, but the multiplicative-noise variance bias is O(dt) and may still
  need a finer step. If the evaluator reports the variance error above 10%, refine dt
  (e.g. to 0.002 or 0.001) until the variance error stabilizes.
- Use `rng = np.random.default_rng(seed)` for reproducibility.
- **dt refined from 0.005 to 0.001**: at dt=0.005 the deterministic forward-Euler drift bias
  alone (no noise) is ~5.3% relative error on mean_X (checked by simulating dE[X]/dt = F@E[X]
  with plain forward Euler), which combined with Monte Carlo sampling noise pushed the
  empirical mean_X error to ~7.2%, above the 5% threshold. Halving dt repeatedly
  (0.005 -> 0.002 -> 0.001) shows the bias shrinking roughly linearly (O(dt), as expected for
  Euler-Maruyama's drift discretization) until it drops below the ~1.8% Monte Carlo noise
  floor set by num_paths=50000 (std_err/|mean_X| ~ sqrt(Var_X/N)/|mean_X| ~ 1.8%). dt=0.001
  (1000 steps) was selected as the point where further refinement gives no material
  improvement, at reasonable computational cost.

## Results

Run: `num_paths=50000, dt=0.001, T=1.0, seed=42` (1000 steps).

Empirical moments:
- mean_X    = -0.117573   (exact -0.1148824, rel. err. 2.34%)
- mean_Y    = -0.151942   (exact -0.1530794, rel. err. 0.74%)
- variance_X = 0.173102   (exact 0.1715989, rel. err. 0.88%)
- variance_Y = 0.173268   (exact 0.1627163, rel. err. 6.48%)

All four relative errors are comfortably within the evaluation thresholds
(mean < 5%, variance < 10%).

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

Ran with num_paths=50000, dt=0.001, T=1.0, seed=42 (1000 steps). Both exact means
(-0.11488 for X, -0.15308 for Y) have |exact_mean| > 0.01, so the mean check applies
to both components and both pass comfortably under the 5% threshold. Both variance
components pass under the 10% threshold; Y's variance error (6.48%) is the tightest
margin, consistent with the O(dt) multiplicative-noise variance discretization bias
noted in the plan — well controlled at dt=0.001. This solver is numerically identical
to plan 2 (2-euler-maruyama-fine), which uses the same dt=0.001 and channel-aggregation
scheme and also scores 10/10 with matching numbers.

### Feedback for solver

None.

</review>
