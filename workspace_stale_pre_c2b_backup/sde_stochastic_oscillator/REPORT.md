# Report — Stochastic Oscillator

## Problem

**Type:** SDE — Stochastic Harmonic Oscillator (family: `stochastic_oscillator`).

Coupled Itô system, state dimension 2 (X = position, Y = velocity):

    dX = Y dt,                 X(0) = 1.0
    dY = -X dt + sigma dW,     Y(0) = 0.0

with sigma = 0.3 over t in [0, 2*pi]. Noise is **additive**, entering only the velocity component (G = [[0],[sigma]]). Linear, non-stiff.

Analytic terminal moments used as ground truth:
- mean_X = X_0 cos t + Y_0 sin t; variance_X = sigma^2/2 (t - sin t cos t)
- mean_Y = -X_0 sin t + Y_0 cos t; variance_Y = sigma^2/2 (t + sin t cos t)

At T = 2*pi, mean_Y ≈ 0, so the relative-mean check on Y is skipped (near-zero threshold 0.01).

## Plans

### 1-euler-maruyama — final score 10/10, iter 1

Euler-Maruyama Monte Carlo (strong order 0.5, weak order 1.0), dt = 0.01, num_paths = 50000, seed = 42, T = 2*pi (629 steps). Milstein was not eligible: additive noise (dg/dstate = 0) plus multi-dimensional state (would require Lévy areas), so Milstein reduces to EM.

Final metrics (empirical vs. exact):

| Component | Quantity | Empirical | Exact | Rel. error | Status |
|---|---|---|---|---|---|
| X | Mean | 1.035287 | 1.000000 | 3.53% | PASS (<5%) |
| X | Variance | 0.288725 | 0.282743 | 2.12% | PASS (<10%) |
| Y | Mean | 0.002206 | ~0.0 | — | PASS (skipped, near-zero) |
| Y | Variance | 0.294114 | 0.282743 | 4.02% | PASS (<10%) |

All checks passed on the first evaluator pass.

## Recommendation

**Best plan: `1-euler-maruyama`** (score 10/10, 1 iteration). Euler-Maruyama is both the correct and sufficient scheme for this additive-noise, multi-dimensional linear SDE. It achieves all moment-accuracy thresholds with dt = 0.01 and 50k paths, with no refinement required.

## Failures

None. The single proposed plan reached the terminal score 10 on its first cycle.
