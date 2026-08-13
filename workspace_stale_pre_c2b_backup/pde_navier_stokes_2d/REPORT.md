# REPORT — pde_navier_stokes_2d

## Problem

**Type:** PDE — multi-field system
**Family:** `navier-stokes` — 2-D incompressible Navier-Stokes, Taylor-Green vortex on the periodic box [0, 2π]², ν = 0.1, t_final = 0.5.

**Analytic solution (Taylor-Green):**
- u = sin(x) cos(y) · exp(−2νt)
- v = −cos(x) sin(y) · exp(−2νt)
- p = 0.25 (cos 2x + cos 2y) · exp(−4νt)

**Scoring contract:** run at N=48 and N=96; `rel_l2_err_max = 0.01`; `min_spatial_order = 1.5`; required fields u, v; pressure p is a gauge field (scored mean-removed); hard structural gate on `divergence_norm` (div u ≈ 0, else CONSTRAINT_VIOLATION).

## Plans

| Plan | Scheme | Final score | Iter | rel L2 err (u, N=96) | Observed order | max\|div u\| |
|---|---|---|---|---|---|---|
| 1-pseudo-spectral | Fourier pseudo-spectral, integrating-factor viscous term, RK4 advection (2/3 dealias), per-stage Leray projection | 10 | 1 | 5.57e-16 | inf (super-conv waiver) | 2.2e-16 |
| 2-fd-projection-fft | Chorin projection, collocated grid, 2nd-order central FD, RK2 (Heun), FFT pressure-Poisson | 10 | 1 | 1.18e-3 | 1.967 | ~1e-15 (FFT check 6.3e-15) |
| 3-mac-staggered-projection | Staggered MAC grid, RK2 (Heun), FFT pressure-Poisson; grad/div exact adjoints | 10 | 1 | 2.26e-3 | 1.958 | ~1e-15 (FFT check 1.4e-5) |

All three cleared the accuracy tolerance (<1%), the spatial-order floor (≥1.5), and the divergence-free structural gate on the first solver↔evaluator cycle. No plan failed to reach score 10; no residual errors remain.

### Notes per plan
- **1-pseudo-spectral** — The Taylor-Green field contains only wavenumbers |k| = 1, 2, which a spectral method resolves exactly at any N, so accuracy is machine-precision (~1e-15) at both resolutions. Order reports as `inf` under the harness's tolerance-relative super-convergence waiver (both grids below 1e-9). Velocity kept divergence-free by construction via per-RK4-stage Leray projection.
- **2-fd-projection-fft** — Clean, measurable 2nd-order convergence (1.97). Divergence-free in the consistent FFT-operator sense; independently verified via spectral divergence at round-off. Uses `linspace(0,2π,N,endpoint=False)` for a self-consistent periodic grid (a sensible deviation from the SOLUTION.md endpoint-duplication text, matching the codebase's periodic-FFT convention).
- **3-mac-staggered-projection** — Strongest structural guarantee: gradient and divergence are exact discrete adjoints on the MAC grid, so the discrete divergence is zero to round-off (not merely to discretization order). Clean 2nd-order convergence (1.96). De-staggering to the collocated report grid uses a 2nd-order half-cell average, preserving scheme order.

## Best plan

**Recommendation: `1-pseudo-spectral`.**

All three plans tie on the formal criteria (score 10, iter 1), so the tiebreak is accuracy: the pseudo-spectral scheme is strictly the most accurate, reaching machine precision (~1e-15 vs. ~1e-3 truncation-limited error for the two 2nd-order projection methods) because the Taylor-Green vortex is band-limited to |k| ≤ 2 and thus resolved exactly. It also enforces the divergence constraint by construction at every RK4 stage.

Caveat for generalization: the pseudo-spectral advantage is specific to smooth, periodic, band-limited fields like Taylor-Green. For under-resolved or non-periodic problems, the **2-fd-projection-fft** and **3-mac-staggered-projection** methods are more robust general-purpose choices, with the MAC scheme (plan 3) offering the strongest divergence-free guarantee.
