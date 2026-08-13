# REPORT — pde_keller_segel_2d

## Problem

- **Type:** PDE — family `reaction-diffusion` (2-D Keller-Segel chemotaxis system).
- **Fields:** two coupled fields — cell density `rho` and chemoattractant `c` — on `[0,1]²`, `t ∈ [0, 0.5]`.
- **Character:** nonlinear (chemotaxis flux `-chi·div(rho·grad c)`), non-stiff, manufactured (MMS) sources, time-dependent Dirichlet BCs.
- **Manufactured exact solution:**
  - `rho = 1 + 0.2·sin(πx)·sin(πy)·e^(-t)` (range [1, 1.2])
  - `c = 1 + 0.1·cos(πx)·cos(πy)·e^(-2t)` (range [0.9, 1.1])
- **Gates:** `rel_l2_err_max = 0.01`, `min_spatial_order ≥ 1.4` on primary field `rho`, hard structural gate `min_density` (positivity), required fields `{rho, c}`.

## Plans

All three cleared every gate and scored **10/10** in a single solver↔evaluator cycle. Errors below are relative-L2 at the fine grid N=96; convergence order is the observed spatial order on primary field `rho`.

| Plan | Scheme | Score | Iter | rho err (N=96) | c err (N=96) | order (rho) | min(rho) | Time steps @N=96 |
|---|---|---|---|---|---|---|---|---|
| 1-explicit-central-rk2 | SSP-RK2 (Heun) + conservative central-FV chemotaxis flux + central diffusion; `dt=0.2 dx²/D` | 10 | 1 | 5.82e-06 | 4.38e-07 | 2.02 | 1.000 ✓ | ~22,563 |
| 2-upwind-fluxlimited-positive | SSP-RK2 + van Leer MUSCL flux-limited upwind chemotaxis (positivity by construction) + central diffusion | 10 | 1 | 5.80e-06 | 4.38e-07 | 2.01 | 1.000 ✓ | ~22,563 |
| 3-imex-adi-upwind | IMEX: Crank-Nicolson diffusion via Peaceman-Rachford ADI (tridiagonal, unconditionally stable) + explicit positivity-preserving van Leer upwind chemotaxis + trapezoidal predictor-corrector; `dt=0.4 dx/a_max` | 10 | 1 | 2.09e-05 | 5.64e-06 | 1.86 | 1.000 ✓ | 48 |

Notes:
- The manufactured solution is smooth and stays well above zero (`rho ≥ 1`), so even the non-upwinded Plan 1 never trips the `min_density` gate; Plans 2 and 3 guarantee positivity structurally regardless.
- Plans 2 and 3 achieve positivity with **no clipping** (no `np.maximum`) — positivity is built into the flux reconstruction.

## Best plan

**Recommendation: `3-imex-adi-upwind`.**

All three plans tie on the protocol criteria (score 10, iter 1), so the tiebreak is on merit for this problem class:
- **Same accuracy class** — rho error 2.1e-05 at N=96, ~3 orders under the 1% tolerance, order 1.86 (clears the 1.4 floor).
- **~470× fewer time steps** than the explicit plans (48 vs ~22,563 at N=96) because ADI removes the `dt~dx²` diffusion stability restriction — `dt` scales linearly with `dx`.
- **Structure-preserving** — positivity-preserving upwind chemotaxis flux, appropriate for the Keller-Segel class where blow-up/negative-density is the central numerical risk under sharper conditions than this benchmark.

Plan 2 is the recommended runner-up: it matches Plan 1's 2nd-order accuracy while also being positivity-preserving, at the same explicit cost. Plan 1 is the highest-accuracy plain scheme but carries no positivity mechanism, making it the least robust choice for stiffer/steeper chemotaxis regimes.

## Failures

None. All three plans reached the terminal condition (score 10) on the first cycle with no remaining errors.
