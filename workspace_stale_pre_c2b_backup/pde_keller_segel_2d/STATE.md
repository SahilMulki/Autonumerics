---
phase: done
problem_type: pde
problem_spec: workspace/pde_keller_segel_2d/problem_spec.json
plans:
  1-explicit-central-rk2:
    one-sentence: Explicit SSP-RK2 with 2nd-order conservative central-difference finite-volume chemotaxis flux plus central diffusion, dt=0.2 dx^2/D; no positivity mechanism.
    iter: 1
    score: 10
  2-upwind-fluxlimited-positive:
    one-sentence: Explicit SSP-RK2 with van Leer flux-limited upwind chemotaxis flux (positivity-preserving, 2nd order in smooth regions) plus central diffusion.
    iter: 1
    score: 10
  3-imex-adi-upwind:
    one-sentence: IMEX with Crank-Nicolson ADI diffusion (unconditionally stable tridiagonal) coupled to explicit positivity-preserving upwind chemotaxis, dt=0.4 dx/a_max.
    iter: 1
    score: 10
best_plan: 3-imex-adi-upwind
---
