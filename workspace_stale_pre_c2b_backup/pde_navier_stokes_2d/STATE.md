---
phase: done
problem_type: pde
problem_spec: workspace/pde_navier_stokes_2d/problem_spec.json
plans:
  1-pseudo-spectral:
    one-sentence: Fourier pseudo-spectral with integrating-factor viscous term, RK4 advection, and Fourier-space Leray projection for machine-precision divergence-free velocity.
    iter: 1
    score: 10
  2-fd-projection-fft:
    one-sentence: Chorin projection on collocated grid, 2nd-order central FD advection/diffusion, explicit RK2, FFT pressure-Poisson solve.
    iter: 1
    score: 10
  3-mac-staggered-projection:
    one-sentence: Staggered MAC grid with RK2 and FFT pressure-Poisson; matched grad/div operators give exactly-zero discrete divergence.
    iter: 1
    score: 10
best_plan: 1-pseudo-spectral
---
