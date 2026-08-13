# Autonumerics Benchmark Report

_Generated 2026-08-12 14:20_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **26.4m**

## Executive summary

- **Problems:** 35 (28 PDE, 7 SDE); 34 with exact/reference ground truth.
- **Runs completed:** 35/35.
- **Pipeline self-reported pass (10/10):** 33/35 (94%).
- **Independently confirmed pass:** 33/34 (97% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| PASS (verified) | 33 | 94% |
| FAIL | 1 | 3% |
| SELF-SCORE ONLY | 1 | 3% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 5 | 5/5 | 5/5 |
| PDE | Tier 2 | 5 | 5/5 | 5/5 |
| PDE | Tier 3 | 18 | 16/18 | 16/17 |
| SDE | Tier 1 | 3 | 3/3 | 3/3 |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 4 | 4/4 | 4/4 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

None. Every pipeline self-score agreed with the independent ground-truth check.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| P01 | pde_heat_1d | PDE | 1 | 10 | 4-spectral-sine-series | 4 | relL2=1.88e-16, p=inf | PASS (verified) | 9.2m |
| P02 | pde_heat_2d | PDE | 1 | 10 | 4-fourth-order-compact-cn | 4 | relL2=5.11e-08, p=3.98 | PASS (verified) | 10.1m |
| P03 | pde_wave_1d | PDE | 1 | 10 | 4-fd-4th-order-rk4 | 4 | relL2=6.48e-14, p=inf | PASS (verified) | 10.1m |
| P04 | pde_advection_1d | PDE | 1 | 10 | 3-spectral-fft-rk4 | 10 | relL2=3.96e-07, p=4.00 | PASS (verified) | 14.1m |
| P05 | pde_poisson_2d | PDE | 1 | 10 | 3-compact-4th-order | 4 | relL2=1.04e-09, p=4.05 | PASS (verified) | 10.7m |
| P06 | pde_laplace_2d | PDE | 2 | 10 | 3-fd-compact-4th-order | 3 | relL2=2.51e-14, p=inf | PASS (verified) | 7.0m |
| P07 | pde_convection_diffusion_bl | PDE | 2 | 10 | 3-exponential-fitting | 8 | relL2=1.01e-16, p=inf | PASS (verified) | 21.3m |
| P08 | pde_helmholtz_2d | PDE | 2 | 10 | 3-fd-compact-4th-order | 3 | relL2=2.56e-10, p=inf | PASS (verified) | 9.3m |
| P09 | pde_anisotropic_diffusion | PDE | 2 | 10 | 3-fd-4th-order | 4 | relL2=4.16e-09, p=4.04 | PASS (verified) | 10.1m |
| P10 | pde_wave_2d | PDE | 2 | 10 | 4-spectral-sine-galerkin | 4 | relL2=1.01e-14, p=inf | PASS (verified) | 9.8m |
| P11 | pde_burgers_inviscid | PDE | 3 | 6 | 4-weno5-rk3 | 13 | relL2=4.01e-02 | FAIL | 7.6m |
| P12 | pde_fokker_planck_ou | PDE | 3 | 10 | 4-spectral-fft-split | 4 | relL2=7.66e-06, p=2.02 | PASS (verified) | 16.3m |
| P13 | pde_fractional_diffusion | PDE | 3 | 10 | 1-l1-implicit-2nd-order | 3 | relL2=1.72e-04, p=2.03 | PASS (verified) | 23.9m |
| P14 | pde_black_scholes_call | PDE | 3 | 10 | 4-fourth-order-compact-cn | 4 | relL2=3.83e-06, p=2.08 | PASS (verified) | 13.4m |
| P15 | pde_kuramoto_sivashinsky | PDE | 3 | 7 | 1-etdrk4-spectral | 4 | no GT | SELF-SCORE ONLY | 25.9m |
| P16 | pde_stefan_1d_similarity | PDE | 3 | 10 | 1-front-fixing-cn | 2 | relL2=3.81e-06, p=2.02 | PASS (verified) | 18.1m |
| P17 | pde_monge_ampere_2d | PDE | 3 | 10 | 2-standard-fd-newton-convex-seed | 2 | relL2=1.62e-06, p=2.01 | PASS (verified) | 10.8m |
| P18 | pde_porous_medium_2d | PDE | 3 | 10 | 1-fd-explicit-conservative | 1 | relL1=4.85e-04, p=1.90 | PASS (verified) | 12.0m |
| P19 | pde_poisson_lshape | PDE | 3 | 10 | 3-singular-enrichment | 1 | relL2=1.52e-16, p=inf | PASS (verified) | 20.3m |
| P20 | pde_cahn_hilliard_2d | PDE | 3 | 10 | 1-spectral-imex-cnab2 | 1 | relL2=1.64e-03, p=1.95 | PASS (verified) | 16.4m |
| P21 | pde_heston_2d | PDE | 3 | 10 | 1-fd-explicit-ftcs | 5 | relL2=1.16e-05, p=2.13 | PASS (verified) | 60s |
| P22 | pde_fichera_3d | PDE | 3 | 10 | 2-fd-uniform-iterative-cg | 1 | relL2=3.26e-04, p=1.65 | PASS (verified) | 8.5m |
| P23 | pde_acoustic_3d_layered | PDE | 3 | 10 | 1-fd-explicit-leapfrog | 2 | relL2=1.93e-03, p=2.05 | PASS (verified) | 14.6m |
| P24 | pde_navier_stokes_2d | PDE | 3 | 10 | 1-spectral-projection | 2 | relL2=2.31e-15, p=inf [u:2.3e-15 v:2.3e-15 p:4.7e-15] | PASS (verified) | 11.1m |
| P25 | pde_mhd_2d | PDE | 3 | 10 | 2-projection-semi-implicit | 2 | relL2=8.88e-04, p=1.68 [u:8.9e-04 v:8.9e-04 Bx:5.7e-04 By:5.7e-04 p:2.1e-02] | PASS (verified) | 3.0m |
| P26 | pde_keller_segel_2d | PDE | 3 | 10 | 1-fd-explicit-upwind | 2 | relL2=2.57e-06, p=1.64 [rho:2.6e-06 c:3.4e-07] | PASS (verified) | 29s |
| P27 | pde_elasticity_2d | PDE | 3 | 10 | 2-q1-fem-sri | 2 | relL2=9.03e-05, p=2.03 [u:9.0e-05 v:9.0e-05] | PASS (verified) | 8.6m |
| P28 | pde_maxwell_3d | PDE | 3 | 10 | 3-yee-pseudospectral | 2 | relL2=4.05e-05, p=2.02 [Ex:4.0e-05 Ey:4.0e-05 Ez:2.7e-16 Hx:1.4e-04 Hy:1.4e-04 Hz:1.4e-04] | PASS (verified) | 12.4m |
| S01 | sde_gbm | SDE | 1 | 10 | 1-euler-maruyama | 2 | varErr=1.8%, meanErr=0.2% | PASS (verified) | 2.9m |
| S02 | sde_ornstein_uhlenbeck | SDE | 1 | 10 | 1-euler-maruyama | 1 | varErr=2.4%, meanErr=0.5% | PASS (verified) | 2.5m |
| S03 | sde_bm_with_drift | SDE | 1 | 10 | 1-euler-maruyama | 1 | varErr=1.5%, meanErr=0.2% | PASS (verified) | 2.4m |
| S11 | sde_cir_feller_violated | SDE | 3 | 10 | 2-milstein | 2 | varErr=1.8%, meanErr=4.4% | PASS (verified) | 4.2m |
| S14 | sde_gbm_2d_high_corr | SDE | 3 | 10 | 1-euler-maruyama | 1 | varErr=0.1%, meanErr=0.1% | PASS (verified) | 2.8m |
| S17 | sde_multichannel_stiff_m13 | SDE | 3 | 10 | 1-euler-maruyama | 1 | varErr=6.9%, meanErr=2.4% | PASS (verified) | 5.4m |
| S20 | sde_ginzburg_landau_s6 | SDE | 3 | 10 | 1-tamed-euler-maruyama | 3 | varErr=1.6%, meanErr=2.1% (ref) | PASS (verified) | 8.6m |

## Tier-by-tier detail

### PDE

**Tier 1**

- `P01 pde_heat_1d` (heat) — **PASS (verified)**. Textbook parabolic IVBP; explicit FTCS and implicit Crank-Nicolson both apply. Independent: relL2=1.88e-16, p=inf.
- `P02 pde_heat_2d` (heat) — **PASS (verified)**. Separable 2D diffusion; tests 2D grid assembly and CFL in two dimensions. Independent: relL2=5.11e-08, p=3.98.
- `P03 pde_wave_1d` (wave) — **PASS (verified)**. Second-order-in-time hyperbolic PDE; needs a stable leap-frog / Newmark step. Independent: relL2=6.48e-14, p=inf.
- `P04 pde_advection_1d` (advection) — **PASS (verified)**. Transports a narrow Gaussian; first-order upwind smears it, so it stresses numerical diffusion. Independent: relL2=3.96e-07, p=4.00.
- `P05 pde_poisson_2d` (poisson) — **PASS (verified)**. Steady elliptic solve; 2D sparse assembly and a direct/iterative linear solve. Independent: relL2=1.04e-09, p=4.05.

**Tier 2**

- `P06 pde_laplace_2d` (laplace) — **PASS (verified)**. Non-separable-looking BC data; solution grows like sinh in y, so it is not symmetric in x/y. Independent: relL2=2.51e-14, p=inf.
- `P07 pde_convection_diffusion_bl` (convection-diffusion) — **PASS (verified)**. eps=1e-3 boundary layer near x=1; centered differences oscillate, so it needs upwinding or a fine/graded mesh. Independent: relL2=1.01e-16, p=inf.
- `P08 pde_helmholtz_2d` (helmholtz) — **PASS (verified)**. k=10 makes the discrete operator indefinite; naive iterative solvers stall. Independent: relL2=2.56e-10, p=inf.
- `P09 pde_anisotropic_diffusion` (anisotropic-diffusion) — **PASS (verified)**. 100:1 diffusion-tensor anisotropy stresses conditioning and mesh aspect ratio. Independent: relL2=4.16e-09, p=4.04.
- `P10 pde_wave_2d` (wave) — **PASS (verified)**. 2D hyperbolic with a sqrt(2) modal frequency; CFL couples both spatial directions. Independent: relL2=1.01e-14, p=inf.

**Tier 3**

- `P11 pde_burgers_inviscid` (burgers-inviscid) — **FAIL**. Entropy shock; a conservative/Godunov scheme captures the speed but smears the front, so <1% L2 is very hard. Independent: relL2=4.01e-02.
- `P12 pde_fokker_planck_ou` (fokker-planck) — **PASS (verified)**. Drift-diffusion in conservation form; must conserve mass and track a moving, narrowing Gaussian. Independent: relL2=7.66e-06, p=2.02.
- `P13 pde_fractional_diffusion` (fractional-diffusion) — **PASS (verified)**. Caputo derivative of order 1/2 needs a history-aware (L1) scheme and the Mittag-Leffler solution; far outside the manual. Independent: relL2=1.72e-04, p=2.03.
- `P14 pde_black_scholes_call` (black-scholes) — **PASS (verified)**. Backward-in-time terminal-value problem with a non-smooth payoff kink; convection-dominated near S=Smax. Independent: relL2=3.83e-06, p=2.08.
- `P15 pde_kuramoto_sivashinsky` (kuramoto-sivashinsky) — **SELF-SCORE ONLY**. Fourth-order, chaotic, no closed form; stiff and requires an ETDRK/IMEX spectral integrator. No analytic ground truth. No closed form; pipeline self-score only.
- `P16 pde_stefan_1d_similarity` (stefan-free-boundary) — **PASS (verified)**. Moving free boundary (phase change); the front's kink limits the L2 rate and a naive fixed-grid scheme mislocates it. Independent: relL2=3.81e-06, p=2.02.
- `P17 pde_monge_ampere_2d` (monge-ampere) — **PASS (verified)**. Fully nonlinear elliptic det(D^2 u)=f; needs a convexity-preserving (monotone wide-stencil) scheme, not a linear solve. Independent: relL2=1.62e-06, p=2.01.
- `P18 pde_porous_medium_2d` (porous-medium) — **PASS (verified)**. Degenerate diffusion u_t=Delta(u^2) with a compact-support free boundary; scored in the L1 norm because the moving edge pollutes the L2 rate. Independent: relL1=4.85e-04, p=1.90.
- `P19 pde_poisson_lshape` (poisson-reentrant-corner) — **PASS (verified)**. Re-entrant corner: u ~ r^{2/3} is not in H^2, so uniform low-order methods lose their rate; the order gate is the whole point. Independent: relL2=1.52e-16, p=inf.
- `P20 pde_cahn_hilliard_2d` (cahn-hilliard) — **PASS (verified)**. Fourth-order stiff phase-field operator; explicit stepping needs dt~h^4, so an energy-stable / IMEX scheme is required. Independent: relL2=1.64e-03, p=1.95.
- `P21 pde_heston_2d` (heston) — **PASS (verified)**. 2D convection-diffusion with a cross-derivative and a non-smooth payoff kink; degenerate as v->0. Reference is a semi-closed characteristic-function integral. Independent: relL2=1.16e-05, p=2.13.
- `P22 pde_fichera_3d` (fichera-corner) — **PASS (verified)**. 3-D re-entrant vertex (cube minus an octant); the regularized singular solution has a sharp near-corner layer, and only in-domain nodes are scored. Independent: relL2=3.26e-04, p=1.65.
- `P23 pde_acoustic_3d_layered` (acoustic-layered) — **PASS (verified)**. 3-D wave with a discontinuous (layered) wave speed and high spatial frequency; numerical dispersion/phase pollution unless enough points per wavelength are used. Independent: relL2=1.93e-03, p=2.05.
- `P24 pde_navier_stokes_2d` (navier-stokes) — **PASS (verified)**. Incompressible NS system (u, v, p); a scheme that fails to enforce div u = 0 pollutes the solution -- the divergence constraint is a hard gate, separate from field accuracy. Independent: relL2=2.31e-15, p=inf [u:2.3e-15 v:2.3e-15 p:4.7e-15].
- `P25 pde_mhd_2d` (mhd) — **PASS (verified)**. Coupled velocity + magnetic field; the solenoidal constraint div B = 0 is a distinct hard gate from div u = 0, and the Lorentz/induction coupling is stiff. Independent: relL2=8.88e-04, p=1.68 [u:8.9e-04 v:8.9e-04 Bx:5.7e-04 By:5.7e-04 p:2.1e-02].
- `P26 pde_keller_segel_2d` (keller-segel) — **PASS (verified)**. Chemotaxis system (density rho, chemoattractant c); an aggregating drift makes naive schemes produce negative density -- positivity is a hard gate. Independent: relL2=2.57e-06, p=1.64 [rho:2.6e-06 c:3.4e-07].
- `P27 pde_elasticity_2d` (elasticity-locking) — **PASS (verified)**. lambda/mu = 1e4: a standard displacement scheme locks (volumetric over-stiffening), so the displacement error plateaus under refinement instead of converging. Independent: relL2=9.03e-05, p=2.03 [u:9.0e-05 v:9.0e-05].
- `P28 pde_maxwell_3d` (maxwell) — **PASS (verified)**. Full 3-D Maxwell system (E, H each 3-vectors); a scheme that does not preserve div E = div H = 0 accumulates spurious charge -- two hard 3-D structural gates plus phase accuracy. Independent: relL2=4.05e-05, p=2.02 [Ex:4.0e-05 Ey:4.0e-05 Ez:2.7e-16 Hx:1.4e-04 Hy:1.4e-04 Hz:1.4e-04].

### SDE

**Tier 1**

- `S01 sde_gbm` (geometric_brownian_motion) — **PASS (verified)**. Canonical multiplicative-noise SDE; Euler-Maruyama and Milstein both apply. Independent: varErr=1.8%, meanErr=0.2%.
- `S02 sde_ornstein_uhlenbeck` (ornstein_uhlenbeck) — **PASS (verified)**. Additive noise, linear mean-reverting drift; EM suffices at a modest step. Independent: varErr=2.4%, meanErr=0.5%.
- `S03 sde_bm_with_drift` (bm_with_drift) — **PASS (verified)**. Constant drift and additive noise; exactly integrable, so EM should be essentially exact in the mean. Independent: varErr=1.5%, meanErr=0.2%.

**Tier 3**

- `S11 sde_cir_feller_violated` (cox_ingersoll_ross) — **PASS (verified)**. 2 kappa theta = 1 < sigma^2 = 4, so paths hit zero frequently; naive Euler biases the variance badly. Independent: varErr=1.8%, meanErr=4.4%.
- `S14 sde_gbm_2d_high_corr` (gbm_2d_correlated) — **PASS (verified)**. rho=0.95 makes the correlation matrix nearly singular; the Cholesky factor is ill-conditioned and volatility is higher. Independent: varErr=0.1%, meanErr=0.1%.
- `S17 sde_multichannel_stiff_m13` (multichannel_stiff_linear) — **PASS (verified)**. 13 non-commuting multiplicative noise channels on a spiral drift; a correct solver must assemble all 13 channels and resolve the step-size-sensitive terminal variance, well beyond the 1-2 channel baseline. Independent: varErr=6.9%, meanErr=2.4%.
- `S20 sde_ginzburg_landau_s6` (ginzburg_landau) — **PASS (verified)**. Same superlinear family at higher volatility; naive Euler produces partial NaNs and a badly biased variance. Scored against a discretization-free reference. Independent: varErr=1.6%, meanErr=2.1% (ref).

## Scheme coverage and efficiency

Winning scheme among verified passes:

| Scheme (plan slug) | Times best |
|---|---|
| euler-maruyama | 5 |
| fourth-order-compact-cn | 2 |
| fd-compact-4th-order | 2 |
| spectral-sine-series | 1 |
| fd-4th-order-rk4 | 1 |
| spectral-fft-rk4 | 1 |
| compact-4th-order | 1 |
| exponential-fitting | 1 |
| fd-4th-order | 1 |
| spectral-sine-galerkin | 1 |
| spectral-fft-split | 1 |
| l1-implicit-2nd-order | 1 |
| front-fixing-cn | 1 |
| standard-fd-newton-convex-seed | 1 |
| fd-explicit-conservative | 1 |
| singular-enrichment | 1 |
| spectral-imex-cnab2 | 1 |
| fd-explicit-ftcs | 1 |
| fd-uniform-iterative-cg | 1 |
| fd-explicit-leapfrog | 1 |
| projection-semi-implicit | 1 |
| milstein | 1 |
| tamed-euler-maruyama | 1 |
| spectral-projection | 1 |
| fd-explicit-upwind | 1 |
| q1-fem-sri | 1 |
| yee-pseudospectral | 1 |

- Mean solver-evaluator iterations per problem (summed over plans): 3.2.
- Mean wall time per completed run: 10.4m; max 25.9m.

## Failures and errors

- `P11 pde_burgers_inviscid` — **FAIL**: rel L2 = 4.006e-02.  
  log: `benchmark/results/logs/pde_burgers_inviscid.log`

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- Ground truth for all 34 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
