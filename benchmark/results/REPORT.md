# Autonumerics Benchmark Report

_Generated 2026-08-29 17:45_

Pipeline invocation: `claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"`

Active time, most recent run invocation: **26.3m**

## Executive summary

- **Problems:** 28 (22 PDE, 6 SDE); 27 with exact/reference ground truth.
- **Runs completed:** 28/28.
- **Pipeline self-reported pass (10/10):** 26/28 (93%).
- **Independently confirmed pass:** 25/27 (93% of independently-checkable problems).
- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was independently confirmed.

### Verdict distribution

| Verdict | Count | Share |
|---|---|---|
| PASS (verified) | 25 | 89% |
| UNDERCLAIM | 2 | 7% |
| SELF-SCORE ONLY | 1 | 4% |

### Independently-verified pass rate by type and tier

| Type | Tier | N | Pipeline pass | Independently verified |
|---|---|---|---|---|
| PDE | Tier 1 | 5 | 5/5 | 5/5 |
| PDE | Tier 2 | 5 | 5/5 | 5/5 |
| PDE | Tier 3 | 12 | 12/12 | 11/11 |
| SDE | Tier 1 | 2 | 2/2 | 2/2 |
| SDE | Tier 2 | 0 | 0/0 | n/a |
| SDE | Tier 3 | 4 | 2/4 | 2/4 |

> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care (boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, superlinear/tamed drift, and blow-up/domain stability). Independently verified counts both exact/reference moment passes and stability (no blow-up) passes.

## Independent-verification discrepancies

### S11 sde_cir_feller_violated — UNDERCLAIM

Pipeline stopped at **8/10**, but the independent check **passes** (varErr=1.8%, meanErr=5.0%). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

### S17 sde_multichannel_stiff_m13 — UNDERCLAIM

Pipeline stopped at **6/10**, but the independent check **passes** (varErr=6.9%, meanErr=2.4%). The solution was actually good enough; the pipeline under-scored or ran out of iterations.

## Full results

| ID | Slug | Type | Tier | Pipe | Best plan | Iters | Independent metric | Verdict | Time |
|---|---|---|---|---|---|---|---|---|---|
| P01 | pde_heat_1d | PDE | 1 | 10 | 3-fd4-rk4 | 3 | relL2=2.05e-09, p=4.04 | PASS (verified) | 8.2m |
| P02 | pde_heat_2d | PDE | 1 | 10 | 2-crank-nicolson-sparse | 3 | relL2=9.94e-05, p=2.00 | PASS (verified) | 10.3m |
| P03 | pde_wave_1d | PDE | 1 | 10 | 2-newmark-implicit | 2 | relL2=3.26e-09, p=4.07 | PASS (verified) | 10.3m |
| P04 | pde_advection_1d | PDE | 1 | 10 | 1-fd4-central-rk4 | 3 | relL2=1.97e-04, p=4.00 | PASS (verified) | 18.4m |
| P05 | pde_poisson_2d | PDE | 1 | 10 | 2-fd4-compact-mehrstellen | 3 | relL2=1.04e-09, p=4.05 | PASS (verified) | 6.1m |
| P06 | pde_laplace_2d | PDE | 2 | 10 | 2-fd4-compact-9point | 3 | relL2=7.98e-14, p=inf | PASS (verified) | 6.4m |
| P07 | pde_convection_diffusion_bl | PDE | 2 | 10 | 1-exponential-fitting | 3 | relL2=1.44e-15, p=inf | PASS (verified) | 3.0m |
| P08 | pde_helmholtz_2d | PDE | 2 | 10 | 2-fd4-compact-direct | 3 | relL2=2.56e-10, p=inf | PASS (verified) | 7.5m |
| P09 | pde_anisotropic_diffusion | PDE | 2 | 10 | 2-fd4-wide-stencil | 3 | relL2=4.16e-09, p=4.04 | PASS (verified) | 6.5m |
| P10 | pde_wave_2d | PDE | 2 | 10 | 2-fourth-order-space | 3 | relL2=6.11e-09, p=4.05 | PASS (verified) | 6.7m |
| P11 | pde_burgers_inviscid | PDE | 3 | 10 | 2-godunov-subcell | 3 | relL2=9.64e-08 | PASS (verified) | 22.4m |
| P12 | pde_fokker_planck_ou | PDE | 3 | 10 | 1-fd-explicit-flux | 1 | relL2=1.34e-03, p=2.04 | PASS (verified) | 2.4m |
| P13 | pde_fractional_diffusion | PDE | 3 | 10 | 3-l1-graded-fem-p1 | 3 | relL2=4.92e-05, p=2.03 | PASS (verified) | 7.6m |
| P14 | pde_black_scholes_call | PDE | 3 | 10 | 1-fd-explicit-rk2 | 2 | relL2=1.37e-05, p=2.03 | PASS (verified) | 5.7m |
| P15 | pde_kuramoto_sivashinsky | PDE | 3 | 10 | 3-ifrk4-spectral | 3 | no GT | SELF-SCORE ONLY | 6.3m |
| P16 | pde_stefan_1d_similarity | PDE | 3 | 10 | 2-landau-explicit-rk2 | 2 | relL2=4.75e-07, p=2.21 | PASS (verified) | 15.8m |
| P17 | pde_monge_ampere_2d | PDE | 3 | 10 | 2-fd2-newton | 2 | relL2=1.62e-06, p=2.01 | PASS (verified) | 19.8m |
| P18 | pde_porous_medium_2d | PDE | 3 | 10 | 1-fd-explicit-conservative | 2 | relL1=5.44e-04, p=1.91 | PASS (verified) | 25.3m |
| P20 | pde_cahn_hilliard_2d | PDE | 3 | 10 | 3-fd4-imex-sbdf2 | 2 | relL2=1.09e-04, p=4.04 | PASS (verified) | 20.9m |
| P21 | pde_heston_2d | PDE | 3 | 10 | 2-fully-implicit-bdf2 | 1 | relL2=1.11e-03, p=0.03 | PASS (verified) | 15.5m |
| P24 | pde_navier_stokes_2d | PDE | 3 | 10 | 3-fourier-pseudospectral | 2 | relL2=1.03e-15, p=inf [u:1.0e-15 v:1.0e-15 p:2.1e-15] | PASS (verified) | 18.8m |
| P25 | pde_mhd_2d | PDE | 3 | 10 | 2-streamfunction-vorticity-ct | 2 | relL2=6.06e-04, p=2.03 [u:6.1e-04 v:6.1e-04 Bx:3.2e-04 By:3.2e-04 p:6.9e+00] | PASS (verified) | 13.7m |
| S02 | sde_ornstein_uhlenbeck | SDE | 1 | 10 | 1-euler-maruyama | 1 | varErr=0.8%, meanErr=0.5% | PASS (verified) | 4.9m |
| S03 | sde_bm_with_drift | SDE | 1 | 10 | 1-euler-maruyama | 1 | varErr=0.4%, meanErr=0.2% | PASS (verified) | 5.1m |
| S11 | sde_cir_feller_violated | SDE | 3 | 8 | 2-milstein | 4 | varErr=1.8%, meanErr=5.0% | UNDERCLAIM | 3.5m |
| S14 | sde_gbm_2d_high_corr | SDE | 3 | 10 | 1-euler-maruyama | 1 | varErr=0.1%, meanErr=0.1% | PASS (verified) | 8.8m |
| S17 | sde_multichannel_stiff_m13 | SDE | 3 | 6 | 1-euler-maruyama | 2 | varErr=6.9%, meanErr=2.4% | UNDERCLAIM | 12.0m |
| S20 | sde_ginzburg_landau_s6 | SDE | 3 | 10 | 3-log-transform-euler | 3 | varErr=0.0%, meanErr=0.5% (ref) | PASS (verified) | 2.1m |

## Tier-by-tier detail

### PDE

**Tier 1**

- `P01 pde_heat_1d` (heat) — **PASS (verified)**. Textbook parabolic IVBP; explicit FTCS and implicit Crank-Nicolson both apply. Independent: relL2=2.05e-09, p=4.04.
- `P02 pde_heat_2d` (heat) — **PASS (verified)**. Separable 2D diffusion; tests 2D grid assembly and CFL in two dimensions. Independent: relL2=9.94e-05, p=2.00.
- `P03 pde_wave_1d` (wave) — **PASS (verified)**. Second-order-in-time hyperbolic PDE; needs a stable leap-frog / Newmark step. Independent: relL2=3.26e-09, p=4.07.
- `P04 pde_advection_1d` (advection) — **PASS (verified)**. Transports a narrow Gaussian; first-order upwind smears it, so it stresses numerical diffusion. Independent: relL2=1.97e-04, p=4.00.
- `P05 pde_poisson_2d` (poisson) — **PASS (verified)**. Steady elliptic solve; 2D sparse assembly and a direct/iterative linear solve. Independent: relL2=1.04e-09, p=4.05.

**Tier 2**

- `P06 pde_laplace_2d` (laplace) — **PASS (verified)**. Non-separable-looking BC data; solution grows like sinh in y, so it is not symmetric in x/y. Independent: relL2=7.98e-14, p=inf.
- `P07 pde_convection_diffusion_bl` (convection-diffusion) — **PASS (verified)**. eps=1e-3 boundary layer near x=1; centered differences oscillate, so it needs upwinding or a fine/graded mesh. Independent: relL2=1.44e-15, p=inf.
- `P08 pde_helmholtz_2d` (helmholtz) — **PASS (verified)**. k=10 makes the discrete operator indefinite; naive iterative solvers stall. Independent: relL2=2.56e-10, p=inf.
- `P09 pde_anisotropic_diffusion` (anisotropic-diffusion) — **PASS (verified)**. 100:1 diffusion-tensor anisotropy stresses conditioning and mesh aspect ratio. Independent: relL2=4.16e-09, p=4.04.
- `P10 pde_wave_2d` (wave) — **PASS (verified)**. 2D hyperbolic with a sqrt(2) modal frequency; CFL couples both spatial directions. Independent: relL2=6.11e-09, p=4.05.

**Tier 3**

- `P11 pde_burgers_inviscid` (burgers-inviscid) — **PASS (verified)**. Entropy shock; a conservative/Godunov scheme captures the speed but smears the front, so <1% L2 is very hard. Independent: relL2=9.64e-08.
- `P12 pde_fokker_planck_ou` (fokker-planck) — **PASS (verified)**. Drift-diffusion in conservation form; must conserve mass and track a moving, narrowing Gaussian. Independent: relL2=1.34e-03, p=2.04.
- `P13 pde_fractional_diffusion` (fractional-diffusion) — **PASS (verified)**. Caputo derivative of order 1/2 needs a history-aware (L1) scheme and the Mittag-Leffler solution; far outside the manual. Independent: relL2=4.92e-05, p=2.03.
- `P14 pde_black_scholes_call` (black-scholes) — **PASS (verified)**. Backward-in-time terminal-value problem with a non-smooth payoff kink; convection-dominated near S=Smax. Independent: relL2=1.37e-05, p=2.03.
- `P15 pde_kuramoto_sivashinsky` (kuramoto-sivashinsky) — **SELF-SCORE ONLY**. Fourth-order, chaotic, no closed form; stiff and requires an ETDRK/IMEX spectral integrator. No analytic ground truth. No closed form; pipeline self-score only.
- `P16 pde_stefan_1d_similarity` (stefan-free-boundary) — **PASS (verified)**. Moving free boundary (phase change); the front's kink limits the L2 rate and a naive fixed-grid scheme mislocates it. Independent: relL2=4.75e-07, p=2.21.
- `P17 pde_monge_ampere_2d` (monge-ampere) — **PASS (verified)**. Fully nonlinear elliptic det(D^2 u)=f; needs a convexity-preserving (monotone wide-stencil) scheme, not a linear solve. Independent: relL2=1.62e-06, p=2.01.
- `P18 pde_porous_medium_2d` (porous-medium) — **PASS (verified)**. Degenerate diffusion u_t=Delta(u^2) with a compact-support free boundary; scored in the L1 norm because the moving edge pollutes the L2 rate. Independent: relL1=5.44e-04, p=1.91.
- `P20 pde_cahn_hilliard_2d` (cahn-hilliard) — **PASS (verified)**. Fourth-order stiff phase-field operator; explicit stepping needs dt~h^4, so an energy-stable / IMEX scheme is required. Independent: relL2=1.09e-04, p=4.04.
- `P21 pde_heston_2d` (heston) — **PASS (verified)**. 2D convection-diffusion with a cross-derivative and a non-smooth payoff kink; degenerate as v->0. Reference is a semi-closed characteristic-function integral. Independent: relL2=1.11e-03, p=0.03.
- `P24 pde_navier_stokes_2d` (navier-stokes) — **PASS (verified)**. Incompressible NS system (u, v, p); a scheme that fails to enforce div u = 0 pollutes the solution -- the divergence constraint is a hard gate, separate from field accuracy. Independent: relL2=1.03e-15, p=inf [u:1.0e-15 v:1.0e-15 p:2.1e-15].
- `P25 pde_mhd_2d` (mhd) — **PASS (verified)**. Coupled velocity + magnetic field; the solenoidal constraint div B = 0 is a distinct hard gate from div u = 0, and the Lorentz/induction coupling is stiff. Independent: relL2=6.06e-04, p=2.03 [u:6.1e-04 v:6.1e-04 Bx:3.2e-04 By:3.2e-04 p:6.9e+00].

### SDE

**Tier 1**

- `S02 sde_ornstein_uhlenbeck` (ornstein_uhlenbeck) — **PASS (verified)**. Additive noise, linear mean-reverting drift; EM suffices at a modest step. Independent: varErr=0.8%, meanErr=0.5%.
- `S03 sde_bm_with_drift` (bm_with_drift) — **PASS (verified)**. Constant drift and additive noise; exactly integrable, so EM should be essentially exact in the mean. Independent: varErr=0.4%, meanErr=0.2%.

**Tier 3**

- `S11 sde_cir_feller_violated` (cox_ingersoll_ross) — **UNDERCLAIM**. 2 kappa theta = 1 < sigma^2 = 4, so paths hit zero frequently; naive Euler biases the variance badly. Independent: varErr=1.8%, meanErr=5.0%.
- `S14 sde_gbm_2d_high_corr` (gbm_2d_correlated) — **PASS (verified)**. rho=0.95 makes the correlation matrix nearly singular; the Cholesky factor is ill-conditioned and volatility is higher. Independent: varErr=0.1%, meanErr=0.1%.
- `S17 sde_multichannel_stiff_m13` (multichannel_stiff_linear) — **UNDERCLAIM**. 13 non-commuting multiplicative noise channels on a spiral drift; a correct solver must assemble all 13 channels and resolve the step-size-sensitive terminal variance, well beyond the 1-2 channel baseline. Independent: varErr=6.9%, meanErr=2.4%.
- `S20 sde_ginzburg_landau_s6` (ginzburg_landau) — **PASS (verified)**. Same superlinear family at higher volatility; naive Euler produces partial NaNs and a badly biased variance. Scored against a discretization-free reference. Independent: varErr=0.0%, meanErr=0.5% (ref).

## Scheme coverage and efficiency

Winning scheme among verified passes:

| Scheme (plan slug) | Times best |
|---|---|
| euler-maruyama | 3 |
| crank-nicolson-sparse | 1 |
| godunov-subcell | 1 |
| fd4-imex-sbdf2 | 1 |
| fully-implicit-bdf2 | 1 |
| fourier-pseudospectral | 1 |
| streamfunction-vorticity-ct | 1 |
| fd4-rk4 | 1 |
| newmark-implicit | 1 |
| fd4-central-rk4 | 1 |
| log-transform-euler | 1 |
| fd4-compact-mehrstellen | 1 |
| fd4-compact-9point | 1 |
| exponential-fitting | 1 |
| fd4-compact-direct | 1 |
| fd4-wide-stencil | 1 |
| fourth-order-space | 1 |
| fd-explicit-flux | 1 |
| l1-graded-fem-p1 | 1 |
| fd-explicit-rk2 | 1 |
| landau-explicit-rk2 | 1 |
| fd2-newton | 1 |
| fd-explicit-conservative | 1 |

- Mean solver-evaluator iterations per problem (summed over plans): 2.4.
- Mean wall time per completed run: 10.5m; max 25.3m.

## Failures and errors

No failures, overclaims, blow-ups, run errors, or unverifiable results.

## Methodology and reproducibility

- **Pipeline pass** = the conductor's best plan reached the terminal self-score of 10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near zero, mean rel err < 5%).
- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and compares the raw numerical output to ground truth authored and validated in `benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based relative L2 norm from `pde_manual.md`.
- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic solution); *reference* (no elementary moment formula — the ground truth is a discretization-free Monte-Carlo of the known exact solution, compared within an SE-aware tolerance); *stability* (no ground truth — the independent check only confirms a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).
- Ground truth for all 27 exact/reference problems is validated independently (SDE moments against moment-ODE integration, matrix-exponential / affine systems, and direct simulation; the Ginzburg-Landau reference anchored to the literature value E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).
- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript is saved under `benchmark/results/logs/`.
