# Hard SDE Benchmark Candidates (from deep literature review)

Source: `agon-artifacts/topics/0713-hard-sde-benchmark-landscape.md` (16 deep-read papers,
2026-07-14). This file re-organizes that catalogue **by how each problem maps onto the
AutoNumerics ground-truth contract**, which is the thing that decides how each one gets scored.

## The ground-truth contract (why this file is organized the way it is)

`benchmark/problems.py` scores every SDE problem by comparing the pipeline's terminal moments
against an **independently authored** ground truth:

- SDE problems expose `moments(T) -> {"mean", "variance"}` (or `mean_X/variance_X/mean_Y/variance_Y`
  for 2-D state), validated in `validate_ground_truth.py`.
- Problems with no exact ground truth set `has_ground_truth=False` → verifier reports
  "pipeline self-score only".

So each candidate below is tagged with a **GT-FIT**:

- **EXACT** — terminal moments are computable to machine precision (closed form, quadrature over
  a 1-D integral, matrix exponential of a linear moment system, or affine-process moment ODEs).
  These slot straight into the existing harness. **Implement these first.**
- **REFERENCE** — no elementary moment formula, but the *exact solution path* is known, so you can
  produce a discretization-error-free reference by Monte-Carlo-ing the exact solution at large N
  (report the MC standard error as the tolerance). Needs a small harness extension to accept a
  reference-with-tolerance instead of an exact value.
- **SELF-SCORE** — no ground truth at all; still valuable as a *stability* benchmark (does the
  agent's scheme blow up / go negative / silently diverge?). Set `has_ground_truth=False`.
- **NEW-VERIFIER** — ground truth exists but is *not a terminal moment* (an option price, a BSDE
  scalar value, or a convergence-*rate* check). Does not fit `moments()`; needs a new verifier type.

Current suite for context: 15 SDEs, all EXACT-moment classical families (GBM, OU, CIR, BM,
linear-additive, 2-D GBM, oscillator). Everything below is strictly harder than that baseline.

---

## Tier A — EXACT ground truth (implement these first)

### A1. Multi-channel stiff linear test system  ★ strongest fit
- **SDE**: `dX = F X dt + Σ_r G_r X dW_r`, X ∈ ℝ^d, scan noise-channel count m ∈ {9, 11, 13}.
- **Why hard**: stiff + many noise channels. Counterintuitive result from the source — *noise-channel
  count, not drift stiffness, is the dominant stability killer*, and more implicitness does **not**
  monotonically help. Explicit schemes need impractically small dt.
- **Correct method**: split-step / drift-implicit Milstein; verify with the closed-form
  `S = S^det + S^stoch` Kronecker-product mean-square-stability matrix (test ρ(S) < 1).
- **GT-FIT: EXACT.** Linear SDE ⇒ the mean solves `d E[X]/dt = F E[X]` (matrix exponential) and the
  second-moment matrix solves a linear Lyapunov-type ODE built from F and the G_r. Author these as
  `moments(T)` via `scipy.linalg.expm`. Also gives a free analytic verifier (the stability matrix).
- **Source**: 1411.7080 · wiki: `~/.agon/wiki/1411.7080.md`
- **Suggested**: tier 3, several slugs `sde_multichannel_stiff_m{9,11,13}`.

### A2. Noise-free quintic with random initial condition  ★ great adversarial case
- **SDE**: `dX = −X^5 dt`, X_0 = ξ ~ N(0, σ̄²), σ̄ ∈ {0.1, 1/3}.
- **Exact solution**: `X_t = ξ / (1 + 4 t ξ^4)^{1/4}`.
- **Why hard**: MLMC-Euler diverges **almost surely**; and it *looks* convergent for small N (up to
  N ≈ 2^22) before blowing up — a "false convergence at practical N" trap. Excellent test of whether
  the agent diagnoses asymptotic instability instead of eyeballing a short run.
- **Correct method**: tamed Euler (+ MLMC), rate N^{−1/2+ε}.
- **GT-FIT: EXACT.** Terminal moments = E[X_t^k] over ξ ~ N(0, σ̄²) — a 1-D Gaussian integral of the
  closed-form solution; evaluate by Gauss–Hermite quadrature to machine precision inside `moments(T)`.
- **Source**: 1105.0226 · wiki: `~/.agon/wiki/1105.0226.md`
- **Suggested**: tier 3, `sde_mlmc_quintic_falseconv`.

### A3. Affine jump / squared-volatility families  (partial — derive the affine moment ODEs)
- **alpha-CIR**: `dX = (a − kX)dt + σ₁√|X| dW + σ₂|X_{t−}|^{1/α} dZ`, Z compensated α-stable, α ∈ (1,2).
  Naive EM produces a negative discriminant with positive probability every step; the |discriminant|
  fix drops the rate from ½ to log(n)^{−1}. **Source**: 1804.08070.
- **log-Heston, Feller-violated (hvr: a=0.1, σ=1.0)**: `dX=(r−½Y)dt+√Y(ρdW+√(1−ρ²)dB)`,
  `dY=(a−bY)dt+σ√Y dW`. The fast Ninomiya–Victoir scheme is not merely inaccurate but **ill-defined**
  (σ²>4a) — the agent must detect the regime and fall back to exact CIR sampling. **Source**: 2407.17151.
- **Squared-vol family** (CIR / Aït-Sahalia / Lewis 3/2): `dX=[δ+γX−αX^a]dt+βX^b dW`; Lewis-3/2 only has
  finite moments up to p₀=2α/β²+1 when α>β² — a "does the agent check moment existence first" diagnostic.
  **Source**: 1203.5809.
- **GT-FIT: EXACT (for the affine ones), with derivation work.** CIR and Heston are affine processes, so
  their conditional moments satisfy Riccati/linear ODEs solvable in closed form — this is how the current
  `cir_moments` already works, and it extends to Heston's (X,Y) and to affine jump-diffusions (the jump
  compensator adds a known term to the moment ODE). The log-Heston *reference in the paper* is an option
  price (that part is NEW-VERIFIER), but for benchmarking terminal moments you author the affine formulas.
- **wiki**: `~/.agon/wiki/1804.08070.md`, `2407.17151.md`, `1203.5809.md`
- **Suggested**: tier 3, `sde_alpha_cir_jump`, `sde_log_heston_feller_violated`, `sde_lewis_32`.

---

## Tier B — REFERENCE ground truth (exact solution known; MC-reference + small harness extension)

### B1. Stochastic Ginzburg–Landau  ★ clean superlinear canary
- **SDE**: `dX = (η + σ²/2)X − λX³ dt + σ X dW`, X_0=1, T=1; parameter sweep σ ∈ {2,4,5,6,7} gives a
  ready-made difficulty ladder (σ=2,4 mild bias → σ=5 near-halved estimate → σ=6 partial NaN → σ=7 total NaN).
- **Exact solution**: `X_t = ξ·exp(ηt+σW_t) / √(1 + 2ξ²λ ∫₀ᵗ exp(2ηs+2σW_s) ds)`.
- **Correct method**: tamed / truncated Euler.
- **GT-FIT: REFERENCE.** No elementary moment formula (the path integral resists it), but you can simulate
  the *exact solution* (no discretization error, only MC error) at large N to get reference moments with a
  known tolerance. This is the strongest kind of REFERENCE ground truth — the reference itself has zero
  scheme bias, so it cleanly exposes a wrong scheme.
- **Source**: 0905.0273 (σ-sweep, Table 1) / 1105.0226 · wiki: `~/.agon/wiki/0905.0273.md`, `1105.0226.md`
- **Suggested**: tier 3, `sde_ginzburg_landau_sweep` (one problem per σ, or one with the ladder in the description).

---

## Tier C — SELF-SCORE only (no ground truth; use as stability / no-blow-up benchmarks)

These have `has_ground_truth=False` but are still strong benchmark items: the pass criterion is
"the scheme stays finite / positive / non-divergent," which is exactly what a naive one-shot solver
gets wrong. Good for showing the agentic approach avoids silent blow-up.

- **C1. Quintic-drift SDE**: `dX = −X^5 dt + X dW`, X_0=1, T=1. Naive EM diverges (HJK theorem). Correct:
  tamed Euler, >1000× faster than implicit at matched accuracy. Source 1010.3756 / 0905.0273.
- **C2. FENE boundary-blowup**: `dX = −X/(1−‖X‖²) dt + dW` on ‖X‖<1 (axis 1 × axis 9). Adaptive Euler with a
  boundary-aware step beats tamed/implicit by a documented margin. Source 1609.08101.
- **C3. Cross-validated cubic-drift**: `dx = x(1−x²)dt + σ(1−x²)dw`. Validated by two independent groups.
  Difficulty knob ρ sets required moment order — usable as a programmatic difficulty score. Source 1601.02695.
- **C4. 12-example superlinear catalogue** (van der Pol, Duffing–van der Pol, Lorenz, Brusselator, SIR,
  Lotka–Volterra, squared-vol family, Langevin), each with a hand-built Lyapunov function. Source 1203.5809.
- **C5. Mean-field FitzHugh–Nagumo network**: cubic superlinear drift + √-diffusion (needs full-truncation
  positivity fix) + mean-field coupling. Naive uniform EM shows literal "particle corruption" (blow-up).
  Also a *plausible-wrong-choice* trap: taming exponent α=1 gives **zero** discernible rate; α=½/adaptive works.
  Source 2005.06034.
- **C6. 3-species Chemical Langevin Equation**: reaction rates spanning 11 orders of magnitude (stiff). Source 1411.7080 Ex.3.
- wikis: `~/.agon/wiki/{1010.3756,1609.08101,1601.02695,1203.5809,2005.06034}.md`

---

## Tier D — NEW-VERIFIER (ground truth exists but is not a terminal moment — needs harness work)

Only pursue these if you want to extend the benchmark beyond moment-checking. Each needs a new
verifier type, so they are more work than Tier A–C.

- **D1. Deep BSDE 6-PDE catalogue** (d up to 100): Allen–Cahn (GT 0.052802), HJB (GT 4.5901 via Cole–Hopf),
  borrowing-spread option (GT 21.299), Burgers-type — 3 with **exact analytic** scalar ground truth. Ground
  truth is a scalar solution value u(0,x), not a moment. Tests coarse method-*family* selection ("this needs
  a deep-BSDE / regression-MC solver, not dt-tuning"). Source 1706.04702 · wiki `~/.agon/wiki/1706.04702.md`.
- **D2. Rough-volatility weak-order check**: `dX = f(Ŵ_t)dB_t`, Ŵ Liouville fBm, H ∈ (0,½]. The trap: weak
  order is min(3H+½, 1) with a log correction exactly at H=1/6 — *not* the "obvious" 2H or H+½. Ground truth
  is the convergence *rate*, so scoring = a log-log slope fit + H=1/6 log-correction detection. Source 2212.01591.
- **D3. Lévy-area method audit** (m ≥ 3 non-commutative noise): score whether the Milstein implementation uses
  no Lévy area (silently degrades to Euler order), naive KPW truncation (polynomially worse cost), Wiktorsson
  (never Pareto-optimal), or Mrongowius–Rößler (correct). Oracle: open-source `LevyArea.jl`/`.m`. This is a
  *method-audit* verifier, not a numeric compare. Sources 2101.09542, 2201.08424.

---

## Suggested first pass (highest value / least harness change)

1. **A1 multi-channel stiff linear** — pure EXACT via matrix exponential, and it carries the most
   counterintuitive "wrong method" story (implicitness doesn't help; noise-channel count is the killer).
2. **A2 MLMC false-convergence quintic** — EXACT via quadrature, and the "looks convergent then diverges"
   trap is the single best demonstration that a one-shot solver is fooled where an agent that checks
   asymptotics is not.
3. **B1 Ginzburg–Landau σ-sweep** — REFERENCE via exact-solution MC; the σ ladder gives a built-in
   difficulty gradient in one family.
4. **A3 affine (alpha-CIR + Feller-violated log-Heston)** — EXACT moments via affine ODEs (reuse the
   pattern already in `cir_moments`); positivity/Feller violation is the failure mode.
5. Add a handful of **Tier C** as `has_ground_truth=False` stability benchmarks (C1 quintic, C2 FENE,
   C5 FitzHugh–Nagumo) — cheap to add, strong "naive blows up" demonstrations.

Defer Tier D until you decide to add non-moment verifiers, and (optionally) run a targeted axis-5
(structure-preserving / ergodic) deep-lit tick if you want more EXACT-moment cases — ergodic SDEs
have known stationary moments, which fit this harness better than most of the catalogue.
