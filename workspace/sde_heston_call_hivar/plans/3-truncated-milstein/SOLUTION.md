---
id: 3
plan_slug: truncated-milstein
scheme: milstein
scheme_family: milstein
levy_area_treatment: not_required
expected_strong_order: 0.22
expected_weak_order: 1.0
strategy: Truncated (positivity-preserving) scalar Milstein on the autonomous V equation driven by the combined increment dW2, with log-Euler on S; no Levy area needed because the Milstein correction is applied only to the scalar V channel.
---

## SDE Reference

Heston, Ito, zero interest rate, on t in [0, T], T = 1:

    dS = sqrt(V) * S dW1,                          S(0) = S0
    dV = kappa * (eta - V) dt + sigma * sqrt(V) dW2,  V(0) = V0
    corr(dW1, dW2) = rho

| Parameter | Value | Note |
|---|---|---|
| kappa | 2.58 | mean reversion |
| eta | 0.043 | long-run variance |
| sigma | 1.0 | vol-of-vol (problem.md writes the V diffusion as sqrt(V) dW2) |
| rho | -0.36 | corr(dW1, dW2) |
| r | 0.0 | no drift in S, no discounting anywhere |
| V0 (`Y_0`) | 0.164 | well above eta: V decays toward eta over [0, 1] |
| S0 for `terminal_paths` (`X_0`) | 100.0 | |
| K | 100.0 | strike |
| spots | 75, 100, 125 | one price per spot |

Feller is violated: 2*kappa*eta = 0.22188 < sigma^2 = 1 (CIR dimension delta = 0.44376 < 1), so V
reaches 0 and spends real time near it.

Noise in two independent channels Z1, Z2:

    dW1 = dZ1
    dW2 = rho * dZ1 + sqrt(1 - rho**2) * dZ2

Quantity of interest (the whole external grade): `price_S{75,100,125} = mean((S_T - 100)^+)` for each
spot, undiscounted. Tolerances: 1.5% / 0.4% / 0.2% relative (0.0199 / 0.0412 / 0.0589 in price
units). `price_S125` is the binding one.

## Numerical Scheme

- **Method**: truncated Milstein for V (Hefter and Herzwurm 2018) + log-Euler for S
- **dt**: 0.002 (the contract value; use the `dt` passed in, `Nt = max(1, round(T/dt))`, `dt = T/Nt` -> Nt = 500)
- **num_paths**: 2000000 (graded call), then 500000 (second call, same seed)
- **seed**: 42
- **T**: 1.0
- **Milstein correction** (V channel only, from `problem_spec.json → milstein_correction`):
  `(sigma**2/4)*(dW2**2 - dt)`. The joint 2-D Milstein is not available (`milstein_eligible: false`,
  reason `levy_area`: the commutativity condition fails in the S component). The V equation alone is
  autonomous scalar CIR driven by the single combined increment dW2, so its scalar Milstein
  correction is well defined and needs no Levy area. S is advanced with log-Euler, which has no
  Milstein term. Declared through `levy_area_treatment: not_required` in the frontmatter.
- **Expected strong order**: 0.22 (spec-declared `verification.expected_strong_order`; with delta =
  0.444 no equidistant-increment scheme exceeds delta/2 for V, and truncated Milstein is the scheme
  that attains that lower bound (Hefter, Herzwurm and Mueller-Gronbach 2019). The nominal Milstein
  1.0 is not achievable here; the kernel band admits 0.0-0.62)
- **Expected weak order**: 1.0 (diagnostic only; may read below 1)
- **Reference for scoring**: analytic moments (`analytic_moments.has_analytic_solution == true`: E[S_T] = 100, Var[S_T] = 768.581477379, E[V_T] = 0.0521686544868, Var[V_T] = 0.0115699466208), plus `verification.functional_reference` for the three prices and the cross moment E[S_T V_T] (covariance -0.775885) as a correlation probe

Per step n (state: `logR` = log(S/S0), starting at 0; `V` >= 0, starting at V0):

    dW1    = dZ1
    dW2    = rho*dZ1 + sqrt(1 - rho**2)*dZ2
    sq     = sqrt(V)                                   # V >= 0 always under this scheme
    logR  += -0.5*V*dt + sq*dW1                        # left-point V (Ito)
    V      = maximum( maximum(sq + 0.5*sigma*dW2, 0)**2
                      + (kappa*eta - 0.25*sigma**2 - kappa*V)*dt , 0 )

Expanding the square shows the V update is exactly Milstein,
`V + kappa*(eta - V)*dt + sigma*sqrt(V)*dW2 + (sigma**2/4)*(dW2**2 - dt)`, with two truncations: the
inner `max(., 0)` on `sqrt(V) + sigma*dW2/2` and the outer `max(., 0)` on the result. Here
`kappa*eta - sigma**2/4 = 0.111 - 0.25 < 0`, so the outer truncation is active and required. Use `V`
from the start of the step in the S update.

At T: `R = exp(logR)`; `price_S{s0} = mean(maximum(s0*R - 100, 0))` for s0 in (75, 100, 125);
`terminal_paths = column_stack([100*R, V])`.

## Implementation Notes

- **The V = 0 boundary is the point of the problem.** This scheme keeps V >= 0 by construction, so
  `sqrt` is always safe (gate `finite`) and the reported V needs no output clip (gate
  `variance_nonnegative`). Its truncations are part of a scheme with a proven optimal L1 rate on
  Feller-violated CIR, not an absorption fix applied to an Euler state. Do **not** add a separate
  `V = max(V, 0)` or `|V|` on top, and do not drop either `max` from the update.
- **Sign and argument check on the correction**: the Milstein term is `+ (sigma**2/4)*(dW2**2 - dt)`,
  evaluated with `V_n`, never `V_{n+1}`, and with the *combined* increment dW2, not dZ2. A sign error
  or using dZ2 alone shows up as wrong V moments and a wrong cross moment E[S_T V_T].
- **Correlation**: rho enters the terminal statistics only through E[S_T V_T] (the spec's
  `cross_moment`, expected value 4.44098). Dropping or flipping it fails that check.
- **S via log-Euler** keeps S > 0 (gate `asset_positive`) and makes each step an exact conditional
  martingale, so E[S_T] = S0 up to MC noise (invariant `martingale`, reference 100.0).
- **One simulation serves all three spots.** Simulate R = S/S0 once and price `(s0*R_T - 100)^+` for
  each spot; the three prices share their noise, as problem.md intends.
- **Memory**: draw increments one step at a time: `Z = rng.standard_normal((num_paths, 2))`, scaled
  by sqrt(dt), from `rng = np.random.default_rng(seed)`. A full `(2e6, 500, 2)` block is 16 GB. Keep
  state as float64 arrays of length num_paths; reuse buffers.
- **Verification hooks** (keep the contract signature `solve_sde(num_paths, dt, T, seed=42)` and add
  keyword-only optionals with defaults):
  - `dW`: shape `(num_paths, Nt, 2)`, the **independent** channel increments (dZ1, dZ2), already
    scaled by sqrt(dt). When given, take `Nt = dW.shape[1]`, `dt = T/Nt`, build dW2 from them as
    above, and draw no randomness of your own.
  - `observables`: `{name: phi}` with `phi` taking an `(num_paths, 2)` array of (S, V) for S0 = 100.
    Accumulate the trapezoid path integral `0.5*(phi(X_prev)+phi(X))*dt` and return it under
    `path_integrals`. Build that array only when observables are requested.
- **Return dict**: `terminal_paths` (num_paths, 2) = (S, V) for S0 = 100; `empirical_mean` and
  `empirical_variance` (ddof=1) as Python lists of floats; `functionals` =
  `{"price_S75", "price_S100", "price_S125"}` as Python floats; also `num_paths`, `dt`, `T`.
- **Two-count check**: the 2e6 and 5e5 calls with the same seed must give different prices that agree
  within MC noise. Do not cache, round or hard-code a price; never read
  `verification.functional_reference`.
- **No work at import**; numpy and the standard library only. `dt` is fixed by the contract.
- **Accuracy risk**: the S125 bias must stay well under ~0.1% (payoff s.e. is about 0.07% at 2e6
  paths). The truncated Milstein improves V's pathwise accuracy near zero, but the price bias also
  depends on the left-point `int V dt` in the S step. If the evaluator reports a price bias, the
  first remedy is the S-step quadrature (for example, trapezoid V in the `-0.5*int V dt` term with the
  matching variance in the diffusion term), not a smaller step.
- **Expected ceiling**: the Dynkin ladder at 2e6 paths needs a 64 GB increment array, above the
  kernel's 12 GB budget, so `dynkin_ok` will be `"unavailable"` and the score caps at 9. A resource
  limit stated in the spec, not a solver defect.

## Results

Revision after review (price_S100 bias -0.50% from the V law). solver.py is still truncated Milstein on V driven by dW2 plus log-Euler on S, with a mean-preserving correction of the two truncations: the inner `max` mean deficit D1(V_n) is added to the drift, and the outer `max` is followed by a rescale by target/(target + E[clip gain]), target = V_n + kappa*(eta - V_n)*dt, so the one-step conditional mean equals the untruncated Milstein mean. Both corrections are closed forms in Phi/phi (A&S erfc, numpy only), V >= 0 by construction, no extra clip. `dW` (num_paths, Nt, 2) and `observables` hooks work. Seed 42, dt = 0.002 (Nt = 500), T = 1. Run time about 8 min for both calls.

| num_paths | mean [S, V] | variance [S, V] |
|---|---|---|
| 2000000 | [100.0271, 0.052018] | [769.33, 0.011599] |
| 500000 | [100.0517, 0.051923] | [774.74, 0.011500] |

Prices (undiscounted), with relative difference from `verification.functional_reference` as reported by my own check (not used in the solver):

| num_paths | price_S75 | price_S100 | price_S125 |
|---|---|---|---|
| 2000000 | 1.33164 (+0.5%) | 10.31289 (+0.12%) | 29.44497 (+0.06%) |
| 500000 | 1.33373 | 10.32578 | 29.47345 |

Against the analytic moments: E[V_T] 0.05202 vs 0.05217 (-0.3%, was -3%), Var[V_T] 0.01160 vs 0.01157, Var[S_T] 769.3 vs 768.6. Prior version had E[V_T] 0.05059 and price_S100 10.2484. Known remaining defect: the terminal V law still carries an atom at 0 (about 18.6% at 1e6 paths in a 1e6-path experiment, down from 25.8%), a property of any explicit scheme at delta = 0.44 and dt = 0.002 that the mean correction does not remove; only its first moment is repaired.

## Evaluation

<metrics>
score: 1
certified: false
provenance: none
solver_sha256: 2a4f3769477ed04cff1305a7d3e0ffcfbdc18092b4125eaba384ca09e20ffb23
wall_time_s: 300.0600
</metrics>

<review score=1>

**Score: 1/10**

The kernel ran the current `solver.py` (sha 2a4f3769...) at the contract call `solve_sde(num_paths=2000000, dt=0.002, T=1, seed=42)` and it did not return within the 300 s wall-clock budget (`crash.reason: timeout`). Nothing downstream of the solve was measured. No agent cap was applied.

### Numerical Accuracy
- provenance:              none  (no reference route was reached; the solve never returned)
- mean_rel_err:            not_run (no terminal paths)
- variance_rel_err:        not_run (no terminal paths)
- mc_se_rel:               not_run   resolved: not_run
- observed_order:          not_run (expected 0.22)
- weak_order:              not_run
- dynkin:                  not_run
- terminal_law:            not declared in the spec
- shape:                   not_run
- constraints:             not_run (finite, asset_positive, variance_nonnegative, martingale)
- prices:                  not_run. The previous revision's own figures in Results (price_S100 10.31289, +0.12%) were not reproduced by the kernel and are not evidence here.
- resolution_evidence:     not_used_as_accuracy_evidence
- bound_by:                rubric - the solver crashed or returned non-finite output (certification cap 1: the solver crashed)

### Feedback for solver
- **This is a runtime failure, not an accuracy verdict.** The previous revision of this file (sha 3b7a499a..., kept as `solver.scored.py`) finished the graded call in about 90 s. The mean-preserving revision adds an `_erfc`/`_Phi`/`_pdf` evaluation (several `np.exp`, a `np.where`, fancy-index gather/scatter on the `near` mask) every one of the 500 steps over 2e6 paths, and by your own Results note both calls take about 8 min. The 2e6 call alone is over the 300 s budget. The budget is a kernel default (`timeout_s`, source `kernel_default`), and `dt` and `num_paths` are fixed by the contract, so the cost has to come out of the step.
- Where the time goes, from the source: per step you compute `_inner_deficit` and `_outer_gain` (each with `_Phi` twice or more, each `_Phi` an `_erfc` with a degree-5 polynomial and an `exp`) on the `near` subset, plus the boolean-mask copies `a[near]`, `D1[near] = ...`, `gain[near] = ...`, plus `np.column_stack` allocation only when observables are on. Options that keep the same correction: (a) tabulate `D1(a)` and `gain` on a grid in `a` and the clip parameter and use `np.interp`, which replaces the transcendental calls with a lookup; (b) since the correction only matters for small `a`, tighten the `near` cutoff and check how many paths it actually selects at the V0 = 0.164 start versus near V = 0; (c) preallocate buffers and use in-place ufuncs (`out=`) to cut the temporaries on 2e6-length arrays; (d) fuse the two Gaussian draws into one `standard_normal((num_paths, 2))` as you already do, and avoid `np.where` in `_erfc` by evaluating on `|x|` and reflecting with arithmetic.
- **Do not trade away the accuracy fix to get under the budget.** The prior kernel pass on the 3b7a499a revision (before the mean correction) found price_S100 at 10.2484 against the reference 10.30017 (-0.503%, z = -4.2), with 25.8% of terminal V exactly 0 and E[V_T] 3% low. Your revision reports an improvement to E[V_T] 0.05202 (-0.3%) and price_S100 +0.12%, but that is your own run and has not been confirmed by the kernel. Once the solve fits the budget it will be graded on the moments, the strong-order ladder, the cross moment and the martingale invariant, and the three gated prices must still be checked against `verification.functional_reference` with their payoff standard errors.
- **Known limitation you flagged that will not clear by speeding up:** the terminal V law still carries an atom at zero (about 18.6% in your 1e6-path experiment). Only the first moment is repaired, so the second moment and the shape of V_T, and prices that depend on the V law near 0, can still be off. If you can only meet the budget by dropping the mean correction, say so, and consider instead sampling V exactly (noncentral chi-square, df = 0.444, as the spec's `terminal_law_note` describes) with a conditional S step, which would remove the atom and the dt bias together.
- Keep the `dW` hook of shape (num_paths, Nt, 2) and the `observables` hook exactly as they are, and keep `Nt = dW.shape[1]` on the hook path.
- Expected ceiling after the crash is fixed: `dynkin_ok` is declared unavailable by the spec (64 GB increment array against a 12 GB budget), which caps the rubric at 9.
- Frontmatter: `levy_area_treatment: not_required` is a legitimate declaration for a scalar Milstein on V with log-Euler on S. A previous pass recorded the kernel not reading that key from frontmatter, which bound the kernel score at 8 independently of accuracy. That is a kernel/spec plumbing gap, not a solver defect, and it was not exercised this time because the solve crashed first.

</review>
