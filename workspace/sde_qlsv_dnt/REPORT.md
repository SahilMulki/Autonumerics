# REPORT — sde_qlsv_dnt

**Type:** SDE (Heston variance V driving a quadratic local-vol asset S; 2 states, 2 noise channels, non-commuting noise so Milstein is ineligible). **Path B** (no closed form).

## Ambiguities
- R36: problem.md — "They must differ ... and they must agree to within the sampling noise the two counts imply." No numeric bar given; the formulator used |p_400k − p_100k| ≤ 2·sqrt(se400² + se100²).

## Provenance of the scoring
**`self_convergence`** — the weakest tier. The kernel had no reference for S (moment ODE, Kolmogorov, stationary density all unavailable), so no plan is certified and the kernel ceiling is 8. The three graded quantities (no-touch probabilities for S0 = 75/100/125) are **not read by the kernel**; they were never graded against the 2-D backward-Kolmogorov reference the spec describes.

## Plans (both: Euler–Maruyama family, full-truncation V, Brownian-bridge continuous barrier weights)
| Plan | Scheme | Score | Cert | Provenance | est. rel. err | Order (obs / expected) | Iter | State |
|---|---|---|---|---|---|---|---|---|
| 2-euler-maruyama-drift-implicit | drift-implicit positive-root step in R=1/(c·S), uses supplied dW | 7 | no | self_convergence | 1.7e-2 (estimate) | 0.317 / 0.5 | 2 | stopped (plateau) |
| 1-euler-maruyama-bessel | exact Bessel-3 step in R, extra auxiliary Gaussians | 7 | no | self_convergence | 1.7e-2 (estimate) | 0.317 / 0.5 | 2 | stopped (plateau) |

Solver-reported functionals (400k paths, dt=0.002, seed 42), plan 2: 0.93383 / 0.91433 / 0.59207 (s.e. 0.00039/0.00044/0.00077); plan 1: 0.93392 / 0.91464 / 0.59236. The two independent schemes agree within ~0.05%. Solver-side checks: at ρ=0 both match the exact series (0.9466/0.8853/0.5773) within MC error; at ρ=−0.36 dt=0.002 vs 0.0005 differs ≤0.13%. 100k vs 400k path-count check passes for both. These are not kernel-graded evidence.

## Recommendation
**Plan 2** — tied on score and provenance; ranked first by SQI (0.692 vs 0.538), decided by solve cost (61 s vs 77 s) and full use of the supplied dW (fully coupled strong-order study).

## Why 7 and not higher
Binding rule is the rubric row via `invariants_ok: false`, caused solely by the spec's non-gate, report-only `variance_nonnegative` (18.5% of V_T negative — intrinsic to full truncation when Feller fails; clipping would be wrong). Without it the self-convergence ceiling is 8. Observed strong order 0.317 is inside the two-sided band around 0.5 (Feller-violated CIR rate).

## Verification gaps
- No kernel reference for S; moments/variance of S not measured (S variance s.e. ~8.5%, heavy tail; S is a strict local martingale).
- **The ρ=−0.36 2-D Kolmogorov PDE reference was never run**; the actual grade (0.8% on the three probabilities) is unverified. Only ρ=0 exact-series and refinement-bias checks exist (solver-side).
- Dynkin check `unavailable`: finest ladder level needs 12.8 GB vs 12 GB budget (400k paths × 2000 steps × 2). No circularity break earned.
- Plan 1's auxiliary Bessel draws are not coupled across CRN levels; its bridge weight drops the double-crossing term (unquantified against reference).
- Single scheme family (euler-maruyama): `single_family_pool: true`. The agreement gate (≥9 scores) did not apply; the second break could not be earned. Both plans read the same spec operator, so a mis-transcribed equation would pass unanimously.
- E[V_T] is ~1% low (≈2–3 s.e.) vs exact 0.04838 — full-truncation bias.
- Spec defects noted by evaluators: Kolmogorov route declared as `functional_reference` (not kernel-routable); non-gate invariant still lowers `invariants_ok`.
- Final gate warning (plan 2): `state_clip` — `Rn = maximum(... Rn ...)` at solver.py line 47 clips the R state (positivity guard in the implicit-step root); full truncation of V itself is inside coefficients only.
