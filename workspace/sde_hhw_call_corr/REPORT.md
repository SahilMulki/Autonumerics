# REPORT — sde_hhw_call_corr

**Problem:** SDE, Heston–Hull–White (S, V, R; three correlated noise channels, multiplicative, Milstein ineligible: Lévy area). Goal: three European call prices (S0 = 75/100/125) at 1,000,000 paths, dt = 0.004, T = 10, tolerance 0.3%.

**Ambiguities:** none recorded (32 requirements, all `mapped`, none ambiguous/dropped).

**Provenance of scoring (read first):** Path B — no closed form, no usable moment ODE/Kolmogorov/stationary reference. Both plans are `self_convergence`: the strongest evidence the kernel can give is a ceiling of 8, and neither certified. **The 0.3% price accuracy is NOT established by this run.** The prices below are unverified estimates.

## Plans
| Plan | Scheme | Score | Provenance | est. err (estimate) | Obs. strong order (S/V/R) | Iter | State |
|---|---|---|---|---|---|---|---|
| 1-euler-maruyama-full-truncation | EM, full-truncation V, log-S, shared left-point rate | 7 | self_convergence | 8.9e-07 | 0.033 / 0.077 / 1.015 | 2 | stopped (plateau) |
| 2-qe-martingale-corrected | Andersen QE V (driven by dW2), martingale-corrected log-S, exact R, trapezoid discount | 7 | self_convergence | 8.7e-06 | 0.037 / 0.087 / 0.999 | 2 | stopped (plateau) |

Prices at 1e6 paths (S75 / S100 / S125): plan 1 = 37.578 / 57.480 / 78.646; plan 2 = 37.575 / 57.479 / 78.647 (control-variate SE ≈ 0.03–0.04). Hand-checked: Black–Scholes limit (σv=σr=0) matches exact to ~0.01%; no-arbitrage bounds, monotonicity/convexity hold; 1e6 vs 250k path counts agree within noise (0.1–0.2%).

**Best plan (by `rank`): 1-euler-maruyama-full-truncation.** Score and provenance tie; decided by sqi (0.692 vs 0.616), driven by the structural-margin and cost terms. This is a tie-break, not a claim that plan 1 is more accurate; the two price sets agree to ~0.01%.

## Why 7 and not higher
- Score 7 is the kernel's row for "Dynkin did not reach a verdict" with no reference; the 9 row needed `dynkin_ok`. Dynkin was **unavailable** (finest level needs ~240 GB vs 12 GB budget at 1e6 paths — not a solver fault).
- Plan 1: V Richardson check unstable (shift 0.94 s.e., 8.3%); `invariants_ok: false` because full-truncation V is negative on 64.9% of terminal paths (`variance_nonnegative`, a diagnostic the spec marks informational but which the kernel counts).
- Both plans stopped after two cycles: the solver agents made no code change (evaluator found no defect; hash unchanged) so a re-evaluation would be identical. Plateau rule applied without re-running the evaluator.

## Agreement gate (Path B)
`compare` between plans 1 and 2 at 1e6 paths, dt 0.004: **INCONCLUSIVE** — moment-space difference 3.6e-2 (independent seeds), 2.3e-2 (shared seed) vs tol 0.1; MC interval straddles the tolerance, so neither a break nor a cap. The kernel tagged both plans `euler-maruyama` family, so the pool is effectively **single-family** (`single_family_pool: true`); the second circularity break could not be earned. Both plans read the same formulator operator, so a mis-transcribed equation would pass unanimously.

## Verification gaps
- No reference for S; `moments_ok`/`resolved` unavailable.
- Prices/functionals (0.3%) and the 250k-vs-1M path-count check are not graded by the kernel on the SDE path; they were checked by hand by the evaluators.
- Dynkin unavailable (memory budget). `scheme_family` left undeclared deliberately (a declared family imposes a two-sided strong-order band that a correct R component, order ≈1, would fail; V's order is ~0.1 near the Feller-violating regime).
- Discretisation bias at the 0.1–0.3% level is not resolved: CRN dt ladder (240k paths) moves within ±0.07%, ≈1.3 s.e.
- Plan 2 E[D]=0.7979 vs Vasicek 0.79684 is within 1 s.e. (0.0015).
- Gate warning: plan 1 `maximum(..., out=Vp)` flagged as a state clip (line 80); intended as full truncation of V.
- Plan 2 runs ~120–155 s using forked workers; serial fallback would exceed the 300 s limit.
- Formulator issue: declared invariant `variance_nonnegative` is violated by the spec-endorsed full-truncation scheme.
