# Report: sde_heston_call_hivar

**Type:** SDE (Heston stochastic volatility, 2-D, correlated noise, Feller violated). Path A (closed-form moments).

## Ambiguities
- R30: problem.md says "this problem has no terminal-moment formula, so the quantity of interest below is the whole grade". The formulator kept the three call prices as the external grade (evaluation_thresholds.functionals) but also declared exact terminal moments (CIR moments for V, E[S]=S0, E[S^2] by Riccati, E[SV]) validated via moment ODE (3.5e-13) and the Heston characteristic function (2e-16). The solved problem is unchanged; the evidence is an addition.

## Provenance
All scored plans: `analytic` (closed form validated against the spec's own generator). No plan is `certified`: the Dynkin generator check is `unavailable` (64 GB increment array vs 12 GB budget at the contract's 2e6 paths, dt=0.002). This caps the score at 9 for every plan. It is a resource limit, not a solver defect, and `num_paths` is a hard-gated contract value.

## Plans
| Plan | Scheme | Score | Prov. | est. rel err (measured) | Obs. strong order | iter | Status |
|---|---|---|---|---|---|---|---|
| 1-full-truncation-euler | Full-truncation EM on V, log-Euler S | 9 | analytic | 2.5e-3 | 0.336 (expected 0.22) | 2 | stopped (plateau; refine left solver unchanged) |
| 2-andersen-qe | Andersen QE + martingale-corrected S | 9 | analytic | 2.4e-3 | 0.357 (S) / 0.273 (V) | 2 | stopped (plateau; solver unchanged) |
| 3-truncated-milstein | Milstein on V, truncated, log-Euler S | 1 | none | not measured | – | 2 | stopped (regressed) |

Prices (evaluator-graded against functional_reference at 2e6 paths; the kernel does not grade SDE functionals): 
- Plan 1 relative errors +5.1e-3 / +1.5e-3 / +7.4e-4 for S0=75/100/125 (tolerances 1.5e-2 / 4e-3 / 2e-3): pass.
- Plan 2 +3.7e-3 / +1.1e-3 / +6.7e-4: pass.
- Both 2e6 and 5e5 runs agree within MC noise. Offsets are ~1 payoff s.e., so discretization bias is below resolution, not shown to be zero.

Plan 3: iteration 1 scored 4 (evaluator `agent_cap` from kernel 8): price_S100 was -0.503% vs 0.4% tolerance, with an atom of 25.8% at V=0 and E[V_T] 3% low. The refine added mean-preserving corrections that fixed the prices in the solver's own run, but made the solver exceed the kernel's 300 s timeout (score 1, crash). The `solver.py` on disk is the timed-out version that the score 1 describes. The kernel also did not read `levy_area_treatment: not_required` from frontmatter, which bound plan 3 at 8 (kernel plumbing gap, not a solver defect).

## Recommendation
**Best plan: 1-full-truncation-euler.** Tied with plan 2 on score (9) and provenance (analytic); decided by the SQI (0.708 vs 0.556), driven by solve cost (80 s vs 217 s; plan 2 is within ~28% of the 300 s timeout). Both are equally accurate within error bars. Plan 2 has smaller price biases and does not rely on truncation, but costs 2.7x more.

## Verification gaps
- Dynkin generator check unavailable (memory budget) for every plan; no plan certified. Nothing establishes by kernel check that paths come from this SDE beyond the moments, cross moment E[SV] (correlation correct), order study and martingale invariant.
- Price check was done by the evaluator by hand; the kernel's functional checks only run for PDEs.
- Weak order indeterminate (noise floor); `order` input to SQI not reported.
- Plan 2 has an atom at V_T=0 (17.8% of paths) that the true law lacks; harmless for the graded statistics but would fail a declared terminal law. Plan 1's V uses full truncation (V+ only at output).
- Gate findings (warnings): `state_clip` on plans 1 and 3 (the `maximum(..., out=Vp)` is the output V+ / auxiliary clip, which plan 1's evaluator inspected and judged not a state-clip defect).
- No Path B agreement gate (Path A).
- Plan 3's refined solver was never successfully evaluated; its fix is unconfirmed by the kernel.
