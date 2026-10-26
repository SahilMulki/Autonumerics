# REPORT: sde_sabr_cev_call_ii_lowvol

**Problem:** SDE, SABR (beta=1/2, rho=-0.6, nu=0.8, T=10), 2-D state (S, sigma), S=0 attainable and absorbing. Graded on three undiscounted call prices (K1,K2,K3) to 1% against an undisclosed reference. Path B (no closed form).

**Ambiguities:** none (33 ledger entries, all mapped).

## Provenance of scoring — read first
Both plans: **provenance `none`, score 2, certified false.** No reference route was available (moment ODE does not close, Kolmogorov is 2-D, not ergodic, Dynkin refused: 160 GB increment array vs 12 GB cap). The score reflects missing evidence, not a detected solver defect; no gate was violated, constraints and invariants were clean, and no agent_cap was applied. **The graded call prices were not verified by the kernel** (functional checks are not wired into the SDE path). The result is NOT a validated pass.

## Plans
| Plan | Scheme | Score | Prov. | est. err | Order | iter |
|---|---|---|---|---|---|---|
| 1-euler-maruyama-absorbing | EM on S with permanent absorption, exact log-step sigma, put-call-parity pricing | 2 | none | n/a (2.2e-14 is the sigma roundoff artefact) | S column 0.513±0.058; sigma column abstained | 2 (cycle 2 solver made no code change) |
| 2-qe-martingale-absorbing | Andersen QE martingale-preserving step on S, exact sigma, parity pricing (family `other`) | 2 | none | n/a (2.3e-14, same artefact) | S column 0.453±0.049; aggregate unresolved | 2 (cycle 2 solver made no code change) |

Prices at 1e6 paths, dt=0.004 (parity estimator): plan 1: 0.049853 / 0.043231 / 0.035146; plan 2: 0.049941 / 0.043346 / 0.035297 (K1/K2/K3). Differences ~0.2–0.4%, inside the 1% tolerance. Raw (non-parity) call estimators run ~0.9–1.2% higher for plan 1; the estimator choice could decide pass/fail against an unknown reference. About 58% of paths are absorbed. E[S_T]: plan 1 0.070426 (+0.6%, EM not a martingale; kept out of prices by parity), plan 2 0.070062.

## Best plan
`1-euler-maruyama-absorbing`, chosen by the ranker on the sqi tiebreak (both score 2/none). That tiebreak is mainly solve cost (171 s vs 199 s) and is NOT a statement that plan 1 is more accurate; plan 2 preserves the martingale property better, and arguably is the more faithful scheme.

## Plateau / stopping
Neither plan certified. After cycle 1 both scored 2 and the evaluators requested no code change, so the cycle-2 solvers left solver.py unchanged (hashes identical to the scored files). Re-evaluation would reproduce the same result, so both were marked `stopped` (plateaued) without a second evaluator pass.

## Path B agreement gate
`cli.py compare` (1e6 paths, dt=0.004; families euler-maruyama vs other): **INCONCLUSIVE** — 5.2e-3 difference on shared seed, 0.97 in moment space on independent seeds (dominated by heavy-tailed sigma variance), worst tol 0.1. Neither a break nor a cap; recorded in agreements.json. Neither plan reached score >=9, so no winner stands on this gate. Caveat: both plans read the same formulator spec, so a mis-transcribed equation would pass unanimously.

## Verification gaps
- Call prices vs reference: not checkable by the kernel.
- Dynkin: unavailable (memory budget); D1/operator validation: unavailable.
- Sigma-column order study abstains (exact log step); aggregate order unresolved.
- Martingale invariant on sigma: tail noise (lognormal, log-variance 6.4).
- Final gate (`cli.py gate`) produced no findings beyond the "reference unavailable" warning.
