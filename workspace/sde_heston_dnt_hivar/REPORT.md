# REPORT — sde_heston_dnt_hivar

**Problem:** SDE, Heston stochastic volatility (2 states S,V; 2 correlated noise channels; Feller ratio 0.222 < 1, so V hits 0). Graded quantity: three continuously-monitored double-no-touch survival probabilities (barriers 50/150, from S0 = 75, 100, 125), rel. tol 0.6%, 400k paths (secondary 100k).

## Ambiguities
- **R27** — problem.md: "Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade." Formulator: terminal moments are in fact exact (E[S]=100, Var S=768.581, CIR mean/var of V) and validated, but are used only as supporting evidence; the graded quantity is the three functionals.

## Provenance of scoring — read this first
Best plan: **9/10, NOT certified, provenance `analytic`**, but the analytic evidence covers **terminal moments only**. The kernel does not read `evaluation_thresholds.functionals`, so **the three graded no-touch probabilities were not verified against any reference by the kernel**. 9 (not 10) because `dynkin_ok` never reached a verdict: the finest Dynkin level needs 12.8 GB of increments vs the 12 GB budget at 400k paths (predicted in the spec; not a solver defect; the path count was not reduced since 400k is the stated grading count).

## Plans (both: scheme_family euler-maruyama, dt=0.002, 400k paths, seed 42)
| Plan | Scheme | Score | Cert | Provenance | est. rel err (measured vs exact moments) | obs. strong order | iters | solve_s |
|---|---|---|---|---|---|---|---|---|
| 2-euler-maruyama-cir-milstein | full-truncation V + scalar Milstein term on V, log-Euler S, per-step Brownian-bridge barrier correction | 9 | no | analytic | 1.8e-3 | 0.242 | 2 (stopped, plateau) | 51.0 |
| 1-euler-maruyama | full-truncation Euler V, log-Euler S, same bridge correction | 9 | no | analytic | 8.5e-3 | 0.287 | 2 (stopped, plateau) | 77.3 |

No-touch estimates (S75 / S100 / S125), 400k paths: plan 2: 0.7914 / 0.8603 / 0.5969; plan 1: 0.7909 / 0.8595 / 0.5969. The two plans agree to ≤ 8e-4 absolute (≤ 1.0e-3 relative), well inside the 0.6% tolerance, but they are the same family and share a seed, so this is weak evidence. 100k-path runs differ from the 400k runs by 6e-4–8e-4 (sampling noise; plan 1's 100k run reuses part of the 400k noise).
Self-checks run by the solvers (not kernel-verified): dt vs dt/4 under common random numbers moves the bridge-corrected values by ≤ 0.15% (plan 2) / ≤ 0.14% (plan 1); grid-only monitoring reads 0.75–2.9% higher, confirming the bridge correction is active.

## Recommendation
**Plan 2 (2-euler-maruyama-cir-milstein).** Tied on score (9), certification (no) and provenance (analytic); decided by the quality index (sqi 0.701 vs 0.563), driven by lower solve cost (51 s vs 77 s) and smaller moment error.

## Plan status
Neither plan certified; both capped at 9 by the Dynkin memory refusal. Cycle 2 solvers made no code change (solver.py sha256 unchanged), so evaluators were not re-run; both treated as plateaued/stopped at iter 2.

## Verification gaps
- Graded no-touch functionals: no kernel-side check; `functional_reference_pde` (2-D Kolmogorov survival PDE) and the QoI plan (bridge consistency, two-path-count, cross-plan agreement) were not executed by the kernel. Only solver-side dt-flatness checks exist.
- Dynkin check: `unavailable` (memory budget) → no certification possible at 400k paths.
- Cross-plan `compare` not run (Path A; both plans are the same family anyway).
- `invariants_not_reported` (both plans): `variance_nonnegative_report`, `no_touch_monotone_structure`.
- Strong order measured 0.24–0.29 vs expected 0.5 (inside the ±0.4 band; consistent with full truncation under a Feller violation); weak order indeterminate.
- Martingale check on S has thin headroom for plan 2 (|E[S_T]−100| = 0.0715 vs bar 0.0877).
- Raw V is negative at T on ~23% of paths; terminal V is reported as max(V,0) (documented).
- Both plans read the same formulator spec; a mis-transcribed equation would pass unanimously.
- Final `gate` run: no findings.
