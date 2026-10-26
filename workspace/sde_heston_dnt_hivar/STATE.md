phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_heston_dnt_hivar/problem_spec.json
ambiguities:
  - R27: 'problem.md: "Scoring is independent of any self-report: this problem has no terminal-moment formula, so the quantity of interest below is the whole grade." Formulator: terminal moments are in fact exact (validated), kept as supporting evidence only; the graded quantity is the three no-touch functionals (evaluation_thresholds.functionals, rel_tol 0.006).'
plans:
  1-euler-maruyama:
    one-sentence: Euler-Maruyama, full-truncation V, log-Euler S, Brownian-bridge continuous barrier correction, dt=0.002, 400k paths.
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 8.533e-03
    state: stopped
  2-euler-maruyama-cir-milstein:
    one-sentence: Same as plan 1 with scalar Milstein correction on the V equation only (EM family).
    iter: 2
    score: 9
    certified: false
    provenance: analytic
    est_err: 1.805e-03
    state: stopped
note: cycle 2 solvers returned byte-identical solver.py (sha256 unchanged); score is a deterministic function of that file, so evaluators were not re-run. Both plans treated as plateaued (iter 2, no improvement) -> stopped.
best_plan: 2-euler-maruyama-cir-milstein
best_provenance: analytic
