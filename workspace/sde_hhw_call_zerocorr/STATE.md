phase: running
problem_type: sde
path: B
problem_spec: workspace/sde_hhw_call_zerocorr/problem_spec.json
ambiguities:
  - R30: 'problem.md says "this problem has no terminal-moment formula, so the quantity of interest below is the whole grade"; formulator kept the grade as the 3 call prices (evaluation_thresholds.functionals) and used E[S_T] only as a surrogate diagnostic, since Var(S_T) is unresolvable at 1e6 paths'
plans:
  1-euler-maruyama-full-truncation:
    one-sentence: Euler-Maruyama with full-truncation V, log-Euler S, pathwise trapezoid discounting, control variates; dt=0.004, 1e6 paths.
    iter: 2
    score: 3
    certified: false
    provenance: surrogate
    est_err: 2.255e-03
    state: stopped
  2-qe-log-euler:
    one-sentence: Andersen QE scheme for V with log-Euler S and exact OU for R; dt=0.004, 1e6 paths.
    iter: 2
    score: 3
    certified: false
    provenance: surrogate
    est_err: 4.5e-04
    state: await_solver
