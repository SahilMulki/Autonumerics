---
phase: done
problem_type: sde
path: A
problem_spec: workspace/sde_kubo_oscillator/problem_spec.json
ambiguities:
  - R7: 'The harness checks that the sample you return has conserved it; a scheme that reads the coefficients as Ito coefficients solves a different equation, whose amplitude grows. (tolerance unstated; formulator encoded a gated constraint: mean amplitude deviation over paths < 0.25)'
  - R8: 'This problem is graded **path by path**, not in distribution. (no evaluation_thresholds key for a path-by-path reference; carried via exact moments, CRN strong-order study, and recorded exact_pathwise_solution)'
  - R9: 'gates the relative strong error `sqrt(mean((X_num - X_exact)^2)) / sqrt(mean(X_exact^2))` over paths at < 20% (no kernel key for a path-by-path error tolerance; not a kernel threshold)'
pool_findings: none
agreement_gate: all three pairs INCONCLUSIVE at num_paths=2000, dt=0.04 (rel diff 5.7e-2 / 7.4e-2 / 7.0e-2 vs tol 0.05); recorded in agreements.json
ranker_decided_by: tie
plans:
  1-cayley-midpoint:
    one-sentence: Stochastic implicit midpoint (Cayley) on the Stratonovich form; explicit rotation by 2*arctan((a*dt+sigma*dW)/2), preserves |X|=1 to round-off.
    iter: 2
    score: 6
    certified: false
    provenance: analytic
    est_err: 3.40e-02
    resolved: false
    mc_se_rel: 3.43e-02
    unresolved_retry: true
    unresolved_retry_instruction: 'functional_limited_by unavailable (no functional declared) -> raise num_paths until mc_se_rel is below the accuracy the tolerance demands'
    stopped_reason: unresolved (resolved false after the unresolved retry; solvers had no lever on the graded path count)
    state: stopped
  2-stochastic-heun:
    one-sentence: Stochastic Heun predictor-corrector on the Stratonovich form, no Ito correction.
    iter: 2
    score: 6
    certified: false
    provenance: analytic
    est_err: 9.34e-02
    resolved: false
    mc_se_rel: 3.40e-02
    unresolved_retry: true
    unresolved_retry_instruction: 'functional_limited_by unavailable (no functional declared) -> raise num_paths until mc_se_rel is below the accuracy the tolerance demands'
    stopped_reason: unresolved (resolved false after the unresolved retry; solvers had no lever on the graded path count)
    state: stopped
  3-milstein:
    one-sentence: Milstein on the Ito form with drift -0.5*sigma^2*X and correction 0.5*(-sigma^2 X)(dW^2 - dt).
    iter: 2
    score: 6
    certified: false
    provenance: analytic
    est_err: 1.502e-01
    resolved: false
    mc_se_rel: 3.18e-02
    unresolved_retry: true
    unresolved_retry_instruction: 'functional_limited_by unavailable (no functional declared) -> raise num_paths until mc_se_rel is below the accuracy the tolerance demands'
    stopped_reason: unresolved (resolved false after the unresolved retry; solvers had no lever on the graded path count)
    state: stopped
best_plan: 1-cayley-midpoint
best_provenance: analytic
---
