---
phase: running
problem_type: sde
path: A
problem_spec: workspace/sde_kubo_oscillator/problem_spec.json
ambiguities:
  - R20: 'problem.md gates a pathwise relative strong error < 20%; no evaluation_thresholds key
      carries a pathwise strong-error tolerance and the solver contract returns no scalar
      functional, so it is carried instead by the exact moments (5% mean / 10% variance), the CRN
      strong-order floors (0.5 EM / 1.0 Milstein) and the exact pathwise solution recorded at
      analytic_moments.exact_pathwise_solution. Recorded as a vocabulary gap, not a relaxation.'
pool_findings: 'cli.py gate at plan-creation: ok, no findings. Plan-creator declined an
  exact-transition plan on purpose (problem.md excludes a solver that evaluates the closed-form
  rotation as transcription; R21 leaves exact_sampling_components undeclared), declined a
  variance-reduction slot (no declared functional), and declared no boundary_treatment (spec
  declares no barrier; conserved amplitude is carried as the ungated martingale constraint).'
plans:
  1-stratonovich-midpoint:
    one-sentence: Implicit midpoint on the Stratonovich coefficients, closed-form Cayley rotation; conserves |X| pathwise to roundoff, strong order 1.
    iter: 1
    score: 3
    certified: false
    provenance: analytic
    est_err: 2.34e-02
    note: 'Dynkin fail, z=8.24 vs bar 2.43, on the exactly-conserved test function X1^2+X2^2 only
      (residual -2.6e-11 +/- 3.1e-12, i.e. roundoff scale); X1 and X2 clean. Evaluator records this
      as a probable kernel roundoff-guard false positive and applied no cap. Order 0.963 +/- 0.011
      against a floor of 1.0; moments well inside tolerance.'
    state: await_solver
  2-milstein:
    one-sentence: Explicit Milstein on the Ito-converted drift with -0.5*sigma^2*X*(dW^2-dt); no Levy area since noise_dimension=1, strong order 1.
    iter: 1
    score: 5
    certified: false
    provenance: analytic
    est_err: 9.70e-02
    bias_retry: true
    bias_retry_moment: variance (moment_bias_frac_tol 0.9033)
    note: 'Bound by the non-gate martingale invariant: E|X_T|^2 = 1.0728, drift 0.0728 vs bar
      7.9e-4. Dynkin unresolved (windows disagree by 14 s.e.). Variance error 9.70% against a 10%
      bar.'
    state: await_solver
  3-euler-maruyama:
    one-sentence: Plain Euler-Maruyama on the Ito form, expected strong order 0.5; baseline.
    iter: 1
    score: 5
    certified: false
    provenance: analytic
    est_err: 6.04e-02
    bias_retry: true
    bias_retry_moment: variance (moment_bias_frac_tol 0.6341)
    note: 'Bound by the same non-gate martingale invariant: drift 0.0513 vs bar 2.82e-3. Dynkin
      passed marginally (median z 2.12 over 3 seeds vs bar 2.43). Order 0.531 +/- 0.012.'
    state: await_solver
