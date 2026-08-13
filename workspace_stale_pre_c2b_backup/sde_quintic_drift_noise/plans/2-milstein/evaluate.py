"""Evaluator for plan 2-milstein (sde_quintic_drift_noise).

This problem (dX = -X^5 dt + X dW) has NO closed-form solution:
problem_spec.json sets analytic_moments.has_analytic_solution = false and both
mean/variance expressions to null. It is a *stability* benchmark (see
problem_spec.json -> implementation_notes and problem.md): the independent
check is only that a correctly-tamed scheme keeps the terminal states finite,
since plain (untamed) Euler-Maruyama / Milstein is known to diverge for this
superlinear drift (Hutzenthaler-Jentzen-Kloeden). This mirrors
benchmark/verify.py::verify_sde_stability, which uses
stability_check = {"type": "finite"} for this exact problem slug.

Because there is no analytic mean/variance, the standard
mean_rel_err / var_rel_err thresholds from project_manual.md do not apply here
(there is nothing to compare the empirical moments against). Instead:

  - PRIMARY pass criterion: all terminal_paths finite (no Inf/NaN).
  - SECONDARY (diagnostic only, not required for pass): re-run at a smaller
    dt (dt/2) and confirm the empirical mean/variance are of the same order
    of magnitude (i.e. the scheme is not just "finite by luck" at this dt but
    numerically stable under refinement).
"""

import numpy as np

from solver import solve_sde

# Hyperparameters (from SOLUTION.md "Numerical Scheme" section / problem_spec.json defaults)
NUM_PATHS = 50000
DT = 0.01
T = 1.0
SEED = 42

SCHEME = "milstein (tamed drift + Milstein diffusion correction)"


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    terminal_paths = np.asarray(result["terminal_paths"], dtype=float)
    empirical_mean = float(result["empirical_mean"])
    empirical_var = float(result["empirical_variance"])

    finite_mask = np.isfinite(terminal_paths)
    finite_fraction = float(finite_mask.mean()) if terminal_paths.size else 0.0
    all_finite = bool(finite_mask.all()) and terminal_paths.size > 0
    moments_finite = np.isfinite(empirical_mean) and np.isfinite(empirical_var)

    # Diagnostic refinement check: halve dt, confirm still finite and moments
    # are in the same ballpark (loose factor-of-3 sanity band, not a strict
    # threshold -- this is just to catch "finite but secretly blowing up".
    refine_result = solve_sde(num_paths=NUM_PATHS, dt=DT / 2.0, T=T, seed=SEED)
    refine_paths = np.asarray(refine_result["terminal_paths"], dtype=float)
    refine_finite = bool(np.isfinite(refine_paths).all()) and refine_paths.size > 0
    refine_mean = float(refine_result["empirical_mean"])
    refine_var = float(refine_result["empirical_variance"])

    mean_consistent = refine_finite and abs(refine_mean - empirical_mean) < 0.5 * max(abs(empirical_mean), 1.0)
    var_consistent = refine_finite and abs(refine_var - empirical_var) < 3.0 * max(abs(empirical_var), 1e-6)
    refinement_stable = refine_finite and mean_consistent and var_consistent

    overall_pass = bool(all_finite and moments_finite)

    print("=== Evaluation Results ===")
    print(f"Scheme:              {SCHEME}")
    print(f"dt:                  {result['dt']}")
    print(f"num_paths:           {result['num_paths']}")
    print(f"T:                   {result['T']}")
    print()
    print("Ground truth: NONE (has_analytic_solution = false in problem_spec.json).")
    print("This is a STABILITY benchmark -- pass/fail is determined by finiteness")
    print("of the terminal states, not by comparison to an analytic mean/variance.")
    print()
    print(f"Empirical mean:       {empirical_mean:.6f}")
    print(f"Empirical variance:   {empirical_var:.6f}")
    print(f"Fraction finite:      {finite_fraction:.6f}")
    print(f"All terminal finite:  {all_finite}  ({'PASS' if all_finite else 'FAIL'})")
    print(f"Reported moments finite: {moments_finite}  ({'PASS' if moments_finite else 'FAIL'})")
    print()
    print("--- Diagnostic: refinement check at dt/2 (not required for pass) ---")
    print(f"dt/2 = {DT / 2.0}")
    print(f"Refined mean:         {refine_mean:.6f}")
    print(f"Refined variance:     {refine_var:.6f}")
    print(f"Refined run finite:   {refine_finite}")
    print(f"Moments stable under refinement: {refinement_stable}")
    print()
    print(f"Overall (stability check): {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
