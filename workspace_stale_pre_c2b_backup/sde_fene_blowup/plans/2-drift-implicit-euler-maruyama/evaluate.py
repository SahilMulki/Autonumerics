"""
Evaluator for the FENE nonlinear-spring SDE (sde_fene_blowup), plan 2:
drift-implicit Euler-Maruyama.

This problem has NO analytic moments (problem_spec.json ->
analytic_moments.has_analytic_solution = false). It is a STABILITY
benchmark, not an accuracy benchmark. The pass criterion (from
problem_spec.json -> evaluation_thresholds) is:

    - every terminal state X(T) must be finite (no NaN / Inf)
    - every terminal state must satisfy |X(T)| < 1 (stay inside the
      open domain (-1, 1) enforced by the confining drift)

Empirical mean/variance are reported for reference only; there is no
exact value to compare them against, so no mean/variance thresholds
apply here.
"""

import numpy as np

from solver import solve_sde

# Hyperparameters from SOLUTION.md / problem_spec.json
NUM_PATHS = 50000
DT = 0.01
T = 1.0
SEED = 42

STABILITY_DOMAIN = (-1.0, 1.0)


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)
    X = np.asarray(result["terminal_paths"])

    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]
    dt_used = result["dt"]

    all_finite = bool(np.all(np.isfinite(X)))
    # If not finite, np.abs/np.max would choke on inf but not crash; guard anyway.
    if all_finite:
        max_abs = float(np.max(np.abs(X)))
    else:
        finite_mask = np.isfinite(X)
        n_nonfinite = int(np.size(X) - np.count_nonzero(finite_mask))
        max_abs = float(np.max(np.abs(X[finite_mask]))) if np.any(finite_mask) else float("inf")

    within_domain = bool(all_finite and np.all(np.abs(X) < STABILITY_DOMAIN[1]))

    overall_pass = all_finite and within_domain

    print("=== Evaluation Results (Stability Benchmark) ===")
    print(f"Scheme:              drift-implicit Euler-Maruyama")
    print(f"dt:                  {dt_used}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print()
    print("This SDE has no analytic moments (has_analytic_solution=false).")
    print("Pass criterion: all terminal states finite AND |X(T)| < 1.")
    print()
    print(f"Empirical mean:      {empirical_mean:.6f}  (reference only, no exact value)")
    print(f"Empirical variance:  {empirical_var:.6f}  (reference only, no exact value)")
    print()
    print(f"All terminal states finite:        {all_finite}")
    if not all_finite:
        print(f"  Number of non-finite paths:       {n_nonfinite}")
    print(f"Max |X(T)| over all paths:          {max_abs:.12f}")
    print(f"All |X(T)| < 1 (in stability domain): {within_domain}")
    print()
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")

    return {
        "all_finite": all_finite,
        "within_domain": within_domain,
        "max_abs": max_abs,
        "empirical_mean": empirical_mean,
        "empirical_var": empirical_var,
        "overall_pass": overall_pass,
    }


if __name__ == "__main__":
    main()
