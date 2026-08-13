"""
Evaluator for sde_fene_blowup / plan 3 (tamed-euler-maruyama).

This problem has NO analytic moments (problem_spec.json ->
analytic_moments.has_analytic_solution == false). It is a STABILITY
benchmark, not an accuracy benchmark. The pass criterion (from
problem_spec.json -> evaluation_thresholds) is:

  - all terminal states X(T) must be finite (no NaN / Inf)
  - all terminal states must satisfy |X(T)| < 1  (stay inside the open
    domain (-1, 1) imposed by the singular FENE drift)

Empirical mean/variance are reported for reference only (no exact
values to compare against).
"""

import numpy as np

from solver import solve_sde

# Hyperparameters from SOLUTION.md
NUM_PATHS = 50000
DT = 0.005
T = 1.0
SEED = 42

# Thresholds from problem_spec.json -> evaluation_thresholds
DOMAIN_LOWER, DOMAIN_UPPER = -1.0, 1.0


def main():
    result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

    X = np.asarray(result["terminal_paths"])
    empirical_mean = result["empirical_mean"]
    empirical_var = result["empirical_variance"]
    dt_used = result["dt"]

    all_finite = bool(np.all(np.isfinite(X)))
    # Guard against inf/nan when computing max abs (avoid crash on print)
    if all_finite:
        max_abs = float(np.max(np.abs(X)))
        all_in_domain = bool(np.all(np.abs(X) < 1.0))
        n_violations = int(np.sum(np.abs(X) >= 1.0))
    else:
        finite_mask = np.isfinite(X)
        max_abs = float(np.max(np.abs(X[finite_mask]))) if np.any(finite_mask) else float("nan")
        all_in_domain = False
        n_violations = int(np.sum(~finite_mask | (np.abs(np.nan_to_num(X, nan=2.0, posinf=2.0, neginf=2.0)) >= 1.0)))

    stability_pass = all_finite and all_in_domain
    overall_pass = stability_pass

    print("=== Evaluation Results ===")
    print("Scheme:              tamed-euler-maruyama (with reflection)")
    print(f"dt:                  {dt_used}")
    print(f"num_paths:           {NUM_PATHS}")
    print(f"T:                   {T}")
    print()
    print("--- Stability benchmark (no analytic moments) ---")
    print(f"Domain:              ({DOMAIN_LOWER}, {DOMAIN_UPPER})")
    print(f"All finite:          {all_finite}  ({'PASS' if all_finite else 'FAIL'})")
    print(f"Max |X(T)|:          {max_abs:.12f}")
    print(f"All |X(T)| < 1:      {all_in_domain}  ({'PASS' if all_in_domain else 'FAIL'})")
    print(f"# domain violations: {n_violations}")
    print()
    print(f"Empirical mean:      {empirical_mean:.6f}  (reference only, no exact value)")
    print(f"Empirical variance:  {empirical_var:.6f}  (reference only, no exact value)")
    print()
    print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")


if __name__ == "__main__":
    main()
