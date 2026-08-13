import numpy as np

from solver import solve_sde

# Hyperparameters from SOLUTION.md "Results" section (selected final configuration)
NUM_PATHS = 50000
DT = 0.0002
T = 1.0
SEED = 42

# Analytic (reference) moments from problem_spec.json -> analytic_moments
# No elementary closed-form; reference is a discretization-free Monte-Carlo of the
# exact pathwise solution, stable across seeds/grid refinement.
EXACT_MEAN = 0.659
EXACT_VAR = 1.117

# Thresholds from problem_spec.json -> evaluation_thresholds
VAR_THRESHOLD = 0.10
MEAN_THRESHOLD = 0.05
NEAR_ZERO_MEAN_THRESHOLD = 0.01

scheme = "euler-maruyama (tamed drift)"

result = solve_sde(num_paths=NUM_PATHS, dt=DT, T=T, seed=SEED)

empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]
dt_used = result["dt"]

finite = np.all(np.isfinite(result["terminal_paths"]))

mean_rel_err = abs(empirical_mean - EXACT_MEAN) / max(abs(EXACT_MEAN), 1e-10)
var_rel_err = abs(empirical_var - EXACT_VAR) / max(abs(EXACT_VAR), 1e-10)

variance_passes = var_rel_err < VAR_THRESHOLD
near_zero_mean = abs(EXACT_MEAN) < NEAR_ZERO_MEAN_THRESHOLD
mean_passes = near_zero_mean or (mean_rel_err < MEAN_THRESHOLD)

overall_pass = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt_used}")
print(f"num_paths:           {NUM_PATHS}")
print(f"T:                   {T}")
print(f"All finite:          {finite}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {EXACT_MEAN:.6f}")
print(f"Mean rel. error:     {mean_rel_err:.4f}  "
      f"({'PASS' if mean_passes else 'FAIL'} | "
      f"{'skipped (near-zero)' if near_zero_mean else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {EXACT_VAR:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
