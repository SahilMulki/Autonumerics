import sys
import os
import numpy as np

# Add plan directory to path for solver import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_sde

# Hyperparameters from SOLUTION.md (solver updated to dt=0.001, Milstein correction)
num_paths = 200000
dt = 0.001
T = 1.0
seed = 42

# Parameters from problem_spec.json
X_0 = 1.0
mu = 0.05
sigma = 1.0

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var = result["empirical_variance"]

# Analytic moments at T
t = T
exact_mean = X_0 * np.exp(mu * t)
exact_var = X_0**2 * np.exp(2 * mu * t) * (np.exp(sigma**2 * t) - 1)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err = abs(empirical_var - exact_var) / max(abs(exact_var), 1e-10)

# Thresholds from problem_spec.json
var_threshold = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

# Pass/fail logic
variance_passes = var_rel_err < var_threshold
near_zero_mean = abs(exact_mean) < near_zero_threshold
mean_passes = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass = variance_passes and mean_passes

# Print results
print("=== Evaluation Results ===")
print(f"Scheme:              euler-maruyama (with Milstein correction)")
print(f"dt:                  {result['dt']}")
print(f"num_paths:           {result['num_paths']}")
print(f"T:                   {result['T']}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
skip_note = " (skipped (near-zero))" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{skip_note})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
