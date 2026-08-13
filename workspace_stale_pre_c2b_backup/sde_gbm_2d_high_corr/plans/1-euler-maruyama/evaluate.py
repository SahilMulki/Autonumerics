import sys
import os
import numpy as np

# Add solver directory to path for import
sys.path.insert(0, os.path.dirname(__file__))
from solver import solve_sde

# Hyperparameters from problem_spec.json / SOLUTION.md
num_paths = 50000
dt        = 0.01
T         = 1.0
seed      = 42

# Parameters
X_0, Y_0   = 1.0, 1.0
mu1, sigma1 = 0.10, 0.30
mu2, sigma2 = 0.10, 0.30

# Thresholds from problem_spec.json
mean_threshold = 0.05
var_threshold  = 0.10
near_zero_threshold = 0.01

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]      # list of 2 floats
empirical_var  = result["empirical_variance"]  # list of 2 floats

# Exact moments at t=T
t = T
exact_mean_X     = X_0 * np.exp(mu1 * t)
exact_variance_X = X_0**2 * np.exp(2*mu1*t) * (np.exp(sigma1**2 * t) - 1)
exact_mean_Y     = Y_0 * np.exp(mu2 * t)
exact_variance_Y = Y_0**2 * np.exp(2*mu2*t) * (np.exp(sigma2**2 * t) - 1)

# Per-component relative errors
mean_rel_err_X = abs(empirical_mean[0] - exact_mean_X) / max(abs(exact_mean_X), 1e-10)
var_rel_err_X  = abs(empirical_var[0]  - exact_variance_X) / max(abs(exact_variance_X), 1e-10)

mean_rel_err_Y = abs(empirical_mean[1] - exact_mean_Y) / max(abs(exact_mean_Y), 1e-10)
var_rel_err_Y  = abs(empirical_var[1]  - exact_variance_Y) / max(abs(exact_variance_Y), 1e-10)

# Pass/fail per component
near_zero_mean_X = abs(exact_mean_X) < near_zero_threshold
mean_passes_X    = near_zero_mean_X or (mean_rel_err_X < mean_threshold)
variance_passes_X = var_rel_err_X < var_threshold

near_zero_mean_Y = abs(exact_mean_Y) < near_zero_threshold
mean_passes_Y    = near_zero_mean_Y or (mean_rel_err_Y < mean_threshold)
variance_passes_Y = var_rel_err_Y < var_threshold

overall_pass = variance_passes_X and mean_passes_X and variance_passes_Y and mean_passes_Y

print("=== Evaluation Results ===")
print(f"Scheme:              euler-maruyama")
print(f"dt:                  {dt}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print("--- Component X ---")
print(f"Empirical mean X:      {empirical_mean[0]:.6f}")
print(f"Exact mean X:          {exact_mean_X:.6f}")
print(f"Mean rel. error X:     {mean_rel_err_X:.4f}  "
      f"({'PASS' if mean_passes_X else 'FAIL'}"
      f"{' | skipped (near-zero)' if near_zero_mean_X else ''})")
print()
print(f"Empirical variance X:  {empirical_var[0]:.6f}")
print(f"Exact variance X:      {exact_variance_X:.6f}")
print(f"Variance rel. error X: {var_rel_err_X:.4f}  ({'PASS' if variance_passes_X else 'FAIL'})")
print()
print("--- Component Y ---")
print(f"Empirical mean Y:      {empirical_mean[1]:.6f}")
print(f"Exact mean Y:          {exact_mean_Y:.6f}")
print(f"Mean rel. error Y:     {mean_rel_err_Y:.4f}  "
      f"({'PASS' if mean_passes_Y else 'FAIL'}"
      f"{' | skipped (near-zero)' if near_zero_mean_Y else ''})")
print()
print(f"Empirical variance Y:  {empirical_var[1]:.6f}")
print(f"Exact variance Y:      {exact_variance_Y:.6f}")
print(f"Variance rel. error Y: {var_rel_err_Y:.4f}  ({'PASS' if variance_passes_Y else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
