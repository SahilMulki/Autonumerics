import sys
import os
import numpy as np

# Add plan directory to path for importing solver
plan_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plan_dir)

from solver import solve_sde

# Hyperparameters from problem_spec.json evaluation_thresholds
num_paths = 50000
dt = 0.01
T = 1.0
seed = 42

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]
dt_used        = result["dt"]

# CIR parameters
X_0   = 0.5
kappa = 1.0
theta = 0.5
sigma = 2.0
t = T

# Analytic moments at T
exact_mean = theta + (X_0 - theta) * np.exp(-kappa * t)
exact_var  = (sigma**2 / kappa) * (
    X_0 * (np.exp(-kappa * t) - np.exp(-2 * kappa * t))
    + 0.5 * theta * (1 - np.exp(-kappa * t))**2
)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Thresholds
var_threshold  = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

variance_passes = (var_rel_err < var_threshold)
near_zero_mean  = abs(exact_mean) < near_zero_threshold
mean_passes     = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass    = variance_passes and mean_passes

print("=== Evaluation Results ===")
print(f"Scheme:              euler-maruyama")
print(f"dt:                  {dt_used}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{' | skipped (near-zero)' if near_zero_mean else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
