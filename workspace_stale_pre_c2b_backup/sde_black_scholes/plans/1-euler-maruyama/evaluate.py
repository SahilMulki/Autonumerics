import sys
import os
import numpy as np

# Add plan directory to path so solver.py can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from solver import solve_sde

# Hyperparameters from SOLUTION.md / problem_spec.json defaults
num_paths = 50000
dt        = 0.01
T         = 1.0
seed      = 42

# Scheme label
scheme = "Euler-Maruyama"

# Run solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]

# Black-Scholes parameters
X_0   = 100.0
r     = 0.05
sigma = 0.20
t     = T

# Analytic moments
exact_mean = X_0 * np.exp(r * t)
exact_var  = X_0**2 * np.exp(2 * r * t) * (np.exp(sigma**2 * t) - 1)

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Pass/fail thresholds
var_threshold  = 0.10
mean_threshold = 0.05

variance_passes = var_rel_err < var_threshold

near_zero_mean = abs(exact_mean) < 0.01
mean_passes    = near_zero_mean or (mean_rel_err < mean_threshold)

overall_pass = variance_passes and mean_passes

# Print structured summary
print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {result['dt']}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
skip_note = "skipped (near-zero)" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{' | ' + skip_note if skip_note else ''})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
