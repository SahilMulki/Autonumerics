import sys
import os
import numpy as np

# Add the plan directory to the path so we can import solver.py
sys.path.insert(0, os.path.dirname(__file__))
from solver import solve_sde

# Hyperparameters from problem_spec.json evaluation_thresholds
num_paths = 50000
dt = 0.01
T = 1.0
seed = 42

# Scheme info
scheme = "euler-maruyama"

# Run the solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]
empirical_var  = result["empirical_variance"]

# Analytic moments from problem_spec.json
# mean_expression: X_0 + mu * t
# variance_expression: sigma**2 * t
X_0   = 1.0
mu    = 0.5
sigma = 0.3
t     = T

exact_mean = X_0 + mu * t          # 1.5
exact_var  = sigma**2 * t          # 0.09

# Relative errors
mean_rel_err = abs(empirical_mean - exact_mean) / max(abs(exact_mean), 1e-10)
var_rel_err  = abs(empirical_var  - exact_var)  / max(abs(exact_var),  1e-10)

# Thresholds from problem_spec.json
var_threshold  = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

# Pass/fail logic
variance_passes = var_rel_err < var_threshold
near_zero_mean  = abs(exact_mean) < near_zero_threshold
mean_passes     = near_zero_mean or (mean_rel_err < mean_threshold)
overall_pass    = variance_passes and mean_passes

# Print structured summary
print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {result['dt']}")
print(f"num_paths:           {result['num_paths']}")
print(f"T:                   {result['T']}")
print()
print(f"Empirical mean:      {empirical_mean:.6f}")
print(f"Exact mean:          {exact_mean:.6f}")
mean_skip_str = " (skipped: near-zero)" if near_zero_mean else ""
print(f"Mean rel. error:     {mean_rel_err:.4f}  ({'PASS' if mean_passes else 'FAIL'}{mean_skip_str})")
print()
print(f"Empirical variance:  {empirical_var:.6f}")
print(f"Exact variance:      {exact_var:.6f}")
print(f"Variance rel. error: {var_rel_err:.4f}  ({'PASS' if variance_passes else 'FAIL'})")
print()
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
