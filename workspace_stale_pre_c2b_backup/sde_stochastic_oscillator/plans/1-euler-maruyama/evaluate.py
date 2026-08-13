import sys
import os
import numpy as np

# Add the plan directory to the path so we can import solver.py
plan_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, plan_dir)

from solver import solve_sde

# Hyperparameters from problem_spec.json
num_paths = 50000
dt = 0.01
T = 2 * np.pi
seed = 42

# Thresholds from problem_spec.json
var_threshold = 0.10
mean_threshold = 0.05
near_zero_threshold = 0.01

# Run the solver
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)

empirical_mean = result["empirical_mean"]    # [mean_X, mean_Y]
empirical_var  = result["empirical_variance"] # [var_X, var_Y]

# Parameters
X_0 = 1.0
Y_0 = 0.0
sigma = 0.3
t = T

# Analytic moments at T
exact_mean_X     = X_0 * np.cos(t) + Y_0 * np.sin(t)
exact_mean_Y     = -X_0 * np.sin(t) + Y_0 * np.cos(t)
exact_variance_X = sigma**2 / 2 * (t - np.sin(t) * np.cos(t))
exact_variance_Y = sigma**2 / 2 * (t + np.sin(t) * np.cos(t))

exact_means = [exact_mean_X, exact_mean_Y]
exact_vars  = [exact_variance_X, exact_variance_Y]
labels      = ["X", "Y"]

print("=== Evaluation Results ===")
print(f"Scheme:              euler-maruyama")
print(f"dt:                  {result['dt']:.6f}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T:.6f}")
print()

component_passes = []

for i, label in enumerate(labels):
    emp_mean = empirical_mean[i]
    emp_var  = empirical_var[i]
    ex_mean  = exact_means[i]
    ex_var   = exact_vars[i]

    mean_rel_err = abs(emp_mean - ex_mean) / max(abs(ex_mean), 1e-10)
    var_rel_err  = abs(emp_var  - ex_var)  / max(abs(ex_var),  1e-10)

    near_zero_mean = abs(ex_mean) < near_zero_threshold
    mean_passes    = near_zero_mean or (mean_rel_err < mean_threshold)
    var_passes     = var_rel_err < var_threshold
    comp_passes    = mean_passes and var_passes
    component_passes.append(comp_passes)

    skip_str = " (skipped — near-zero)" if near_zero_mean else ""
    mean_status = "PASS" if mean_passes else "FAIL"
    var_status  = "PASS" if var_passes  else "FAIL"

    print(f"--- Component {label} ---")
    print(f"Empirical mean {label}:      {emp_mean:.6f}")
    print(f"Exact mean {label}:          {ex_mean:.6f}")
    print(f"Mean rel. error {label}:     {mean_rel_err:.4f}  ({mean_status}{skip_str})")
    print()
    print(f"Empirical variance {label}:  {emp_var:.6f}")
    print(f"Exact variance {label}:      {ex_var:.6f}")
    print(f"Variance rel. error {label}: {var_rel_err:.4f}  ({var_status})")
    print()

overall_pass = all(component_passes)
print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
