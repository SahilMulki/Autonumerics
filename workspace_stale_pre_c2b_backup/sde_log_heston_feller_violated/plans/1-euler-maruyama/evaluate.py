import numpy as np
from solver import solve_sde

# --- Hyperparameters (from SOLUTION.md / problem_spec.json) ---
num_paths = 50000
dt = 0.001
T = 1.0
seed = 42

# --- Parameters (from problem_spec.json) ---
X_0 = 0.0
Y_0 = 0.1
r = 0.05
a = 0.1
b = 1.0
sigma = 1.0
rho = -0.9

# --- Thresholds (from problem_spec.json evaluation_thresholds) ---
var_threshold = 0.10
mean_threshold = 0.05
near_zero_mean_threshold = 0.01

# --- Run solver ---
result = solve_sde(num_paths=num_paths, dt=dt, T=T, seed=seed)
scheme = "euler-maruyama (full-truncation)"

emp_mean_X, emp_mean_Y = result["empirical_mean"]
emp_var_X, emp_var_Y = result["empirical_variance"]
dt_used = result["dt"]

# --- Exact moments at t = T (closed-form, from problem_spec.json) ---
t = T
exact_mean_X = 0.0
exact_variance_X = 0.13731143072354426
exact_mean_Y = 0.1
exact_variance_Y = 0.04323323583816936

# --- Check finiteness ---
finite = all(np.isfinite(v) for v in [emp_mean_X, emp_mean_Y, emp_var_X, emp_var_Y])

def rel_err(empirical, exact):
    return abs(empirical - exact) / max(abs(exact), 1e-10)

components = {}

# --- X component ---
mean_rel_err_X = rel_err(emp_mean_X, exact_mean_X)
var_rel_err_X = rel_err(emp_var_X, exact_variance_X)
near_zero_mean_X = abs(exact_mean_X) < near_zero_mean_threshold
mean_passes_X = near_zero_mean_X or (mean_rel_err_X < mean_threshold)
variance_passes_X = var_rel_err_X < var_threshold
components["X"] = dict(
    empirical_mean=emp_mean_X, exact_mean=exact_mean_X, mean_rel_err=mean_rel_err_X,
    near_zero_mean=near_zero_mean_X, mean_passes=mean_passes_X,
    empirical_var=emp_var_X, exact_var=exact_variance_X, var_rel_err=var_rel_err_X,
    variance_passes=variance_passes_X,
)

# --- Y component ---
mean_rel_err_Y = rel_err(emp_mean_Y, exact_mean_Y)
var_rel_err_Y = rel_err(emp_var_Y, exact_variance_Y)
near_zero_mean_Y = abs(exact_mean_Y) < near_zero_mean_threshold
mean_passes_Y = near_zero_mean_Y or (mean_rel_err_Y < mean_threshold)
variance_passes_Y = var_rel_err_Y < var_threshold
components["Y"] = dict(
    empirical_mean=emp_mean_Y, exact_mean=exact_mean_Y, mean_rel_err=mean_rel_err_Y,
    near_zero_mean=near_zero_mean_Y, mean_passes=mean_passes_Y,
    empirical_var=emp_var_Y, exact_var=exact_variance_Y, var_rel_err=var_rel_err_Y,
    variance_passes=variance_passes_Y,
)

overall_pass = finite and all(c["mean_passes"] and c["variance_passes"] for c in components.values())

print("=== Evaluation Results ===")
print(f"Scheme:              {scheme}")
print(f"dt:                  {dt_used}")
print(f"num_paths:           {num_paths}")
print(f"T:                   {T}")
print(f"Finite outputs:      {finite}")
print()

for name, c in components.items():
    print(f"--- Component {name} ---")
    print(f"Empirical mean:      {c['empirical_mean']:.6f}")
    print(f"Exact mean:          {c['exact_mean']:.6f}")
    skip_note = 'skipped (near-zero)' if c['near_zero_mean'] else ''
    print(f"Mean rel. error:     {c['mean_rel_err']:.4f}  ({'PASS' if c['mean_passes'] else 'FAIL'} | {skip_note})")
    print()
    print(f"Empirical variance:  {c['empirical_var']:.6f}")
    print(f"Exact variance:      {c['exact_var']:.6f}")
    print(f"Variance rel. error: {c['var_rel_err']:.4f}  ({'PASS' if c['variance_passes'] else 'FAIL'})")
    print()

print(f"Overall: {'PASS' if overall_pass else 'FAIL'}")
