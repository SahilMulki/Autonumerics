import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from solver import solve_pde

result = solve_pde()
u_num = result["numerical_solution"]
x = result["grid"]["x"]
t = result["t_final"]

alpha = 0.1
u_exact = np.exp(-alpha * np.pi**2 * t) * np.sin(np.pi * x)

l2_num = np.sqrt(np.mean((u_num - u_exact)**2))
l2_denom = np.sqrt(np.mean(u_exact**2)) + 1e-14
rel_l2_error = l2_num / l2_denom

rel_l2_threshold = 0.01
passes = rel_l2_error < rel_l2_threshold

print("=== PDE Evaluation Results ===")
print(f"Scheme:            Crank-Nicolson implicit")
print(f"Grid:              Nx={result['Nx']}")
print(f"dt:                {result['dt']:.6f}")
print(f"t_final:           {t}")
print()
print(f"Relative L2 error: {rel_l2_error:.6f}  ({'PASS' if passes else 'FAIL'})")
print(f"Threshold:         {rel_l2_threshold:.4f}")
print()
print(f"Max pointwise error: {np.max(np.abs(u_num - u_exact)):.6e}")
print(f"Max |u_exact|:       {np.max(np.abs(u_exact)):.6e}")
print()
print(f"Overall: {'PASS' if passes else 'FAIL'}")
