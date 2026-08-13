import json
import os

import numpy as np

from solver import solve_pde

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, "..", "..", "problem_spec.json")

with open(SPEC_PATH) as f:
    spec = json.load(f)

thresholds = spec.get("evaluation_thresholds", {})


def rel_err(u_num, u_exact, metric="l2", mask=None):
    d = u_num - u_exact
    ref = u_exact
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d ** 2)) / (np.sqrt(np.mean(ref ** 2)) + 1e-14))


grid_N = thresholds.get("grid_N", 64)
min_ord = thresholds.get("min_spatial_order", 1.0)
order_on = thresholds.get("order_check", True)
tol = thresholds.get("rel_l2_err_max", 0.01)
metric = thresholds.get("metric", "l2")
axes_nm = thresholds.get("axes", ["x", "y", "z"])
fields = thresholds.get("fields")
primary = thresholds.get("primary_field")
required = thresholds.get("required_fields")
gauge = set(thresholds.get("gauge_fields", []))
diagnostics = thresholds.get("diagnostics", [])

Ns = [grid_N, 2 * grid_N] if order_on else [grid_N]

field_errs_per_grid = []
viol = []
final_min_rho = None

for N in Ns:
    result = solve_pde(N)
    coords = np.meshgrid(*[result["grid"][a] for a in axes_nm if a in result["grid"]], indexing="ij")
    X, Y = coords
    t = result.get("t_final")

    # --- exact fields (from problem_spec.json analytic_solution) ---
    exact = {
        "rho": 1.0 + 0.2 * np.sin(np.pi * X) * np.sin(np.pi * Y) * np.exp(-t),
        "c": 1.0 + 0.1 * np.cos(np.pi * X) * np.cos(np.pi * Y) * np.exp(-2.0 * t),
    }
    num = result["fields"]

    fe = {}
    for name in fields:
        un = np.asarray(num[name], float).reshape(exact[name].shape)
        ue = exact[name]
        if name in gauge:
            un = un - np.mean(un)
            ue = ue - np.mean(ue)
        fe[name] = rel_err(un, ue, metric, None)
    field_errs_per_grid.append(fe)

    # structural diagnostic: min_density gate on rho (checked at every grid, but
    # only the finest grid's violation determines the gate outcome per the manual;
    # we check on the finest grid below)
    if N == Ns[-1]:
        final_min_rho = float(np.min(num["rho"]))

# --- structural diagnostics (evaluated on the finest grid) ---
for diag in diagnostics:
    if diag.get("name") == "min_density":
        if final_min_rho is not None and final_min_rho < -1e-8:
            if diag.get("gate", False):
                viol.append("min_density (rho min = %.6e < 0)" % final_min_rho)

req = required or (fields or ["u"])
e_fine = max(field_errs_per_grid[-1][f] for f in req)
converged = e_fine < tol

primary_field = primary or (fields or ["u"])[0]
p_coarse = field_errs_per_grid[0][primary_field]
p_fine = field_errs_per_grid[-1][primary_field]
if not order_on or p_fine < 1e-9 or p_coarse < 1e-9:
    order = float("inf")
    order_ok = True
else:
    order = float(np.log2(p_coarse / p_fine))
    order_ok = order >= min_ord

passes = converged and order_ok and not viol

print("=== PDE Evaluation Results ===")
print("Scheme:            finite-difference-explicit (central FV + SSP-RK2)")
print(f"Resolutions:       N = {Ns}   metric = {metric}")
print(f"Per-field error:   {field_errs_per_grid[-1]} (fine grid; required max = {e_fine:.6e}, tol {tol})")
print(f"Observed order:    {order:.3f}  (primary '{primary_field}', min {min_ord}; {'ok' if order_ok else 'TOO LOW'})")
print(f"Constraints:       {'ok' if not viol else 'VIOLATED: ' + ', '.join(viol)}")
print(f"min(rho) fine grid: {final_min_rho:.8f}")
print(f"Coarse-grid errors: {field_errs_per_grid[0]}")
print(f"Overall: {'PASS' if passes else 'FAIL'}")
