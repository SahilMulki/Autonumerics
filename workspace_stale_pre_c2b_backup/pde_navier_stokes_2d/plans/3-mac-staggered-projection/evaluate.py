"""
Evaluator for plan 3 (MAC staggered-grid projection) on pde_navier_stokes_2d.

Runs solve_pde(N) at N = grid_N and 2*grid_N (Taylor-Green vortex, periodic
[0,2pi]^2), compares (u, v, p) against the exact decaying Taylor-Green
solution, estimates the observed spatial order on the primary field u, and
checks the hard divergence-free gate on the returned collocated fields.
"""

import json
import os

import numpy as np

from solver import solve_pde

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC_PATH = os.path.join(HERE, "..", "..", "problem_spec.json")

with open(SPEC_PATH) as f:
    spec = json.load(f)

thresholds = spec.get("evaluation_thresholds", {})
nu = spec.get("parameters", {}).get("nu", 0.1)

grid_N = thresholds.get("grid_N", 48)
min_ord = thresholds.get("min_spatial_order", 1.0)
order_on = thresholds.get("order_check", True)
tol = thresholds.get("rel_l2_err_max", 0.01)
metric = thresholds.get("metric", "l2")
axes_nm = thresholds.get("axes", ["x", "y"])
fields = thresholds.get("fields", ["u", "v", "p"])
primary = thresholds.get("primary_field", "u")
required = thresholds.get("required_fields", ["u", "v"])
gauge = set(thresholds.get("gauge_fields", []))
diagnostics_cfg = thresholds.get("diagnostics", [{"name": "divergence_norm", "gate": True}])

Ns = [grid_N, 2 * grid_N] if order_on else [grid_N]


def rel_err(u_num, u_exact, metric="l2", mask=None):
    d = u_num - u_exact
    ref = u_exact
    if mask is not None:
        d, ref = d[mask], ref[mask]
    if metric == "l1":
        return float(np.mean(np.abs(d)) / (np.mean(np.abs(ref)) + 1e-14))
    return float(np.sqrt(np.mean(d**2)) / (np.sqrt(np.mean(ref**2)) + 1e-14))


def exact_fields(X, Y, t, nu):
    return {
        "u": np.sin(X) * np.cos(Y) * np.exp(-2 * nu * t),
        "v": -np.cos(X) * np.sin(Y) * np.exp(-2 * nu * t),
        "p": 0.25 * (np.cos(2 * X) + np.cos(2 * Y)) * np.exp(-4 * nu * t),
    }


field_errs_per_grid = []
viol = []
div_norms = []

for N in Ns:
    result = solve_pde(N)
    x = result["grid"]["x"]
    y = result["grid"]["y"]
    X, Y = np.meshgrid(x, y, indexing="ij")
    t = result.get("t_final", spec.get("time_interval", {}).get("T", 0.5))

    exact = exact_fields(X, Y, t, nu)
    num = result["fields"]

    fe = {}
    for name in fields:
        un = np.asarray(num[name], float).reshape(exact[name].shape)
        ue = exact[name]
        if name in gauge:
            un = un - np.mean(un)
            ue = ue - np.mean(ue)
        fe[name] = rel_err(un, ue, metric)
    field_errs_per_grid.append(fe)

    # --- structural diagnostic: divergence of the returned collocated (u,v) ---
    u_out = np.asarray(num["u"], float)
    v_out = np.asarray(num["v"], float)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # central periodic difference divergence on the collocated grid
    div = (np.roll(u_out, -1, axis=0) - np.roll(u_out, 1, axis=0)) / (2 * dx) + (
        np.roll(v_out, -1, axis=1) - np.roll(v_out, 1, axis=1)
    ) / (2 * dy)
    vel_scale = np.sqrt(np.mean(u_out**2 + v_out**2)) + 1e-14
    div_rel = float(np.sqrt(np.mean(div**2)) / vel_scale)

    # also cross-check against the solver's own internal MAC-grid diagnostic
    internal_div = result.get("_diagnostics", {}).get("max_mac_divergence")

    div_norms.append({"collocated_FD_rel_div_norm": div_rel, "internal_mac_max_div": internal_div})

    for d in diagnostics_cfg:
        if d.get("name") == "divergence_norm" and d.get("gate", False):
            # gate on the collocated FD divergence (what a downstream consumer of
            # the returned fields would see); generous relative tolerance since
            # de-staggering introduces an O(dx^2) truncation on top of round-off.
            if div_rel > 1e-3:
                viol.append(f"divergence_norm@N={N} (rel={div_rel:.3e} > 1e-3)")

req = required or fields
e_fine = max(field_errs_per_grid[-1][f] for f in req)
converged = e_fine < tol

p_coarse = field_errs_per_grid[0][primary]
p_fine = field_errs_per_grid[-1][primary]
if not order_on or p_fine < 1e-9 or p_coarse < 1e-9:
    order = float("inf")
    order_ok = True
else:
    order = float(np.log2(p_coarse / p_fine))
    order_ok = order >= min_ord

passes = converged and order_ok and not viol

print("=== PDE Evaluation Results ===")
print("Scheme:            MAC staggered-grid projection (RK2 advection-diffusion + FFT pressure-Poisson)")
print(f"Resolutions:       N = {Ns}   metric = {metric}")
print(f"Per-field error:   {field_errs_per_grid[-1]} (fine grid; required max = {e_fine:.6e}, tol {tol})")
print(f"Observed order:    {order:.3f}  (primary '{primary}', min {min_ord}; {'ok' if order_ok else 'TOO LOW'})")
print(f"Divergence diag:   {div_norms}")
print(f"Constraints:       {'ok' if not viol else 'VIOLATED: ' + ', '.join(viol)}")
print(f"Overall: {'PASS' if passes else 'FAIL'}")
