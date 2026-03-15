# evaluator.py (UPDATED, PDE-agnostic)

import time
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

# Type alias for solver: callable returning {"u":..., "coords":..., "t":... or "t_final":...}
SolverFn = Callable[..., Dict[str, Any]]


def extract_final_state_and_time(
    result: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, np.ndarray], Optional[float]]:
    """
    Normalize solver output into:
      - u_final: ndarray at final time (or steady-state)
      - coords: {"x":..., "y":..., "z":...}
      - t_final: float or None

    Accepted result formats:
      - Time-dependent:
          {"u": u, "coords": coords, "t": t_array} where u has shape (Nt, ...)
          {"u": u, "coords": coords, "t_final": float} where u is already final state
      - Steady-state:
          {"u": u, "coords": coords, "t": None}
          {"u": u, "coords": coords, "t_final": None}
    """
    if "u" not in result or "coords" not in result:
        raise KeyError("Solver result must contain keys 'u' and 'coords'.")

    u = result["u"]
    coords = result["coords"]

    # Determine final time
    if "t_final" in result:
        t_final = result["t_final"]
        t_final = None if t_final is None else float(t_final)
        u_final = u
        return u_final, coords, t_final

    t = result.get("t")
    if t is None:
        # steady-state
        return u, coords, None

    # time-dependent with time array
    t = np.asarray(t, dtype=float)
    if t.ndim != 1 or t.size == 0:
        raise ValueError("If provided, result['t'] must be a non-empty 1D array.")
    t_final = float(t[-1])

    u = np.asarray(u)
    if u.ndim < 2:
        # If u is 1D but time exists, it must be (Nt,) (rare)
        if u.shape[0] != t.size:
            raise ValueError("Time-dependent result has inconsistent shapes between u and t.")
        u_final = u[-1]
    else:
        # Typical: u shape (Nt, Nx[,Ny[,Nz]])
        if u.shape[0] != t.size:
            raise ValueError("Time-dependent result must have u.shape[0] == len(t).")
        u_final = u[-1]

    return u_final, coords, t_final


def compute_l2_and_max_error_from_exact(
    u_num_final: np.ndarray,
    u_exact: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute L2 and max error between numerical and exact solutions
    on the same grid, using mean-square L2.
    """
    u_num_final = np.asarray(u_num_final)
    u_exact = np.asarray(u_exact)

    if u_num_final.shape != u_exact.shape:
        raise ValueError(f"Shape mismatch: u_num {u_num_final.shape} vs u_exact {u_exact.shape}")

    diff = (u_num_final - u_exact).ravel()
    l2_error = np.sqrt(np.mean(diff**2))
    max_error = np.max(np.abs(diff))
    return float(l2_error), float(max_error)


def evaluate_plan(
    pde_spec: Dict[str, Any],
    plan: Dict[str, Any],
    build_solver: Callable[[Dict[str, Any], Dict[str, Any]], SolverFn],
    analytic_solution: Callable[
        [Dict[str, np.ndarray], Optional[float], Dict[str, Any]], Optional[np.ndarray]
    ],
) -> Dict[str, Any]:
    """
    Build + run solver for a single plan, measure runtime, and compute errors if possible.

    - analytic_solution(coords, t_final, pde_spec) -> ndarray or None
      If None, error metrics are skipped (set to None) and failed=False.
    """
    plan_id = plan.get("plan_id")

    # 1) Build solver using LLM code + critic
    try:
        solver_fn = build_solver(pde_spec, plan)
    except Exception as e:
        return {
            "plan_id": plan_id,
            "metrics": {
                "L2_error": None,
                "max_error": None,
                "runtime_sec": float("inf"),
                "failed": True,
                "failure_reason": f"build_solver exception: {e}",
                "analytic_skipped": True,
            },
            "plan": plan,
        }

    # 2) Run solver and measure runtime
    t_start = time.perf_counter()
    try:
        result = solver_fn()
        runtime_sec = time.perf_counter() - t_start
    except Exception as e:
        return {
            "plan_id": plan_id,
            "metrics": {
                "L2_error": None,
                "max_error": None,
                "runtime_sec": float("inf"),
                "failed": True,
                "failure_reason": f"solver runtime exception: {e}",
                "analytic_skipped": True,
            },
            "plan": plan,
        }

    # 3) Normalize result to final state
    try:
        u_num_final, coords, t_final = extract_final_state_and_time(result)
    except Exception as e:
        return {
            "plan_id": plan_id,
            "metrics": {
                "L2_error": None,
                "max_error": None,
                "runtime_sec": runtime_sec,
                "failed": True,
                "failure_reason": f"result format exception: {e}",
                "analytic_skipped": True,
            },
            "plan": plan,
        }

    # 4) Compute error metrics if analytic is available/compatible
    try:
        u_exact = analytic_solution(coords, t_final, pde_spec)
        if u_exact is None:
            l2_error = None
            max_error = None
            failed = False
            failure_reason = ""
            analytic_skipped = True
        else:
            l2_error, max_error = compute_l2_and_max_error_from_exact(u_num_final, u_exact)
            failed = False
            failure_reason = ""
            analytic_skipped = False
    except Exception as e:
        # Analytic exists but evaluation failed (bad expression, mismatch, etc.)
        l2_error = None
        max_error = None
        failed = True
        failure_reason = f"analytic/error computation exception: {e}"
        analytic_skipped = False

    # 5) Extract discretization info for LLM
    spatial = plan.get("spatial_discretization", {}) or {}
    time_step = plan.get("time_stepping", {}) or {}

    Nx = spatial.get("Nx")
    Ny = spatial.get("Ny")
    Nz = spatial.get("Nz")
    method = time_step.get("method")
    dt = time_step.get("dt")
    Nt = time_step.get("Nt")
    spatial_scheme = spatial.get("scheme")
    spatial_order = spatial.get("order")

    return {
        "plan_id": plan_id,
        "dimension": int(pde_spec.get("spatial_dimension", 1)),
        "spatial_resolution": {"Nx": Nx, "Ny": Ny, "Nz": Nz},
        "time_stepping": {"method": method, "dt": dt, "Nt": Nt},
        "scheme": {
            "spatial_scheme": spatial_scheme,
            "spatial_order": spatial_order,
        },
        "metrics": {
            "L2_error": l2_error,
            "max_error": max_error,
            "runtime_sec": runtime_sec,
            "failed": failed,
            "failure_reason": failure_reason,
            "analytic_skipped": analytic_skipped,
        },
        "plan": plan,
    }
