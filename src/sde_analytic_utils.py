# sde_analytic_utils.py
"""Analytic moment evaluator and accuracy metrics for SDE Monte Carlo results.

This module is the SDE counterpart to analytic_utils.py but completely
different in structure: instead of evaluating expressions on a spatial grid,
it evaluates scalar moment expressions E[X(t)] and Var[X(t)] at a given time t,
then compares them to the empirical Monte Carlo estimates.

The comparison signal (moment relative error) plays the same role that L2 error
plays in the PDE pipeline: it tells the plan scorer and plan selector how
accurately each (scheme, dt, num_paths) combination reproduces the true
distribution of X(T).

The SDE analytic_solution field in the spec is expected to have the form:
    {
        "type": "moments",
        "mean":     "<Python/NumPy expression in t, X_0, param names>",
        "variance": "<Python/NumPy expression in t, X_0, param names>"
    }

All five Phase 1 problems store their moments in this format.
"""

from __future__ import annotations

import re
from typing import Any, Optional

import numpy as np


# =============================================================================
# Expression preprocessing
# =============================================================================

# Replaces standalone '^' with '**' (LaTeX/math convention vs Python).
# Also replaces standalone bare 'i' (imaginary unit) with '1j'. SDE moment
# expressions should already be in NumPy syntax, but this guards against
# expressions copy-pasted from math textbooks.
_I_TOKEN = re.compile(r"(?<![A-Za-z0-9_])i(?![A-Za-z0-9_])")


def _preprocess_expr(expr: str) -> str:
    """Normalize math-style syntax to Python: '^' -> '**', bare 'i' -> '1j'."""
    expr = expr.replace("^", "**")
    expr = _I_TOKEN.sub("1j", expr)
    return expr


# =============================================================================
# Namespace builder
# =============================================================================

def _build_moment_namespace(sde_spec: dict, t: float) -> dict[str, Any]:
    """Build the eval() namespace for a moment expression evaluated at time t.

    The namespace includes:
      - np    : numpy module (for np.exp, np.sqrt, etc.)
      - t     : the evaluation time (scalar float)
      - X_0   : initial condition value (extracted from sde_spec["initial_condition"])
      - <key> : every name/value pair from sde_spec["parameters"]

    This mirrors the convention used in the SDE problem descriptions and
    the formulator's SDE_FORMULATOR_SYSTEM where moment expressions are
    written in terms of these exact variable names.

    Example: for the OU process with parameters {"theta": 1.5, "mu": 0.0, "sigma": 0.5}
    and initial condition {"X_0": 2.0}, the expression
        "mu + (X_0 - mu) * np.exp(-theta * t)"
    evaluates correctly because the namespace contains all four symbols.
    """
    # Store t as-is: callers pass either a scalar float (terminal comparison)
    # or a numpy array (vectorised path evaluation in evaluate_moment_paths).
    ns: dict[str, Any] = {"np": np, "t": t}

    # Extract the scalar initial condition value and expose it as X_0.
    # sde_spec["initial_condition"] is normalised by _post_process_sde_spec to
    # {"X_0": float} or {"X_0": [float, ...]} — we take the first (and for
    # Phase 1, only) value.
    ic = sde_spec.get("initial_condition", {})
    if isinstance(ic, dict):
        for key, val in ic.items():
            if isinstance(val, (list, tuple)):
                ns[key] = float(val[0])
            else:
                try:
                    ns[key] = float(val)
                except (TypeError, ValueError):
                    pass
    elif isinstance(ic, (int, float)):
        # Bare scalar: store as X_0
        ns["X_0"] = float(ic)

    # Inject all problem parameters (mu, sigma, theta, a, b, c, …).
    params = sde_spec.get("parameters", {}) or {}
    for key, val in params.items():
        if isinstance(val, (int, float, complex, np.number)):
            ns[key] = val

    return ns


# =============================================================================
# Safe expression evaluator
# =============================================================================

def _eval_moment_expr(expr: Optional[str], ns: dict[str, Any]) -> Optional[float]:
    """Evaluate a scalar moment expression string in the given namespace.

    Returns the float result, or None if:
      - expr is None / empty
      - eval raises any exception (e.g., missing variable, division by zero)

    Never raises — callers treat None as "no analytic value available".
    """
    if not isinstance(expr, str) or not expr.strip():
        return None
    try:
        result = eval(_preprocess_expr(expr), ns)
        return float(result)
    except Exception:
        return None


# =============================================================================
# Public API — terminal moment comparison
# =============================================================================

def evaluate_sde_moments(sde_spec: dict, solver_result: dict) -> dict:
    """Compare terminal Monte Carlo moment estimates to exact closed-form values.

    This is the primary accuracy metric for the SDE pipeline. It is called
    after run_generated_sde_solver() succeeds for a given plan, and its output
    is stored in the plan's "metrics" dict that the plan scorer and selector
    consume.

    Args:
        sde_spec:      SDE specification from formulate_problem().
        solver_result: Dict returned by solve_sde() (from the generated code).
                       Must contain at minimum "terminal_mean", "terminal_variance",
                       "num_paths", "scheme", and "t".

    Returns a dict with the following keys (all always present):

        has_analytic_moments  — bool: True if exact mean and/or variance
                                are available for comparison
        t_final               — float: terminal time T used for evaluation
        empirical_mean        — float: E[X(T)] from Monte Carlo
        empirical_variance    — float: Var[X(T)] from Monte Carlo
        exact_mean            — float or None: closed-form E[X(T)]
        exact_variance        — float or None: closed-form Var[X(T)]
        mean_absolute_error   — float or None: |empirical - exact| for mean
        variance_absolute_error — float or None: |empirical - exact| for variance
        mean_relative_error   — float or None: relative error for mean;
                                computed as |emp - exact| / (|exact| + 1e-10)
                                so BM (exact mean = 0) gets a meaningful value
        variance_relative_error — float or None: relative error for variance
        num_paths             — int: number of Monte Carlo paths
        scheme                — str: "euler_maruyama" or "milstein"
    """
    # --- Pull empirical values from solver result ---
    emp_mean = float(solver_result.get("terminal_mean", float("nan")))
    emp_var  = float(solver_result.get("terminal_variance", float("nan")))
    num_paths = int(solver_result.get("num_paths", 0))
    scheme    = str(solver_result.get("scheme", "unknown"))

    # Terminal time: use the last element of the t array returned by the solver.
    # Fall back to sde_spec if the solver didn't return t.
    t_arr = solver_result.get("t", None)
    if t_arr is not None and hasattr(t_arr, "__len__") and len(t_arr) > 0:
        t_final = float(np.asarray(t_arr)[-1])
    else:
        t_final = float(sde_spec.get("time_interval", {}).get("t_final", 1.0))

    # --- Try to get exact moment expressions from analytic_solution ---
    anal = sde_spec.get("analytic_solution", None)
    exact_mean: Optional[float] = None
    exact_var:  Optional[float] = None

    if isinstance(anal, dict) and anal.get("type") == "moments":
        ns = _build_moment_namespace(sde_spec, t_final)
        exact_mean = _eval_moment_expr(anal.get("mean"),     ns)
        exact_var  = _eval_moment_expr(anal.get("variance"), ns)

    has_analytic = (exact_mean is not None) or (exact_var is not None)

    # --- Compute absolute and relative errors ---
    # Relative error denominator: |exact| + eps avoids division by zero.
    # eps = 1e-10 is small enough that it doesn't distort non-zero exact values.
    _eps = 1e-10

    mean_abs_err: Optional[float] = None
    mean_rel_err: Optional[float] = None
    var_abs_err:  Optional[float] = None
    var_rel_err:  Optional[float] = None

    if exact_mean is not None and np.isfinite(emp_mean):
        mean_abs_err = abs(emp_mean - exact_mean)
        mean_rel_err = mean_abs_err / (abs(exact_mean) + _eps)

    if exact_var is not None and np.isfinite(emp_var):
        var_abs_err = abs(emp_var - exact_var)
        var_rel_err = var_abs_err / (abs(exact_var) + _eps)

    return {
        "has_analytic_moments":   has_analytic,
        "t_final":                t_final,
        "empirical_mean":         emp_mean,
        "empirical_variance":     emp_var,
        "exact_mean":             exact_mean,
        "exact_variance":         exact_var,
        "mean_absolute_error":    mean_abs_err,
        "variance_absolute_error": var_abs_err,
        "mean_relative_error":    mean_rel_err,
        "variance_relative_error": var_rel_err,
        "num_paths":              num_paths,
        "scheme":                 scheme,
    }


# =============================================================================
# Public API — full moment path comparison
# =============================================================================

def evaluate_moment_paths(sde_spec: dict, solver_result: dict) -> dict:
    """Evaluate exact moment functions along the full time grid and compare to
    the solver's mean_path and var_path arrays.

    This provides a richer accuracy picture than terminal-only comparison: it
    shows whether the moment error is uniformly small or concentrated near T,
    which is useful for the plan scorer and the reasoning agent.

    Args:
        sde_spec:      SDE specification from formulate_problem().
        solver_result: Dict returned by solve_sde(). Must contain "t",
                       "mean_path", and "var_path".

    Returns a dict with:
        t                — ndarray (Nt+1,): time grid
        mean_path_exact  — ndarray (Nt+1,) or None: exact E[X(t_k)] at each step
        var_path_exact   — ndarray (Nt+1,) or None: exact Var[X(t_k)] at each step
        mean_path_error  — ndarray (Nt+1,) or None: |empirical - exact| for mean
        var_path_error   — ndarray (Nt+1,) or None: |empirical - exact| for variance
        max_mean_error   — float or None: max of mean_path_error
        max_var_error    — float or None: max of var_path_error
        mean_rel_error_path — ndarray (Nt+1,) or None: relative error at each step
        var_rel_error_path  — ndarray (Nt+1,) or None: relative error at each step
    """
    t_arr      = np.asarray(solver_result.get("t", []))
    mean_path  = np.asarray(solver_result.get("mean_path", []))
    var_path   = np.asarray(solver_result.get("var_path", []))

    # Degenerate case: solver didn't return a time array
    if t_arr.size == 0:
        return {
            "t": t_arr,
            "mean_path_exact": None, "var_path_exact":  None,
            "mean_path_error": None, "var_path_error":  None,
            "max_mean_error":  None, "max_var_error":   None,
            "mean_rel_error_path": None, "var_rel_error_path": None,
        }

    anal = sde_spec.get("analytic_solution", None)
    mean_expr = None
    var_expr  = None

    if isinstance(anal, dict) and anal.get("type") == "moments":
        mean_expr = anal.get("mean")
        var_expr  = anal.get("variance")

    # Evaluate the moment expressions at each time step in t_arr.
    # We do this with a vectorised eval approach: build the namespace once,
    # then inject a numpy array for t so the expression operates over all steps
    # simultaneously. This avoids a Python loop over Nt time steps.
    _eps = 1e-10

    mean_exact_arr: Optional[np.ndarray] = None
    var_exact_arr:  Optional[np.ndarray] = None

    if mean_expr is not None or var_expr is not None:
        # Build the namespace with t as the full time array (vectorised eval).
        # The moment expressions use np.exp, np.sqrt, etc., which all broadcast
        # over arrays, so eval("X_0 * np.exp(mu * t)", ns) with t as an ndarray
        # returns an ndarray of the same shape as t — no loop needed.
        ns_vec = _build_moment_namespace(sde_spec, t=t_arr)  # t injected as array
        ns_vec["t"] = t_arr  # override scalar with array for vectorised eval

        if mean_expr is not None:
            try:
                raw = eval(_preprocess_expr(mean_expr), ns_vec)
                mean_exact_arr = np.broadcast_to(
                    np.asarray(raw, dtype=float), t_arr.shape
                ).copy()
            except Exception:
                mean_exact_arr = None

        if var_expr is not None:
            try:
                raw = eval(_preprocess_expr(var_expr), ns_vec)
                var_exact_arr = np.broadcast_to(
                    np.asarray(raw, dtype=float), t_arr.shape
                ).copy()
            except Exception:
                var_exact_arr = None

    # --- Absolute and relative errors along the path ---
    mean_err_arr:     Optional[np.ndarray] = None
    var_err_arr:      Optional[np.ndarray] = None
    mean_rel_err_arr: Optional[np.ndarray] = None
    var_rel_err_arr:  Optional[np.ndarray] = None
    max_mean_err:     Optional[float]      = None
    max_var_err:      Optional[float]      = None

    if mean_exact_arr is not None and mean_path.size == mean_exact_arr.size:
        mean_err_arr     = np.abs(mean_path - mean_exact_arr)
        mean_rel_err_arr = mean_err_arr / (np.abs(mean_exact_arr) + _eps)
        max_mean_err     = float(mean_err_arr.max())

    if var_exact_arr is not None and var_path.size == var_exact_arr.size:
        var_err_arr     = np.abs(var_path - var_exact_arr)
        var_rel_err_arr = var_err_arr / (np.abs(var_exact_arr) + _eps)
        max_var_err     = float(var_err_arr.max())

    return {
        "t":                   t_arr,
        "mean_path_exact":     mean_exact_arr,
        "var_path_exact":      var_exact_arr,
        "mean_path_error":     mean_err_arr,
        "var_path_error":      var_err_arr,
        "max_mean_error":      max_mean_err,
        "max_var_error":       max_var_err,
        "mean_rel_error_path": mean_rel_err_arr,
        "var_rel_error_path":  var_rel_err_arr,
    }
