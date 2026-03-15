import re
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

AnalyticExpr = Union[str, Sequence[str], Dict[str, str]]
AnalyticSpec = Union[
    str,
    Dict[str, Any],  # wrapper: {"expression": ..., "space_variables": ..., ...}
    Sequence[str],  # list of expressions (multi-field)
    Dict[str, str],  # dict of name -> expression (multi-field)
]

# Regex to safely replace standalone i with 1j (avoid clobbering identifiers like "pi", "sin", "phi", etc.)
_I_TOKEN = re.compile(r"(?<![A-Za-z0-9_])i(?![A-Za-z0-9_])")


def _preprocess_expr(expr: str) -> str:
    """
    Make common math-style expressions compatible with Python/NumPy:
      - '^' -> '**'
      - standalone 'i' -> '1j' (imag unit)
    """
    expr = expr.replace("^", "**")
    expr = _I_TOKEN.sub("1j", expr)
    return expr


def analytic_solution(
    coords: Dict[str, np.ndarray],
    t_final: Optional[float],
    pde_spec: Dict[str, Any],
    *,
    raise_on_error: bool = False,
    expected_num_fields: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    PDE-agnostic analytic solution evaluator (multi-D + multi-field).

    Accepts pde_spec["analytic_solution"] in formats (but not limited to):
      A) "np.sin(np.pi*x) * np.exp(-t)"
      B) {"expression": "...", "space_variables": ["x","y"], ...}
      C) {"expression": ["expr0","expr1",...], "stack_axis": 0, ...}
      D) {"expression": {"u":"...", "v":"...", ....}, "field_order":["u","v", ...], "stack_axis":0, ...}
      E) ["expr0","expr1",...]
      F) {"u":"...", "v":"..."}  (order deterministic but prefer field_order, variables here can vary)

    Coordinate handling (general):
      - If coords[var] are 1D axes -> builds meshgrid with indexing="ij"
      - If coords[var] are already same-shaped grids -> uses them directly

    Returns:
      scalar field: ndarray of shape (N1, N2, ...)
      multi-field: ndarray stacked along stack_axis, e.g. (n_vars, N1, N2, ...)

    Note:
      - The exact solution can also be implicit solutions e.g. u(x,t) = x*sin(u).
    """
    anal: AnalyticSpec = pde_spec.get("analytic_solution")
    if anal is None:
        return None

    # wrapper defaults
    space_vars = None
    stack_axis = 0
    field_order = None

    # unwrap common wrapper dict
    if isinstance(anal, dict) and (
        "expression" in anal or "space_variables" in anal or "fields" in anal
    ):
        space_vars = anal.get("space_variables", None)
        stack_axis = anal.get("stack_axis", 0)
        field_order = anal.get("field_order", None)

        expr = anal.get("expression", None)
        if expr is None and "fields" in anal:
            expr = anal["fields"]

        # unwrap nesting patterns sometimes produced by configs
        if isinstance(expr, dict) and "expr" in expr:
            expr = expr["expr"]
        if isinstance(expr, list) and len(expr) == 1:
            expr = expr[0]

        anal = expr

    # infer spatial variables from coords if not provided
    if not space_vars:
        space_vars = [k for k in ("x", "y", "z") if k in coords]

    if not space_vars:
        return None

    # Pull coordinate arrays
    coord_arrays = [np.asarray(coords[v]) for v in space_vars]

    # Decide whether coords are axes (1D) or already-gridded (same-shape arrays)
    all_1d = all(a.ndim == 1 for a in coord_arrays)
    same_shape = (
        all(a.shape == coord_arrays[0].shape for a in coord_arrays) and coord_arrays[0].ndim >= 1
    )

    if all_1d:
        grids = np.meshgrid(*coord_arrays, indexing="ij")
        grid_shape = grids[0].shape
    elif same_shape:
        # already-gridded coordinates (e.g., X,Y mesh)
        grids = coord_arrays
        grid_shape = grids[0].shape
    else:
        # Mixed / unusual: try meshgrid if they are 1D-like, otherwise give up
        if raise_on_error:
            shapes = {v: np.asarray(coords[v]).shape for v in space_vars}
            raise ValueError(f"Unsupported coord shapes for space_vars={space_vars}: {shapes}")
        return None

    # Safe-ish namespace for eval
    ns: Dict[str, Any] = {"np": np, "__builtins__": {}}

    # Common math functions without "np." prefix (helps user-provided expressions)
    ns.update(
        {
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "sech": lambda x: 1.0 / np.cosh(x),
            "tanh": np.tanh,
            "sqrt": np.sqrt,
            "log": np.log,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e,
            "i": 1j,
            "I": 1j,
            "π": np.pi,
            "max": np.maximum,
            "min": np.minimum,
            "lambda": lambda x: x,  # dummy lambda function for compatibility
        }
    )

    # parameters dict (optional, common in benchmark specs)
    params = pde_spec.get("parameters", {}) or {}
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, (int, float, complex, np.number)):
                ns[k] = v

    # numeric coefficients
    coeffs = pde_spec.get("coefficients", {}) or {}
    if isinstance(coeffs, dict):
        for k, v in coeffs.items():
            if isinstance(v, (int, float, complex, np.number)):
                ns[k] = v

    # top-level numeric params
    for k, v in pde_spec.items():
        if isinstance(v, (int, float, complex, np.number)):
            ns[k] = v

    # spatial vars: x/y/z become full grids
    for var, grid in zip(space_vars, grids, strict=False):
        ns[var] = grid

    # time
    if t_final is not None:
        ns["t"] = float(t_final)

    def _eval_one(expr_str: str) -> np.ndarray:
        if not isinstance(expr_str, str):
            raise TypeError(f"Expression must be a string, got {type(expr_str)}")

        expr_str = _preprocess_expr(expr_str)

        out = eval(expr_str, ns)
        out = np.asarray(out)

        # If expression returned scalar, broadcast to grid shape (preserve dtype, incl. complex)
        if out.shape == ():
            out = np.full(grid_shape, out, dtype=out.dtype)

        # Ensure broadcast-compatible shape
        if out.shape != grid_shape:
            try:
                out = np.broadcast_to(out, grid_shape)
            except Exception as e:
                raise ValueError(
                    f"Expression produced shape {out.shape}, expected {grid_shape}. Expr={expr_str!r}"
                ) from e

        return out

    try:
        # Case 1: single expression -> scalar field
        if isinstance(anal, str):
            return _eval_one(anal)

        # Case 2: list/tuple of expressions -> stack
        if isinstance(anal, (list, tuple)):
            fields = [_eval_one(e) for e in anal]
            return np.stack(fields, axis=stack_axis)

        # Case 3: dict name->expr -> stack, using field_order if provided
        if isinstance(anal, dict):
            if field_order:
                items = [(k, anal[k]) for k in field_order if k in anal]
                missing = [k for k in field_order if k not in anal]
                if missing:
                    raise KeyError(f"analytic_solution missing fields: {missing}")
            else:
                # Prefer common PDE variable order if present; fallback to sorted
                canonical = ["u", "v", "w", "p", "eta", "phi"]
                keys = list(anal.keys())
                ordered = [k for k in canonical if k in keys]
                if not ordered:
                    ordered = sorted(keys)
                items = [(k, anal[k]) for k in ordered]

            # If caller knows numeric channel count, truncate to match
            if expected_num_fields is not None and expected_num_fields > 0:
                items = items[:expected_num_fields]

            fields = [_eval_one(expr) for _, expr in items]
            return np.stack(fields, axis=stack_axis)

        return None

    except Exception:
        if raise_on_error:
            raise
        return None


def implicit_solution_residual(
    coords: Dict[str, np.ndarray],
    t_final: Optional[float],
    pde_spec: Dict[str, Any],
    u_final: np.ndarray,
    *,
    raise_on_error: bool = False,
) -> Optional[np.ndarray]:
    """
    Evaluate implicit residual r = F(u_final, x, y, z, t, params) on the grid.

    Requires:
      pde_spec["analytic_solution"]["type"] == "implicit"
      pde_spec["analytic_solution"]["equation/expression"] = string for F(...)=0
    """
    anal = pde_spec.get("analytic_solution")
    if not isinstance(anal, dict) or anal.get("type", None) != "implicit":
        return None

    eqn = anal.get("equation", None)
    if eqn is None:
        eqn = anal.get("expression", None)
    if not isinstance(eqn, str) or not eqn.strip():
        return None

    # space variables
    space_vars = anal.get("space_variables", None)
    if not space_vars:
        space_vars = [k for k in ("x", "y", "z") if k in coords]
    if not space_vars:
        return None

    # Build grids exactly like analytic_solution()
    coord_arrays = [np.asarray(coords[v]) for v in space_vars]
    all_1d = all(a.ndim == 1 for a in coord_arrays)
    same_shape = (
        all(a.shape == coord_arrays[0].shape for a in coord_arrays) and coord_arrays[0].ndim >= 1
    )

    if all_1d:
        grids = np.meshgrid(*coord_arrays, indexing="ij")
        grid_shape = grids[0].shape
    elif same_shape:
        grids = coord_arrays
        grid_shape = grids[0].shape
    else:
        if raise_on_error:
            shapes = {v: np.asarray(coords[v]).shape for v in space_vars}
            raise ValueError(f"Unsupported coord shapes for space_vars={space_vars}: {shapes}")
        return None

    # Namespace like analytic_solution()
    ns: Dict[str, Any] = {"np": np, "__builtins__": {}}
    ns.update(
        {
            "exp": np.exp,
            "sin": np.sin,
            "cos": np.cos,
            "tan": np.tan,
            "sinh": np.sinh,
            "cosh": np.cosh,
            "tanh": np.tanh,
            "sqrt": np.sqrt,
            "log": np.log,
            "abs": np.abs,
            "pi": np.pi,
            "e": np.e,
            "i": 1j,
            "I": 1j,
            "π": np.pi,
            "max": np.maximum,
            "min": np.minimum,
        }
    )

    # parameters
    params = pde_spec.get("parameters", {}) or {}
    if isinstance(params, dict):
        for k, v in params.items():
            if isinstance(v, (int, float, complex, np.number)):
                ns[k] = v

    # coefficients
    coeffs = pde_spec.get("coefficients", {}) or {}
    if isinstance(coeffs, dict):
        for k, v in coeffs.items():
            if isinstance(v, (int, float, complex, np.number)):
                ns[k] = v

    # top-level numeric
    for k, v in pde_spec.items():
        if isinstance(v, (int, float, complex, np.number)):
            ns[k] = v

    # spatial vars (grids)
    for var, grid in zip(space_vars, grids, strict=False):
        ns[var] = grid

    # time
    if t_final is not None:
        ns["t"] = float(t_final)

    # the unknown itself
    u_arr = np.asarray(u_final)

    # If u_final is multi-field
    # For now: if it has same shape as grid -> scalar u.
    # If it is (C, ...) channels-first, user must write equation using u0,u1,... or provide mapping.
    if u_arr.shape == grid_shape:
        ns["u"] = u_arr
    elif u_arr.ndim == len(grid_shape) + 1 and u_arr.shape[1:] == grid_shape:
        # channels-first: define u0,u1,... and also u as the full array
        ns["u"] = u_arr
        for c in range(u_arr.shape[0]):
            ns[f"u{c}"] = u_arr[c]
    else:
        if raise_on_error:
            raise ValueError(
                f"u_final shape {u_arr.shape} not compatible with grid shape {grid_shape}"
            )
        return None

    # Evaluate F(...)
    try:
        eqn_py = _preprocess_expr(eqn)  # reuse your preprocess
        r = eval(eqn_py, ns)
        r = np.asarray(r)
        if r.shape == ():
            r = np.full(grid_shape, r, dtype=r.dtype)
        if r.shape != grid_shape:
            r = np.broadcast_to(r, grid_shape)
        return r
    except Exception:
        if raise_on_error:
            raise
        return None
