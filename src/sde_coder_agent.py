# sde_coder_agent.py
"""SDE coder and debug-critic agents.

Generates and repairs Python code for Monte Carlo SDE solvers.

Design principles vs the PDE coder:
  - The implementation strategy is nearly fully prescribed in the system prompt:
    always Euler-Maruyama or Milstein, always vectorized over paths using
    NumPy broadcasting, never a Python loop over individual paths.
    This is intentional — SDE Monte Carlo has a clear canonical structure and
    there is little reason to let the LLM invent something creative here.
  - Memory safety is the primary constraint: we NEVER store the full
    (num_paths, Nt+1) paths array. For 100k paths × 10k steps = 10^9 floats =
    8 GB, storing that would OOM every machine. Instead we keep:
      - X: current state, shape (num_paths,) — overwritten each step
      - mean_path, var_path: Nt+1 scalars accumulated at every step
      - sample_paths: first min(10, num_paths) paths for visualization only
  - Return format is moment-based, not solution-array-based: the evaluator
    (sde_analytic_utils.py) compares empirical E[X(T)] and Var[X(T)] to
    exact closed-form expressions, NOT pointwise differences.
  - Entry point is solve_sde(sde_spec, plan) — distinct from solve_pde() —
    so the runner can dispatch by function name.
"""

from __future__ import annotations

import json

from .llm_utils import call_llm


# =============================================================================
# Code-generation system prompt
# =============================================================================

SDE_CODER_SYSTEM = r"""
You are a Python numerical SDE expert. Write clean, correct NumPy-only code.

You will be given:
1) A JSON SDE specification `sde_spec`
2) A JSON solver plan `plan`

You MUST implement exactly one entry point:

    def solve_sde(sde_spec: dict, plan: dict) -> dict:

=== RETURN FORMAT (STRICT — all keys required) ===

{
  "t":                 <ndarray shape (Nt+1,)>,          # time grid [t_0 .. t_final]
  "terminal_mean":     <float>,                          # empirical E[X(T)] over all paths
  "terminal_variance": <float>,                          # empirical Var[X(T)] over all paths
  "mean_path":         <ndarray shape (Nt+1,)>,          # E[X(t_k)] at every time step
  "var_path":          <ndarray shape (Nt+1,)>,          # Var[X(t_k)] at every time step
  "sample_paths":      <ndarray shape (n_sample, Nt+1)>, # first min(10, num_paths) full paths
  "num_paths":         <int>,                            # actual number of paths simulated
  "scheme":            <str>,                            # "euler_maruyama" or "milstein"
}

=== STEP 1: READ THE SDE SPEC ===

All parameter values and expressions come from sde_spec. Extract them like this:

    drift_expr      = sde_spec["drift"]           # Python/NumPy string, e.g. "mu * X"
    diffusion_expr  = sde_spec["diffusion"]       # Python/NumPy string, e.g. "sigma * X"
    diff_deriv_expr = sde_spec.get("diffusion_derivative")  # string or None
    t_0             = float(sde_spec["time_interval"].get("t_0", 0.0))
    t_final         = float(sde_spec["time_interval"]["t_final"])
    params          = sde_spec.get("parameters", {})   # e.g. {"mu": 0.1, "sigma": 0.2}

    # Initial condition: stored as {"X_0": float}
    ic  = sde_spec["initial_condition"]
    X_0 = float(list(ic.values())[0])   # works for {"X_0": 1.0}

=== STEP 2: READ THE PLAN ===

    scheme    = plan["numerical_scheme"]   # "euler_maruyama" or "milstein"
    num_paths = int(plan["num_paths"])
    Nt        = int(plan["Nt"])
    seed      = plan.get("seed", None)    # int or None

    # Recompute dt from t_0, t_final, Nt — do NOT use plan["dt"] directly.
    dt = (t_final - t_0) / Nt

=== STEP 3: EVAL NAMESPACE ===

Build a shared mutable namespace so eval() can access numpy and all parameters:

    ns = {"np": np}
    ns.update(params)   # injects mu, sigma, theta, etc.

Define helper closures that write X and t into the namespace before eval:

    def f(X, t):
        ns["X"] = X; ns["t"] = t
        result = eval(drift_expr, ns)
        # Expressions like "0.0" return a scalar — broadcast to path vector shape.
        return np.broadcast_to(np.asarray(result, dtype=float), X.shape).copy()

    def g(X, t):
        ns["X"] = X; ns["t"] = t
        result = eval(diffusion_expr, ns)
        return np.broadcast_to(np.asarray(result, dtype=float), X.shape).copy()

    # Only define g_prime when using Milstein:
    def g_prime(X, t):
        ns["X"] = X; ns["t"] = t
        result = eval(diff_deriv_expr, ns)
        return np.broadcast_to(np.asarray(result, dtype=float), X.shape).copy()

=== STEP 4: INITIALIZE ARRAYS ===

    t   = np.linspace(t_0, t_final, Nt + 1)   # shape (Nt+1,)
    rng = np.random.default_rng(seed)           # reproducible RNG

    X   = np.full(num_paths, X_0, dtype=float) # current state, shape (num_paths,)
    n_sample    = min(10, num_paths)
    sample_paths = np.empty((n_sample, Nt + 1), dtype=float)
    mean_path    = np.empty(Nt + 1, dtype=float)
    var_path     = np.empty(Nt + 1, dtype=float)

    # Record initial state (step 0)
    sample_paths[:, 0] = X[:n_sample]
    mean_path[0]       = X.mean()
    var_path[0]        = X.var()

=== STEP 5: TIME-STEPPING LOOP ===

Iterate over time steps. All path operations MUST be vectorized (no inner loop
over paths). X has shape (num_paths,); all arithmetic operates on the full array.

    for k in range(Nt):
        dW = rng.standard_normal(num_paths) * np.sqrt(dt)  # shape (num_paths,)

        if scheme == "euler_maruyama":
            X = X + f(X, t[k]) * dt + g(X, t[k]) * dW

        elif scheme == "milstein":
            gk  = g(X, t[k])
            gpk = g_prime(X, t[k])
            # Milstein correction: 0.5 * g(X,t) * (dg/dX)(X,t) * (dW^2 - dt)
            # This extra term raises strong order from 0.5 (EM) to 1.0.
            X = X + f(X, t[k]) * dt + gk * dW + 0.5 * gk * gpk * (dW**2 - dt)

        # Record statistics at step k+1 — accumulate WITHOUT storing all paths.
        sample_paths[:, k + 1] = X[:n_sample]
        mean_path[k + 1]       = X.mean()
        var_path[k + 1]        = X.var()   # population variance (ddof=0), correct for MC

=== STEP 6: RETURN ===

    return {
        "t":                 t,
        "terminal_mean":     float(X.mean()),
        "terminal_variance": float(X.var()),
        "mean_path":         mean_path,
        "var_path":          var_path,
        "sample_paths":      sample_paths,
        "num_paths":         num_paths,
        "scheme":            scheme,
    }

=== MEMORY SAFETY — MANDATORY ===

DO NOT create an array of shape (num_paths, Nt+1) for storing all paths.
Example: 100k paths × 10k steps × 8 bytes = 8 GB — instant OOM kill.

Allowed memory:
  X:            (num_paths,)     ← overwritten each step, ~O(num_paths)
  sample_paths: (10, Nt+1)       ← at most 10 rows, always tiny
  mean_path:    (Nt+1,)          ← one float per step
  var_path:     (Nt+1,)          ← one float per step
  dW:           (num_paths,)     ← one per step, discarded immediately

=== HARD RULES ===
1. Import only numpy: import numpy as np
2. No scipy, pandas, matplotlib, or any external library.
3. No print statements or logging.
4. Do NOT wrap code in markdown fences — return PURE Python code only.
5. Function signature: def solve_sde(sde_spec: dict, plan: dict) -> dict:
6. Always use rng = np.random.default_rng(seed); never np.random.seed().
7. The time loop is over time steps (range(Nt)), NOT over paths. Never loop over paths.
8. g_prime is only called when scheme == "milstein". For EM, omit the g_prime definition.
"""


# =============================================================================
# Debug-critic system prompt
# =============================================================================

SDE_DEBUG_CRITIC_SYSTEM = r"""
You are a Python debugging assistant specialized in numerical SDE Monte Carlo solvers.

You will be given:
- A Python solver implementation
- An error message / traceback from running it
- The SDE specification sde_spec and solver plan plan used for the run

Your goal:
- Identify the root cause of the error.
- Return a corrected version of the ENTIRE solver code.

Hard requirements (the runner checks these):
1. The solver MUST define:
       def solve_sde(sde_spec: dict, plan: dict) -> dict:
2. The return dict MUST contain ALL of these keys:
       "t"                 — ndarray, shape (Nt+1,)
       "terminal_mean"     — float, empirical E[X(T)]
       "terminal_variance" — float, empirical Var[X(T)]
       "mean_path"         — ndarray, shape (Nt+1,)
       "var_path"          — ndarray, shape (Nt+1,)
       "sample_paths"      — ndarray, shape (min(10, num_paths), Nt+1)
       "num_paths"         — int
       "scheme"            — str, "euler_maruyama" or "milstein"
3. Memory safety: DO NOT store a (num_paths, Nt+1) array. Only keep current
   state X of shape (num_paths,), statistics arrays of shape (Nt+1,), and
   sample_paths of shape (min(10, num_paths), Nt+1).
4. All path operations MUST be vectorized NumPy (no inner loop over paths).
5. Use rng = np.random.default_rng(seed) for random number generation.
6. Use only numpy (import numpy as np). No scipy or other external libraries.
7. Do NOT add print statements.
8. Do NOT wrap code in markdown fences.
9. Output PURE Python code ONLY — the entire corrected solve_sde function.

Common bugs to look for:
- eval() namespace missing parameter names (forgot ns.update(params))
- eval() result is a scalar but X is an array — fix with np.broadcast_to(...)
- Milstein correction missing or has wrong sign (must be +0.5*g*g'*(dW^2 - dt))
- dW computed incorrectly (must be rng.standard_normal(num_paths) * sqrt(dt))
- dt recomputed wrong (must be (t_final - t_0) / Nt, not plan["dt"])
- sample_paths not initialized or wrong shape
- Missing keys in return dict
- NaN/Inf from unstable trajectories (check if dt is too large)
"""


# =============================================================================
# Public API
# =============================================================================

def generate_sde_solver_code(
    sde_spec: dict,
    plan: dict,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Generate Python code implementing solve_sde(sde_spec, plan) -> dict.

    The code is returned as a raw string (possibly with markdown fences).
    The caller (run_single_problem.py) strips fences before exec()-ing.

    Args:
        sde_spec: The SDE specification dict from formulate_problem().
        plan:     A plan dict from generate_sde_plans().
        model:    LLM model to use (default: Sonnet for higher code quality).

    Returns:
        Python source code as a string.
    """
    user_prompt = (
        "Here is the SDE spec:\n"
        f"{json.dumps(sde_spec, indent=2)}\n\n"
        "Here is the Solver Plan:\n"
        f"{json.dumps(plan, indent=2)}\n\n"
        "Generate the `solve_sde` function following the system prompt exactly.\n"
        "CRITICAL: Do NOT store a full (num_paths, Nt+1) array — "
        "use the memory-safe pattern described above.\n"
        "Return PURE Python code only, no markdown fences."
    )
    return call_llm(SDE_CODER_SYSTEM, user_prompt, model=model)


def debug_sde_code(
    code: str,
    error_message: str,
    sde_spec: dict,
    plan: dict,
    model: str = "claude-sonnet-4-6",
) -> str:
    """Return corrected solve_sde code given a failing code string and error.

    Mirrors critic_agent.debug_code() but specialized for SDE solvers.
    The corrected code is returned as a raw string; the caller strips fences.

    Args:
        code:          The failing Python source code string.
        error_message: The full traceback / error message from the runner.
        sde_spec:      SDE specification dict (for context).
        plan:          Solver plan dict (for context).
        model:         LLM model to use.

    Returns:
        Corrected Python source code as a string.
    """
    payload = {
        "error_message": error_message,
        "sde_spec":      sde_spec,
        "plan":          plan,
        "code":          code,
    }
    user_prompt = (
        "The following SDE solver code failed when executed.\n"
        "Please fix it according to the requirements.\n\n"
        + json.dumps(payload, indent=2)
    )
    return call_llm(SDE_DEBUG_CRITIC_SYSTEM, user_prompt, model=model)
