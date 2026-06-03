# formulator.py
import json
import re
from typing import Optional

try:
    from .llm_utils import call_llm
except ImportError:
    from llm_utils_seeded import call_llm


# =============================================================================
# Problem-type detection
# =============================================================================

# Phrases that unambiguously signal a stochastic differential equation.
# These are deliberately specific to avoid false-positives on PDE language:
# words like "diffusion" or "noise" alone appear in deterministic PDE text,
# so they are NOT included here.
_SDE_KEYWORD_PHRASES = [
    "stochastic differential equation",
    "stochastic differential",
    " sde",        # space-prefix avoids matching "inside" etc.
    "(sde)",
    "wiener process",
    "brownian motion",
    "itô",
    "ito sde",
    "ito integral",
    "stratonovich",
    "langevin equation",
    "langevin sde",
    "stochastic noise",
]

# Notation patterns that appear in Itô SDEs but almost never in PDE text.
# These are checked case-sensitively because dW, dX are conventionally capitalised.
_SDE_NOTATION_PATTERNS = [
    " dW",   # Wiener increment (space-prefix avoids "width")
    "(dW",
    "dX =",  # Itô differential (requires '=' to avoid matching array indices)
    "dX=",
    "dY =",  # common second component in 2D SDEs
    "dY=",
]


def _is_sde_problem(description: str) -> bool:
    """
    Lightweight heuristic: returns True if `description` describes an SDE.

    We classify before the LLM call to pick the right system prompt, which
    avoids paying for a separate classification call.  The check is conservative:
    it only triggers on vocabulary that is genuinely SDE-specific, so PDEs with
    stochastic *forcing* in their description won't be misrouted unless they also
    use canonical SDE notation.
    """
    lower = description.lower()
    if any(phrase in lower for phrase in _SDE_KEYWORD_PHRASES):
        return True
    if any(notation in description for notation in _SDE_NOTATION_PATTERNS):
        return True
    return False


# =============================================================================
# Shared JSON cleaning (used by both PDE and SDE paths)
# =============================================================================

def _clean_json_response(resp: str) -> str:
    """Strip markdown fences and inline comments from an LLM JSON response."""
    resp = resp.strip()

    if "```" in resp:
        resp = re.sub(r"```[a-zA-Z0-9]*", "", resp).replace("```", "")

    # Remove // inline comments that the LLM sometimes adds despite being told not to.
    lines = resp.split("\n")
    clean_lines = []
    for line in lines:
        if "//" in line:
            line = line.split("//")[0]
        clean_lines.append(line)
    resp = "\n".join(clean_lines)

    return resp.strip()


# =============================================================================
# PDE formulator  (original — renamed constant, logic unchanged)
# =============================================================================

# Renamed from FORMULATOR_SYSTEM → PDE_FORMULATOR_SYSTEM for clarity now that
# a parallel SDE_FORMULATOR_SYSTEM exists.  The public API (formulate_problem)
# is unchanged so no callers need updating.
PDE_FORMULATOR_SYSTEM = r"""
You are a mathematical modeler specializing in partial differential equations (PDEs).

Goal:
Convert a natural-language PDE problem description into a STRICT JSON object.

Supported Scopes:
- Any PDE family (Heat, Wave, Advection, Fluids, Reaction-Diffusion, or Custom/Generic).
- Any spatial dimension (1D, 2D, 3D, or higher).
- Time-dependent OR steady-state.

CRITICAL RULES (Anti-Hallucination):
1. **Spatial Variables**:
   - If dimension is 1-3: Use ["x"], ["x", "y"], or ["x", "y", "z"].
   - If dimension > 3: Use ["x1", "x2", ..., "xd"].
   - NEVER include "t" (time) in `spatial_variables`.
2. **Coefficients**:
   - Only include coefficients explicitly stated in the description.
   - Do NOT invent parameters.
3. **Analytic Solution**:
   - Only include if the description provides a known exact solution.
4. **Format**:
   - Output STRICT JSON.
   - Check that there's no comma missing or extra commas.
   - **NO COMMENTS** (e.g., do not use // or # inside the JSON).
   - **NO TRAILING COMMAS**.
5. **Analytic Solution**:
    - The expression only contains the right-hand side (RHS) of the solution.
    - Make sure to use Python syntax for math functions (e.g., np.sin, np.exp) and operators.
    - If the description provides a solution in math notation, convert it to Python syntax (e.g., '^' to '**', standalone 'i' to '1j').
    - Ensure that the variables in the analytic solution match those defined in `spatial_variables` and `time_variable`.
    - Make sure it can be use to evalute the relative l2 norm.
    - If it's multiple fields, expression must be a JSON object mapping field names to RHS expressions. No labels inside a single string.
    - Set the "type" field to "explicit" if it's a direct formula, or "implicit" if it's an implicit relation (e.g., u(x,t) = x*sin(u)).
6. **governing_equation**:
    - Provide the PDE in a clear string format, e.g., "u_t = nu u_xx" or "u_t + u u_x = nu u_xx".
    - Use standard mathematical notation with Python syntax for functions and operators. like u_x for partial derivatives instead of ∂_x u, and u_t for time derivative instead of ∂_t u.
    - Fill in the * sign if there's multiplication but shows like "u u_x" instead of "u*u_x".
7. **Initial condition**:
    - Provide the initial condition in a clear string format.
    - Use Python syntax for math functions and operators.

JSON Output Schema:
{
  "equation_type": "string",
  "governing_equation": "string",
  "spatial_dimension": 1,
  "time_dependent": true,
  "spatial_variables": ["x"],
  "time_variable": "t",
  "domain": {
    "bounds": {
       "x": [0.0, 1.0]
    }
  },
  "initial_condition": "string_expression_or_null"
  "boundary_conditions": {
    "type": "string",
    "values": {}
  },
  "parameters": {},
  "analytic_solution": {
       "type": "string",
       "expression": "string",
       "space_variables": ["x"]
  },
  "notes": "string_or_null"
}
"""


def _post_process_pde_spec(spec: dict) -> dict:
    """Normalize PDE spec defaults and fix common LLM mistakes."""

    dim = spec.get("spatial_dimension", 1)
    sv = spec.get("spatial_variables", [])

    if "t" in sv:
        sv = [v for v in sv if v != "t"]
        spec["spatial_variables"] = sv

    if not sv:
        if dim == 1:
            sv = ["x"]
        elif dim == 2:
            sv = ["x", "y"]
        elif dim == 3:
            sv = ["x", "y", "z"]
        else:
            sv = [f"x{i + 1}" for i in range(dim)]
        spec["spatial_variables"] = sv

    dom = spec.get("domain", {})
    if dom is None:
        dom = {}

    bounds = dom.get("bounds", {})
    if not isinstance(bounds, dict):
        bounds = {}

    for v in sv:
        if v not in bounds:
            bounds[v] = [0.0, 1.0]
        if not isinstance(bounds[v], list) or len(bounds[v]) != 2:
            bounds[v] = [0.0, 1.0]

    dom["bounds"] = bounds

    for v in ["x", "y", "z"]:
        if v in bounds:
            dom[f"{v}_min"] = bounds[v][0]
            dom[f"{v}_max"] = bounds[v][1]

    spec["domain"] = dom

    if "parameters" not in spec:
        spec["parameters"] = {}

    return spec


# =============================================================================
# SDE formulator  (new)
# =============================================================================

SDE_FORMULATOR_SYSTEM = r"""
You are a mathematical modeler specializing in stochastic differential equations (SDEs).

Goal:
Convert a natural-language SDE problem description into a STRICT JSON object.

Standard Itô SDE Form (scalar, 1D):
    dX(t) = f(X, t) dt + g(X, t) dW(t)
where:
    f(X, t)  — drift coefficient
    g(X, t)  — diffusion coefficient
    W(t)     — standard scalar Wiener process (Brownian motion)

Multi-dimensional extension (state_dimension > 1):
    dX_i = f_i(X, t) dt + sum_j G_ij(X, t) dW_j(t)
In this case, drift is a list of expressions and diffusion is a list-of-lists matrix.
For Phase 1 (scalar SDEs), always use the scalar form.

CRITICAL RULES (Anti-Hallucination):
1. **Parameters**: Extract ONLY values explicitly stated in the description.
   Do NOT invent numerical values (e.g., do not default mu=0.05 if mu is unspecified).
2. **Analytic solution**: Include ONLY if the description explicitly provides one
   (exact path formula, exact moments, or named distribution).
3. **Drift / Diffusion expressions**: Write in Python/NumPy syntax.
   Use `X` for the scalar state variable, `t` for time, and the exact parameter
   names from the `parameters` dict (e.g., `mu`, `sigma`, `theta`).
   Examples: drift="mu * X", diffusion="sigma", drift="theta * (mu - X)".
4. **diffusion_derivative**: This is dg/dX — the partial derivative of the diffusion
   coefficient with respect to the state X.  It is required by the Milstein scheme.
   - Provide it ONLY when g(X,t) has a simple closed-form derivative w.r.t. X.
   - Examples: g=sigma*X → "sigma";  g=sigma → "0.0";  g=sigma*np.sqrt(X) → "sigma / (2.0 * np.sqrt(X))"
   - If the derivative is complex or unknown, set to null.
5. **noise_structure**: Set to "additive" if g does not depend on X (constant diffusion),
   "multiplicative" if g depends on X, or "mixed" for systems with both types of noise.
6. **Format**: Output STRICT JSON ONLY.  No comments.  No trailing commas.
7. **Analytic solution expressions**: Use Python/NumPy syntax.  Available variables in
   the evaluation namespace are: `t` (time), `X_0` (scalar initial condition),
   and all keys from the `parameters` dict.  Use `np.` prefix for NumPy functions.

JSON Output Schema:
{
  "equation_type": "SDE",
  "sde_type": "ito",
  "governing_equation": "string",
  "drift": "string",
  "diffusion": "string",
  "diffusion_derivative": "string or null",
  "state_dimension": 1,
  "noise_dimension": 1,
  "noise_type": "wiener",
  "noise_structure": "string",
  "time_interval": {
    "t_0": 0.0,
    "t_final": 1.0
  },
  "initial_condition": {
    "X_0": 1.0
  },
  "parameters": {},
  "analytic_solution": {
    "type": "moments",
    "mean": "string or null",
    "variance": "string or null",
    "terminal_distribution": "string or null"
  },
  "notes": "string or null"
}

Analytic solution type guidance:
- "moments"      : provide exact mean(t) and/or variance(t) as Python expressions.
- "explicit"     : provide the exact sample-path formula X(t) = ... (only when the
                   Wiener path W(t) does not appear, i.e., the formula is deterministic).
- "distribution" : name the terminal distribution family (e.g., "lognormal", "gaussian")
                   and optionally provide moment expressions.
Set analytic_solution to null if no closed-form information is given.
"""


def _post_process_sde_spec(spec: dict) -> dict:
    """
    Normalize SDE spec defaults and fix common LLM output inconsistencies.

    Mirrors _post_process_pde_spec() in structure: make every field safe to
    read downstream without guard clauses scattered across the pipeline.
    """

    # ── Top-level discriminator ───────────────────────────────────────────────
    # Always force this so the pipeline router in run_single_problem.py can rely
    # on it without checking the raw LLM text again.
    spec["equation_type"] = "SDE"

    # ── SDE type ─────────────────────────────────────────────────────────────
    # Default to Itô (by far the most common in physics/finance literature).
    if spec.get("sde_type") not in ("ito", "stratonovich"):
        spec["sde_type"] = "ito"

    # ── Dimensions ───────────────────────────────────────────────────────────
    # state_dimension: the dimension of the state vector X(t).
    # noise_dimension: the dimension of the Wiener process W(t).
    # For most Phase 1 problems both are 1; for correlated multi-D SDEs they may differ.
    if not isinstance(spec.get("state_dimension"), int) or spec["state_dimension"] < 1:
        spec["state_dimension"] = 1
    if not isinstance(spec.get("noise_dimension"), int) or spec["noise_dimension"] < 1:
        spec["noise_dimension"] = spec["state_dimension"]

    # ── Time interval ─────────────────────────────────────────────────────────
    # Normalise to {"t_0": float, "t_final": float}.
    # LLMs sometimes output "T", "t_end", or "t_max" instead of "t_final".
    ti = spec.get("time_interval")
    if not isinstance(ti, dict):
        ti = {}
    for alias in ("T", "t_end", "t_max"):
        if alias in ti and "t_final" not in ti:
            ti["t_final"] = ti.pop(alias)
    ti.setdefault("t_0", 0.0)
    ti.setdefault("t_final", 1.0)
    spec["time_interval"] = ti

    # ── Initial condition ─────────────────────────────────────────────────────
    # Normalise to {"X_0": scalar_or_list}.
    # LLMs sometimes output a bare float instead of a dict.
    ic = spec.get("initial_condition")
    if ic is None:
        ic = {"X_0": 1.0}
    elif isinstance(ic, (int, float)):
        ic = {"X_0": float(ic)}
    elif isinstance(ic, list):
        # Multi-dimensional IC supplied as a list — wrap it.
        ic = {"X_0": ic}
    spec["initial_condition"] = ic

    # ── Parameters ───────────────────────────────────────────────────────────
    if not isinstance(spec.get("parameters"), dict):
        spec["parameters"] = {}

    # ── Noise type ───────────────────────────────────────────────────────────
    spec.setdefault("noise_type", "wiener")

    # ── Noise structure (additive vs multiplicative) ───────────────────────────
    # If the LLM did not set this (or set an invalid value), infer it from the
    # diffusion expression.  The rule: if the diffusion coefficient g(X,t) contains
    # the state variable X, the noise is multiplicative (Milstein is then beneficial).
    # A simple regex word-boundary check handles the common 1D case correctly.
    if spec.get("noise_structure") not in ("additive", "multiplicative", "mixed"):
        diffusion_expr = spec.get("diffusion", "")
        if isinstance(diffusion_expr, str):
            has_state_var = bool(re.search(r"\bX\b", diffusion_expr))
            spec["noise_structure"] = "multiplicative" if has_state_var else "additive"
        else:
            # Multi-dimensional diffusion: default conservative (multiplicative)
            # so the planner doesn't prematurely skip Milstein consideration.
            spec["noise_structure"] = "multiplicative"

    # ── Diffusion derivative ──────────────────────────────────────────────────
    # Default to None.  The planner will only propose Milstein plans when this
    # is a non-null string, because Milstein requires dg/dX.
    spec.setdefault("diffusion_derivative", None)

    # ── Analytic solution ─────────────────────────────────────────────────────
    # Normalise to a consistent structure so sde_analytic_utils can read it
    # without further guards.
    anal = spec.get("analytic_solution")
    if not anal or anal == {}:
        spec["analytic_solution"] = None
    elif isinstance(anal, dict):
        # Infer type from whichever fields are present if the LLM omitted it.
        if "type" not in anal:
            if "mean" in anal or "variance" in anal:
                anal["type"] = "moments"
            elif "expression" in anal:
                anal["type"] = "explicit"
            else:
                anal["type"] = "moments"
        # Ensure all expected keys exist so downstream code can .get() safely.
        anal.setdefault("mean", None)
        anal.setdefault("variance", None)
        anal.setdefault("terminal_distribution", None)
        spec["analytic_solution"] = anal

    return spec


# =============================================================================
# Internal formulation helpers
# =============================================================================

def _formulate_pde(problem_description: str) -> dict:
    """
    PDE path: call the LLM with PDE_FORMULATOR_SYSTEM and post-process the output.
    Retries up to 3 times on JSON parse errors (the main failure mode).
    """
    user_prompt = (
        "Here is a natural-language description of a PDE problem:\n"
        f'"""{problem_description}"""\n\n'
        "Please extract the PDE specification details into JSON."
    )

    max_retries = 3
    last_error = None

    for _attempt in range(max_retries):
        try:
            resp = call_llm(PDE_FORMULATOR_SYSTEM, user_prompt, model="claude-sonnet-4-6")
            clean_resp = _clean_json_response(resp)
            spec = json.loads(clean_resp)
            spec = _post_process_pde_spec(spec)
            return spec
        except (json.JSONDecodeError, Exception) as e:
            last_error = e

    print(f"[Formulator/PDE] Failed after {max_retries} attempts.")
    raise ValueError("PDE formulator failed to produce valid JSON.") from last_error


def _formulate_sde(problem_description: str) -> dict:
    """
    SDE path: call the LLM with SDE_FORMULATOR_SYSTEM and post-process the output.
    Same retry structure as the PDE path for consistency.
    """
    user_prompt = (
        "Here is a natural-language description of a stochastic differential equation (SDE):\n"
        f'"""{problem_description}"""\n\n'
        "Please extract the SDE specification details into JSON."
    )

    max_retries = 3
    last_error = None

    for _attempt in range(max_retries):
        try:
            resp = call_llm(SDE_FORMULATOR_SYSTEM, user_prompt, model="claude-sonnet-4-6")
            clean_resp = _clean_json_response(resp)
            spec = json.loads(clean_resp)
            spec = _post_process_sde_spec(spec)
            return spec
        except (json.JSONDecodeError, Exception) as e:
            last_error = e

    print(f"[Formulator/SDE] Failed after {max_retries} attempts.")
    raise ValueError("SDE formulator failed to produce valid JSON.") from last_error


# =============================================================================
# Public entry point
# =============================================================================

def formulate_problem(problem_description: str, seed: Optional[int] = None) -> dict:
    """
    Convert a natural-language problem description to a structured spec dict.

    Routes to the PDE or SDE formulator based on a keyword heuristic, then
    calls the LLM with the appropriate system prompt.  The returned dict always
    has an "equation_type" key ("PDE"-family string or "SDE") so downstream
    pipeline stages can branch without re-reading the raw description.

    The `seed` argument is kept for API compatibility but is unused — the LLM
    temperature (0.2) provides sufficient determinism for the formulation step.
    """
    del seed

    if _is_sde_problem(problem_description):
        return _formulate_sde(problem_description)
    else:
        return _formulate_pde(problem_description)
