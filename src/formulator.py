# formulator.py (ROBUST JSON FIX)
import json
import re
from typing import Optional

try:
    from .llm_utils import call_llm
except ImportError:
    from llm_utils_seeded import call_llm


FORMULATOR_SYSTEM = r"""
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


def _clean_json_response(resp: str) -> str:
    """Attempt to extract valid JSON from LLM markdown/text."""
    resp = resp.strip()

    if "```" in resp:
        resp = re.sub(r"```[a-zA-Z0-9]*", "", resp).replace("```", "")

    lines = resp.split("\n")
    clean_lines = []
    for line in lines:
        if "//" in line:
            line = line.split("//")[0]
        clean_lines.append(line)
    resp = "\n".join(clean_lines)

    return resp.strip()


def _post_process_pde_spec(spec: dict) -> dict:
    """Normalize defaults and fix common LLM mistakes."""

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


def formulate_problem(problem_description: str, seed: Optional[int] = None) -> dict:
    """
    Main entry point: Converts generic text description -> Structured PDE Spec.
    Includes RETRY logic for robust JSON parsing.
    """
    del seed

    user_prompt = (
        "Here is a natural-language description of a PDE problem:\n"
        f'"""{problem_description}"""\n\n'
        "Please extract the PDE specification details into JSON."
    )

    max_retries = 3
    last_error = None

    for _attempt in range(max_retries):
        try:
            resp = call_llm(FORMULATOR_SYSTEM, user_prompt, model="gpt-4.1")
            clean_resp = _clean_json_response(resp)
            spec = json.loads(clean_resp)
            spec = _post_process_pde_spec(spec)
            return spec

        except json.JSONDecodeError as e:
            last_error = e
            continue
        except Exception as e:
            last_error = e
            continue

    print(f"[Formulator] Failed after {max_retries} attempts.")
    raise ValueError("Formulator failed to produce valid JSON.") from last_error
