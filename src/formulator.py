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

===========================================================================
SCALAR SDE (state_dimension = 1) — standard form:
    dX(t) = f(X, t) dt + g(X, t) dW(t)

  drift:               string expression in Python/NumPy using X, t, and parameter names
  diffusion:           string expression in Python/NumPy using X, t, and parameter names
  diffusion_derivative: dg/dX as a string (for Milstein scheme), or null
  initial_condition:   {"X_0": float}
  analytic_solution:   {"type": "moments", "mean": "expr", "variance": "expr"}

===========================================================================
MULTI-DIMENSIONAL SDE (state_dimension > 1):
    dX_i(t) = f_i(X, t) dt + sum_j G_ij(X, t) dW_j(t)
where X = [X_0-component, X_1-component, ...], W = [W_1, ..., W_noise_dim] are
independent Wiener processes, and G is the d_state × d_noise diffusion matrix.

  state_variables:     list of Python variable names for each state component,
                       e.g. ["X", "Y"] for a 2D system.
  drift:               list of strings, one per state component.
                       Use the names from state_variables in expressions.
                       Example: ["Y", "-X"]  for the stochastic oscillator
  diffusion:           list-of-lists G[i][j] — d_state rows × d_noise cols.
                       G[i][j] is the Python/NumPy expression for the coefficient
                       multiplying the j-th noise source in the i-th component.
                       Examples:
                         Single noise source (noise_dim=1), 2D state:
                           [["0.0"], ["sigma"]]   ← only Y component is driven
                         Independent diagonal noise (noise_dim=2), 2D state:
                           [["sigma1 * X", "0.0"], ["0.0", "sigma2 * Y"]]
                         Cholesky-correlated noise (noise_dim=2), 2D state:
                           [["sigma1 * X", "0.0"],
                            ["rho * sigma2 * Y", "np.sqrt(1 - rho**2) * sigma2 * Y"]]
  diffusion_derivative: null for multi-dimensional SDEs (Milstein is not used)
  initial_condition:   dict with one key per component: {"X_0": float, "Y_0": float}
  analytic_solution:   use per-component keys:
                         "mean_X": "expr",  "variance_X": "expr",
                         "mean_Y": "expr",  "variance_Y": "expr",
                         "covariance_XY": "expr"  (optional)

CHOLESKY FACTORIZATION RULE:
When two Brownian motions W1, W2 have correlation ρ, express Y-component noise as:
  sigma_Y * (rho * dW1 + sqrt(1-rho^2) * dW2)
which gives G[1][0] = "rho * sigma_Y * Y" and G[1][1] = "np.sqrt(1-rho**2) * sigma_Y * Y".
Treat dW1, dW2 as INDEPENDENT standard Brownian increments Z1, Z2 in the Cholesky form.

===========================================================================
CRITICAL RULES:
1. **Parameters**: Extract ONLY values explicitly stated in the description.
   Do NOT invent numerical values (e.g., do not default mu=0.05 if mu is unspecified).
2. **Analytic solution**: Include ONLY if the description explicitly provides one
   (exact path formula, exact moments, or named distribution).
3. **Drift / Diffusion expressions**: Python/NumPy syntax only.
   For scalar SDEs use X for the state variable.
   For multi-D SDEs use the names from state_variables (e.g., X, Y).
   Examples: drift="mu * X", diffusion="sigma", drift="theta * (mu - X)".
4. **diffusion_derivative**: dg/dX for scalar SDEs only.
   - Provide when g(X,t) has a closed-form derivative: g=sigma*X → "sigma";
     g=sigma → "0.0"; g=sigma*np.sqrt(X) → "sigma / (2.0 * np.sqrt(X))"
   - Set null for multi-dimensional SDEs and when derivative is unknown.
5. **noise_structure**: "additive" if g does not depend on state (all G[i][j] are
   constants), "multiplicative" if any G[i][j] depends on the state.
6. **Format**: Output STRICT JSON ONLY.  No comments.  No trailing commas.
7. **Analytic solution expressions**: Python/NumPy syntax. Available namespace:
   t (time), all initial_condition keys (e.g. X_0, Y_0), all parameter keys.
   Use np. prefix for NumPy functions (e.g. np.exp, np.cos, np.sqrt).

===========================================================================
JSON Output Schema (scalar SDE, state_dimension=1):
{
  "equation_type": "SDE",
  "sde_type": "ito",
  "governing_equation": "string",
  "state_dimension": 1,
  "noise_dimension": 1,
  "state_variables": ["X"],
  "noise_type": "wiener",
  "noise_structure": "string",
  "drift": "string",
  "diffusion": "string",
  "diffusion_derivative": "string or null",
  "time_interval": {"t_0": 0.0, "t_final": 1.0},
  "initial_condition": {"X_0": 1.0},
  "parameters": {},
  "analytic_solution": {
    "type": "moments",
    "mean": "string or null",
    "variance": "string or null",
    "terminal_distribution": "string or null"
  },
  "notes": "string or null"
}

JSON Output Schema (multi-D SDE, state_dimension=2 example):
{
  "equation_type": "SDE",
  "sde_type": "ito",
  "governing_equation": "string",
  "state_dimension": 2,
  "noise_dimension": 1,
  "state_variables": ["X", "Y"],
  "noise_type": "wiener",
  "noise_structure": "additive",
  "drift": ["Y", "-X"],
  "diffusion": [["0.0"], ["sigma"]],
  "diffusion_derivative": null,
  "time_interval": {"t_0": 0.0, "t_final": 6.28},
  "initial_condition": {"X_0": 1.0, "Y_0": 0.0},
  "parameters": {"sigma": 0.3},
  "analytic_solution": {
    "type": "moments",
    "mean_X": "X_0 * np.cos(t) + Y_0 * np.sin(t)",
    "mean_Y": "-X_0 * np.sin(t) + Y_0 * np.cos(t)",
    "variance_X": "sigma**2 / 2 * (t - np.sin(t) * np.cos(t))",
    "variance_Y": "sigma**2 / 2 * (t + np.sin(t) * np.cos(t))"
  },
  "notes": "string or null"
}

Analytic solution type guidance:
- "moments"      : provide exact mean(t) and/or variance(t) as Python expressions.
                   For multi-D: use mean_X/Y, variance_X/Y keys.
- "explicit"     : exact sample-path formula X(t) = ... (only deterministic formulas).
- "distribution" : name the terminal distribution family (e.g., "lognormal", "gaussian").
Set analytic_solution to null if no closed-form information is given.
"""


def _post_process_sde_spec(spec: dict) -> dict:
    """Normalize SDE spec defaults and fix common LLM output inconsistencies.

    Mirrors _post_process_pde_spec() in structure: make every field safe to
    read downstream without guard clauses scattered across the pipeline.

    Phase 2 additions: handles state_dimension > 1 by normalizing
    state_variables, multi-component IC, list-of-lists diffusion matrix,
    and per-component analytic solution keys (mean_X, variance_X, etc.).
    """

    # ── Top-level discriminator ───────────────────────────────────────────────
    # Always force this so the pipeline router in run_single_problem.py can rely
    # on it without checking the raw LLM text again.
    spec["equation_type"] = "SDE"

    # ── SDE type ─────────────────────────────────────────────────────────────
    if spec.get("sde_type") not in ("ito", "stratonovich"):
        spec["sde_type"] = "ito"

    # ── Dimensions ───────────────────────────────────────────────────────────
    # state_dimension: dimension of the state vector X(t); 1 for scalar problems.
    # noise_dimension: dimension of the Wiener process W(t); defaults to state_dimension.
    if not isinstance(spec.get("state_dimension"), int) or spec["state_dimension"] < 1:
        spec["state_dimension"] = 1
    d = spec["state_dimension"]
    if not isinstance(spec.get("noise_dimension"), int) or spec["noise_dimension"] < 1:
        spec["noise_dimension"] = d

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
    # Scalar: {"X_0": float}.
    # Multi-D: {"X_0": float, "Y_0": float, ...} — one key per component.
    # LLMs sometimes output a bare float or a list; normalise both cases.
    ic = spec.get("initial_condition")
    if ic is None:
        ic = {"X_0": 1.0}
    elif isinstance(ic, (int, float)):
        ic = {"X_0": float(ic)}
    elif isinstance(ic, list):
        # List IC: [1.0, 0.0] → map to {"X_0": 1.0, "Y_0": 0.0} using default names.
        # Default names: X, Y, Z for d<=3, else X0, X1, ...
        default_names = (["X", "Y", "Z"][:d] if d <= 3 else [f"X{i}" for i in range(d)])
        ic = {f"{v}_0": float(val) for v, val in zip(default_names, ic)}
    spec["initial_condition"] = ic

    # ── State variables ───────────────────────────────────────────────────────
    # Derive from IC keys when possible: {"X_0": ..., "Y_0": ...} → ["X", "Y"].
    # The regex strips trailing underscores and digits from IC keys.
    if not spec.get("state_variables"):
        ic_keys = list(spec["initial_condition"].keys())
        if len(ic_keys) == d:
            state_vars = [re.sub(r"[_\d]+$", "", k) or k for k in ic_keys]
        else:
            state_vars = (["X", "Y", "Z"][:d] if d <= 3 else [f"X{i}" for i in range(d)])
        spec["state_variables"] = state_vars

    # ── Parameters ───────────────────────────────────────────────────────────
    if not isinstance(spec.get("parameters"), dict):
        spec["parameters"] = {}

    # ── Noise type ───────────────────────────────────────────────────────────
    spec.setdefault("noise_type", "wiener")

    # ── Multi-D diffusion normalisation ───────────────────────────────────────
    # For state_dimension > 1, diffusion must be a list-of-lists G[i][j].
    # The formulator sometimes outputs a flat list of strings; promote it here.
    if d > 1:
        noise_dim = spec["noise_dimension"]
        diff = spec.get("diffusion")
        if isinstance(diff, list) and len(diff) == d and diff:
            if isinstance(diff[0], str):
                # Flat list of d strings. Interpret based on noise_dimension:
                # noise_dim == 1  → each element is G[i][0] (single noise source)
                # noise_dim == d  → diagonal matrix (each component its own noise)
                if noise_dim == 1:
                    spec["diffusion"] = [[expr] for expr in diff]
                else:
                    # Diagonal: G[i][i] = diff[i], all off-diagonal = "0.0"
                    mat = []
                    for i in range(d):
                        row = ["0.0"] * noise_dim
                        if i < noise_dim:
                            row[i] = diff[i]
                        mat.append(row)
                    spec["diffusion"] = mat

    # ── Noise structure (additive vs multiplicative) ───────────────────────────
    # Infer from diffusion expressions if not set or invalid.
    if spec.get("noise_structure") not in ("additive", "multiplicative", "mixed"):
        state_vars = spec.get("state_variables", ["X"])
        diff = spec.get("diffusion", "")

        if isinstance(diff, str):
            # Scalar: check if g(X) references the state variable
            has_state_dep = bool(re.search(r"\bX\b", diff))
        elif isinstance(diff, list):
            # Multi-D: collect all expression strings and check for any state var reference
            all_exprs = []
            for row in diff:
                if isinstance(row, list):
                    all_exprs.extend(row)
                elif isinstance(row, str):
                    all_exprs.append(row)
            has_state_dep = any(
                bool(re.search(rf"\b{v}\b", expr))
                for v in state_vars
                for expr in all_exprs
                if isinstance(expr, str)
            )
        else:
            has_state_dep = False

        spec["noise_structure"] = "multiplicative" if has_state_dep else "additive"

    # ── Diffusion derivative ──────────────────────────────────────────────────
    # Only relevant for scalar Milstein; null for multi-D (Lévy area not implemented).
    spec.setdefault("diffusion_derivative", None)

    # ── Analytic solution ─────────────────────────────────────────────────────
    # Normalise to a consistent structure so sde_analytic_utils can read it
    # without further guards.
    anal = spec.get("analytic_solution")
    if not anal or anal == {}:
        spec["analytic_solution"] = None
    elif isinstance(anal, dict):
        # Normalize the type field. The LLM sometimes returns "distribution"
        # or other non-standard strings instead of the canonical "moments".
        current_type = anal.get("type")
        # Multi-D analytic solutions use mean_X/Y, variance_X/Y keys.
        has_multid_keys = any(
            k.startswith("mean_") or k.startswith("variance_")
            for k in anal.keys()
        )
        if current_type not in ("moments", "explicit"):
            if "mean" in anal or "variance" in anal or has_multid_keys:
                anal["type"] = "moments"
            elif "expression" in anal:
                anal["type"] = "explicit"
            else:
                anal["type"] = "moments"
        # For scalar SDEs, ensure standard keys exist so downstream can .get() safely.
        # For multi-D, skip these defaults (they would add incorrect null scalar keys).
        if not has_multid_keys:
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
