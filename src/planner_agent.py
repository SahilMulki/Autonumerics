# planner_agent.py

import json
import re

from .llm_utils import call_llm


def _extract_json(text: str) -> str:
    """Strip code fences then find the first balanced JSON object or array."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener, closer = ch, ("}" if ch == "{" else "]")
            break
    if start is None:
        raise ValueError(f"No JSON found in LLM response: {text[:200]!r}")

    depth, in_str, escape = 0, False, False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            escape = (not escape and c == "\\")
            if not escape and c == '"':
                in_str = False
        else:
            if c == '"':
                in_str = True
            elif c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]

    raise ValueError("Could not extract balanced JSON from LLM response.")

PLANNER_SYSTEM = r"""
You are an expert in numerical PDEs.

Given a PDE specification `pde_spec`, propose a diverse set of numerical solver plans.
You are NOT restricted to any fixed list of methods.

IMPORTANT: You MUST output a JSON LIST of exactly `num_plans` plan objects.
Each plan object MUST follow this schema (keys required unless noted):

{
  "plan_id": <string>,                 # unique
  "description": <string>,             # short human-readable
  "spatial_discretization": {
    "scheme": <string>,                # e.g., "finite_difference", "finite_volume", "spectral", "fem"
    "Nx": <int>,                       # required
    "Ny": <int or null>,               # null for 1D
    "Nz": <int or null>,               # null for 1D/2D
    "order": <int or null>,            # optional accuracy order
    "extra_parameters": { ... }        # optional free-form
  },
  "time_stepping": {
    "method": <string or null>,        # null if steady-state PDE (e.g., Poisson)
    "dt": <float or null>,             # may be null if Nt is provided or if steady-state
    "Nt": <int or null>,               # may be null if dt is provided or if steady-state
    "t_final": <float or null>,        # can be copied from pde_spec["time_interval"]["t_max"]
    "explicit_or_implicit": <string>,  # e.g., "explicit", "implicit", "imex", or null for steady-state
    "order": <int or null>,
    "extra_parameters": { ... }
  }
}

Guidelines:
- For time-dependent PDEs, include a time-stepping method; for steady-state PDEs, set time_stepping.method = null.
- Ensure grid sizes are consistent with pde_spec["dimension"].
- Include a mix of low/medium/high resolution plans and explicit/implicit options when appropriate.
- Keep plans realistic (avoid methods that obviously don't apply or violate well-posedness, stability condition, numerical sanity. take in consider of convergence and consistency).
- Make sure it outputs valid JSON that can be parsed by `json.loads()` without errors.
- You can change '' to "" if that helps with valid JSON.
- Make sure the generated plans has well-posedness, (e.g the numerical scheme must match the PDE type (hyperbolic, parabolic, elliptic, mixed), boundary conditions must be compatible with the PDE and discretization)
- Avoid generating plans that are obviously unstable or inconsistent (e.g., explicit method with very large dt for a diffusion problem, or cfl is appropriate given the domain and discretization).
- Make sure the plans respect to consistency and convergence principles (e.g., higher-order methods should have smaller dt or finer grid to be stable and achieve their accuracy potential).
- Make sure the plans didn't violate basic numerical sanity (e.g., time step must not exceed theoretical stability limits.)
"""


def generate_plans(pde_spec: dict, num_plans: int = 25, model: str = "claude-haiku-4-5-20251001") -> list[dict]:
    user_prompt = json.dumps({"pde_spec": pde_spec, "num_plans": num_plans}, indent=2)
    resp = call_llm(PLANNER_SYSTEM, user_prompt, model=model)
    return json.loads(_extract_json(resp))
