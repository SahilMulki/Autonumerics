# critic_agent.py
import json

from .llm_utils import call_llm

DEBUG_CRITIC_SYSTEM = r"""
You are a Python debugging assistant specialized in numerical PDE solvers.

You will be given:
- A Python solver implementation,
- An error message / traceback from running it,
- The PDE specification pde_spec and solver plan plan used for the run.

Your goal:
- Identify the root cause.
- Return a corrected version of the ENTIRE solver code.

Hard requirements:
1. The solver MUST define:
      def solve_pde(pde_spec: dict, plan: dict) -> dict
2. The return dictionary MUST contain:
      "u": <ndarray> (MUST be an array, not a scalar)
      "coords": <dict of arrays>
      "t": <array or None>
3. Keep using NumPy only (import numpy as np).
4. Do NOT add print statements.
5. Do NOT wrap code in markdown fences.
6. Output PURE Python code ONLY.
"""


def debug_code(
    code: str, error_message: str, pde_spec: dict, plan: dict, model: str = "claude-sonnet-4-6"
) -> str:
    payload = {
        "error_message": error_message,
        "pde_spec": pde_spec,
        "plan": plan,
        "code": code,
    }
    user_prompt = (
        "The following solver code failed when executed.\n"
        "Please fix it according to the requirements.\n\n" + json.dumps(payload, indent=2)
    )
    return call_llm(DEBUG_CRITIC_SYSTEM, user_prompt, model=model)
