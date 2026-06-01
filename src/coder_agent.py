# coder_agent.py (MEMORY-SAFE VERSION)  -- UPDATED (minimal changes)
import json

from .llm_utils import call_llm

CODER_SYSTEM = r"""
You are a Python numerical PDE expert. Write clean, correct NumPy-only code.

You will be given:
1) A JSON PDE specification `pde_spec`.
2) A JSON solver plan `plan`.

You MUST implement exactly one entry point:

    def solve_pde(pde_spec: dict, plan: dict) -> dict:

Return format (STRICT):
{
  "u": u,                           # ndarray solution (see memory rules below)
  "coords": {"x": x, ...},          # dictionary of 1D coordinate arrays
  "t": t_array,                     # 1D array of time steps
  "residual": residual_grid         # REQUIRED: ndarray of pointwise PDE residual (same shape as u)
}

CRITICAL RULES:
1. **Dynamic Parameters**: Read Nx, dt, etc. from `plan`.
2. **Compute Residual (REQUIRED)**:
   - After computing the final solution `u`, compute the **pointwise PDE residual grid** by plugging `u` back into the governing equation.
   - Return the **residual grid** (array). Do NOT return a scalar residual or precomputed norm.
   - The evaluator will compute L2 / relative norms consistently across plans.
3. **Robustness**: If plan doesn't specify dt, use CFL to estimate it.

**MEMORY SAFETY RULES (PREVENT SIGKILL):**
- **Do NOT store the full time history** if Nt > 1000 or the grid is large (>128^2).
- Storing `u` for every step will crash the machine (OOM).
- Instead, **return only the final state** `u_final`, OR decimate the output (e.g., store every 100th step).
- If the plan implies a matrix inversion (implicit method), use `np.linalg.solve` on sparse structures or iterative approaches if possible; avoid building dense `(Nx*Ny, Nx*Ny)` matrices.

"""


def generate_solver_code(pde_spec: dict, plan: dict, model: str = "claude-sonnet-4-6") -> str:
    user_prompt = (
        "Here is the PDE spec:\n"
        f"{json.dumps(pde_spec, indent=2)}\n\n"
        "Here is the Solver Plan:\n"
        f"{json.dumps(plan, indent=2)}\n\n"
        "Generate the `solve_pde` function.\n"
        "IMPORTANT: You must implement a post-processing step to calculate `residual` as a pointwise residual grid (ndarray), not a scalar.\n"
        "WARNING: Implement memory-saving logic (do not store every time step) to avoid Killed: 9 errors."
    )
    return call_llm(CODER_SYSTEM, user_prompt, model=model)
