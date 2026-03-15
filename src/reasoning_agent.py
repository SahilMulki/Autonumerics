import json

from .llm_utils import call_llm

REASONER_SYSTEM = r"""
You are a Professor of Numerical Analysis.

Your Task:
You are given the results of a numerical PDE experiment. A solver plan has been identified as the best-performing plan.
You must explain WHY this plan succeeded for THIS PDE, using the provided numerical metrics and PDE properties.

Metrics Semantics (IMPORTANT):
- Relative L2 Error: Measures accuracy against the analytic solution (when available).
- Relative Residual: Measures how well the numerical solution satisfies the PDE operator, normalized by solution magnitude.
- Absolute Residual: Measures raw PDE imbalance and reflects operator scaling and conditioning.

IMPORTANT CONSTRAINT:
The solver plan provided is the final selected plan.
You must ONLY explain the performance of this plan.
Do NOT compare against or reference other plans.
Do NOT suggest alternative schemes.

Guidelines:
- Be problem-specific: reference PDE type (elliptic/parabolic/hyperbolic), smoothness, stiffness, boundary conditions.
- Do NOT explain methods generically.
- If L2_error is available, this means the analytic solution is known; if only residual_l2 or relative residual error is available meaning analytic solution is unknown, prioritize explaining residuals.
- Explicitly link EACH reported metric to theory.
- If relative residual is small but relative L2 is large, explain this discrepancy.
- If analytic solution exists, prioritize relative L2; otherwise prioritize residuals.
- Discuss runtime only in relation to stability constraints and solver structure.

Tone:
Analytical, evidence-based, concise.
Max length: 200 words.

Output:
Plain text only.
"""


def analyze_theoretical_fit(
    pde_spec: dict,
    plan: dict,
    metrics: dict,
    model: str = "gpt-4.1",
) -> str:
    """
    Generates a PDE-specific theoretical explanation for the solver's performance
    using relative L2 error, relative residual, and absolute residual.
    """

    payload = {
        "pde_context": {
            "family": pde_spec.get("family"),
            "equation": pde_spec.get("governing_equation"),
            "dimension": pde_spec.get("dimension"),
            "time_dependent": pde_spec.get("time_dependent"),
            "linearity": "linear" if pde_spec.get("linear") else "nonlinear",
            "boundary_conditions": pde_spec.get("boundary_conditions", {}).get("type"),
            "analytic_solution": pde_spec.get("analytic_solution"),
        },
        "winning_plan": {
            "plan_id": plan.get("plan_id"),
            "spatial_scheme": plan.get("spatial_discretization", {}).get("scheme"),
            "time_integrator": plan.get("time_stepping", {}).get("method"),
            "resolution": {
                "Nx": plan.get("spatial_discretization", {}).get("Nx"),
                "Ny": plan.get("spatial_discretization", {}).get("Ny"),
            },
        },
        "empirical_metrics": {
            "relative_l2_error": metrics.get("relative_l2"),
            "relative_residual": metrics.get("relative_residual"),
            "absolute_residual": metrics.get("absolute_residual"),
            "runtime_seconds": metrics.get("runtime_s"),
        },
    }

    user_prompt = (
        "Below are the PDE description, solver configuration, and observed numerical metrics.\n"
        "Explain, using numerical analysis theory, why this solver performed well for this PDE.\n\n"
        f"{json.dumps(payload, indent=2)}"
    )

    return call_llm(REASONER_SYSTEM, user_prompt, model=model)
