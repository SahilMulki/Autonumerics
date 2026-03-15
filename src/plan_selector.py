# plan_selector.py

import json
from typing import Any, Dict, List

from .llm_utils import call_llm

PLAN_SCORER_SYSTEM = r"""
You are an expert in numerical PDEs and scientific computing.

Context:
The scoring criteria below reflect standard, well-established numerical
analysis principles (stability, consistency, efficiency), not empirical tuning.

Task:
Given (1) a PDE specification pde_spec, (2) a list of candidate solver plans,
and optionally (3) LLM-generated problem/plan features,
assign a numeric score to EACH plan, reflecting your expectation of:
- stability/robustness for the PDE type and BC/IC,
- accuracy (order + resolution),
- efficiency (computational cost),
- overall suitability.

Return EXACTLY a JSON list of objects:
[
  {"plan_id": "...", "score": <number 0..100>, "rationale": "<1-3 sentences>"},
  ...
]

Rules:
- Higher score = better.
- If the PDE seems stiff or diffusion-dominated, prefer stable implicit/Crank–Nicolson-type approaches.
- If the PDE is advection-dominated, prefer upwind/WENO/TVD-type spatial schemes.
- Penalize extremely expensive plans unless accuracy gain is justified.
- You may use the provided feature_info if available.
- Do NOT output any extra text.
"""

PLAN_SELECTOR_SYSTEM = r"""
You are an expert in numerical PDEs and scientific computing.

Task:
Given
(1) a structured PDE problem description and
(2) a list of evaluated solver plans with error metrics and runtime,

select the most appropriate solver plan by balancing accuracy and efficiency.

Evaluation principles:
- If any plan that violates basic stability/accuracy principles (e.g., explicit method with very large dt for a diffusion problem), discard it regardless of other metrics.
- If any plan contains inappropriate boundary conditions, domain or discretization, for the PDE type, discard it.
- Rule out plans that are obviously inconsistent or violate numerical sanity (e.g., time step must not exceed theoretical stability limits).
- When an analytic solution is available, prioritize smaller relative L2 error.
- When no analytic solution is available, prioritize smaller residual errors
  (both absolute residual error and relative residual error).
- Prefer faster runtime when accuracy is comparable.
- A moderately slower method is acceptable if it is significantly more accurate.
- Compare only metrics that are present for a given plan.

Output JSON ONLY in the following format:

{
  "best_plan_id": "...",
  "reasoning": "...",
  "ranking": [
    {"plan_id": "...", "rank": 1},
    {"plan_id": "...", "rank": 2}
  ]
}

Rules:
- rank = 1 must correspond to best_plan_id.
- Do not hallucinate plans or metrics.
- Do not output any extra text.
"""


def score_plans_with_llm(
    pde_spec: Dict[str, Any],
    plans: List[Dict[str, Any]],
    feature_info: Dict[str, Any] | None = None,
    model: str = "gpt-4.1",
) -> List[Dict[str, Any]]:
    payload = {
        "pde_spec": pde_spec,
        "plans": plans,
        "feature_info": feature_info,
    }
    user_prompt = (
        "Here is the PDE specification, candidate plans, and optional features.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nPlease score every plan."
    )
    resp_str = call_llm(PLAN_SCORER_SYSTEM, user_prompt, model=model)
    return json.loads(resp_str)


def choose_best_plan_with_llm(
    pde_spec: Dict[str, Any],
    evaluated_plans: List[Dict[str, Any]],
    model: str = "gpt-4.1",
) -> Dict[str, Any]:
    payload = {
        "pde_spec": pde_spec,
        "evaluated_plans": evaluated_plans,
    }

    user_prompt = (
        "Here is the PDE specification and the list of evaluated solver plans.\n\n"
        + json.dumps(payload, indent=2)
        + "\n\nPlease choose the best plan according to the instructions."
    )

    resp_str = call_llm(PLAN_SELECTOR_SYSTEM, user_prompt, model=model)
    return json.loads(resp_str)
