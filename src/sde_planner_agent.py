# sde_planner_agent.py
"""SDE-specific planner: generates Monte Carlo solver plans for Itô SDEs.

Phase 1 capability: Euler-Maruyama (EM) and Milstein schemes only.

Key design decisions vs the PDE planner:
  - Plan schema is completely different: no spatial grid, no implicit/explicit
    distinction — instead we have dt, Nt, num_paths, numerical_scheme.
  - Milstein is only proposed when (a) noise_structure == "multiplicative" AND
    (b) diffusion_derivative is non-null in the SDE spec. When noise is additive,
    Milstein's correction term is zero so it reduces to EM — proposing it would
    just waste diversity budget.
  - convergence_target distinguishes pathwise accuracy studies ("strong") from
    moment/distribution comparisons ("weak"). Phase 1 primarily targets weak
    convergence since we compare E[X(T)] and Var[X(T)] to exact moments.
  - Post-processing enforces invariants that the LLM might violate: Nt is always
    recomputed from dt (not trusted from LLM output), Milstein is blocked for
    additive noise, and num_paths is floored at a safe minimum.
"""

from __future__ import annotations

import json
import re

from .llm_utils import call_llm


# =============================================================================
# JSON extraction helper (same logic as planner_agent._extract_json)
# =============================================================================

def _extract_json(text: str) -> str:
    """Strip markdown code fences then extract the first balanced JSON object/array.

    The LLM sometimes wraps its output in ```json ... ``` fences or adds
    explanation text before/after the JSON. This handles both cases.
    """
    text = text.strip()
    # Remove opening fence (e.g., ```json or ```python)
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    # Remove closing fence
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Find the first { or [ — that is where the JSON starts
    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener, closer = ch, ("}" if ch == "{" else "]")
            break
    if start is None:
        raise ValueError(f"No JSON found in LLM response: {text[:200]!r}")

    # Walk forward tracking string context and brace depth to find the matching closer
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


# =============================================================================
# System prompt
# =============================================================================

SDE_PLANNER_SYSTEM = r"""
You are an expert in numerical methods for stochastic differential equations (SDEs).

Given an SDE specification `sde_spec` and a requested number of plans `num_plans`,
propose a diverse set of Monte Carlo solver plans. You are restricted to
Euler-Maruyama (EM) and Milstein schemes (Phase 1 capability).

IMPORTANT: You MUST output a JSON LIST of exactly `num_plans` plan objects.
Each plan object MUST follow this schema (all keys required):

{
  "plan_id": <string>,            — unique identifier, e.g. "em_dt0.01_n5000"
  "description": <string>,        — short human-readable description
  "numerical_scheme": <string>,   — "euler_maruyama" or "milstein" (see rules below)
  "dt": <float>,                  — time step size; must be > 0 and < t_final
  "Nt": <int>,                    — number of time steps = round(t_final / dt)
  "num_paths": <int>,             — number of Monte Carlo sample paths
  "convergence_target": <string>, — "strong" (pathwise) or "weak" (moments/distribution)
  "seed": <int or null>,          — RNG seed for reproducibility; null means no fixed seed
  "strong_order": <float>,        — theoretical strong convergence order of the scheme
  "weak_order": <float>,          — theoretical weak convergence order of the scheme
  "extra_parameters": {}          — optional free-form dict; leave empty if not needed
}

SCHEME SELECTION RULES (STRICT — violating these produces incorrect code):
1. Euler-Maruyama (EM):
   - Always allowed for any SDE (scalar or multi-dimensional).
   - Strong order: 0.5, Weak order: 1.0.
   - Set strong_order=0.5 and weak_order=1.0.

2. Milstein:
   - ONLY propose Milstein when ALL THREE conditions hold:
       (a) sde_spec["state_dimension"] == 1  (scalar SDEs only)
       (b) sde_spec["noise_structure"] == "multiplicative"
       (c) sde_spec["diffusion_derivative"] is non-null
   - For state_dimension > 1: Milstein requires the Lévy area correction term
     which is not implemented — always use Euler-Maruyama for multi-D SDEs.
   - When noise_structure == "additive", dg/dX = 0, so the Milstein correction
     term vanishes — Milstein reduces to EM. Do NOT propose it in this case.
   - When diffusion_derivative is null, the correction cannot be computed.
   - Strong order: 1.0, Weak order: 1.0.
   - Set strong_order=1.0 and weak_order=1.0.

PARAMETER GUIDELINES:
- dt must satisfy 0 < dt < t_final where t_final = sde_spec["time_interval"]["t_final"].
- Nt must equal round(t_final / dt).  Compute it exactly — the coder will recompute
  it anyway, but keeping it consistent avoids confusion.
- Cover at least two decades in dt (e.g., include 0.01 and 0.001) to explore
  time-step accuracy.
- Cover at least two orders of magnitude in num_paths (e.g., 1000 and 100000) to
  explore the variance-cost tradeoff in the Monte Carlo estimator.
- For weak convergence (moment comparison): larger dt may still be acceptable if
  num_paths is large — the dominant error is Monte Carlo variance, not time-step bias.
- For strong convergence (pathwise): use smaller dt — pathwise accuracy requires
  resolving individual trajectories precisely.
- Include at least one plan with num_paths >= 50000 to serve as a reliable baseline
  with low Monte Carlo variance.
- Include at least one plan with dt <= 0.001 for high time-step accuracy.
- Avoid dt > 0.1 when t_final <= 1.0 — too coarse to be meaningful.

DIVERSITY REQUIREMENTS:
- Cover the accuracy-cost tradeoff spectrum: cheap plans (large dt, few paths)
  through expensive plans (small dt, many paths).
- If Milstein is allowed: include at least 2 Milstein plans and at least 2 EM plans,
  with different (dt, num_paths) combinations for each scheme.
- If Milstein is NOT allowed (additive noise): use only EM, but vary (dt, num_paths)
  to create num_plans distinct plans.
- Mix convergence_target values: some plans labeled "weak" (moment accuracy),
  some labeled "strong" (pathwise accuracy).

PLAN_ID FORMAT (recommended): "<scheme>_dt<dt>_n<num_paths>"
Example: "em_dt0.01_n10000" or "milstein_dt0.001_n50000"

OUTPUT RULES:
- Output MUST be valid JSON. Output ONLY the JSON array, nothing else.
- Do NOT use Markdown or code fences.
- Use double quotes for all strings.
- Do NOT use trailing commas.
- All numeric fields must be valid JSON numbers (no NaN, no Infinity).
"""


# =============================================================================
# Post-processing: enforce invariants the LLM might violate
# =============================================================================

# Theoretical orders for each supported scheme — used to fix or fill in orders
# when the LLM gets them wrong or omits them.
_SCHEME_ORDERS: dict[str, tuple[float, float]] = {
    "euler_maruyama": (0.5, 1.0),   # (strong_order, weak_order)
    "milstein":       (1.0, 1.0),
}


def _post_process_sde_plans(plans: list[dict], sde_spec: dict) -> list[dict]:
    """Sanitize and correct LLM-generated SDE plans.

    Enforces:
    - Valid scheme names (unknown → euler_maruyama)
    - Milstein blocked when noise is not multiplicative or diffusion_derivative is null
    - dt > 0 and dt < t_final; Nt recomputed from dt (not trusted from LLM)
    - num_paths >= 100 (safety floor)
    - convergence_target in {"strong", "weak"}
    - seed is int or None
    - Correct strong_order and weak_order for the (possibly corrected) scheme
    - Unique plan_ids (suffix _2, _3 if duplicate)
    - extra_parameters defaults to {}
    """
    t_final = float(
        sde_spec.get("time_interval", {}).get("t_final", 1.0)
    )

    # Milstein eligibility: all three conditions must hold.
    # The Lévy area correction required for multi-D Milstein is not implemented,
    # so multi-D problems are restricted to Euler-Maruyama.
    state_dim = sde_spec.get("state_dimension", 1)
    noise_is_multiplicative = sde_spec.get("noise_structure", "additive") == "multiplicative"
    has_diffusion_derivative = sde_spec.get("diffusion_derivative") not in (None, "null", "")
    milstein_allowed = (state_dim == 1) and noise_is_multiplicative and has_diffusion_derivative

    seen_ids: dict[str, int] = {}  # track plan_id occurrences to deduplicate
    cleaned: list[dict] = []

    for i, plan in enumerate(plans):
        if not isinstance(plan, dict):
            continue

        # --- scheme ---
        scheme = str(plan.get("numerical_scheme", "euler_maruyama")).lower().strip()
        if scheme not in _SCHEME_ORDERS:
            # Unknown scheme: fall back to EM
            scheme = "euler_maruyama"
        if scheme == "milstein" and not milstein_allowed:
            # LLM ignored the additive-noise rule — silently downgrade to EM
            scheme = "euler_maruyama"

        # --- dt and Nt ---
        raw_dt = plan.get("dt", 0.01)
        try:
            dt = float(raw_dt)
        except (TypeError, ValueError):
            dt = 0.01
        # dt must be strictly positive and strictly less than t_final
        dt = max(dt, 1e-6)
        dt = min(dt, t_final * 0.999)
        # Recompute Nt to match dt exactly — the LLM rounds differently sometimes
        Nt = max(1, round(t_final / dt))

        # --- num_paths ---
        raw_paths = plan.get("num_paths", 1000)
        try:
            num_paths = int(raw_paths)
        except (TypeError, ValueError):
            num_paths = 1000
        num_paths = max(100, num_paths)  # safety floor

        # --- convergence_target ---
        conv = str(plan.get("convergence_target", "weak")).lower().strip()
        if conv not in ("strong", "weak"):
            conv = "weak"

        # --- seed ---
        raw_seed = plan.get("seed", None)
        seed: int | None
        if raw_seed is None or raw_seed == "null":
            seed = None
        else:
            try:
                seed = int(raw_seed)
            except (TypeError, ValueError):
                seed = None

        # --- orders: always set from scheme, not from LLM (LLM may be wrong) ---
        strong_order, weak_order = _SCHEME_ORDERS[scheme]

        # --- description ---
        desc = str(plan.get("description", f"Plan {i+1}: {scheme}, dt={dt}, n={num_paths}"))

        # --- plan_id: deduplicate by appending _2, _3, ... ---
        raw_id = str(plan.get("plan_id", f"{scheme}_dt{dt}_n{num_paths}"))
        if raw_id in seen_ids:
            seen_ids[raw_id] += 1
            plan_id = f"{raw_id}_{seen_ids[raw_id]}"
        else:
            seen_ids[raw_id] = 1
            plan_id = raw_id

        # --- extra_parameters ---
        extra = plan.get("extra_parameters", {})
        if not isinstance(extra, dict):
            extra = {}

        cleaned.append({
            "plan_id":            plan_id,
            "description":        desc,
            "numerical_scheme":   scheme,
            "dt":                 dt,
            "Nt":                 Nt,
            "num_paths":          num_paths,
            "convergence_target": conv,
            "seed":               seed,
            "strong_order":       strong_order,
            "weak_order":         weak_order,
            "extra_parameters":   extra,
        })

    return cleaned


# =============================================================================
# Public API
# =============================================================================

def generate_sde_plans(
    sde_spec: dict,
    num_plans: int = 7,
    model: str = "claude-haiku-4-5-20251001",
) -> list[dict]:
    """Generate Monte Carlo solver plans for the given SDE specification.

    Mirrors the interface of planner_agent.generate_plans() but returns
    SDE-specific plan dicts (no spatial grid; has num_paths, dt, scheme).

    Args:
        sde_spec:  The SDE specification dict from formulate_problem().
                   Must contain at least: time_interval.t_final,
                   noise_structure, diffusion_derivative.
        num_plans: Number of plans to generate (default 7, fewer than PDE's 25
                   because each SDE plan is expensive at runtime — a plan with
                   100k paths × 1000 steps touches 10^8 RNG draws).
        model:     LLM model to use for plan generation.

    Returns:
        List of plan dicts with keys:
            plan_id, description, numerical_scheme, dt, Nt, num_paths,
            convergence_target, seed, strong_order, weak_order, extra_parameters.
    """
    user_prompt = json.dumps({"sde_spec": sde_spec, "num_plans": num_plans}, indent=2)
    resp = call_llm(SDE_PLANNER_SYSTEM, user_prompt, model=model)

    raw = _extract_json(resp)
    plans = json.loads(raw)

    # Enforce invariants: scheme eligibility, dt/Nt consistency, num_paths floor, etc.
    return _post_process_sde_plans(plans, sde_spec)
