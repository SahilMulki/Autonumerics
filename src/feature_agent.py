# feature_agent.py

from __future__ import annotations

import json
import re
from typing import Any

from .llm_utils import call_llm

FEATURE_SYSTEM = """
You are an expert in numerical PDEs and machine learning for algorithm selection.

Your task: Given a PDE specification and a list of solver plans, you must:

1. Propose a set of numeric features that are useful for:
   - characterizing the PDE problem (problem-level features),
   - characterizing each solver plan (plan-level features),
   - estimating numerical stability, accuracy, and cost.

2. Compute the feature values for:
   - the single PDE problem (problem_features),
   - each solver plan (plan_features[plan_id]).

RULES (STRICT):
- Output MUST be valid JSON. Output ONLY the JSON object and nothing else.
- Do NOT use Markdown. Do NOT add explanations outside JSON.
- Use double quotes for all strings.
- Do NOT use trailing commas.
- All "value" fields must be numeric JSON numbers (no NaN/Infinity).
- Keep "description" strings short and do not include newlines.

Output STRICT JSON with the following structure:

{
  "problem_features": [
    {
      "name": <string>,
      "value": <float>,
      "description": <string>
    }
  ],
  "plan_features": [
    {
      "plan_id": <plan_id>,
      "features": [
        {
          "name": <string>,
          "value": <float>,
          "description": <string>
        }
      ]
    }
  ]
}
"""


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _extract_first_json_obj(text: str) -> str:
    """
    Extract the first top-level JSON object/array from possibly noisy text.
    """
    text = text.strip()
    if (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    ):
        return text

    start = None
    for i, ch in enumerate(text):
        if ch in "{[":
            start = i
            opener = ch
            closer = "}" if ch == "{" else "]"
            break
    if start is None:
        raise ValueError("No JSON start token found in LLM response.")

    depth = 0
    in_str = False
    escape = False
    for j in range(start, len(text)):
        c = text[j]
        if in_str:
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_str = False
            continue
        else:
            if c == '"':
                in_str = True
                continue
            if c == opener:
                depth += 1
            elif c == closer:
                depth -= 1
                if depth == 0:
                    return text[start : j + 1]

    raise ValueError("Could not extract a balanced JSON object/array from LLM response.")


def _minimal_json_repairs(s: str) -> str:
    """
    Fix a few common LLM JSON issues without being too destructive.
    """
    s = s.strip()
    s = s.replace("“", '"').replace("”", '"').replace("’", "'")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    s = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", s)
    return s


def _safe_json_loads(text: str) -> Any:
    text = _strip_code_fences(text)
    text = _extract_first_json_obj(text)
    text = _minimal_json_repairs(text)
    return json.loads(text)


def _sanitize_feature_payload(payload: dict) -> dict:
    """
    Ensure the output matches expected structure and is JSON-serializable.
    Also normalizes numeric values to float.
    """
    if not isinstance(payload, dict):
        raise ValueError("Parsed JSON is not a dict.")

    if "problem_features" not in payload or "plan_features" not in payload:
        raise ValueError("Missing required keys problem_features / plan_features.")

    def fix_feature_list(lst):
        if not isinstance(lst, list):
            raise ValueError("Feature list must be a list.")
        out = []
        for item in lst:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            desc = str(item.get("description", "")).replace("\n", " ").strip()
            val = item.get("value", 0.0)
            try:
                val = float(val)
            except Exception:
                val = 0.0
            out.append({"name": name, "value": val, "description": desc})
        return out

    payload["problem_features"] = fix_feature_list(payload["problem_features"])

    pf = payload["plan_features"]
    if not isinstance(pf, list):
        raise ValueError("plan_features must be a list.")

    cleaned_pf = []
    for entry in pf:
        if not isinstance(entry, dict):
            continue
        plan_id = entry.get("plan_id", None)
        if plan_id is None:
            plan_id = entry.get("id", entry.get("plan", None))
        features = fix_feature_list(entry.get("features", []))
        cleaned_pf.append({"plan_id": plan_id, "features": features})

    payload["plan_features"] = cleaned_pf
    return payload


def llm_extract_features(
    pde_spec: dict, plans: list[dict], *, model: str = "gpt-4.1", max_tries: int = 3
) -> dict:
    """
    Calls the LLM to generate problem_features + plan_features and returns a dict.
    Robust to minor JSON formatting issues and includes auto-repair retries.
    """
    user_prompt = json.dumps(
        {"pde_spec": pde_spec, "plans": plans},
        indent=2,
        ensure_ascii=False,
    )

    resp = call_llm(FEATURE_SYSTEM, user_prompt, model=model)

    for attempt in range(1, max_tries + 1):
        try:
            payload = _safe_json_loads(resp)
            payload = _sanitize_feature_payload(payload)
            return payload
        except Exception as e:
            if attempt == max_tries:
                print("\n[feature_agent] Failed to parse LLM JSON after retries.")
                print("Last error:", repr(e))
                print("\n==== RAW LLM RESPONSE (BEGIN) ====")
                print(resp)
                print("==== RAW LLM RESPONSE (END) ====\n")
                raise

            repair_prompt = (
                "Your previous output was INVALID JSON.\n\n"
                f"Error: {repr(e)}\n\n"
                "Return corrected VALID JSON ONLY, matching the required schema exactly.\n"
                "Do not add any extra keys. Do not add any text outside JSON.\n\n"
                "Here is your invalid output:\n"
                f"{resp}"
            )
            resp = call_llm(FEATURE_SYSTEM, repair_prompt, model=model)

    raise RuntimeError("Unexpected control flow in llm_extract_features.")
