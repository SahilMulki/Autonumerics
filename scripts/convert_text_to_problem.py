import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_utils import call_llm

SYSTEM_PROMPT = """
You convert PDE benchmark descriptions into a STRICT standardized JSON object.

Return ONLY valid JSON. No markdown, no explanations, no code fences.

Target schema:

{
  "id": string,
  "family": string,
  "dimension": int,
  "order": int,
  "time_dependent": bool,
  "linear": bool,
  "stiff": bool,
  "parameters": object,
  "domain": object,
  "t_span": [t0, t1],
  "boundary_conditions": {"type": string, ...},
  "analytic_solution": null OR {
      "expression": string,
      "space_variables": [string]
  } OR {
      "fields": {name: string OR [string]},
      "space_variables": [string]
  },
  "description": string
}

Rules:

- Use snake_case for "id".
- Use stable "family" names like "navier_stokes", "heat", "burgers".
- Use floats for numeric parameters.
- Domain bounds may be numbers or strings like "2*pi".
- Use axes among: x, y, z.
- Periodic -> {"type": "periodic"}.
- Dirichlet from exact -> {"type": "dirichlet", "value": "from_analytic_solution"}.
- Multiple exact fields -> use "fields".
- No analytic solution -> analytic_solution = null.
- Keep description text essentially unchanged (only trim whitespace).

Return JSON ONLY.
""".strip()

REQUIRED_KEYS = {
    "id",
    "family",
    "dimension",
    "order",
    "time_dependent",
    "linear",
    "stiff",
    "parameters",
    "domain",
    "t_span",
    "boundary_conditions",
    "analytic_solution",
    "description",
}

DEFAULT_DESCRIPTION = """
Solve the 2D incompressible Navier-Stokes equations on [0,2pi]x[0,2pi], t in [0,1], with periodic BCs.
Take viscosity nu=0.01 and use the Taylor-Green vortex exact solution:
u=sin(x)cos(y)exp(-2 nu t),  v=-cos(x)sin(y)exp(-2 nu t),
p=-(1/4)(cos(2x)+cos(2y))exp(-4 nu t).
""".strip()


def validate_problem(problem: Dict[str, Any]) -> None:
    missing = REQUIRED_KEYS - set(problem.keys())
    if missing:
        raise ValueError(f"Missing required keys: {sorted(missing)}")

    if not isinstance(problem["dimension"], int) or problem["dimension"] < 1:
        raise ValueError("dimension must be a positive integer")

    if not isinstance(problem["order"], int) or problem["order"] < 1:
        raise ValueError("order must be a positive integer")

    t_span = problem["t_span"]
    if not (isinstance(t_span, list) and len(t_span) == 2):
        raise ValueError("t_span must be a 2-element list [t0, t1]")


def generate_problem_from_description(
    description: str,
    model: str = "claude-sonnet-4-6",
) -> Dict[str, Any]:
    raw = call_llm(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=description,
        model=model,
    )

    try:
        problem = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM did not return valid JSON.\nRaw output:\n{raw}") from e

    validate_problem(problem)
    return problem


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a PDE description into a normalized JSON spec."
    )
    parser.add_argument("--description", help="Inline PDE description text.")
    parser.add_argument(
        "--description-file", help="Path to a text file containing the PDE description."
    )
    parser.add_argument("--model", default="claude-sonnet-4-6", help="LLM model to use.")
    return parser.parse_args(argv)


def _read_description(args: argparse.Namespace) -> str:
    if args.description:
        return args.description.strip()
    if args.description_file:
        return Path(args.description_file).read_text(encoding="utf-8").strip()
    return DEFAULT_DESCRIPTION


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    description = _read_description(args)
    problem = generate_problem_from_description(description, model=args.model)
    print(json.dumps(problem, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
