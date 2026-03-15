import argparse
import datetime
import json
import os
import random
import re
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analytic_utils import analytic_solution, implicit_solution_residual
from src.coder_agent import generate_solver_code
from src.critic_agent import debug_code
from src.feature_agent import llm_extract_features
from src.formulator import formulate_problem
from src.plan_selector import choose_best_plan_with_llm, score_plans_with_llm
from src.planner_agent import generate_plans
from src.problem_lists_0125 import get_problem_by_id
from src.reasoning_agent import analyze_theoretical_fit

VERBOSE = bool(int(os.getenv("RUNNER_VERBOSE", "0")))
DEFAULT_PROBLEM_ID = "reaction_diffusion_2d_linear_dirichlet"


def _save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _save_text(text: str, path: str):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _sanitize(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in str(s))


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _make_coarse_plan(plan: dict, scale_factor: int = 4) -> dict:
    """Creates a low-res version of the plan for stability/syntax checking."""
    coarse = json.loads(json.dumps(plan))
    spatial = coarse.get("spatial_discretization", {})

    for key in ["Nx", "Ny", "Nz"]:
        if spatial.get(key):
            try:
                val = int(spatial[key])
                spatial[key] = max(16, val // scale_factor)
            except (ValueError, TypeError):
                pass

    coarse["description"] = f"[COARSE PRE-CHECK] {coarse.get('description', '')}"
    return coarse


def _coords_from_result(result: dict):
    coords = result.get("coords", {}) or {}
    out = {}
    for k in ["x", "y", "z", "x1", "x2", "x3"]:
        if k in coords and coords[k] is not None:
            out[k] = np.asarray(coords[k])
    return out


def _infer_cell_volume(coords: dict, verbose: bool = False) -> float:
    del verbose

    spacings = []
    for _k, arr in (coords or {}).items():
        a = np.asarray(arr)
        if a.ndim == 1 and a.size > 1:
            h = float(np.mean(np.diff(a)))
            if np.isfinite(h) and h > 0:
                spacings.append(h)
    if not spacings:
        return 1.0
    return float(np.prod(spacings))


def compute_residual_l2(pde_spec: dict, result: dict) -> tuple[dict, str]:
    """
    Returns residual metrics:
      metrics["residual"]          = absolute L2 residual ||r||_{L2(Ω)}
      metrics["relative_residual"] = ||r|| / (||u|| + eps)

    - If solver returns scalar residual, treat it as relative_residual and set residual=None.
    - If solver returns residual field array, compute both.
    - Works for real/complex (uses |.|^2).
    """
    del pde_spec

    if "residual" not in result or result["residual"] is None:
        return {"residual": None, "relative_residual": None}, "solver did not return 'residual'"

    try:
        res_arr = np.asarray(result["residual"])

        if res_arr.ndim == 0:
            return {
                "residual": None,
                "relative_residual": float(res_arr),
            }, "solver returned scalar residual"

        r = res_arr
        coords_raw = _coords_from_result(result)
        dv = _infer_cell_volume(coords_raw, verbose=VERBOSE)

        abs_res = float(np.sqrt(np.sum(np.abs(r.ravel()) ** 2) * dv))

        u = np.asarray(result.get("u", 0.0))
        u_l2 = float(np.sqrt(np.sum(np.abs(u.ravel()) ** 2) * dv)) + 1e-12

        rel_res = abs_res / u_l2
        return {
            "residual": abs_res,
            "relative_residual": float(rel_res),
        }, "computed from residual array"

    except Exception as e:
        return {"residual": None, "relative_residual": None}, f"error parsing solver residual: {e}"


class SolverTimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    del signum, frame
    raise SolverTimeoutError("Solver execution timed out (exceeded 120 seconds).")


def run_generated_solver(code: str, pde_spec: dict, plan: dict) -> dict:
    """Execute generated code and strictly validate its output structure and stability."""
    clean_code = _strip_code_fences(code)
    env: Dict[str, Any] = {}

    try:
        exec(clean_code, env, env)
    except SyntaxError as e:
        raise SyntaxError(f"Generated code has syntax error: {e}") from e

    if "solve_pde" in env and callable(env["solve_pde"]):
        try:
            result = env["solve_pde"](pde_spec, plan)
        except Exception as e:
            raise RuntimeError(f"Solver execution failed with runtime error: {e}") from e

        if not isinstance(result, dict):
            raise TypeError(f"Solver returned {type(result)}, expected dict.")
        if "u" not in result:
            raise KeyError("Solver result is missing the required 'u' key.")

        u_val = np.asarray(result["u"])

        if u_val.ndim == 0:
            raise ValueError("Solver returned a scalar value for 'u'. Must be an array.")

        coords = result.get("coords", {})
        if coords:
            spatial_dims = len([k for k in ["x", "y", "z", "x1", "x2", "x3"] if k in coords])
            if u_val.ndim < spatial_dims:
                raise ValueError(
                    f"Solver returned 'u' with shape {u_val.shape}, fewer dimensions "
                    f"than the {spatial_dims} spatial coordinates provided."
                )

        if not np.all(np.isfinite(u_val)):
            raise ValueError(
                "Numerical Instability Detected: Solution contains NaNs or Infinities."
            )

        if "residual" not in result or result["residual"] is None:
            raise ValueError(
                "Solver result is missing 'residual' (L2 error check). You MUST compute and return it."
            )

        return result

    raise ValueError("Generated code did not define solve_pde(pde_spec, plan).")


def run_with_critic(
    initial_code: str,
    pde_spec: dict,
    plan: dict,
    max_retries: int = 10,
    code_log_dir: Optional[str] = None,
    phase_name: str = "Dense",
) -> Tuple[dict, str, int]:
    code = initial_code
    last_err = None

    if code_log_dir is not None:
        os.makedirs(code_log_dir, exist_ok=True)
        _save_text(code, os.path.join(code_log_dir, f"code_{phase_name}_start.py"))

    signal.signal(signal.SIGALRM, _timeout_handler)

    for attempt in range(max_retries + 1):
        print(f"[{phase_name}] attempt {attempt}/{max_retries} (Time Limit: 120s)", flush=True)

        try:
            signal.alarm(120)
            result = run_generated_solver(code, pde_spec, plan)
            signal.alarm(0)

            if code_log_dir is not None:
                _save_text(code, os.path.join(code_log_dir, f"code_{phase_name}_final.py"))
            return result, code, attempt + 1

        except Exception as e:
            signal.alarm(0)
            last_err = traceback.format_exc()

            if isinstance(e, ValueError) and "scalar" in str(e):
                last_err = f"Output Error: {str(e)}\nPlease reshape 'u' to be an array."
            elif isinstance(e, ValueError) and "Numerical Instability" in str(e):
                last_err = f"Stability Error: {str(e)}\nHint: Reduce time step (dt) or check CFL."
            elif isinstance(e, ValueError) and "residual" in str(e):
                last_err = f"Compliance Error: {str(e)}\nPlease implement the residual calculation."
            elif isinstance(e, SolverTimeoutError):
                last_err = "TimeoutError: Execution exceeded 120s. Optimize loops."

            if attempt >= max_retries:
                break

            raw_correction = debug_code(code, last_err, pde_spec, plan)
            code = _strip_code_fences(raw_correction)

            if code_log_dir is not None:
                _save_text(
                    code, os.path.join(code_log_dir, f"code_{phase_name}_fix_{attempt + 1}.py")
                )

    raise RuntimeError(
        f"Solver failed after {max_retries + 1} attempts in {phase_name}.\nFailure: {last_err}"
    )


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Auto-demo single-problem pipeline.")
    parser.add_argument(
        "--problem-id",
        default=DEFAULT_PROBLEM_ID,
        help="Problem identifier from problem_lists_0125.py",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Global random seed for the run.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None):
    args = parse_args(argv)

    global_seed = args.seed
    random.seed(global_seed)
    np.random.seed(global_seed)

    problem_id = args.problem_id
    try:
        problem = get_problem_by_id(problem_id)
        problem_description = problem.get("description")
    except Exception as e:
        print(f"Error loading problem ID '{problem_id}': {e}")
        return

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    problem_id_safe = _sanitize(problem_id)
    out_dir = PROJECT_ROOT / "outputs_new_paper" / f"run_{problem_id_safe}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_root = out_dir / "code"
    code_root.mkdir(parents=True, exist_ok=True)

    _save_json({"problem_id": problem_id, "seed": global_seed}, str(out_dir / "run_metadata.json"))

    print("=== Step 1: Formulator ===")
    pde_spec = formulate_problem(problem_description, seed=global_seed)
    pde_spec["id"] = problem.get("id", problem_id)
    pde_spec["family"] = problem.get("family", None)
    pde_spec["dimension"] = problem.get("dimension", pde_spec.get("spatial_dimension"))
    pde_spec["time_dependent"] = problem.get("time_dependent", pde_spec.get("time_dependent"))

    if "analytic_solution" in pde_spec and pde_spec["analytic_solution"]:
        sv = pde_spec["analytic_solution"].get("space_variables", [])
        if "t" in sv:
            print("[Runner] Removed 't' from spatial_variables to prevent KeyError.")
            pde_spec["analytic_solution"]["space_variables"] = [v for v in sv if v != "t"]
    print(json.dumps(pde_spec, indent=2))
    _save_json(pde_spec, str(out_dir / "pde_spec.json"))

    print("\n=== Step 2: Planner ===")
    plans = generate_plans(pde_spec, num_plans=10)
    print(f"Generated {len(plans)} plans.")

    print("\n=== Step 3: LLM feature extraction ===")
    try:
        feature_info = llm_extract_features(pde_spec, plans, max_tries=3)
    except Exception:
        print("[runner] Feature extraction failed; continuing without LLM features.")
        feature_info = {"problem_features": [], "plan_features": []}

    print("\n=== Step 4: LLM plan scoring ===")
    scored = score_plans_with_llm(pde_spec, plans, feature_info=feature_info)
    score_map = {d["plan_id"]: d for d in scored if isinstance(d, dict) and "plan_id" in d}

    for pl in plans:
        sc = score_map.get(pl.get("plan_id"), {})
        pl["_llm_score"] = sc.get("score", None)
        pl["_llm_score_rationale"] = sc.get("rationale", "")

    plans_sorted = sorted(
        plans,
        key=lambda p: (p.get("_llm_score") is not None, p.get("_llm_score", -1)),
        reverse=True,
    )
    _save_json(scored, str(out_dir / "plans_scored.json"))

    top_k = min(5, len(plans_sorted))
    selected_plans = plans_sorted[:top_k]
    print(f"Selected top-{top_k} plans by LLM score:")
    for p in selected_plans:
        print(f"  {p.get('plan_id')}: score={p.get('_llm_score')}  {p.get('description', '')[:80]}")

    print("\n=== Step 5: Coarse-to-Fine CodeGen (With Fresh Restarts) ===")
    evaluated_plans = []

    for plan in selected_plans:
        plan_id = plan.get("plan_id")
        plan_code_dir = code_root / f"plan_{_sanitize(plan_id)}"
        print("\n" + "=" * 60)
        print(f"Evaluating plan_id={plan_id}")
        print("=" * 60)

        max_restarts = 2
        run_data = None
        last_error = None

        for restart_idx in range(max_restarts):
            if restart_idx > 0:
                print(
                    f"\n[Restart Triggered] Previous attempt failed. Generating FRESH code "
                    f"(Attempt {restart_idx + 1}/{max_restarts})..."
                )

            t0 = time.time()
            try:
                code = generate_solver_code(pde_spec, plan)

                coarse_plan = _make_coarse_plan(plan, scale_factor=4)
                print("  >> Phase 1: Running Coarse Grid (1/4 res)...")
                _, fixed_code, c_atts = run_with_critic(
                    code,
                    pde_spec,
                    coarse_plan,
                    max_retries=3,
                    code_log_dir=str(plan_code_dir),
                    phase_name=f"Coarse_R{restart_idx}",
                )
                print("  >> Coarse Validated. Promoting code to Dense run.")

                print("  >> Phase 2: Running Target Dense Grid...")
                result, _final_code, d_atts = run_with_critic(
                    fixed_code,
                    pde_spec,
                    plan,
                    max_retries=5,
                    code_log_dir=str(plan_code_dir),
                    phase_name=f"Dense_R{restart_idx}",
                )

                runtime_s = time.time() - t0
                total_attempts = c_atts + d_atts

                coords = result.get("coords", {}) or {}
                u_all = np.asarray(result["u"])
                t_arr = result.get("t")
                if t_arr is None:
                    t_arr = coords.get("t")

                u_final = u_all
                t_final = None
                if t_arr is not None:
                    t_arr = np.asarray(t_arr)
                    t_final = float(t_arr[-1])
                    nt = len(t_arr)
                    if u_all.ndim > 0 and nt > 1:
                        if u_all.shape[0] == nt:
                            u_final = u_all[-1]
                        elif u_all.shape[-1] == nt:
                            u_final = u_all[..., -1]

                analytic_skipped = True
                residual_done = False
                l2, mx = None, None
                residual_metrics, residual_note = {"residual": None, "relative_residual": None}, ""
                try:
                    anal = pde_spec.get("analytic_solution", None)
                    if isinstance(anal, dict) and anal.get("type") == "implicit":
                        r_impl = implicit_solution_residual(coords, t_final, pde_spec, u_final)
                        if r_impl is not None:
                            dv = _infer_cell_volume(coords)
                            abs_l2 = float(np.sqrt(np.sum(np.abs(r_impl.ravel()) ** 2) * dv))
                            u_arr = np.asarray(u_final)
                            den = float(np.sqrt(np.sum(np.abs(u_arr.ravel()) ** 2) * dv)) + 1e-12
                            rel_l2 = abs_l2 / den
                            residual_done = True
                            abs_res = abs_l2
                            rel_res = rel_l2
                            residual_note = "[implicit-analytic residual] evaluated "
                            if abs_res is not None or rel_res is not None:
                                print(
                                    f"[Implicit Residual] abs_l2={abs_res} rel_l2_u={rel_res} ({residual_note})"
                                )
                            else:
                                print(
                                    f"[Implicit Residual skipped] {residual_note}. Keys={list(result.keys())}"
                                )
                    else:
                        expected_num_fields = (
                            u_final.shape[0] if np.asarray(u_final).ndim > 1 else None
                        )
                        u_exact = analytic_solution(
                            coords,
                            t_final,
                            pde_spec,
                            expected_num_fields=expected_num_fields,
                        )
                        if u_exact is not None:
                            u_exact = np.asarray(u_exact)
                            u_final_arr = np.asarray(u_final)

                            if (
                                u_exact.shape != u_final_arr.shape
                                and u_exact.size == u_final_arr.size
                            ):
                                u_exact = u_exact.reshape(u_final_arr.shape)

                            if u_exact.shape != u_final_arr.shape:
                                raise ValueError(
                                    f"u_exact shape {u_exact.shape} != u_final shape {u_final_arr.shape}"
                                )

                            dv = _infer_cell_volume(coords)
                            diff = (u_final_arr - u_exact).ravel()
                            num = float(np.sqrt(np.sum(np.abs(diff) ** 2) * dv))
                            den = float(np.sqrt(np.sum(np.abs(u_exact.ravel()) ** 2) * dv)) + 1e-12

                            l2 = num / den
                            err = u_final_arr - u_exact

                            if err.ndim >= 2 and err.shape[0] in (2, 3):
                                mx = float(np.max(np.sqrt(np.sum(np.abs(err) ** 2, axis=0))))
                            else:
                                mx = float(np.max(np.abs(err)))

                            analytic_skipped = False
                            abs_res = None
                            rel_res = None
                        else:
                            print(
                                f"[Analytic] returned None for id={pde_spec.get('id')} family={pde_spec.get('family')}"
                            )
                except Exception as e:
                    print(
                        f"[Analytic error] id={pde_spec.get('id')} family={pde_spec.get('family')}: {e}"
                    )

                if analytic_skipped and (not residual_done):
                    try:
                        residual_metrics, residual_note = compute_residual_l2(pde_spec, result)
                        abs_res = residual_metrics.get("residual", None)
                        rel_res = residual_metrics.get("relative_residual", None)

                        if abs_res is not None or rel_res is not None:
                            print(
                                f"[Residual] abs_l2={abs_res} rel_l2_u={rel_res} ({residual_note})"
                            )
                        else:
                            print(f"[Residual skipped] {residual_note}. Keys={list(result.keys())}")

                    except Exception as e:
                        print(f"[Residual error] {e}. Keys={list(result.keys())}")

                run_data = {
                    "plan_id": plan_id,
                    "metrics": {
                        "runtime_s": runtime_s,
                        "l2_error": l2,
                        "max_error": mx,
                        "residual_l2": abs_res,
                        "relative_residual": rel_res,
                        "num_attempts": total_attempts,
                        "restarts_used": restart_idx,
                    },
                    "plan": plan,
                }

                if not analytic_skipped and (not residual_done):
                    print(f"Runtime: {runtime_s:.3f}s, Explicit Analytic Available, L2: {l2:.3e}")
                elif analytic_skipped and residual_done:
                    print(
                        f"Runtime: {runtime_s:.3f}s, Explicit Analytic Skipped, "
                        f"[Implicit Residual]: abs_l2={abs_res} rel_l2_u={rel_res}"
                    )
                elif analytic_skipped and (not residual_done):
                    print(
                        f"Runtime: {runtime_s:.3f}s, Explicit Analytic Skipped, Residual: {residual_metrics}"
                    )
                else:
                    print("Check your code logic! This case should not happen.")
                break

            except Exception as e:
                print(f"[Run Failed] Restart {restart_idx}: {e}")
                last_error = str(e)

        if run_data:
            evaluated_plans.append(run_data)
        else:
            print(
                f"[CRITICAL FAILURE] Plan {plan_id} failed after {max_restarts} fresh generations."
            )
            evaluated_plans.append({"plan_id": plan_id, "error": last_error})

    _save_json(evaluated_plans, str(out_dir / "evaluated_plans.json"))

    print("\n=== Step 6: Final Selection & Theoretical Analysis ===")
    decision = choose_best_plan_with_llm(pde_spec, evaluated_plans)
    print(json.dumps(decision, indent=2))
    _save_json(decision, str(out_dir / "final_decision.json"))

    best_id = decision.get("best_plan_id")
    best_plan_data = next((p for p in evaluated_plans if p.get("plan_id") == best_id), None)

    if best_plan_data:
        print(f"\n[Analysis] Generating theoretical justification for '{best_id}'...")
        try:
            justification = analyze_theoretical_fit(
                pde_spec,
                best_plan_data.get("plan", {}),
                best_plan_data.get("metrics", {}),
            )
            print("\n" + "=" * 60)
            print("THEORETICAL ANALYSIS")
            print("=" * 60)
            print(justification)
            print("=" * 60 + "\n")
            _save_text(justification, str(out_dir / "theoretical_justification.txt"))
        except Exception as e:
            print(f"Reasoning generation failed: {e}")
    else:
        print("Could not find metrics for the selected plan.")


if __name__ == "__main__":
    main()
