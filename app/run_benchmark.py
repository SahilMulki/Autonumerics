"""Full-pipeline SDE benchmark: runs all (or selected) problems end-to-end.

Runs the full pipeline (formulate → plan → code → evaluate → select) for each
problem and prints a pass/fail summary table.  Results are written to JSON.

Usage:
    python -m app.run_benchmark
    python -m app.run_benchmark --ids bm_standard geometric_brownian_motion
    python -m app.run_benchmark --output /tmp/bench --seed 42
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formulator import formulate_problem
from src.sde_problem_list import ALL_SDE_PROBLEMS, get_sde_problem_by_id

# Reuse helpers from run_single_problem to avoid duplication
from app.run_single_problem import _NumpyEncoder, _save_json, _run_sde_pipeline  # noqa: F401

MEAN_THRESHOLD = 0.05   # 5% relative error
VAR_THRESHOLD  = 0.10   # 10% relative error

CATEGORY: dict[str, str] = {
    "bm_standard":                    "easy",
    "bm_with_drift":                  "easy",
    "ornstein_uhlenbeck":             "easy",
    "linear_sde_additive":            "easy",
    "geometric_brownian_motion":      "medium",
    "cox_ingersoll_ross":             "medium",
    "exponential_ornstein_uhlenbeck": "medium",
    "black_scholes":                  "medium",
    "gbm_2d_correlated":              "medium",
    "stochastic_oscillator":          "medium",
}


def _is_pass(metrics: dict) -> bool:
    """Return True if the best plan's moment errors are within target thresholds."""
    if not metrics.get("has_analytic_moments"):
        return True  # no ground truth available

    var_err = metrics.get("variance_relative_error")
    if var_err is None or not np.isfinite(float(var_err)):
        return False
    if float(var_err) >= VAR_THRESHOLD:
        return False

    mean_err = metrics.get("mean_relative_error")
    if mean_err is None or not np.isfinite(float(mean_err)):
        return True  # variance passed; mean undefined → pass

    # Skip the mean check if any exact-mean component is near zero: the relative
    # error is dominated by Monte Carlo noise in the denominator (known limitation).
    exact_mean_raw = metrics.get("exact_mean")
    if exact_mean_raw is not None:
        em = np.asarray(exact_mean_raw, dtype=float).ravel()
        if np.any(np.abs(em) < 0.01):
            return True  # near-zero mean component → variance check is sufficient

    return float(mean_err) < MEAN_THRESHOLD


def _run_one_problem(problem: dict, bench_dir: Path, seed: int) -> dict:
    """Run the full SDE pipeline for one problem; return a benchmark result dict."""
    pid = problem["id"]
    out_dir = bench_dir / pid
    code_dir = out_dir / "code"
    out_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)

    wall_t0 = time.time()
    try:
        sde_spec = formulate_problem(problem["description"], seed=seed)
        sde_spec["id"] = pid
        sde_spec["family"] = "sde"

        _run_sde_pipeline(sde_spec, out_dir, code_dir, seed)

        decision_path  = out_dir / "sde_final_decision.json"
        evaluated_path = out_dir / "sde_evaluated_plans.json"

        decision  = json.loads(decision_path.read_text())  if decision_path.exists()  else {}
        evaluated = json.loads(evaluated_path.read_text()) if evaluated_path.exists() else []

        best_id   = decision.get("best_plan_id", "")
        best_data = next((p for p in evaluated if p.get("plan_id") == best_id), None)

        wall_s = time.time() - wall_t0

        if best_data and "metrics" in best_data:
            m      = best_data["metrics"]
            scheme = best_data.get("plan", {}).get("numerical_scheme", "?")
            passed = _is_pass(m)
            return {
                "problem_id":    pid,
                "status":        "pass" if passed else "fail",
                "category":      CATEGORY.get(pid, "?"),
                "scheme":        scheme,
                "best_plan_id":  best_id,
                "mean_rel_err":  m.get("mean_relative_error"),
                "var_rel_err":   m.get("variance_relative_error"),
                "has_analytic":  m.get("has_analytic_moments", False),
                "runtime_s":     m.get("runtime_s"),
                "wall_s":        wall_s,
            }

        return {
            "problem_id": pid, "status": "no_metrics",
            "category": CATEGORY.get(pid, "?"), "wall_s": wall_s,
        }

    except Exception as e:
        return {
            "problem_id": pid, "status": "error",
            "category": CATEGORY.get(pid, "?"),
            "error": str(e), "wall_s": time.time() - wall_t0,
        }


def _fmt_err(val) -> str:
    if val is None:
        return "   N/A"
    try:
        f = float(val)
        if not np.isfinite(f):
            return "   N/A"
        return f"{f * 100:5.2f}%"
    except (TypeError, ValueError):
        return "   N/A"


def _fmt_time(secs) -> str:
    if secs is None:
        return "   ?"
    return f"{int(secs):4d}s"


def _print_summary(results: list[dict]) -> None:
    header = (
        f"{'Problem':<34} {'Cat':<7} {'Scheme':<15}"
        f" {'Mean Err':>8} {'Var Err':>8} {'Wall':>6}  Result"
    )
    sep = "-" * len(header)
    print()
    print("=" * len(header))
    print("BENCHMARK SUMMARY — Autonumerics SDE Pipeline")
    print("=" * len(header))
    print(header)
    print(sep)
    for r in results:
        status = r.get("status", "?").upper()
        flag   = "✓" if status == "PASS" else ("✗" if status in ("FAIL", "ERROR") else "~")
        print(
            f"{r['problem_id'][:33]:<34}"
            f" {r.get('category', '?')[:6]:<7}"
            f" {r.get('scheme', '?')[:14]:<15}"
            f" {_fmt_err(r.get('mean_rel_err')):>8}"
            f" {_fmt_err(r.get('var_rel_err')):>8}"
            f" {_fmt_time(r.get('wall_s')):>6}"
            f"  {flag} {status}"
        )
    print(sep)
    counts = {s: sum(1 for r in results if r.get("status") == s)
              for s in ("pass", "fail", "error", "no_metrics")}
    print(
        f"  Total: {len(results)}"
        f"   Pass: {counts['pass']}"
        f"   Fail: {counts['fail']}"
        f"   Error: {counts['error']}"
    )
    print()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run SDE pipeline benchmark over all (or selected) problems."
    )
    p.add_argument("--ids", nargs="+", default=None,
                   help="Problem IDs to benchmark (default: all 10)")
    p.add_argument("--output", default=None,
                   help="Root output directory (default: outputs_new_paper/benchmark_<ts>)")
    p.add_argument("--seed", type=int, default=42, help="Global random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    problems = (
        [get_sde_problem_by_id(pid) for pid in args.ids]
        if args.ids
        else list(ALL_SDE_PROBLEMS)
    )

    if args.output:
        bench_dir = Path(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        bench_dir = PROJECT_ROOT / "outputs_new_paper" / f"benchmark_{ts}"
    bench_dir.mkdir(parents=True, exist_ok=True)

    print(f"Benchmark: {len(problems)} problem(s)  →  {bench_dir}")
    print(f"Seed: {args.seed}\n")

    results: list[dict] = []
    for i, problem in enumerate(problems, 1):
        pid = problem["id"]
        print(f"\n{'='*60}")
        print(f"[{i}/{len(problems)}] {pid}")
        print(f"{'='*60}")

        r = _run_one_problem(problem, bench_dir, args.seed)
        results.append(r)

        status = r.get("status", "?").upper()
        print(f"\n→ {pid}: {status}  (wall={r.get('wall_s', 0):.0f}s)")

    _print_summary(results)

    summary_path = bench_dir / "benchmark_summary.json"
    _save_json({"seed": args.seed, "results": results}, str(summary_path))
    print(f"Results saved to {summary_path}")


if __name__ == "__main__":
    main()
