"""Weak convergence rate verification for SDE solvers.

Generates validated solver code once per scheme, then runs it at multiple dt
values to measure weak error |E[X_h(T)] - E[X_exact(T)]| as a function of dt.
A log-log regression estimates the empirical weak convergence order (expected ~1.0
for both EM and Milstein).

Multiplicative-noise problems (GBM, CIR, Exp-OU, Black-Scholes) run both EM and
Milstein; additive-noise problems run EM only (Milstein correction = 0 for additive).

Usage:
    python -m app.run_convergence
    python -m app.run_convergence --ids geometric_brownian_motion ornstein_uhlenbeck
    python -m app.run_convergence --num-paths 200000 --no-plot
    python -m app.run_convergence --output /tmp/conv --seed 7
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.formulator import formulate_problem
from src.sde_coder_agent import generate_sde_solver_code
from src.sde_analytic_utils import evaluate_sde_moments
from src.sde_problem_list import get_sde_problem_by_id

from app.run_single_problem import (
    _NumpyEncoder,
    _save_json,
    _save_text,
    _strip_code_fences,
    _make_sde_coarse_plan,
    run_sde_with_critic,
)

# ── Default configuration ────────────────────────────────────────────────────

DEFAULT_PROBLEMS = [
    "bm_with_drift",
    "geometric_brownian_motion",
    "ornstein_uhlenbeck",
]

# dt values for the convergence sweep (descending: large dt first)
DT_VALUES = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]

NUM_PATHS        = 100_000   # paths for each dt level in the sweep
VALIDATION_PATHS = 500       # paths used for the critic-loop validation (fast)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_ref_plan(sde_spec: dict, scheme: str, seed: int) -> dict:
    T  = float(sde_spec["time_interval"]["t_final"])
    dt = 0.01
    return {
        "plan_id":            f"{scheme}_convergence_ref",
        "description":        f"Reference plan for convergence code validation ({scheme})",
        "numerical_scheme":   scheme,
        "dt":                 dt,
        "Nt":                 max(1, round(T / dt)),
        "num_paths":          10_000,
        "convergence_target": "weak",
        "seed":               seed,
        "strong_order":       0.5 if scheme == "euler_maruyama" else 1.0,
        "weak_order":         1.0,
        "extra_parameters":   {},
    }


def _formulate_and_validate(
    problem_id: str, scheme: str, out_dir: Path, seed: int
) -> tuple[dict, str]:
    """Formulate the problem and return (sde_spec, validated_code) for `scheme`."""
    problem  = get_sde_problem_by_id(problem_id)
    sde_spec = formulate_problem(problem["description"], seed=seed)
    sde_spec["id"] = problem_id

    ref_plan = _make_ref_plan(sde_spec, scheme, seed)

    print(f"  Generating {scheme} code …")
    code = _strip_code_fences(generate_sde_solver_code(sde_spec, ref_plan))
    _save_text(code, str(out_dir / f"code_{scheme}_initial.py"))

    # Reduce num_paths to VALIDATION_PATHS for a cheap critic-loop validation
    scale = max(1, ref_plan["num_paths"] // VALIDATION_PATHS)
    coarse = _make_sde_coarse_plan(ref_plan, path_scale_factor=scale)

    print(f"  Validating ({coarse['num_paths']} paths) …")
    _, validated_code, attempts = run_sde_with_critic(
        code, sde_spec, coarse,
        max_retries=8, code_log_dir=str(out_dir),
        phase_name=f"Validate_{scheme}",
    )
    print(f"  Validated in {attempts} attempt(s).")
    _save_text(validated_code, str(out_dir / f"code_{scheme}_validated.py"))

    return sde_spec, validated_code


def _run_dt_sweep(
    sde_spec: dict,
    validated_code: str,
    scheme: str,
    dt_values: list[float],
    num_paths: int,
    seed: int,
) -> list[dict]:
    """Execute the validated solver at each dt; return per-dt weak error rows."""
    T = float(sde_spec["time_interval"]["t_final"])

    # Exec the code once; reuse the solve_sde function across all dt levels
    env: dict = {}
    exec(validated_code, env, env)
    solve_fn = env.get("solve_sde")
    if not callable(solve_fn):
        raise RuntimeError("Validated code does not define solve_sde().")

    rows: list[dict] = []
    for dt in dt_values:
        Nt = max(1, round(T / dt))
        plan = {
            "plan_id":            f"conv_dt{dt}",
            "description":        f"Convergence sweep dt={dt}",
            "numerical_scheme":   scheme,
            "dt":                 dt,
            "Nt":                 Nt,
            "num_paths":          num_paths,
            "convergence_target": "weak",
            "seed":               seed,
            "strong_order":       0.5 if scheme == "euler_maruyama" else 1.0,
            "weak_order":         1.0,
            "extra_parameters":   {},
        }

        result  = solve_fn(sde_spec, plan)
        metrics = evaluate_sde_moments(sde_spec, result)

        emp_mean  = np.asarray(metrics["empirical_mean"],    dtype=float)
        emp_var   = np.asarray(metrics["empirical_variance"], dtype=float)
        exact_mean = metrics.get("exact_mean")
        exact_var  = metrics.get("exact_variance")

        mean_abs_err = (
            float(np.max(np.abs(emp_mean - np.asarray(exact_mean, dtype=float))))
            if exact_mean is not None else float("nan")
        )
        var_abs_err = (
            float(np.max(np.abs(emp_var - np.asarray(exact_var, dtype=float))))
            if exact_var is not None else float("nan")
        )

        print(f"    dt={dt:.4f}  Nt={Nt:5d}  |ΔE|={mean_abs_err:.4e}  |ΔVar|={var_abs_err:.4e}")
        rows.append({
            "dt":            dt,
            "Nt":            Nt,
            "mean_abs_err":  mean_abs_err,
            "var_abs_err":   var_abs_err,
            "mean_rel_err":  metrics.get("mean_relative_error"),
            "var_rel_err":   metrics.get("variance_relative_error"),
        })

    return rows


def _fit_slope(dt_vals: list[float], errors: list[float]) -> Optional[float]:
    """Fit a log-log slope via linear regression.  Returns None if < 3 valid points."""
    valid = [(d, e) for d, e in zip(dt_vals, errors) if np.isfinite(e) and e > 0]
    if len(valid) < 3:
        return None
    xs = np.log([v[0] for v in valid])
    ys = np.log([v[1] for v in valid])
    slope, _ = np.polyfit(xs, ys, 1)
    return float(slope)


def _print_table(
    problem_id: str,
    scheme: str,
    rows: list[dict],
    mean_slope: Optional[float],
    var_slope: Optional[float],
) -> None:
    scheme_label = "EM" if scheme == "euler_maruyama" else "Milstein"
    print(f"\n  Weak convergence — {problem_id}  ({scheme_label})")
    print(f"  {'dt':>8}  {'Nt':>6}  {'|ΔE[X]|':>12}  {'|ΔVar[X]|':>12}")
    print(f"  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*12}")
    for r in rows:
        print(
            f"  {r['dt']:>8.4f}  {r['Nt']:>6}"
            f"  {r['mean_abs_err']:>12.4e}  {r['var_abs_err']:>12.4e}"
        )
    print(f"  {'─'*8}  {'─'*6}  {'─'*12}  {'─'*12}")
    mo = f"{mean_slope:.3f}" if mean_slope is not None else "N/A"
    vo = f"{var_slope:.3f}"  if var_slope  is not None else "N/A"
    print(f"  Empirical weak order — E[X]: {mo}   Var[X]: {vo}   (expected ≈ 1.0)")


def _save_plot(
    problem_id: str, scheme: str, rows: list[dict], out_dir: Path
) -> None:
    """Save a log-log convergence plot.  No-op if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    dts       = [r["dt"]           for r in rows]
    mean_errs = [r["mean_abs_err"] for r in rows]
    var_errs  = [r["var_abs_err"]  for r in rows]

    valid_mean = [(d, e) for d, e in zip(dts, mean_errs) if np.isfinite(e) and e > 0]
    valid_var  = [(d, e) for d, e in zip(dts, var_errs)  if np.isfinite(e) and e > 0]

    fig, ax = plt.subplots(figsize=(6, 4))
    if valid_mean:
        ax.loglog([v[0] for v in valid_mean], [v[1] for v in valid_mean],
                  "o-", label="|ΔE[X]|")
    if valid_var:
        ax.loglog([v[0] for v in valid_var], [v[1] for v in valid_var],
                  "s--", label="|ΔVar[X]|")

    # Reference slope-1 line anchored at the first valid mean point
    if valid_mean and dts:
        dt0, err0 = valid_mean[0]
        ref_dts  = np.array([min(dts), max(dts)])
        ref_errs = err0 * (ref_dts / dt0)
        ax.loglog(ref_dts, ref_errs, "k:", alpha=0.6, label="slope=1 reference")

    scheme_label = "EM" if scheme == "euler_maruyama" else "Milstein"
    ax.set_xlabel("dt")
    ax.set_ylabel("Absolute weak error")
    ax.set_title(f"Weak convergence: {problem_id} ({scheme_label})")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    out_path = out_dir / f"convergence_{problem_id}_{scheme}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot → {out_path.name}")


# ── Per-problem entry point ───────────────────────────────────────────────────

def run_convergence_for_problem(
    problem_id: str,
    schemes: list[str],
    dt_values: list[float],
    num_paths: int,
    seed: int,
    out_dir: Path,
    save_plot: bool = True,
) -> dict:
    """Run convergence verification for one problem across all requested schemes."""
    prob_dir = out_dir / problem_id
    prob_dir.mkdir(parents=True, exist_ok=True)

    result: dict = {"problem_id": problem_id, "schemes": {}}

    for scheme in schemes:
        print(f"\n[{problem_id}] scheme={scheme}")
        try:
            sde_spec, validated_code = _formulate_and_validate(
                problem_id, scheme, prob_dir, seed
            )
            rows = _run_dt_sweep(
                sde_spec, validated_code, scheme, dt_values, num_paths, seed
            )
            mean_slope = _fit_slope(dt_values, [r["mean_abs_err"] for r in rows])
            var_slope  = _fit_slope(dt_values, [r["var_abs_err"]  for r in rows])

            _print_table(problem_id, scheme, rows, mean_slope, var_slope)

            if save_plot:
                _save_plot(problem_id, scheme, rows, prob_dir)

            result["schemes"][scheme] = {
                "rows":                  rows,
                "empirical_mean_order":  mean_slope,
                "empirical_var_order":   var_slope,
            }
        except Exception as e:
            print(f"  [FAILED] {e}")
            result["schemes"][scheme] = {"error": str(e)}

    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Weak convergence rate verification for SDE solvers."
    )
    p.add_argument("--ids", nargs="+", default=None,
                   help=f"Problem IDs (default: {DEFAULT_PROBLEMS})")
    p.add_argument("--num-paths", type=int, default=NUM_PATHS,
                   help=f"Paths per dt level (default: {NUM_PATHS})")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip saving matplotlib convergence plots")
    p.add_argument("--output", default=None,
                   help="Output directory (default: outputs_new_paper/convergence_<ts>)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    problem_ids = args.ids or DEFAULT_PROBLEMS

    if args.output:
        out_dir = Path(args.output)
    else:
        ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        out_dir = PROJECT_ROOT / "outputs_new_paper" / f"convergence_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Convergence verification: {problem_ids}")
    print(f"dt_values={DT_VALUES}  num_paths={args.num_paths}  seed={args.seed}")
    print(f"Output: {out_dir}\n")

    all_results: list[dict] = []
    for pid in problem_ids:
        try:
            problem = get_sde_problem_by_id(pid)
        except KeyError:
            print(f"Unknown problem_id: {pid} — skipping.")
            continue

        # Multiplicative-noise scalar SDEs: run both EM and Milstein to compare.
        # Additive-noise or multi-D: EM only (Milstein reduces to EM for additive;
        # multi-D Milstein requires Lévy area correction — not implemented).
        multiplicative = problem.get("noise_structure") == "multiplicative"
        scalar         = problem.get("state_dimension", 1) == 1
        schemes = ["euler_maruyama", "milstein"] if (multiplicative and scalar) else ["euler_maruyama"]

        result = run_convergence_for_problem(
            pid, schemes, DT_VALUES, args.num_paths, args.seed,
            out_dir, save_plot=not args.no_plot,
        )
        all_results.append(result)

    # Final summary table
    print("\n" + "=" * 62)
    print("CONVERGENCE SUMMARY  (expected weak order ≈ 1.0 for EM and Milstein)")
    print("=" * 62)
    for r in all_results:
        for scheme, data in r.get("schemes", {}).items():
            label = "EM       " if scheme == "euler_maruyama" else "Milstein "
            if "error" in data:
                print(f"  {r['problem_id']:<34} {label}  ERROR")
            else:
                mo = data.get("empirical_mean_order")
                vo = data.get("empirical_var_order")
                mo_s = f"{mo:.3f}" if mo is not None else " N/A"
                vo_s = f"{vo:.3f}" if vo is not None else " N/A"
                print(f"  {r['problem_id']:<34} {label}  mean_order={mo_s}  var_order={vo_s}")

    _save_json(all_results, str(out_dir / "convergence_results.json"))
    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
