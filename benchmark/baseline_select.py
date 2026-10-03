"""Pick one answer out of a baseline's matched attempts, the way the conductor does.

The matched C0 baseline gives Claude Opus as many independent attempts at each
problem as the pipeline spent on it (``matched_attempts.json``). That raises the
question the comparison turns on: **which** of those attempts is the baseline's
answer?

Grading the best one by the benchmark's ground truth would hand the baseline an
oracle the pipeline never had -- the pipeline picks its winner with
``verifylib``'s kernel, which never sees the ground truth. So this runs the *same*
kernel over the baseline's attempts and ranks them by the *same* lexicographic rule
``commands/conductor.md`` §3 gives the conductor:

    score desc, provenance (score.PROVENANCE_ORDER), quality index desc,
    solver time asc

(``verifylib.ranking.rank_key`` -- the conductor's key and this one are the same
function, deliberately. A selector that ranked by its own rule would measure the
rule rather than the baseline.)

and reports the selected attempt's harness verdict as the baseline's result. Three
cuts come out of the same data, and the comparison should quote all three:

    first        -- rep1 only; what one completion gets you
    kernel       -- the attempt this selector picks; matched attempts, matched
                    selection rule, no oracle. The headline number.
    pass_at_k    -- any attempt passes; the oracle-selection upper bound, which
                    brackets how much of the gap is selection rather than solving

What the baseline is *given* to make this work, stated plainly because it is the
one asymmetry: the kernel cannot evaluate a solver without a ``problem_spec.json``
(the equation, thresholds, invariants), and a C0 baseline produces no spec -- that
is the formulator's job and the formulator is precisely what C0 lacks. So each
attempt is evaluated against **the pipeline's own spec** for that problem, copied
into a scratch directory beside the attempt's ``solver.py``. The baseline's solver
never sees it; only the selector does. The baseline therefore keeps its handicap
(no decomposition, no planner, no evaluator loop) and loses the one handicap that
would make the question unanswerable. A run where the spec is missing is reported
as ``spec_missing`` rather than silently scored.

Free to re-run: it makes no model calls, only kernel evaluations of solvers that
already exist on disk.

Usage
-----
    uv run python benchmark/baseline_select.py \\
        --results-json benchmark/results/oneshot_no_closed_form.json
    uv run python benchmark/baseline_select.py --results-json <file> --only pde_kdv_zabusky_kruskal
    uv run python benchmark/baseline_select.py --results-json <file> --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import problems as P  # noqa: E402

sys.path.insert(0, REPO_ROOT)
from verifylib.ranking import rank_key  # noqa: E402

SPEC_FILE = "problem_spec.json"
#: Where the selector stages each attempt for the kernel. Not under `workspace/`:
#: `run.isolated_settings` fences every sibling workspace off from a live pipeline
#: run, and a directory of baseline solvers appearing there would be one more thing
#: to fence.
SCRATCH = os.path.join(REPO_ROOT, "workspace_oneshot", "_kernel_select")

PASS_VERDICTS = ("PASS", "STABLE_PASS")


def pipeline_spec(slug):
    """The pipeline's spec for this problem, or None. See the module docstring for
    why the baseline is allowed to borrow it."""
    path = os.path.join(REPO_ROOT, "workspace", slug, SPEC_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        return None


def solution_stub(unit):
    """The minimal ``SOLUTION.md`` the kernel reads: it parses the frontmatter for
    ``scheme_family`` / ``spatial_order`` to choose D1's differencing pair, and falls
    back to 8th-order FD with ``stencil_choice: default`` when they are absent
    (``kernel/plan_meta.py``). A C0 attempt declares neither -- there is no plan
    document -- so the stub declares nothing and the fallback applies uniformly to
    every attempt, which keeps the attempts comparable to each other. It is recorded
    in the result as ``scheme_family_source: unknown`` so a reader can see that the
    kernel guessed rather than being told."""
    return ("---\n"
            f"plan: {unit['_key']}\n"
            "# No scheme_family/spatial_order: a C0 attempt declares no plan, so the\n"
            "# kernel's default stencil choice applies to every attempt alike.\n"
            "---\n\n"
            "# Baseline attempt (C0)\n\n"
            "Staged by `benchmark/baseline_select.py` so the kernel can score this\n"
            "attempt the way it scores a pipeline plan. Not written by an agent.\n")


def stage(unit, spec):
    """Lay out ``<scratch>/<slug>/problem_spec.json`` + ``plans/<key>/{solver.py,
    SOLUTION.md}`` -- the shape ``gate._sibling_spec`` walks up to find."""
    slug = unit["slug"]
    root = os.path.join(SCRATCH, slug)
    plan_dir = os.path.join(root, "plans", unit["_key"].replace("|", "_"))
    os.makedirs(plan_dir, exist_ok=True)
    with open(os.path.join(root, SPEC_FILE), "w") as fh:
        json.dump(spec, fh, indent=1)
    shutil.copyfile(os.path.join(REPO_ROOT, unit["plan_dir"], "solver.py"),
                    os.path.join(plan_dir, "solver.py"))
    with open(os.path.join(plan_dir, "SOLUTION.md"), "w") as fh:
        fh.write(solution_stub(unit))
    return plan_dir


def evaluate(plan_dir, timeout_s=900):
    """``verifylib.cli evaluate --json`` on a staged attempt -> its metrics block."""
    proc = subprocess.run(
        [sys.executable, "-m", "verifylib.cli", "evaluate", plan_dir, "--json"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout_s, check=False)
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        return None, {"error": "kernel produced no JSON",
                      "returncode": proc.returncode,
                      "detail": tail[-3:] if tail else []}


def looks_n_independent(summary):
    """Whether the attempt's error did not respond to ``N`` at all.

    Measured on a real C0 attempt (Opus, KS, rep1, 2026-08-07): it computed on a
    hard-coded internal spectral grid of M = 256 and used the requested ``N`` only as
    the grid it interpolated the answer onto. The solution then differs between
    ``N`` and ``2N`` by 2.2e-16 -- the round-off of the interpolation -- so the
    kernel's ladder reports ``estimated_rel_error: 0.0`` and ``observed_order: inf``
    and caps it at 8 ``self_convergence``. It is not a wrong answer (the harness
    passes it at 2.1e-06); it is an answer whose accuracy the ladder cannot measure.

    This matters twice. The reported ``estimated_rel_error`` of 0.0 is not a
    measurement, and it is the conductor's third ranking criterion -- so an
    N-independent attempt sorts *ahead* of every honestly-measured attempt tied with
    it on score and provenance. The ranking here is deliberately left faithful to
    ``conductor.md`` rather than patched, because the pipeline's own conductor is
    fooled in exactly the same way and hiding that would flatter the pipeline. The
    flag is recorded so the comparison can say how often it happened.

    Also: the benchmark's order check is what actually catches this shape (err_N
    ~= err_2N gives observed order ~0, under any ``min_order``), and KS is the one
    problem in the suite with ``pde_order_check: False`` -- so this attempt passes
    there and the same attempt shape would fail on the other six."""
    err = summary.get("estimated_rel_error")
    order = summary.get("observed_order")
    return bool(err == 0.0 and (order is None or order == float("inf")))


def select(slug, units, dry_run=False):
    """Rank one problem's attempts and return the selection block."""
    spec = pipeline_spec(slug)
    attempts = []
    for unit in units:
        rec = {"key": unit["_key"], "rep": unit["rep"], "manuals": unit["manuals"],
               "harness_verdict": unit["verdict"],
               "harness_passed": unit["verdict"] in PASS_VERDICTS,
               "plan_dir": unit["plan_dir"]}
        solver = os.path.join(REPO_ROOT, unit.get("plan_dir") or "", "solver.py")
        if not unit.get("plan_dir") or not os.path.exists(solver):
            rec["status"] = "no_solver"
        elif spec is None:
            rec["status"] = "spec_missing"
        elif dry_run:
            rec["status"] = "would_evaluate"
        else:
            metrics, err = evaluate(stage(unit, spec))
            if metrics is None:
                rec["status"] = "kernel_error"
                rec["kernel_error"] = err
            else:
                rec["status"] = "ok"
                summary = metrics.get("summary") or {}
                rec["n_independent"] = looks_n_independent(summary)
                rec["kernel"] = {
                    "score": metrics.get("score"), "provenance": metrics.get("provenance"),
                    "tier": (metrics.get("certification") or {}).get("tier"),
                    "bound_by": (metrics.get("certification") or {}).get("bound_by"),
                    "reason": (metrics.get("certification") or {}).get("reason"),
                    "estimated_rel_error": summary.get("estimated_rel_error"),
                    "observed_order": summary.get("observed_order"),
                    "order_floor": summary.get("order_floor"),
                    "d1_outcome": summary.get("d1_outcome"),
                    "wall_time_s": summary.get("wall_time_s"),
                    "summary": summary,
                }
        attempts.append(rec)

    scored = [a for a in attempts if a.get("kernel")]
    # The cost component is pool-relative, so the whole pool goes in: an attempt's
    # quality index is not a property it carries alone.
    pool = [(a["kernel"].get("summary") or {}) for a in scored]
    ranked = sorted(scored, key=lambda a: rank_key(a, pool))
    chosen = ranked[0] if ranked else None
    first = next((a for a in sorted(attempts, key=lambda a: (a["manuals"], a["rep"]))
                  if a["rep"] == 1), None)
    out = {
        "slug": slug,
        "attempts": len(attempts),
        "attempts_kernel_scored": len(scored),
        "ranking": [a["key"] for a in ranked],
        "cuts": {
            "first": {"key": first and first["key"], "passed": bool(first and first["harness_passed"])},
            "kernel": {"key": chosen and chosen["key"],
                       "passed": bool(chosen and chosen["harness_passed"]),
                       "kernel_score": chosen and (chosen.get("kernel") or {}).get("score"),
                       "provenance": chosen and (chosen.get("kernel") or {}).get("provenance"),
                       "n_independent": bool(chosen and chosen.get("n_independent"))},
            "pass_at_k": {"passed": any(a["harness_passed"] for a in attempts),
                          "keys": [a["key"] for a in attempts if a["harness_passed"]]},
        },
        "per_attempt": attempts,
    }
    # The number the whole exercise is for: did the non-oracle selector pick a
    # passing attempt when a passing attempt was available? A False here on a
    # problem whose pass_at_k is True is a *selection* failure, which is exactly
    # the defect `wrong_plan_won` measures on the pipeline side.
    out["selection_lost_a_pass"] = bool(out["cuts"]["pass_at_k"]["passed"]
                                        and not out["cuts"]["kernel"]["passed"])
    out["n_independent_attempts"] = [a["key"] for a in attempts if a.get("n_independent")]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-json", dest="results_json", required=True,
                    help="the oneshot results JSON whose units to select over")
    ap.add_argument("--only", default="", help="comma-separated slugs (default: all in the file)")
    ap.add_argument("--condition", default="c0", help="baseline condition to select over (default c0)")
    ap.add_argument("--manuals", default="both", choices=("naked", "manual", "both"),
                    help="select within one knowledge condition, or pool both (default both). "
                         "Pooling matches the pipeline, whose agents read the manuals.")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the units that would be evaluated; run no kernel")
    ap.add_argument("--out", default=None,
                    help="where to write the selection JSON (default: alongside the results file, "
                         "<name>.selection.json)")
    args = ap.parse_args(argv)

    with open(args.results_json) as fh:
        payload = json.load(fh)
    units = [u for u in payload.get("results", []) if u.get("condition") == args.condition]
    if args.manuals != "both":
        units = [u for u in units if u.get("manuals") == args.manuals]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        units = [u for u in units if u["slug"] in wanted]
    if not units:
        ap.error(f"no {args.condition} units in {args.results_json} matching the filters")

    by_slug = {}
    for u in units:
        by_slug.setdefault(u["slug"], []).append(u)
    order = [p["slug"] for p in P.no_closed_form_problems() if p["slug"] in by_slug]
    order += [s for s in by_slug if s not in order]

    out = {"config": {"results_json": os.path.relpath(args.results_json, REPO_ROOT),
                      "condition": args.condition, "manuals": args.manuals,
                      "ranking": "conductor.md Phase 3 step 2 (score, provenance, "
                                 "estimated_rel_error, order margin, wall time)",
                      "spec_source": "workspace/<slug>/problem_spec.json (the pipeline's; "
                                     "see baseline_select.py docstring)"},
             "problems": {}}
    for slug in order:
        res = select(slug, by_slug[slug], dry_run=args.dry_run)
        out["problems"][slug] = res
        c = res["cuts"]
        chosen = c["kernel"]
        print(f"{slug:34s} attempts={res['attempts']:<2} scored={res['attempts_kernel_scored']:<2} "
              f"first={'PASS' if c['first']['passed'] else 'fail':<4} "
              f"kernel={'PASS' if chosen['passed'] else 'fail':<4} "
              f"(score {chosen['kernel_score']}, {chosen['provenance']}) "
              f"pass@k={'PASS' if c['pass_at_k']['passed'] else 'fail'}"
              f"{'  N-indep' if res['n_independent_attempts'] else ''}"
              f"{'   <- SELECTION LOST A PASS' if res['selection_lost_a_pass'] else ''}")

    if args.dry_run:
        print("\ndry run: no kernel evaluations, nothing written")
        return 0
    path = args.out or (os.path.splitext(args.results_json)[0] + ".selection.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)
    tot = out["problems"]
    for cut in ("first", "kernel", "pass_at_k"):
        n = sum(1 for v in tot.values() if v["cuts"][cut]["passed"])
        print(f"\n{cut:10s} {n}/{len(tot)} problems pass")
    print(f"wrote {os.path.relpath(path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
