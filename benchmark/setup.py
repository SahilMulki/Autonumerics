"""Generate benchmark workspaces from the curated problem set.

For each problem this writes ``workspace/{slug}/problem.md`` -- the single input
the Autonumerics pipeline needs -- and emits ``benchmark/manifest.json`` recording
the (JSON-serializable) metadata for every problem so the runner and report have a
stable record of what the benchmark contains.

Usage:
    uv run python benchmark/setup.py                 # generate all 37 workspaces
    uv run python benchmark/setup.py --only sde_gbm,pde_heat_1d
    uv run python benchmark/setup.py --list          # just print the inventory
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import problems as P  # noqa: E402, I001
import verify as V  # noqa: E402  (single source of truth for the evaluation step)

MANIFEST_PATH = os.path.join(HERE, "manifest.json")


def workspace_dir(slug, workspace_root):
    return os.path.join(workspace_root, slug)


def contract_footer(problem):
    """The solver I/O contract + exact evaluation configuration, appended to every
    problem.md. Keeping this in the problem statement (rather than only in the
    pipeline's private agent prompts) is what makes a one-shot LLM given just
    problem.md a fair comparison: it is told the function to implement, the schema to
    return, and the exact arguments / step it will be scored at."""
    if problem["type"] == "sde":
        d = problem["state_dimension"]
        dtv = V.sde_verify_dt(problem)
        if d == 1:
            schema = (
                "- `terminal_paths`: 1-D numpy array of shape `(num_paths,)` — the state at time T\n"
                "- `empirical_mean`: float — mean of the state at T\n"
                "- `empirical_variance`: float — variance of the state at T (use `ddof=1`)"
            )
        else:
            labels = ", ".join(["X", "Y", "Z"][:d])
            schema = (
                f"- `terminal_paths`: 2-D numpy array of shape `(num_paths, {d})` — the {d} state "
                f"components ({labels}) at time T, in that column order\n"
                "- `empirical_mean`: list of floats, one per component\n"
                "- `empirical_variance`: list of floats, one per component (use `ddof=1`)"
            )
        kind = problem.get("ground_truth_kind", "exact")
        if kind == "stability":
            chk = problem.get("stability_check", {"type": "finite"})
            if chk.get("type") == "domain":
                crit = (f"every returned terminal state is finite **and** stays inside the domain "
                        f"(|state| < {chk.get('abs_bound', 1.0):g})")
            else:
                crit = "every returned terminal state is finite (no Inf/NaN)"
            scoring = (
                "This problem has no closed-form moments. Scoring is independent of any "
                f"self-report: the harness re-runs `solve_sde` at the arguments above and checks "
                f"only that {crit} — exactly what a naive scheme that diverges or escapes the "
                "domain gets wrong."
            )
        else:
            scoring = (
                "Scoring is independent of any self-report: the harness recomputes the terminal "
                "mean and variance at the arguments above and compares them to its own ground "
                "truth (variance relative error < 10%, and — unless the exact mean is within 0.01 "
                "of zero — mean relative error < 5%)."
            )
        return f"""

---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called **once**, with these exact arguments:

    solve_sde(num_paths={problem['num_paths']}, dt={dtv:.4g}, T={problem['T']:.6g}, seed={problem['seed']})

Use the `dt` you are given (do not hardcode a different step). It must return a dict with:

{schema}

Requirements:
- Seed all randomness with `np.random.default_rng(seed)` (not `np.random.seed`).
- Round the step count: `Nt = max(1, round(T / dt))`, then `dt = T / Nt`.
- Keep `solve_sde` importable — no work at module load outside `if __name__ == "__main__"`.
- Use the Python standard library and numpy only.

{scoring}
"""
    # PDE
    dims = problem["dims"]
    axes = problem.get("axes") or ["x", "y", "z"][:dims]
    fields = problem.get("fields")
    metric = problem.get("pde_metric", "l2")
    metric_name = "relative L1" if metric == "l1" else "relative L2"
    shape = "(" + ", ".join(["N"] * dims) + ")"
    axis_list = ", ".join(f'"{a}": <1-D array, length N>' for a in axes)
    tol = V._pde_l2_tol(problem)

    if fields:
        field_lines = "\n".join(f"    - `{f}`: numpy array of shape `{shape}`" for f in fields)
        schema = (f"- `fields`: a dict mapping each field name to its array —\n{field_lines}\n"
                  f'- `grid`: `{{{axis_list}}}`\n'
                  "- `t_final`: float — the time at which the solution is reported")
    else:
        schema = (f"- `numerical_solution`: numpy array of shape `{shape}`\n"
                  f'- `grid`: `{{{axis_list}}}`\n'
                  "- `t_final`: float — the time at which the solution is reported")

    t_eval = problem["t_eval"]
    when = ("This is a steady (time-independent) problem."
            if t_eval is None else
            f"Report the solution at t = {t_eval:.6g} (set `t_final` accordingly).")
    grid_N = V._pde_grid_N(problem)
    order_check = V._pde_order_check(problem)
    min_order = V._pde_min_order(problem)

    # Scoring lines, assembled from the pieces that apply to this problem.
    required = problem.get("required_fields") or (fields or None)
    acc_target = (f"{metric_name} error < {tol * 100:g}%"
                  + (f" for {', '.join('`' + f + '`' for f in required)}" if required and fields else ""))
    if order_check:
        calls = (f"The harness imports `solve_pde` and calls it at two resolutions, "
                 f"**N = {grid_N}** and **N = {2 * grid_N}**.")
        bullets = [
            f"- **accuracy** — {acc_target} at the finer grid (N = {2 * grid_N});",
            f"- **convergence** — the observed order p = log2(err_N / err_2N) on the "
            f"{'primary field' if fields else 'solution'} must be at least **{min_order:g}**. "
            "Refining the grid must actually reduce the error at the expected rate, so a "
            "scheme tuned to one grid — or one that hard-codes an answer — does not pass.",
        ]
    else:
        calls = f"The harness imports `solve_pde` and calls it at **N = {grid_N}**."
        bullets = [f"- **accuracy** — {acc_target} against the hidden reference."]

    if problem.get("gauge_fields"):
        g = ", ".join("`" + f + "`" for f in problem["gauge_fields"])
        bullets.append(f"- the field(s) {g} are compared after subtracting the mean "
                       "(defined only up to a constant).")
    gates = [d for d in problem.get("diagnostics", ()) if d.get("gate")]
    for d in gates:
        bullets.append(f"- **structural constraint (hard gate)** — `{d['name']}`: the "
                       "returned solution must satisfy this or it fails as a "
                       "CONSTRAINT_VIOLATION regardless of accuracy.")
    if problem.get("domain_mask") is not None:
        bullets.append("- the domain is **not the full rectangle**: return values on the "
                       "whole grid, but only the in-domain nodes are scored.")
    scoring = ("Scoring is independent of any self-report and compares against a hidden "
               "reference solution on the grid you return:\n" + "\n".join(bullets))

    libs = ("Use numpy, `scipy.sparse`, and `scipy.sparse.linalg` (and the Python standard "
            "library); no other third-party libraries.")
    return f"""

---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_pde(N: int) -> dict

`N` is the number of grid points per spatial dimension; build your mesh from it
(e.g. `x = np.linspace(a, b, N)`). {calls} It must return a dict with:

{schema}

{when} {scoring}

Requirements:
- Use `np.linspace` for spatial grids (include the boundary points) and honor the `N` you are given.
- Keep `solve_pde` importable — no work at module load outside `if __name__ == "__main__"`.
- {libs}
"""


def write_problem_md(problem, workspace_root):
    """Write workspace/{slug}/problem.md. Returns the path written."""
    wdir = workspace_dir(problem["slug"], workspace_root)
    os.makedirs(wdir, exist_ok=True)
    path = os.path.join(wdir, "problem.md")
    with open(path, "w") as fh:
        fh.write(problem["description"] + contract_footer(problem))
    return path


def manifest_entry(problem):
    """A JSON-serializable summary of one problem (drops the callables)."""
    kind = problem.get("ground_truth_kind", "exact")
    entry = {
        "id": problem["id"],
        "slug": problem["slug"],
        "type": problem["type"],
        "tier": problem["tier"],
        "family": problem["family"],
        "title": problem["title"],
        "challenge": problem["challenge"],
        "has_ground_truth": problem["has_ground_truth"],
        "ground_truth_kind": kind,
        "discriminator": bool(problem.get("discriminator")),
    }
    if problem["type"] == "sde":
        entry.update(
            state_dimension=problem["state_dimension"],
            T=problem["T"],
            dt=problem["dt"],
            eval_dt=V.sde_verify_dt(problem),  # exact step solve_sde is called with
            num_paths=problem["num_paths"],
            seed=problem["seed"],
        )
        if kind == "reference":
            gt = problem["moments"](problem["T"])  # {key: (value, standard_error)}
            entry["ground_truth_moments"] = {k: float(v[0]) for k, v in gt.items()}
            entry["ground_truth_se"] = {k: float(v[1]) for k, v in gt.items()}
        elif kind == "stability":
            entry["stability_check"] = problem.get("stability_check", {"type": "finite"})
        elif problem["has_ground_truth"]:
            entry["ground_truth_moments"] = {
                k: float(v) for k, v in problem["moments"](problem["T"]).items()
            }
    else:
        entry.update(dims=problem["dims"], domain=problem["domain"], t_eval=problem["t_eval"],
                     grid_N=V._pde_grid_N(problem), order_check=V._pde_order_check(problem),
                     min_order=V._pde_min_order(problem), metric=problem.get("pde_metric", "l2"),
                     l2_tol=V._pde_l2_tol(problem), masked_domain=problem.get("domain_mask") is not None)
        if problem.get("fields"):
            entry["fields"] = list(problem["fields"])
            entry["primary_field"] = problem.get("primary_field", problem["fields"][0])
            entry["required_fields"] = problem.get("required_fields") or list(problem["fields"])
            if problem.get("gauge_fields"):
                entry["gauge_fields"] = list(problem["gauge_fields"])
        diags = problem.get("diagnostics")
        if diags:
            entry["diagnostics"] = [{"name": d["name"], "gate": bool(d.get("gate"))} for d in diags]
    return entry


def write_manifest(problem_list, path=MANIFEST_PATH):
    manifest = {
        "n_total": len(problem_list),
        "n_pde": sum(1 for p in problem_list if p["type"] == "pde"),
        "n_sde": sum(1 for p in problem_list if p["type"] == "sde"),
        "pass_thresholds": {
            "pde": {"rel_l2_err_max": 0.01},
            "sde": {"variance_rel_err_max": 0.10, "mean_rel_err_max": 0.05,
                    "near_zero_mean_threshold": 0.01},
        },
        "problems": [manifest_entry(p) for p in problem_list],
    }
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    return path


def select(only):
    if not only:
        return list(P.PROBLEMS)
    wanted = {s.strip() for s in only.split(",") if s.strip()}
    chosen = [p for p in P.PROBLEMS if p["slug"] in wanted or p["id"] in wanted]
    missing = wanted - {p["slug"] for p in chosen} - {p["id"] for p in chosen}
    if missing:
        raise SystemExit(f"unknown problem selector(s): {sorted(missing)}")
    return chosen


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", default="", help="comma-separated slugs or ids to generate")
    ap.add_argument("--workspace", default=os.path.join(REPO_ROOT, "workspace"),
                    help="workspace root (default: <repo>/workspace)")
    ap.add_argument("--list", action="store_true", help="print inventory and exit")
    args = ap.parse_args(argv)

    chosen = select(args.only)

    if args.list:
        for p in chosen:
            gt = "gt" if p["has_ground_truth"] else "no-gt"
            print(f"{p['id']}  T{p['tier']}  {p['type']}  {p['slug']:<32} [{gt}] {p['title']}")
        return

    for p in chosen:
        path = write_problem_md(p, args.workspace)
        print(f"wrote {os.path.relpath(path, REPO_ROOT)}")

    # The manifest always describes the full benchmark, not just the selection.
    mpath = write_manifest(list(P.PROBLEMS))
    print(f"wrote {os.path.relpath(mpath, REPO_ROOT)}  ({len(P.PROBLEMS)} problems)")


if __name__ == "__main__":
    main()
