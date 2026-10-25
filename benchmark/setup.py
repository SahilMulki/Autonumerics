"""Generate benchmark workspaces from the curated problem set.

For each problem this writes ``workspace/{slug}/problem.md`` -- the single input
the Autonumerics pipeline needs -- and emits ``benchmark/manifest.json`` recording
the (JSON-serializable) metadata for every problem so the runner and report have a
stable record of what the benchmark contains.

Usage:
    uv run python benchmark/setup.py                 # stage the 49-problem main suite
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


def functional_gate(problem):
    """The sentence stating the tolerance on every quantity of interest.

    Shared by the PDE and SDE footers since step 5. BENCHOP poses each of its
    problems both ways and compares the two formulations, so the two statements
    have to be the same sentence rather than two paraphrases that drift: a solver
    told a different gate on the SDE formulation would be answering a different
    question."""
    functionals = problem["functional_truth"]
    ftol = V._functional_tol(problem)
    ftols = V._functional_tols(problem)
    names = ", ".join(f"`{k}`" for k in functionals)
    # When a reference value's own uncertainty has widened its gate, the statement
    # says so per value. A solver told only the tightest number would aim at a digit
    # the answer key does not have, and one told only the loosest would under-solve
    # the values that are known precisely.
    if len(set(ftols.values())) > 1:
        per = "; ".join(f"`{k}` within {ftols[k] * 100:g}%" for k in functionals)
        widened = [k for k in functionals
                   if ftols[k] > V._functional_tol(problem, k)]
        why = "" if not widened else (
            f" The gate on {', '.join(f'`{k}`' for k in widened)} is "
            f"{V.REFERENCE_TOL_FACTOR:g}x the reference value's own absolute "
            f"uncertainty, rather than a precision the reference does not have.")
        return (f"{names} must each be within their own relative tolerance of the "
                f"reference value — {per}.{why}")
    return f"{names} must be within {ftol * 100:g}% (relative) of the reference value."


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
        # A price, not a moment: the harness reads the number the solver reports,
        # exactly as on the PDE side, because recovering it from `terminal_paths`
        # would mean the harness choosing the payoff and the estimator and so
        # grading its own method rather than the solver's.
        sde_functionals = problem.get("functional_truth")
        if sde_functionals:
            keys = ", ".join(f'`"{k}"`' for k in sde_functionals)
            schema += (f"\n- `functionals`: a dict of scalar results, containing {keys} "
                       "— each a float")
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
            no_formula = (
                "This problem has no closed-form moments; the harness's ground truth is "
                "computed independently of any formula. "
                if problem.get("closed_form") is False else "")
            moments = (
                "the harness recomputes the terminal mean and variance at the arguments "
                f"above and compares them to its own ground truth (variance relative error "
                f"< {V.SDE_VAR_TOL * 100:g}%, and — unless the exact mean is within 0.01 of "
                f"zero — mean relative error < {V.SDE_MEAN_TOL * 100:g}%)"
                if problem.get("moments") else
                "this problem has no terminal-moment formula, so the quantity of interest "
                "below is the whole grade")
            scoring = (
                f"{no_formula}Scoring is independent of any self-report: {moments}."
            )
            if sde_functionals:
                n_lo = V._sde_level_paths(problem, problem["num_paths"])
                scoring += (
                    f"\n\n**Quantity of interest.** {functional_gate(problem)} It is read "
                    "from the `functionals` dict you return, and it is gated independently "
                    "of the moments: reporting a number without a simulation behind it, or "
                    "a simulation without the number, fails."
                    f"\n\n**Two path counts.** `solve_sde` is called a second time with "
                    f"`num_paths={n_lo}` and the same `dt`, `T` and `seed`, and the two "
                    "reported values of each quantity are compared. They must differ — two "
                    "Monte-Carlo estimates taken at different sample sizes do — and they "
                    "must agree to within the sampling noise the two counts imply. A value "
                    "that is identical at both counts is a constant, not an estimate, and "
                    "fails. Nothing here requires the finer run to be the more accurate "
                    "one; the check is symmetric.")
        # The non-functional rendering is byte-identical to what every SDE problem
        # statement said before step 5, deliberately: a problem statement is an agent
        # *input*, and churn in it makes a later run incomparable with an earlier one
        # for a reason that has nothing to do with the solver.
        once = "**once**" if not sde_functionals else "**twice**"
        second_call = "" if not sde_functionals else (
            f"\n\nand then again with `num_paths="
            f"{V._sde_level_paths(problem, problem['num_paths'])}`, everything else "
            "unchanged (see **Two path counts** below).")
        return f"""

---

## Solver contract (how this problem is run and scored)

Implement, in `solver.py`, exactly this function:

    def solve_sde(num_paths: int, dt: float, T: float, seed: int = 42) -> dict

It is imported and called {once}, with these exact arguments:

    solve_sde(num_paths={problem['num_paths']}, dt={dtv:.4g}, T={problem['T']:.6g}, seed={problem['seed']}){second_call}

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

    t_eval = problem["t_eval"]
    # A steady problem has no time to report the solution at. The line used to
    # read "the time at which the solution is reported" on every problem, and the
    # Schrödinger formulator dutifully ledgered "t_final on a steady problem" as an
    # ambiguity (findings §8).
    t_final_line = ("- `t_final`: float — the time at which the solution is reported"
                    if t_eval is not None else
                    "- `t_final`: `None` (steady problem; the key may be omitted)")
    if fields:
        field_lines = "\n".join(f"    - `{f}`: numpy array of shape `{shape}`" for f in fields)
        schema = (f"- `fields`: a dict mapping each field name to its array —\n{field_lines}\n"
                  f'- `grid`: `{{{axis_list}}}`\n'
                  f"{t_final_line}")
    else:
        schema = (f"- `numerical_solution`: numpy array of shape `{shape}`\n"
                  f'- `grid`: `{{{axis_list}}}`\n'
                  f"{t_final_line}")
    # A problem whose answer is partly a scalar quantity of interest (an eigenvalue, a
    # drag coefficient) asks for it explicitly: the harness reads the number the solver
    # reports rather than trying to recover it from the returned field, which for most
    # such quantities is a second numerical problem in its own right.
    functionals = problem.get("functional_truth")
    if functionals:
        keys = ", ".join(f'`"{k}"`' for k in functionals)
        schema += (f"\n- `functionals`: a dict of scalar results, containing {keys} "
                   "— each a float")

    when = ("This is a steady (time-independent) problem; there is no time to report "
            "the solution at."
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

    if functionals:
        bullets.append(f"- **quantity of interest** — {functional_gate(problem)} This gate is "
                       "independent of the field accuracy: reporting a number without a "
                       "solution behind it, or a solution without the number, fails.")
    if problem.get("normalize_fields"):
        bullets.append("- the returned field is compared **up to a nonzero scalar "
                       "multiple** (normalized and sign-aligned), so any normalization "
                       "and either sign is accepted.")
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
    # On a periodic domain the endpoint convention is the solver's choice and the
    # harness accepts either (its trig interpolation is tested on both), so say so
    # rather than let every formulator log the same ambiguity (KS R14, Burgers R8;
    # findings §8). Stated in the requirements line below.
    periodic = bool(problem.get("periodic")) or "periodic" in problem["description"].lower()
    grid_rule = (
        "Use `np.linspace` for spatial grids and honor the `N` you are given. The domain "
        "is **periodic**, and either endpoint convention is accepted and scored "
        "identically: `np.linspace(a, b, N)` (endpoint-inclusive, the wrap node stored "
        "twice, `u[-1] == u[0]`) or `np.linspace(a, b, N, endpoint=False)` (`N` distinct "
        "nodes). Whichever you return, return every other array (`snapshots`) on that "
        "same grid."
        if periodic else
        "Use `np.linspace` for spatial grids (include the boundary points) and honor the "
        "`N` you are given.")
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
- {grid_rule}
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
    # Only present when False: the no-closed-form suite is the exception, and a key
    # that appears on 50 unrelated entries reads as noise rather than as a flag.
    if problem.get("closed_form") is False:
        entry["closed_form"] = False
    # Same rule, one suite later [step 9]. `closed_form: False` let a reader of the
    # manifest pick the no-closed-form suite out; the externally anchored one was
    # invisible, so every one of its ten entries read as `main` -- the suite whose
    # baseline denominator they must never join. Written only where `problems.py`
    # names it, which leaves the other 56 entries byte-identical.
    if problem.get("suite"):
        entry["suite"] = problem["suite"]
    if problem.get("reference_error"):
        entry["reference_error"] = float(problem["reference_error"])
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
        elif problem["has_ground_truth"] and problem.get("moments"):
            # `has_ground_truth` stopped implying a moment formula in step 5: a
            # problem scored on a price has ground truth and nothing to put here.
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
    # Outside the type branch since step 5: a quantity of interest is a property of
    # the problem, not of the equation it is posed as, and BENCHOP states every one
    # of its problems both ways. Leaving it inside would have meant the manifest
    # described the PDE formulation of a price and said nothing about the SDE one.
    if problem.get("functional_truth"):
        entry["functional_truth"] = {k: float(v)
                                     for k, v in problem["functional_truth"].items()}
        entry["functional_tol"] = V._functional_tol(problem)
        ftols = V._functional_tols(problem)
        # Recorded when the reference's own uncertainty widened a gate, and also
        # when the gates were simply *declared* apart -- in the second case the
        # headline above is the loosest of them, so a reader given only that would
        # think every value was held to it.
        if (any(t > V._functional_tol(problem, n) for n, t in ftols.items())
                or len(set(ftols.values())) > 1):
            entry["functional_tol_used"] = ftols
            entry["reference_abs_err"] = {
                n: V._functional_reference_abs_err(problem, n) for n in ftols}
        if problem["type"] == "sde":
            entry["functional_level_paths"] = V._sde_level_paths(
                problem, problem["num_paths"])
    return entry


def write_manifest(problem_list, path=MANIFEST_PATH):
    by_suite = {}
    for p in problem_list:
        by_suite[problem_suite(p)] = by_suite.get(problem_suite(p), 0) + 1
    manifest = {
        "n_total": len(problem_list),
        "n_pde": sum(1 for p in problem_list if p["type"] == "pde"),
        "n_sde": sum(1 for p in problem_list if p["type"] == "sde"),
        # Each suite is scored against its own denominator, so the three counts are
        # part of what the manifest records rather than something a reader derives.
        "n_by_suite": {k: by_suite[k] for k in SUITES if k in by_suite},
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


SUITES = ("main", "no-closed-form", "benchop")


def problem_suite(problem):
    """Which suite a problem belongs to. ``suite`` is explicit where it exists;
    otherwise it is derived from ``closed_form``, which is how every problem
    authored before there was a third suite keeps the membership it had."""
    named = problem.get("suite")
    if named:
        return named
    return "no-closed-form" if problem.get("closed_form") is False else "main"


def apply_suite_filter(chosen, only, no_closed_form=False, include_all=False,
                       suite=None):
    """Keep the suites apart.

    Each suite carries its own C0/C1/C2 baseline, measured over its own
    denominator, so folding one into another silently would move every rate those
    baselines are quoted against. A suite therefore runs **on its own**: ``main``
    by default, another when asked for by ``--suite``, all of them with
    ``--all-problems``, or any single problem by naming it in ``--only`` -- an
    explicit request always wins, which is what makes a one-problem smoke test
    work. ``--no-closed-form`` is the older spelling of ``--suite no-closed-form``
    and still means exactly that."""
    if only or include_all:
        return chosen
    want = suite or ("no-closed-form" if no_closed_form else "main")
    if want not in SUITES:
        raise SystemExit(f"unknown suite {want!r}; choose from {', '.join(SUITES)}")
    return [p for p in chosen if problem_suite(p) == want]


def guard_results_path(chosen, results_json, suggestion):
    """Refuse to write a run that is not the main sweep into the standing results
    file [step 9].

    `results/README.md` records why: on 2026-09-27 a no-closed-form sweep launched
    with no `--results-json` overwrote the live baseline -- 44 units where there had
    been hundreds -- and the convention written afterwards ("always pass
    `--results-json` for anything that is not the main sweep") was a rule a person
    had to remember. Both runners merge by slug and replace `config` wholesale, so
    the damage is not only mixed records: the surviving ones end up described by the
    new invocation's config, and the suite denominators stop meaning anything.

    Selecting nothing outside `main` is the one case that needs no flag, which keeps
    every main-suite invocation, including a one-problem smoke test, as it was."""
    if results_json:
        return
    outside = sorted({problem_suite(p) for p in chosen} - {"main"})
    if not outside:
        return
    raise SystemExit(
        f"this selection includes the {' and '.join(outside)} "
        f"suite{'s' if len(outside) > 1 else ''}, which must not be "
        f"written into the standing main-suite results file.\n"
        f"Pass an explicit path, e.g. --results-json "
        f"{suggestion.format(suite=outside[0].replace('-', '_'))}\n"
        f"(benchmark/results/README.md, 'Naming conventions')")


def add_suite_arguments(ap):
    """The flags every runner needs, defined once so they cannot drift apart."""
    ap.add_argument("--suite", choices=SUITES, default=None,
                    help="which suite to run (default: main). Each has its own "
                         "baseline denominators, so they never run together unless "
                         "asked")
    ap.add_argument("--no-closed-form", dest="no_closed_form", action="store_true",
                    help="the older spelling of `--suite no-closed-form` (see "
                         "NO_CLOSED_FORM_CANDIDATES.md)")
    ap.add_argument("--all-problems", dest="include_all", action="store_true",
                    help="run every problem, from every suite, together")


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
