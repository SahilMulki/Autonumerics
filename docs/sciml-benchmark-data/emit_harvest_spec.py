"""Emit `benchmark/references/sciml/harvest_spec.json` from the shipped suite.

Why a generated spec rather than constants typed into `harvest.jl`: the comparator
table is only worth having if the Julia schemes are run on *the same problems, at the
same steps, for the same number of paths* as the nine benchmark entries. Typing those
numbers twice is how the two halves drift apart, and a drift of that kind would be
invisible in the output -- the table would still look like a comparison.

So the configuration (horizon, step grid, path count, seed, save grid, gate) is read
out of `sciml_problems()` and written once. What is *not* written is the coefficients:
those stay in `SDEProblemLibrary.jl`, which is their authority. The two halves are tied
together instead by `analytic_checks` -- sample points of this repo's closed forms that
`harvest.jl` evaluates the library's own `analytic` against. If the Python entry and the
Julia problem ever stop being the same problem, that check fails rather than the table
quietly becoming meaningless.

Run:  uv run python docs/sciml-benchmark-data/emit_harvest_spec.py
"""

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "benchmark"))

import problems as P  # noqa: E402

OUT = ROOT / "benchmark" / "references" / "sciml" / "harvest_spec.json"

#: Which published comparator set each entry belongs to. The names are resolved to
#: `StochasticDiffEq.jl` algorithms in `harvest.jl`; keeping them as labels here is
#: what stops this file from needing to know Julia.
SCHEME_SET = {
    "M01": "diagonal",
    "M02": "additive",
    "M03": "diagonal",
    "M04": "diagonal",
    "M05": "diagonal",
    "M06": "stiff",
    "M07": "stiff",
    "M08": "stiff",
    "M09": "weak",
}

#: The library binding each entry is built from, and whether `p` must be supplied.
#: `prob_sde_nltest` is unexported, hence the module-qualified name.
JULIA_PROBLEM = {
    "M01": "SDEProblemLibrary.prob_sde_linear",
    "M02": "SDEProblemLibrary.prob_sde_additive",
    "M03": "SDEProblemLibrary.prob_sde_cubic",
    "M04": "SDEProblemLibrary.prob_sde_wave",
    "M05": "SDEProblemLibrary.prob_sde_nltest",
    "M06": "SDEProblemLibrary.prob_sde_stiffquadito",
    "M07": "SDEProblemLibrary.prob_sde_stiffquadito",
    "M08": "SDEProblemLibrary.prob_sde_stiffquadito",
    # Not in the library at all: the weak page defines it inline, so `harvest.jl`
    # transcribes it from the notebook source instead (plan §2a.5).
    "M09": "noncommutative_inline",
}


def _analytic_checks(entry):
    """Sample points of this repo's closed form, for `harvest.jl` to check the
    library's `analytic` against.

    The points are deterministic and spread over the horizon and over a few sigma of
    the Brownian path, including `W = 0` (where a sign error in a `beta*W` term hides)
    and both signs (where one in a drift term does not).
    """
    fn = entry.get("pathwise_exact")
    if not callable(fn):
        return []
    T = float(entry["T"])
    out = []
    for frac in (0.25, 0.5, 1.0):
        t = T * frac
        for z in (-1.5, 0.0, 0.8):
            W = z * np.sqrt(t)
            x = float(np.asarray(fn(t, W), dtype=float).reshape(-1)[0])
            out.append({"t": t, "W": float(W), "x": x})
    return out


def _stiff_params(entry):
    """`(alpha, beta)` for the stiff triple, read off the `partial` the entry ships
    rather than re-typed -- the same reason the horizons are read and not re-typed."""
    fn = entry.get("pathwise_exact")
    kw = getattr(fn, "keywords", None) or {}
    if "alpha" in kw and "beta" in kw:
        return [float(kw["alpha"]), float(kw["beta"])]
    return None


def build():
    probs = []
    for e in P.sciml_problems():
        pid = e["id"]
        rec = {
            "id": pid,
            "slug": e["slug"],
            "sciml_key": e["sciml_key"],
            "julia_problem": JULIA_PROBLEM[pid],
            "scheme_set": SCHEME_SET[pid],
            "T": float(e["T"]),
            "state_dimension": int(e["state_dimension"]),
            "noise_dimension": int(e.get("noise_dimension") or e["state_dimension"]),
            "num_paths": int(e["num_paths"]),
            "seed": int(e["seed"]),
            "graded_dt": float(e["dt"]),
            "dts": [float(dt) for _, dt in e["work_precision_levels"]],
            "paths_per_level": [int(n) for n, _ in e["work_precision_levels"]],
        }
        sp = _stiff_params(e)
        if sp is not None:
            rec["p"] = sp
        if e.get("ground_truth_kind") == "pathwise":
            rec["kind"] = "pathwise"
            rec["gate"] = {"pathwise_tol": float(e["pathwise_tol"])}
            rec["save_points"] = int(e.get("pathwise_save_points") or 0)
            rec["analytic_checks"] = _analytic_checks(e)
        else:
            rec["kind"] = "weak"
            rec["gate"] = {"functional_tol": dict(e["functional_tol"])}
            rec["moment_truth"] = dict(e["functional_truth"])
            rec["observables"] = sorted(e["functional_truth"])
        probs.append(rec)
    return {
        "_comment": (
            "Generated by docs/sciml-benchmark-data/emit_harvest_spec.py from "
            "benchmark/problems.py sciml_problems(). Do not edit by hand; it is the "
            "input to benchmark/references/sciml/harvest.jl and the only thing that "
            "keeps the comparator runs on the same problems as the suite."
        ),
        "problems": probs,
    }


def main():
    spec = build()
    OUT.write_text(json.dumps(spec, indent=1, sort_keys=False) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(spec['problems'])} problems)")
    for p in spec["problems"]:
        n = len(p.get("analytic_checks", []))
        print(f"  {p['id']}  {p['scheme_set']:9s} T={p['T']:<8g} "
              f"paths={p['num_paths']:<7d} dts={len(p['dts'])}  checks={n}")


if __name__ == "__main__":
    main()
