"""Check the numbers quoted in plan-sciml-benchmark.md §3f/§9a/§10a–10g against the data.

A plan that records measured findings in prose is only as good as the prose matching the
artifacts, and prose drifts silently: a table edited after a re-run, a figure rounded
twice, a count that was true of an earlier sweep. This re-derives each quoted number from
`comparators.json`, the two baseline result files and **all three** step-10 results files, and
checks that the plan still says it. A few claims read a plan's `SOLUTION.md` directly,
because the per-plan record carries only a subset of the `<metrics>` fields.

It is not a test (it reads gitignored result files, so it cannot run in CI) and it is not
a substitute for one. It is the thing to run before trusting a number quoted out of that
document.

Run:  uv run python docs/sciml-benchmark-data/audit_plan_claims.py
Exits non-zero if any claim no longer matches.
"""

import json
import pathlib
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "plan-sciml-benchmark.md"
COMP = ROOT / "benchmark" / "references" / "sciml" / "comparators.json"
BASE = ROOT / "benchmark" / "results" / "baselines"
PROBE = ROOT / "benchmark" / "results" / "pipeline" / "sciml_probe.json"
#: The remaining five of step 10, run into their own results file so the four
#: earlier records are not merged over (§10d).
REST = (ROOT / "benchmark" / "results" / "pipeline" / "sciml_rest"
        / "sciml_probe_rest.json")
#: M09's third attempt, again in its own file (§10e). The `RUN_ERROR` row for the
#: same slug in :data:`PROBE` is the first attempt and is superseded by this one.
M09R = (ROOT / "benchmark" / "results" / "pipeline" / "sciml_m09"
        / "sciml_probe_m09.json")
KS = (ROOT / "docs" / "sciml-benchmark-data" / "measurements"
      / "ks_gate_resolution.json")

if str(ROOT) not in sys.path:  # this file lives two levels down from the repo root
    sys.path.insert(0, str(ROOT))


def _load(path):
    if not path.exists():
        print(f"missing (gitignored result file?): {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def _degenerate(doc):
    return sum(1 for r in doc["results"]
               if (((r.get("verify") or {}).get("pathwise_ladder") or {}).get("degenerate")))


def _field(slug, name):
    """One ``<metrics>`` field across every plan of a problem, in plan order."""
    from verifylib.review import parse_metrics
    out = []
    for plan in sorted((ROOT / "workspace" / slug / "plans").iterdir()):
        sol = plan / "SOLUTION.md"
        if sol.exists():
            block = parse_metrics(sol.read_text(encoding="utf-8")) or {}
            if name in block:
                out.append(block[name])
    return out


def _headrooms(slug):
    """Every plan's ``invariant_worst_headroom`` on one problem (\u00a710d).

    Read from each ``SOLUTION.md`` rather than from the result record: the record's
    per-plan ``kernel`` block carries only a subset of the metrics, and this field
    is not in it.
    """
    from verifylib.review import parse_metrics
    out = []
    for plan in sorted((ROOT / "workspace" / slug / "plans").iterdir()):
        sol = plan / "SOLUTION.md"
        if sol.exists():
            block = parse_metrics(sol.read_text(encoding="utf-8")) or {}
            if "invariant_worst_headroom" in block:
                out.append(float(block["invariant_worst_headroom"]))
    return out


def _family(plan_dir):
    """``(scheme_family, source, declared)`` as the kernel resolves it (§10d).

    Read through ``plan_meta`` rather than by grepping the file, because the finding
    is precisely that a *blank* front-matter field and an inferred one are different
    things with the same spelling in the ``<metrics>`` block.
    """
    from verifylib.kernel import plan_meta
    meta = plan_meta.read(str(plan_dir))
    return (meta["scheme_family"], meta["scheme_family_source"], meta["declared"])


def _bar_ratios():
    """Each buildable law's bootstrap critical value as a multiple of the asymptotic
    Kolmogorov one, at n = 50 000 (§10g). Calls the shipped code, so a change to the
    bar or to B shows up here rather than in a stale table."""
    import numpy as np

    from verifylib.kernel.sde import laws

    n = 50_000
    asymptotic = laws.KS_C95 / np.sqrt(n)
    cir = {"law": "noncentral_chi2",
           "scale": "sigma**2*(1-np.exp(-kappa*T))/(4*kappa)",
           "df": "4*kappa*theta/sigma**2",
           "nc": "X_0*np.exp(-kappa*T)/(sigma**2*(1-np.exp(-kappa*T))/(4*kappa))"}
    out = {}
    for name, spec, params in (
            ("normal", {"law": "normal", "mean": 0.0, "variance": 1.0}, {}),
            ("mild_lognormal", {"law": "lognormal", "log_mean": 0.0,
                                "log_variance": 0.09}, {}),
            ("feller_ok_cir", cir,
             {"X_0": 0.5, "kappa": 1.0, "theta": 1.0, "sigma": 1.0, "T": 1.0}),
            ("m01_lognormal", {"law": "lognormal", "log_mean": -0.0616,
                               "log_variance": 0.7569}, {}),
            ("feller_violating_cir", cir,
             {"X_0": 0.5, "kappa": 1.0, "theta": 0.5, "sigma": 2.0, "T": 1.0})):
        law, _ = laws.build(spec, params)
        out[name] = laws.bootstrap_critical(law, n) / asymptotic
    return out


def _terminal_law_specs():
    """Which live sciml specs declare a ``terminal_law`` at all -- the set the §10g
    fix can possibly move."""
    out = []
    for spec in sorted((ROOT / "workspace").glob("sde_sciml_*/problem_spec.json")):
        doc = json.loads(spec.read_text())
        if (doc.get("verification") or {}).get("terminal_law"):
            out.append(spec.parent.name)
    return out


def _m02_regraded_scores():
    return [json.loads((p / "evaluation.json").read_text()).get("score")
            for p in sorted((ROOT / "workspace" / "sde_sciml_additive"
                             / "plans").iterdir())
            if (p / "evaluation.json").exists()]


def _lilliefors_agreement():
    """How close the bootstrap's normal-law bar is to the published 0.886/sqrt(n),
    per sample size -- the one check on this bar that comes from outside the repo."""
    import numpy as np

    from verifylib.kernel.sde import laws

    law, _ = laws.build({"law": "normal", "mean": 0.0, "variance": 1.0}, {})
    return {n: laws.bootstrap_critical(law, n) / (0.886 / np.sqrt(n))
            for n in (5_000, 20_000, 50_000)}


def _m01_regraded():
    """M01's three plans as the kernel scores them **now**, best-plan-first by the
    harness's measured error (§10g's re-grade table)."""
    _, ordered = _m01_plans_by_error()
    out = []
    for plan in ordered:
        doc = json.loads(((ROOT / "workspace" / "sde_sciml_linear_ito" / "plans"
                           / plan["name"] / "evaluation.json").read_text()))
        out.append((plan["name"], doc.get("score"),
                    (doc.get("checks") or {}).get("terminal_law_ok")))
    return out


def _decl_order_best():
    """``(sciml best-on, sciml n, holdout best-on, holdout n)`` for the plan that
    comes first in declaration order -- the key a resolution floor falls through
    to, and the reason the floor's sign flips between corpora (§10f)."""
    from benchmark import rank_study

    out = []
    for paths in (rank_study.SCIML, rank_study.HOLDOUT):
        best_on = total = 0
        for pool in rank_study.pools(paths):
            graded = sorted((p for p in pool["plans"] if p["error"] is not None),
                            key=lambda p: p["error"])
            first = min(pool["plans"], key=lambda p: p["name"])
            if first["error"] is None:
                continue
            total += 1
            best_on += graded[0]["name"] == first["name"]
        out += [best_on, total]
    return tuple(out)


def _decl_order_mean_rank():
    """``(sciml, holdout)`` mean rank of the declaration-order first plan among the
    graded plans of its pool, 1 being the best (§10f)."""
    from benchmark import rank_study

    out = []
    for paths in (rank_study.SCIML, rank_study.HOLDOUT):
        ranks = []
        for pool in rank_study.pools(paths):
            graded = sorted((p for p in pool["plans"] if p["error"] is not None),
                            key=lambda p: p["error"])
            first = min(pool["plans"], key=lambda p: p["name"])
            if first["error"] is None:
                continue
            ranks.append([p["name"] for p in graded].index(first["name"]) + 1)
        out.append(sum(ranks) / len(ranks))
    return tuple(out)


def _m01_plans_by_error():
    from benchmark import rank_study

    pool, = [p for p in rank_study.pools(rank_study.SCIML) if p["id"] == "M01"]
    return pool, sorted((p for p in pool["plans"] if p["error"] is not None),
                        key=lambda p: p["error"])


def _m01_scores():
    """M01's scores, best plan first: the evaluator scored the two better plans 8
    and the worst one 10, which is what `score` then ranks on (§10f)."""
    _, ordered = _m01_plans_by_error()
    return [p["summary"].get("score") for p in ordered]


def _m01_indices():
    """M01's three quality indices, best plan first -- the counterfactual behind
    10f's claim that the index ranks M01 right and ``score`` overrides it."""
    from verifylib import ranking

    pool, ordered = _m01_plans_by_error()
    sums = [p["summary"] for p in pool["plans"]]
    tol = ranking._tol_of(sums[0], sums[0])
    keys = ranking.comparable_keys(tol, sums)
    return [ranking.sqi(p["summary"], ranking._tol_of(p["summary"], p["summary"]),
                        sums, keys=keys)[0] for p in ordered]


def _cost(doc):
    return sum(float((r.get("run") or {}).get("cost_usd") or 0) for r in doc["results"])


def _study(variant, floor=0.0, holdout=False, source=None):
    """One row of §10f's variant table, re-derived by running the study (§10f).

    Imported rather than shelled out to, so a signature change here is a failure
    rather than a silently different number.
    """
    from benchmark import rank_study

    paths = rank_study.HOLDOUT if holdout else rank_study.SCIML
    base = None if holdout else rank_study.c0_errors()
    # Same variant -> (source, breadth) mapping `rank_study.main` uses. Without it a
    # `regraded` claim silently reads the `<metrics>` blocks and measures `current`.
    breadth = 0
    name = variant
    if ":" in variant and not variant.startswith("floor:"):
        name, n = variant.split(":", 1)
        breadth = int(n)
        name = "abstain" if name == "thin" else name
    # `main()`'s mapping exactly, and no more: SS10f's variant table was measured
    # against the scorer that *shipped*, so `abstain`/`discriminating`/`floor`
    # default to `metrics`. SS10h's rows cross those variants with the re-graded
    # scorer and pass `source="evaluation"` to say so.
    src = source or ("evaluation" if variant in ("regraded", "whitelisted")
                     else "metrics")
    pool_list = rank_study.pools(
        paths, truncated=(variant == "truncated"), source=src)
    if variant == "whitelisted":
        pool_list = rank_study._restrict_to_record(pool_list)
    rows = rank_study.study(pool_list, name, floor, base, breadth)
    return rank_study.summarise(rows), rows


def _fallback_fires():
    """`(fallback fires, shared set of one, only one component varies)` over all 18
    recorded pools (§10h).

    The three are different numbers and the plan's table keeps them apart. The
    fallback needs an **empty** shared set, which never happens. A shared set of
    *one* is M05 alone. M02 shares two components but only one of them *varies*
    across the pool, which is the one-component vote by another route -- and the
    reason `discriminating` and `thin:K` are separate variants.
    """
    from benchmark import rank_study
    from verifylib import ranking

    empty, one, onevary = 0, [], []
    for paths in (rank_study.SCIML, rank_study.HOLDOUT):
        for pool in rank_study.pools(paths, source="evaluation"):
            sums = [q["summary"] for q in pool["plans"] if q["summary"]]
            if len(sums) < 2:
                continue
            tol = ranking._tol_of(sums[0], sums[0])
            parts = [ranking.components(q, tol, sums) for q in sums]
            shared = [k for k in ranking.CERT_COMPONENTS
                      if all(q.get(k) is not None for q in parts)]
            varying = [k for k in shared
                       if max(q[k] for q in parts) - min(q[k] for q in parts) > 1e-12]
            empty += not shared
            if len(shared) == 1:
                one.append(pool["id"])
            if len(varying) == 1:
                onevary.append(pool["id"])
    return empty, sorted(one), sorted(onevary)


def _decl_order_stats():
    """The first-declared plan's rank by harness error, per suite (§10h)."""
    from benchmark import rank_study

    out = {}
    for label, paths in (("sciml", rank_study.SCIML),
                         ("holdout", rank_study.HOLDOUT)):
        ranks = []
        for pool in rank_study.pools(paths, source="evaluation"):
            byerr = sorted([q for q in pool["plans"] if q["error"]],
                           key=lambda q: q["error"])
            first = pool["plans"][0]["name"]
            r = next((i + 1 for i, q in enumerate(byerr) if q["name"] == first), None)
            if r is not None:
                ranks.append((r, len(byerr)))
        out[label] = (round(statistics.mean(r for r, _ in ranks), 2),
                      sum(1 for r, _ in ranks if r == 1),
                      sum(1 for r, n in ranks if r == n))
    return out


def _record_is_lossless():
    """Does a re-rank from the record reproduce one from the full block, row for
    row, on both suites (§10h)? This is the item-3 fix's acceptance line."""
    for holdout in (False, True):
        full = _study("regraded", holdout=holdout)[1]
        cut = _study("whitelisted", holdout=holdout)[1]
        strip = [[{k: v for k, v in r.items() if k != "source"} for r in rows]
                 for rows in (full, cut)]
        if strip[0] != strip[1]:
            return False
    return True


def _whitelist_census():
    """`(fields ranking.py reads that the record drops, whitelist size)` (§10h)."""
    import re

    from benchmark import run

    src = (ROOT / "verifylib" / "ranking.py").read_text(encoding="utf-8")
    read = set(re.findall(r'summary\.get\("([A-Za-z_0-9]+)"', src))
    return sorted(read - set(run.RECORDED_METRICS)), len(run.RECORDED_METRICS)


def _dynkin_bars():
    """The `dynkin_bar` values the pools carry, against the 2.43 fallback (§10h)."""
    from benchmark import rank_study
    from verifylib import ranking

    seen = set()
    for paths in (rank_study.SCIML, rank_study.HOLDOUT):
        for pool in rank_study.pools(paths, source="evaluation"):
            for q in pool["plans"]:
                v = ranking._num((q["summary"] or {}).get("dynkin_bar"))
                if v is not None:
                    seen.add(round(v, 4))
    return sorted(seen)


#: `run.RECORDED_METRICS` as it stood before SS10h, verbatim from `git show
#: HEAD~1:benchmark/run.py`. Hardcoded because it is the *historical* fact the
#: fabrication claim is about: cutting to the current whitelist now correctly
#: abstains, which is the fix, not the defect.
PRE_10H_WHITELIST = (
    "score", "certified", "provenance", "estimated_rel_error",
    "estimated_rel_error_T", "error_horizon", "d1_outcome", "reference_outcome",
    "design_order", "invariant_max_drift", "dynkin_z", "solve_time_s",
    "functional_name", "functional_rel_tol", "functional_rel_error",
    "functional_bias_rel", "functional_se_rel", "functional_limited_by",
    "solver_sha256")


def _whitelist():
    """`run.RECORDED_METRICS` as it stands now."""
    from benchmark import run

    return run.RECORDED_METRICS


def _m05_accuracy(whitelist):
    """M05's accuracy component with each plan's summary cut to `whitelist` (§10h).

    `None` means the kernel's abstention survived the cut. A number means the cut
    manufactured one: `uncertainty()` lost its inputs, so `noise_limited()` returned
    False rather than "unknown" and accuracy fell through to `headroom_fraction`.
    """
    from benchmark import rank_study
    from verifylib import ranking

    pool = next(q for q in rank_study.pools(rank_study.SCIML, source="evaluation")
                if q["id"] == "M05")
    plans = [(q["name"], {k: v for k, v in (q["summary"] or {}).items()
                          if whitelist is None or k in whitelist})
             for q in pool["plans"]]
    sums = [sm for _, sm in plans]
    tol = ranking._tol_of(sums[0], sums[0])
    return [(name, ranking.accuracy(sm, tol)) for name, sm in plans]


def main():
    from verifylib import ranking

    plan = PLAN.read_text()
    comp, probe, ks = _load(COMP), _load(PROBE), _load(KS)
    rest = _load(REST)
    m09r_doc = _load(M09R)
    A = _load(BASE / "oneshot_sciml_scheme_matched.json")
    B = _load(BASE / "oneshot_sciml_unrestricted.json")
    if not all((comp, probe, A, B, ks, rest, m09r_doc)):
        print("cannot audit: re-run step 3, 9 or 10, or ks_gate_resolution.py, first")
        return 2

    rows = comp["rows"]
    pathwise = [r for r in rows if r["kind"] == "pathwise"]
    stages = {}
    for r in rows:
        stages.setdefault(r["scheme"], set()).add(r["stages"])
    checks_total = sum(c["checks_total"] for c in comp["analytic_cross_checks"])
    checks_pass = sum(c["checks_passed"] for c in comp["analytic_cross_checks"])
    worst = max(c["worst_rel"] for c in comp["analytic_cross_checks"])
    m01 = next(r for r in probe["results"] if r["id"] == "M01")
    hbp = m01["harness_by_plan"]
    em = hbp["plans"]["1-euler-maruyama"]["harness_error"]
    srk = hbp["plans"]["3-heun-srk"]["harness_error"]
    wall_min = m01["run"]["wall_seconds"] / 60
    kr, k42 = ks["readings"], ks["per_seed"]["42"]

    # The rev-7 probes. Both are read the same way as M01: the per-plan harness error
    # is the number the plan's tables quote, and the ratio of winner to best is the
    # claim, because that ratio is the defect.
    m06 = next(r for r in probe["results"] if r["id"] == "M06")
    m09 = next(r for r in probe["results"] if r["id"] == "M09")
    h6, h9 = m06["harness_by_plan"], m09["harness_by_plan"]
    m02 = next(r for r in probe["results"] if r["id"] == "M02")
    h2 = m02["harness_by_plan"]
    p2 = h2["plans"]

    # The rev-10 five. `by` indexes both result files together, because the defect
    # tallies in 10d are over every completed entry, not over one invocation.
    five = {r["id"]: r for r in rest["results"]}
    # M09 last, so the completed third attempt overrides the superseded RUN_ERROR
    # row the first attempt left in `probe` (§10e).
    by = {r["id"]: r
          for r in probe["results"] + rest["results"] + m09r_doc["results"]}
    m09r = by["M09"]
    h9r = m09r["harness_by_plan"]
    nc = ROOT / "workspace" / "sde_sciml_weak_noncommutative"
    m05 = five["M05"]
    h5 = m05["harness_by_plan"]
    nl = ROOT / "workspace" / "sde_sciml_nltest" / "plans"
    five_min = sum(r["run"]["wall_seconds"] for r in rest["results"]) / 60
    capped = [i for i, r in by.items()
              if (r["verify"] or {}).get("score_cap") == 3
              and r["verify"]["verified_score"] == 10
              and (r["verify"].get("ledger_violations") or [])]

    #: (label, the value re-derived from the data, a string the plan must contain)
    claims = [
        ("comparator rows", len(rows) == 279, "**279 rows**"),
        ("distinct schemes", len({r["scheme"] for r in rows}) == 19, "19 schemes"),
        ("every cell clean", {r["status"] for r in rows} == {"ok"}, "every cell `ok`"),
        ("pathwise rows", len(pathwise) == 255, "255 pathwise rows"),
        ("analytic cross-checks", checks_total == checks_pass == 72, "all 72 checks pass"),
        ("worst cross-check", f"{worst:.2e}" == "2.18e-16", "2.18e-16"),
        ("EM stages", stages["EM"] == {1.0}, "| EM, SimplifiedEM | 1 | 1 | **1** |"),
        ("RKMil stages", stages["RKMil"] == {2.0}, "| RKMil | 1 | 2 | **2** |"),
        ("SRI stages", stages["SRI"] == {10.0}, "`SRI` costs **10**"),
        ("weak-scheme stages", stages["DRI1"] == stages["RDI2WM"] == {9.0},
         "weak schemes cost **9**"),
        ("baseline units all pass",
         all(r["verdict"] == "PASS" for r in A["results"] + B["results"])
         and len(A["results"]) == len(B["results"]) == 9, "**18/18 PASS**"),
        ("degenerate ladders", (_degenerate(A), _degenerate(B)) == (0, 8),
         "**Degenerate ladders: 0 of 8 in the scheme-matched arm, 8 of 8 in the "
         "unrestricted one.**"),
        ("scheme-matched cost", f"{_cost(A):.2f}" == "0.96", "**$0.96**"),
        ("unrestricted cost", f"{_cost(B):.2f}" == "1.06", "**$1.06**"),
        ("probe verdict", m01["verdict"] == "VERIFIED_PASS",
         "`VERIFIED_PASS`"),
        ("probe wall clock", f"{wall_min:.1f}" == "19.6",
         "**19.6 minutes of wall clock**"),
        ("wrong plan won", hbp["wrong_plan_won"] is True, "**`wrong_plan_won: True`**"),
        ("the plan gap", f"{em / srk:.1f}" == "17.7", "17.7×"),
        ("the noise margin", f"{hbp['harness_error_margin']:.4f}" == "0.0048", "0.0048"),
        # Added after an over-broad line rewrite replaced this row of §2b's measured
        # table with a step-table row. The audit is the thing that caught it, so the
        # row it caught is now one of the claims.
        ("the §2b save-grid table", True, "| 10 | t = 0.30 | **3.5e-13** |"),
        ("the metric label",
         hbp["harness_error_metric"] == "relative strong error / its gate",
         "relative strong error / its gate"),
        # §10a's diagnosis. The three KS statistics are quoted from the run's own
        # evaluation.json files, which live under `workspace/` and every later run
        # overwrites; ks_gate_resolution.py reproduces them to 13 digits from the
        # parameters alone, so the reproduction is what the prose is held to.
        ("KS reproduces the run's EM cell", f"{k42['em']['ks']:.6f}" == "0.004602",
         "| **1-euler-maruyama** | 0.089% | 1.12% | 0.004602 |"),
        ("KS reproduces the run's Milstein cell",
         f"{k42['milstein']['ks']:.6f}" == "0.006141", "| 0.006141 |"),
        ("KS reproduces the run's Heun cell",
         f"{k42['heun']['ks']:.6f}" == "0.006754", "| 0.006754 |"),
        ("the critical value", f"{k42['em']['crit']:.6f}" == "0.006074", "0.006074"),
        ("Milstein's margin, as a fraction of the bar",
         f"{(k42['milstein']['ks'] / k42['em']['crit'] - 1):.3f}" == "0.011", "1.011×"),
        ("an exact solver fails the gate at seed 42",
         k42["exact"]["ks_ok"] is False and k42["exact"]["rel_err"] == 0.0
         and f"{k42['exact']['ks']:.6f}" == "0.007133", "**0.007133**"),
        ("...and worse than the EM that passed",
         k42["exact"]["ks"] > k42["em"]["ks"], "**FAIL, 1.174×**"),
        ("exact-solver failures over the seed sweep",
         (len(kr["exact_solver_gate_failures"]), kr["seeds"]) == (5, 9)
         and kr["exact_solver_gate_failures"] == [42, 3, 5, 7, 8],
         "**5 of 9** (42, 3, 5, 7, 8)"),
        ("the configuration is not systematic",
         kr["em_passes_while_better_fail"] == 1, "| **1/9** |"),
        ("KS ordering is near a coin flip",
         (kr["ks_order_matches_error_order"], kr["ks_order_reversed"]) == (6, 3),
         "| 6/9 |"),
        ("the between-scheme spread",
         f"{kr['between_scheme_spread_min']:.5f}-{kr['between_scheme_spread_max']:.5f}"
         == "0.00062-0.00154", "0.00062–0.00154"),
        ("the exact solver's own KS range",
         f"{kr['exact_ks_min']:.4f}-{kr['exact_ks_max']:.4f}" == "0.0035-0.0138",
         "0.0035–0.0138"),
        ("the law's kurtosis", f"{kr['law_kurtosis']:.2f}" == "50.65", "`κ = 50.65`"),
        ("the predicted divisor error",
         f"{kr['predicted_sd_rel_se']:.4f}" == "0.0158", "**1.58%**"),
        ("the correlation", f"{kr['corr_ks_vs_sd_error']:.2f}" == "0.70", "**+0.70**"),

        # --- rev 7: M06 -----------------------------------------------------
        ("M06 inverted too", h6["wrong_plan_won"] is True, "`wrong_plan_won: true`"),
        ("M06's winner was the first plan",
         h6["winner"] == "1-euler-maruyama", "**1-euler-maruyama** | `euler-maruyama`"),
        ("M06's best plan by harness",
         h6["best_plan_by_harness"] == "3-trapezoidal-drift-milstein",
         "`3-trapezoidal-drift-milstein`"),
        ("M06's inversion factor",
         f"{h6['winner_harness_err'] / h6['best_harness_err']:.1f}" == "5.1", "**5.1×**"),
        ("M06's margin", f"{h6['harness_error_margin']:.4f}" == "0.0111", "0.0111"),
        ("M06's three plans all scored 10",
         sorted(p["score"] for p in m06["pipeline"]["plans"].values()) == [10, 10, 10],
         "**10 / 10 / 10**"),
        ("M06's winner's error, over its gate",
         f"{h6['plans']['1-euler-maruyama']['harness_error']:.4f}" == "0.4955",
         "**0.4955**"),
        ("M06's best plan's error, over its gate",
         f"{h6['plans']['3-trapezoidal-drift-milstein']['harness_error']:.4f}" == "0.0968",
         "**0.0968**"),
        ("M06 verified and passed",
         m06["verdict"] == "VERIFIED_PASS" and m06["verify"]["passed"] is True,
         "`PASS (verified)`"),
        ("M06's strong error against its gate",
         f"{100 * m06['verify']['pathwise_rel_err']:.3f}" == "0.743"
         and f"{100 * m06['verify']['pathwise_tol_used']:g}" == "1.5",
         "strong error 0.743% against the 1.5%"),
        ("M06's wall clock", f"{m06['run']['wall_seconds'] / 60:.1f}" == "21.5",
         "21.5 min"),
        ("the two identical self-estimates M06's plans report",
         (h6["plans"]["1-euler-maruyama"]["kernel"]["estimated_rel_error"]
          == h6["plans"]["2-milstein"]["kernel"]["estimated_rel_error"] == 0.0143),
         "an *identical* self-estimate, 1.43e-2"),
        ("M06's uncapped ledger violation",
         m06["verify"].get("score_cap") == 3
         and m06["verify"]["verified_score"] == 10
         and any(f["path"] == "run_meta.num_paths"
                 for f in m06["verify"].get("ledger_violations") or []),
         "`score_cap: 3` — and grades **10 /"),

        # --- rev 7: M09 -----------------------------------------------------
        ("M09 inverted too", h9["wrong_plan_won"] is True, "`wrong_plan_won: true`"),
        ("M09's winner was the first plan",
         h9["winner"] == "1-euler-maruyama-antithetic",
         "**1-euler-maruyama-antithetic**"),
        ("M09's best plan by harness",
         h9["best_plan_by_harness"] == "2-exact-lognormal-integrating-factor",
         "`2-exact-lognormal-integrating-factor`"),
        ("M09's inversion factor",
         f"{h9['winner_harness_err'] / h9['best_harness_err']:.1f}" == "3.3", "**3.3×**"),
        ("M09's margin", f"{h9['harness_error_margin']:.4f}" == "0.1468", "0.1468"),
        ("M09's three plans all scored 8",
         sorted(p["score"] for p in m09["pipeline"]["plans"].values()) == [8, 8, 8],
         "**8 / 8 / 8**"),
        ("the harness grades M09's two losers 10 and its winner 8",
         (h9["plans"]["1-euler-maruyama-antithetic"]["harness"]["verified_score"] == 8
          and all(h9["plans"][k]["harness"]["verified_score"] == 10 for k in
                  ("2-exact-lognormal-integrating-factor", "3-weak-order-2-ito-taylor"))),
         "it shipped the only one of the three its own harness does not score 10"),
        ("M09's winner's error, over its gate",
         f"{h9['plans']['1-euler-maruyama-antithetic']['harness_error']:.4f}" == "0.5490",
         "**0.5490**"),
        ("M09 was cut by the timeout, and stamped",
         (m09["run"]["timed_out"] is True
          and f"{m09['run']['wall_seconds'] / 60:.1f}" == "60.0"
          and m09["run"]["graded_incomplete"] is True
          and m09["pipeline"]["phase"] == "running"),
         "timed out at 60.0 min, `graded_incomplete: true`"),
        ("M09's three functional gates",
         sorted(m09["verify"]["functional_tol_used"].values()) == [0.016, 0.02, 0.029],
         "gates of 0.020 / 0.016 /"),
        # --- rev 9: M02, the first run under the fix -------------------------
        ("M02 shipped the ranker's winner, not the first plan",
         (m02["pipeline"]["best_plan"] == "3-drift-implicit-trapezoidal"
          and m02["pipeline"].get("best_plan_source") == "state"),
         "`best_plan_source: state`"),
        ("M02's verdict and wall clock",
         (m02["verdict"] == "UNDERCLAIM"
          and f"{m02['run']['wall_seconds'] / 60:.1f}" == "13.0"),
         "**`UNDERCLAIM`,\n13.0 min"),
        ("M02's graded error against its gate",
         (f"{m02['verify']['pathwise_rel_err']:.2e}" == "1.34e-07"
          and f"{m02['verify']['pathwise_tol_used']:g}" == "0.0012"),
         "**1.34e-7** against a\n1.2e-3 gate"),
        ("M02's two good plans beat its Euler-Maruyama by these factors",
         (round(p2["1-euler-maruyama"]["harness_error"]
                / p2["2-stochastic-heun"]["harness_error"]) == 4879
          and round(p2["1-euler-maruyama"]["harness_error"]
                    / p2["3-drift-implicit-trapezoidal"]["harness_error"]) == 1980),
         "1980× and 4879×"),
        ("M02 inverted by this much, against this margin",
         (f"{h2['winner_harness_err'] / h2['best_harness_err']:.2f}" == "2.46"
          and f"{h2['harness_error_margin']:.2e}" == "2.49e-06"),
         "**2.46×** against a\nmargin of 2.49e-6"),
        ("M02's two best plans declare no scheme family at all",
         all(p2[k].get("scheme_family") is None
             for k in ("2-stochastic-heun", "3-drift-implicit-trapezoidal")),
         "declare **no `scheme_family` at all**"),
        ("defects 9 and 10 reproduced on M02, at a different path count",
         (m02["verify"].get("score_cap") == 3
          and m02["verify"]["verified_score"] == 10
          and any("20000" in f.get("message", "")
                  for f in m02["verify"].get("ledger_violations") or [])),
         "the spec asks for **20000**"),
        ("M09's final kernel re-run timed out",
         (m09.get("kernel") or {}).get("status") == "timeout",
         '`{"status": "timeout"}`'),
        # --- rev 10: the remaining five ---------------------------------------
        ("all five completed and none timed out",
         (len(rest["results"]) == 5
          and all(r["run"]["status"] == "completed" and not r["run"]["timed_out"]
                  for r in rest["results"])),
         "**All five completed, none timed out**"),
        ("the five cost this much wall clock",
         f"{five_min:.1f}" == "80.0", "**80.0 min** for five problems"),
        ("the five verdicts",
         ([five[i]["verdict"] for i in ("M03", "M04", "M05", "M07", "M08")]
          == ["VERIFIED_PASS", "UNDERCLAIM", "UNDERCLAIM",
              "VERIFIED_PASS", "UNDERCLAIM"]),
         "2 `VERIFIED_PASS`, 3 `UNDERCLAIM`"),
        ("no overclaim on any completed entry",
         not any(r["verdict"] == "OVERCLAIM" for r in by.values()),
         "**no overclaim anywhere**"),
        ("every one of the five reached Phase 3",
         all(five[i]["pipeline"].get("best_plan_source") == "state" for i in five),
         "`best_plan_source: state` on all\nfive"),
        ("M05 inverted by this much, against this margin",
         (f"{h5['winner_harness_err'] / h5['best_harness_err']:.0f}" == "622"
          and f"{(h5['winner_harness_err'] - h5['best_harness_err']) / h5['harness_error_margin']:.1f}"
          == "44.6"),
         "**622×**, and **44.6× outside**"),
        ("M05 shipped the milstein over the ito-taylor",
         (h5["winner"] == "2-milstein"
          and h5["best_plan_by_harness"] == "3-ito-taylor-1p5"),
         "`2-milstein` at **0.3406** of its gate over `3-ito-taylor-1p5`"),
        ("M05's order check never ran, on any plan",
         (_field("sde_sciml_nltest", "order_check") == ["not_run"] * 3
          and _field("sde_sciml_nltest", "resolved") == [False] * 3),
         "`order_check`\nis **`not_run`**"),
        ("M05's order-1.5 plan withheld its family and the kernel inferred the EM's",
         (_family(nl / "3-ito-taylor-1p5") == (None, "unknown", False)
          and _family(nl / "1-euler-maruyama")
          == ("euler-maruyama", "inferred", False)),
         "**`scheme_family_source: 'inferred'`, `declared: False`**"),
        ("M08's invariant is violated by every plan and still grades 10",
         (all(h < 0 for h in _headrooms("sde_sciml_stiffquad_mixed"))
          and five["M08"]["verify"]["verified_score"] == 10),
         "\u22120.7693 / \u22120.7726 / \u22120.7480"),
        ("defects 9 and 10 now stand at seven of nine completed entries",
         (len(capped) == 7 and "M05" not in capped and "M09" not in capped
          and sum(1 for r in by.values()
                  if r["run"]["status"] == "completed") == 9),
         "**seven of ~~eight~~ nine**"),
        # --- 10e: M09's third attempt ----------------------------------------
        ("M09 completed on the third attempt, and fast",
         (m09r["run"]["status"] == "completed"
          and not m09r["run"]["timed_out"]
          and f"{m09r['run']['wall_seconds']:.1f}" == "312.1"),
         "**The third attempt cost 312.1 s.**"),
        ("M09's verdict and winner",
         (m09r["verdict"] == "VERIFIED_PASS"
          and h9r["winner"] == "4-conditional-mc-rao-blackwell"
          and m09r["kernel"]["score"] == 10
          and m09r["kernel"]["certified"] is True),
         "winner `4-conditional-mc-rao-blackwell`, **score 10**, `certified: true`"),
        ("M09 is the first case where the ranker and the harness name different "
         "plans and the flag is correctly down",
         (h9r["wrong_plan_won"] is False
          and h9r["winner"] != h9r["best_plan_by_harness"]
          and h9r["best_plan_by_harness"] == "2-exact-lognormal-exponential"
          and f"{h9r['best_harness_err']:.4f}" == "0.0571"
          and f"{h9r['winner_harness_err']:.4f}" == "0.0590"
          and f"{h9r['harness_error_margin']:.4f}" == "0.0617"),
         "**0.0571** of gate\nagainst the winner's **0.0590**"),
        ("the winner's advantage is a real variance reduction",
         (f"{h9r['plans']['4-conditional-mc-rao-blackwell']['kernel']['functional_se_rel']:.4g}"
          == "0.0008152"
          and f"{h9r['plans']['2-exact-lognormal-exponential']['kernel']['functional_se_rel']:.4g}"
          == "0.002568"),
         "**8.152e-04** against\nplan 2's **2.568e-03**"),
        ("the winner is the pool's most expensive plan and cost did not decide",
         (max((p["kernel"]["solve_time_s"], n)
              for n, p in h9r["plans"].items())[1]
          == h9r["winner"]),
         "the pool's **most expensive** plan by 3.4\u00d7"),
        ("M09's order component is None on every plan although all three measured one",
         (all("observed_order" not in p["kernel"] for p in h9r["plans"].values())
          and (_field("sde_sciml_weak_noncommutative", "observed_order")
               == [0.9514, 0.5265, 0.5104])
          and (_field("sde_sciml_weak_noncommutative", "order_check")
               == [True, "unresolved", "unresolved"])),
         "the ranking fix's input is\nbeing truncated by the recorder"),
        ("and carrying it would have changed nothing, because two plans abstain",
         (ranking.order({"kind": "sde", "order_check": "unresolved",
                         "observed_order": 0.5265, "design_order": 0.5}) is None
          and ranking.order({"kind": "sde", "order_check": "true",
                             "observed_order": 0.9514, "design_order": 0.5}) == 1.0),
         "The defect is\ntherefore **latent, not active**"),
        ("M09's pool gate produced a fourth plan, and it won",
         ("exact_sampling_components: 1" in (nc / "STATE.md").read_text()
          and m09r["pipeline"]["n_plans"] == 4
          and (nc / "plans" / "4-conditional-mc-rao-blackwell").is_dir()),
         "the conductor **re-dispatched the plan-creator once**"),
        ("M09's conductor recorded a crash recovery",
         "recovery_note:" in (nc / "STATE.md").read_text(),
         "this is the first time in this corpus it has\nfired"),
        ("M09 ran at the spec's own path count, which is why it is not capped",
         (m09r["verify"]["run"]["num_paths"] == 100000
          and json.loads((nc / "problem_spec.json").read_text())
              ["evaluation_thresholds"]["num_paths"] == 100000),
         "matching `evaluation_thresholds.num_paths` *and*\n`graded_num_paths`"),
        ("the re-formulated M09 spec re-spelled the same coined field",
         (("exact_values_at_T"
           in json.loads((nc / "problem_spec.json").read_text())["analytic_moments"])
          and "terminal_moments_at_T" not in (nc / "problem_spec.json").read_text()),
         "**dropped that name entirely**"),
        # --- 10f: the offline re-rank ----------------------------------------
        ("10f's C0 comparison: shipped loses, the pool wins",
         (f"{_study('current')[0]['vs_c0']:.3f}" == "1.177"
          and f"{statistics.geometric_mean([r['best_err'] / r['c0'] for r in _study('current')[1] if r['c0']]):.3f}"
          == "0.377"),
         "**geomean 1.412\u00d7 worse** | **geomean 0.377\u00d7**"),
        ("10f's variant table, every row",
         all(f"{_study(v, f)[0]['geomean']:.3f}" == g
             and _study(v, f)[0]["picks_best"] == pb
             for v, f, g, pb in (("current", 0.0, "3.120", 5),
                                 ("abstain", 0.0, "3.120", 5),
                                 ("discriminating", 0.0, "3.120", 5),
                                 ("noscore", 0.0, "2.267", 6),
                                 ("truncated", 0.0, "14.023", 3))),
         "| **`current`** (with the rev-8 fix) | **5/9** | **3** | **3.120** |"),
        ("the one fix that crosses the line is the scorer's",
         (f"{_study('noscore')[0]['vs_c0']:.3f}" == "0.855"
          and _study("noscore")[0]["outside_margin"] == 2),
         "from **1.177\u00d7 worse than C0 to 0.855\u00d7\nbetter**"),
        ("the floor is harmful on sciml and helps on the holdout",
         (f"{_study('floor', 0.05)[0]['geomean']:.3f}" == "9.944"
          and f"{_study('floor', 0.05, holdout=True)[0]['geomean']:.3f}" == "1.264"
          and _study("floor", 0.05, holdout=True)[0]["picks_best"] == 8
          and f"{_study('abstain', 0.0, holdout=True)[0]['geomean']:.3f}" == "1.519"),
         "| 0.05 | 4/9 \u00b7 9.944 | **8/9 \u00b7 1.264** |"),
        ("the floor's sign flips because of declaration order, measured both ways",
         (_decl_order_best() == (0, 9, 5, 9)
          and [f"{r:.2f}" for r in _decl_order_mean_rank()] == ["2.89", "1.56"]),
         "best plan on 0 of 9 pools**, mean rank **2.89 of"),
        ("M01's index already ranks it correctly; score overrides it",
         ([f"{i:.4f}" for i in _m01_indices()] == ["0.6046", "0.5972", "0.5828"]
          and next(r["decided_by"] for r in _study("current")[1]
                   if r["id"] == "M01") == "score"
          and _m01_scores() == [8.0, 8.0, 10.0]),
         "the **index already ranks it correctly** (0.6046 / 0.5972 / 0.5828"),
        # --- 10g: the terminal-law bar ---------------------------------------
        ("the bootstrap bar tightens three laws and loosens the two it was "
         "mis-grading",
         (all(0.6 < _bar_ratios()[k] < 0.8
              for k in ("normal", "mild_lognormal", "feller_ok_cir"))
          and f"{_bar_ratios()['m01_lognormal']:.2f}" == "2.99"
          and f"{_bar_ratios()['feller_violating_cir']:.2f}" == "40.97"),
         "| normal | 0.00 | **0.66\u00d7** (tighter) |"),
        ("the normal-law bar reproduces Lilliefors' published constant",
         ([f"{v:.4f}" for v in _lilliefors_agreement().values()]
          == ["1.0069", "1.0203", "1.0041"]),
         "**1.0069, 1.0203 and 1.0041**"),
        ("M01 re-graded: the milstein goes 8 to 10 and the heun stays 8",
         (_m01_regraded() == [("3-heun-srk", 8, True),
                              ("2-milstein", 10, True),
                              ("1-euler-maruyama", 10, True)]),
         "| `2-milstein` | 0.006141 | **FAIL, 1.01\u00d7** | pass, **0.34\u00d7** | 8 | **10** |"),
        ("and that takes the arm past the C0 one-shot",
         (f"{_study('regraded')[0]['geomean']:.3f}" == "2.466"
          and f"{_study('regraded')[0]['vs_c0']:.3f}" == "0.931"
          and f"{next(r['regret'] for r in _study('regraded')[1] if r['id'] == 'M01'):.3f}"
          == "2.132"
          and next(r["decided_by"] for r in _study("regraded")[1]
                   if r["id"] == "M01") == "sqi"),
         "| **`regraded` (this fix)** | 5/9 | 3 | **2.466** | **0.931\u00d7 \u2014 better** |"),
        ("only M01 and M02 declare a terminal law, and M02 is unmoved",
         (_terminal_law_specs() == ["sde_sciml_additive", "sde_sciml_linear_ito"]
          and all(v == 9 for v in _m02_regraded_scores())),
         "0.52\u00d7 of the old, **0.786\u00d7** of the new"),
        # --- SS10h: the three ranker defects -------------------------------
        ("the comparable_keys fallback is dead code on all 18 recorded pools",
         (_fallback_fires() == (0, ["M05"], ["M02", "M05"])),
         "The fallback at `ranking.py:559` is **dead code on every pool ever run**"),
        ("abstaining is bit-identical to shipping, so item 1 is a measured no-op",
         ([{k: v for k, v in r.items() if k != "source"}
           for r in _study("abstain", source="evaluation")[1]]
          == [{k: v for k, v in r.items() if k != "source"}
              for r in _study("regraded")[1]]),
         "bit-identical to `regraded` on\n18/18 pools"),
        ("the thin gate is harmful, monotonically",
         (f"{_study('thin:2', source='evaluation')[0]['vs_c0']:.3f}" == "2.174"
          and f"{_study('thin:3', source='evaluation')[0]['vs_c0']:.3f}" == "3.501"
          and f"{_study('thin:2', source='evaluation')[0]['geomean']:.3f}"
          == "5.761"),
         "| `thin:2` | 5/9 | 5.761 | 2.174\u00d7 | 6/9 | 2.015 |"),
        ("declaration order is the worst plan on sciml and the best on holdout",
         (_decl_order_stats() == {"sciml": (2.89, 0, 8),
                                  "holdout": (1.56, 5, 1)}),
         "| sciml | 2.89 / 3 | 0/9 | **8/9** | `1-euler-maruyama`, all nine |"),
        ("ordertie wins on sciml and breaks holdout where its gate first fires",
         (_study("ordertie:2", source="evaluation")[0]["picks_best"] == 6
          and f"{_study('ordertie:2', source='evaluation')[0]['vs_c0']:.3f}"
          == "0.846"
          and _study("ordertie:3", holdout=True,
                     source="evaluation")[0]["picks_best"] == 2
          and f"{_study('ordertie:3', holdout=True, source='evaluation')[0]['geomean']:.3f}"
          == "4.247"),
         "| `ordertie:3` | 6/9 | 2.242 | 0.846\u00d7 | **2/9** | **4.247** |"),
        ("the record carried a wrong dynkin_bar, not an absent one",
         (_dynkin_bars() == [2.4284, 2.531, 2.8049]),
         "the nine\npools carry 2.4284, 2.5310 and 2.8049"),
        ("the old record fabricated M05's accuracy where the kernel abstained",
         (all(v is None for _, v in _m05_accuracy(None))
          and ([f"{v:.3f}" for _, v in _m05_accuracy(PRE_10H_WHITELIST)]
               == ["0.141", "0.008", "0.046"])
          # and the fix is that the current whitelist no longer does
          and all(v is None for _, v in _m05_accuracy(_whitelist()))),
         "scored three unresolved plans at 0.141 / 0.008 / 0.046"),
        ("the whitelist now carries every field the ranker reads",
         (_whitelist_census() == ([], 31)),
         "taking the whitelist from 19 to 31"),
        ("and the record is a lossless ranking input, both suites",
         (_record_is_lossless()
          and f"{_study('truncated')[0]['geomean']:.3f}" == "14.023"
          and f"{_study('truncated')[0]['vs_c0']:.3f}" == "5.291"
          and _study("whitelisted")[0]["picks_best"] == 5),
         "**bit-identical on all 18 pools across both suites**"),
        ("step 10 is complete at nine of nine",
         (sum(1 for r in by.values() if r["verdict"] == "VERIFIED_PASS") == 5
          and sum(1 for r in by.values() if r["verdict"] == "UNDERCLAIM") == 4
          and f"{sum(r['run']['wall_seconds'] for r in by.values()) / 60:.1f}"
          == "139.3"),
         "Verdicts: **5\n`VERIFIED_PASS`, 4 `UNDERCLAIM`, zero overclaims.**"),
    ]

    bad = 0
    for label, from_data, must_say in claims:
        says = must_say in plan
        if from_data and says:
            print(f"ok    {label}")
            continue
        bad += 1
        why = ("the data no longer supports it" if not from_data
               else f"the plan no longer says {must_say!r}")
        print(f"FAIL  {label}: {why}")
    print(f"\n{len(claims) - bad}/{len(claims)} plan claims verified against the data")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
