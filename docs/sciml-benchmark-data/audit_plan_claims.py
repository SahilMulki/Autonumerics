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


#: The two plans the two-sided strong-order band failed, with the band's own numbers as
#: the pre-SS10k `SOLUTION.md` blocks recorded them. `cli.py evaluate` rewrites
#: `evaluation.json` and never `SOLUTION.md`, so the *before* state stays readable after
#: the fix -- the same reason SS10g's re-grade claims read the record rather than the
#: workspace.
OVERSHOT_PRE_10K = (("sde_sciml_linear_ito", "3-heun-srk"),
                    ("sde_sciml_weak_noncommutative", "1-euler-maruyama"))


def _overshoot_before():
    """``[(plan, observed, floor, band, score)]`` from each plan's `SOLUTION.md` -- the
    state the two-sided band left behind (SS10k)."""
    from verifylib.review import parse_metrics
    out = []
    for slug, plan in OVERSHOT_PRE_10K:
        sol = ROOT / "workspace" / slug / "plans" / plan / "SOLUTION.md"
        b = parse_metrics(sol.read_text(encoding="utf-8")) or {}
        out.append((plan, b.get("observed_order"), b.get("order_floor"),
                    b.get("order_band"), b.get("score")))
    return out


def _overshoot_after():
    """The same two plans as the kernel grades them **now** (SS10k)."""
    out = []
    for slug, plan in OVERSHOT_PRE_10K:
        doc = json.loads((ROOT / "workspace" / slug / "plans" / plan
                          / "evaluation.json").read_text())
        out.append((plan, doc.get("score"), (doc.get("checks") or {}).get("orders_ok")))
    return out


def _gate_is_one_sided():
    """``(overshoots_pass, overshoots_flagged, shortfall_verdict, undeclared_flag)``
    straight from `crn.guards`, on the two measured numbers (SS10k)."""
    from verifylib.kernel.sde import crn

    detail = {"s0": 1.0, "s1": 0.5}
    over = [crn.guards(o, 1.0, 0.5, 1.0, detail, declared=True, candidates=[0.5])
            for o in (0.9831, 0.9514)]
    short = crn.guards(0.504, 1.0, 1.0, 1.0, detail, declared=True, candidates=[1.0])
    undecl = crn.guards(2.0, 1.0, 1.0, 1.0, detail, declared=False, candidates=[1.0])
    return (all(g["orders_ok"] is True for g in over),
            all(g["strong_order_above_band"] is True for g in over),
            short["orders_ok"], undecl["strong_order_above_band"])


def _worst_component_is_one_sided():
    """Of two *passing* components the one with the **lower** order is reported, not the
    one furthest from the expectation on either side (SS10k)."""
    from verifylib.kernel.sde.driver import _worst_component

    near = {"orders_ok": True, "strong_order": 0.95, "component": 0}
    far = {"orders_ok": True, "strong_order": 1.4, "component": 1}
    return (_worst_component([near, far], 1.0)["component"],
            _worst_component([far, near], 1.0)["component"])


def _m09_published_order():
    """``(binding component, published observed_order)`` for M09's Euler-Maruyama, whose
    two components measure 0.5104 and 0.9514 (SS10k)."""
    doc = json.loads((ROOT / "workspace" / "sde_sciml_weak_noncommutative" / "plans"
                      / "1-euler-maruyama" / "evaluation.json").read_text())
    det = (doc.get("detail") or {}).get("orders_ok") or {}
    return det.get("component"), round(doc["summary"].get("observed_order"), 4)


def _m09_keys_exclude_order():
    """M09's comparable components under the re-graded scores. `order` must be absent:
    the Euler-Maruyama measures one and both exact samplers abstain, so SS10b's
    whole-pool rule is what stops this fix promoting the pool's worst plan (SS10k)."""
    from verifylib import ranking

    pool = []
    base = ROOT / "workspace" / "sde_sciml_weak_noncommutative" / "plans"
    for plan in sorted(base.iterdir()):
        ev = plan / "evaluation.json"
        if ev.exists():
            pool.append(json.loads(ev.read_text())["summary"])
    return ranking.comparable_keys(0.0, pool)


def _orders_reason(order, floor):
    """The order row's reason string for one (order, floor) pair (SS10k)."""
    from verifylib.kernel.score import _orders_fail_reason

    return _orders_fail_reason({"observed_order": order, "order_floor": floor})


#: M05's three plans. The §24 cost skip fired on exactly these -- see
#: `_not_run_census` -- and `SOLUTION.md` is not rewritten by `cli.py evaluate`, so the
#: pre-SS10l state stays readable there the same way SS10k's does.
M05 = ("sde_sciml_nltest", ("1-euler-maruyama", "2-milstein", "3-ito-taylor-1p5"))

#: What the harness independently measured for each M05 plan, as a fraction of its own
#: gate (`harness_by_plan`, SS10d). The pool's best is the order-1.5 plan, 622x better
#: than the one the ranker ships.
M05_HARNESS = {"1-euler-maruyama": 0.3563349406586618,
               "2-milstein": 0.3405645864862689,
               "3-ito-taylor-1p5": 0.0005473266994761092}


def _m05_blocks(source):
    """``[(plan, summary dict)]`` for M05, from `SOLUTION.md` (pre-SS10l) or
    `evaluation.json` (as the kernel grades it now)."""
    from verifylib.review import parse_metrics

    slug, plans = M05
    out = []
    for plan in plans:
        base = ROOT / "workspace" / slug / "plans" / plan
        if source == "solution":
            out.append((plan, parse_metrics(
                (base / "SOLUTION.md").read_text(encoding="utf-8")) or {}))
        else:
            doc = json.loads((base / "evaluation.json").read_text())
            summary = dict(doc["summary"])
            summary.setdefault("kind", doc.get("kind"))
            summary["score"] = doc.get("score")
            out.append((plan, summary))
    return out


def _m05_before():
    """``[(plan, order_check, observed_order, dynkin_z, score)]`` as the skip left it,
    plus the components the pool was actually ranked on (SS10l)."""
    from verifylib import ranking

    blocks = _m05_blocks("solution")
    rows = [(plan, b.get("order_check"), b.get("observed_order"), b.get("dynkin_z"),
             b.get("score")) for plan, b in blocks]
    return rows, ranking.comparable_keys(0.0, [b for _, b in blocks])


def _m05_after():
    """The same five fields and the same component set, re-graded (SS10l). The score
    must not move: Path A's `rubric_sde_analytic` puts `failed(resolved) -> 6` above
    every row that reads `orders_ok` or `dynkin_ok`."""
    from verifylib import ranking

    blocks = _m05_blocks("evaluation")
    rows = [(plan, b.get("order_check"), round(b["observed_order"], 4),
             round(b["dynkin_z"], 4), b.get("score")) for plan, b in blocks]
    return rows, ranking.comparable_keys(0.0, [b for _, b in blocks])


def _not_run_census():
    """``(plans reading `order_check: not_run`, total SDE plans)`` across every
    `SOLUTION.md` on disk -- the pre-SS10l census, so what SS24's cost skip ever saved
    is a number rather than an argument (SS10l)."""
    from verifylib.review import parse_metrics

    skipped, total = [], 0
    for sol in sorted((ROOT / "workspace").glob("sde_*/plans/*/SOLUTION.md")):
        block = parse_metrics(sol.read_text(encoding="utf-8")) or {}
        if not block:
            continue
        total += 1
        if block.get("order_check") == "not_run":
            skipped.append(sol.parent.name)
    return sorted(skipped), total


def _m05_order_is_informative_and_discarded():
    """``(raw ranking, harness ranking, component values)`` for M05 (SS10l).

    The raw `observed_order` ranks the pool exactly as the harness does, all three
    positions. `ranking.order` measures a *shortfall against a floor* and every plan
    clears its own, so it maps them to 1.0 / 0.998 / 1.0 -- the best plan tied with
    the worst.
    """
    from verifylib import ranking

    blocks = dict(_m05_blocks("evaluation"))
    by_order = sorted(blocks, key=lambda k: -blocks[k]["observed_order"])
    by_err = sorted(blocks, key=lambda k: M05_HARNESS[k])
    return (tuple(by_order), tuple(by_err),
            tuple(round(ranking.order(blocks[k]), 4) for k in M05[1]))


def _m05_structure_outweighs_order():
    """``(index gap to the wrong plan, structure's weighted lead over order)`` (SS10l).

    Both signed from the ranker's winner against the pool's best. `structure` is
    `invariant_worst_headroom`, which is anti-correlated with quality on this pool, and
    it outweighs the one component pointing the other way by a factor the ratio names.
    """
    from verifylib import ranking

    blocks = dict(_m05_blocks("evaluation"))
    keys = ranking.comparable_keys(0.0, list(blocks.values()))
    win, best = blocks["2-milstein"], blocks["3-ito-taylor-1p5"]
    total = sum(ranking.WEIGHTS[k] for k in keys)
    gap = {k: ranking.WEIGHTS[k] * (getattr(ranking, k)(win) - getattr(ranking, k)(best))
           / total for k in keys}
    index = (ranking.sqi(win, 0.0, keys=keys)[0] - ranking.sqi(best, 0.0, keys=keys)[0])
    return round(index, 4), round(gap["structure"] / -gap["order"], 1)


def _route_corpus_split():
    """``{suite: (pools, `order` comparable, accuracy resolved everywhere)}`` (SS10m).

    This is why all three of SS10l's routes leave the holdout bit-identical, and it is
    an instrumented fact rather than a hope: the two corpora are disjoint on exactly
    the two fields the routes read. Route (i) changes the `order` component and `order`
    is comparable on no holdout pool at all; route (ii) gates `structure` on accuracy
    resolution and accuracy resolves on every holdout pool.
    """
    from benchmark import rank_study
    from verifylib import ranking

    out = {}
    for tag, paths in (("sciml", rank_study.SCIML), ("holdout", rank_study.HOLDOUT)):
        pools = n_order = n_acc = 0
        for pool in rank_study.pools(paths, source="evaluation"):
            sums = [q["summary"] for q in pool["plans"] if q["summary"]]
            if len(sums) < 2:
                continue
            pools += 1
            tol = ranking._tol_of(sums[0], sums[0])
            n_order += "order" in ranking.comparable_keys(tol, sums)
            n_acc += bool(ranking.cost_admissible(tol, sums))
        out[tag] = (pools, n_order, n_acc)
    return out


def _relorder_guard_two_is_load_bearing():
    """``((best, M01's pick) with the guard, the same with it off)`` (SS10m).

    `relorder`'s resolution test, measured rather than argued. M01's top two orders are
    0.98647 +/- 0.0096 and 0.98312 +/- 0.0150 -- 0.0033 apart against a 2-sigma bar of
    0.0357 -- and the *lower* of them is that pool's best plan, so switching the guard
    off (`C = 0`) ranks a pair separated by a tenth of its own error bar, in the wrong
    direction, and undoes SS10k.
    """
    from benchmark import rank_study

    base = rank_study.c0_errors()
    pools = rank_study.pools(rank_study.SCIML, source="evaluation")
    out = []
    for ci in (2.0, 0.0):
        rows = rank_study.study(pools, "relorder", 0.0, base, 0, ci)
        summary = rank_study.summarise(rows)
        m01 = next(r for r in rows if r["id"] == "M01")
        out.append((summary["picks_best"], m01["top"], round(m01["regret"], 3)))
    return tuple(out)


def _relorder_ci_is_flat():
    """``[(C, picks_best, geomean)]`` across the sweep SS10m reports (SS10m).

    One free parameter, flat over a 65x range and degrading to the shipped behaviour at
    both ends rather than to something new -- which is what distinguishes it from the
    resolution floor SS10f fitted and withdrew.
    """
    from benchmark import rank_study

    base = rank_study.c0_errors()
    pools = rank_study.pools(rank_study.SCIML, source="evaluation")
    out = []
    for ci in (0.0, 0.2, 2.0, 13.0, 15.0):
        s = rank_study.summarise(rank_study.study(pools, "relorder", 0.0, base, 0, ci))
        out.append((ci, s["picks_best"], round(s["geomean"], 3)))
    return out


def _relorder_shortfall_guard_is_dead():
    """``(largest shortfall over all 18 pools, the gate's band)`` (SS10m).

    `relorder`'s first guard never fires: no plan on either corpus is short of its own
    floor by more than `crn.STRONG_BAND`, so only the second guard is shown to do work.
    Recorded in the same sense SS10h retired the `comparable_keys` fallback.
    """
    from benchmark import rank_study
    from verifylib import ranking
    from verifylib.kernel.sde.crn import STRONG_BAND

    worst = 0.0
    for paths in (rank_study.SCIML, rank_study.HOLDOUT):
        for pool in rank_study.pools(paths, source="evaluation"):
            for q in pool["plans"]:
                s = q["summary"] or {}
                v = ranking._num(s.get("observed_order"))
                f = ranking._num(s.get("order_floor"))
                if v is not None and f is not None:
                    worst = max(worst, max(0.0, f - v))
    return round(worst, 4), STRONG_BAND


def _relorder_flips_one_winner():
    """``[(pool, shipped top, relorder top)]`` for every pool whose winner moves (SS10m).

    The honest size of route (i): byte-identical to `regraded` on eight of the nine
    pools, so the geomean move is arithmetically M05's 622.233 -> 1.000 alone.
    """
    from benchmark import rank_study

    base = rank_study.c0_errors()
    pools = rank_study.pools(rank_study.SCIML, source="evaluation")
    ship = {r["id"]: r["top"] for r in rank_study.study(pools, "regraded", 0.0, base)}
    rel = rank_study.study(pools, "relorder", 0.0, base, 0, 2.0)
    return [(r["id"], ship[r["id"]], r["top"]) for r in rel if r["top"] != ship[r["id"]]]


def _m05_compare_agrees_at_every_path_count():
    """``[(num_paths, verdict, verdict)]`` for M05's two decisive pairs (SS10m).

    Route (iii). `cli.py compare` grades in **moment space**, and M05's 622x is
    pathwise, so more sampling turns `INCONCLUSIVE` into `AGREE` rather than into a
    separation. Twelve sandbox solves per row; this is the one claim here that is not
    free, so it is read from the recorded measurement file rather than re-run.
    """
    path = (ROOT / "docs" / "sciml-benchmark-data" / "measurements"
            / "m05_cross_plan_compare.json")
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    return [(row["num_paths"], row["milstein_vs_taylor"], row["em_vs_taylor"])
            for row in doc["rows"]]


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
    ci = rank_study.REL_ORDER_CI
    name = variant
    if ":" in variant and not variant.startswith("floor:"):
        name, n = variant.split(":", 1)
        # SS10m's two routes carry a standard-error multiplier where `thin`/`ordertie`
        # carry a breadth, so the suffix means different things and the mapping has to
        # branch on the name rather than on the colon.
        if name in ("relorder", "routes"):
            ci = float(n)
        else:
            breadth = int(n)
            name = "abstain" if name == "thin" else name
    # `main()`'s mapping exactly, and no more: SS10f's variant table was measured
    # against the scorer that *shipped*, so `abstain`/`discriminating`/`floor`
    # default to `metrics`. SS10h's rows cross those variants with the re-graded
    # scorer and pass `source="evaluation"` to say so. SS10m's routes are candidate
    # ranking rules priced against the scorer that ships *today*, so they read
    # `evaluation.json` like `regraded` does.
    src = source or ("evaluation" if name in (
        "regraded", "whitelisted", "relorder", "structadm", "routes") else "metrics")
    pool_list = rank_study.pools(
        paths, truncated=(variant == "truncated"), source=src)
    if variant == "whitelisted":
        pool_list = rank_study._restrict_to_record(pool_list)
    rows = rank_study.study(pool_list, name, floor, base, breadth, ci)
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


def _c2_records():
    """The nine completed step-10 records, M09's third attempt last (§10e)."""
    out = {}
    for doc in (_load(PROBE), _load(REST), _load(M09R)):
        for r in (doc or {}).get("results", []):
            out[r["id"]] = r
    return out


def _verdict_axes():
    """`(C2 on the old conjunction, C2 on the fix, C2 harness, C0, underclaims)` (§10i).

    Re-derived by driving the real verdict functions rather than quoted, so the claim
    fails if either one changes.
    """
    import compare as _C
    import report as _R

    old = ("VERIFIED_PASS", "STABLE_PASS")   # `_PASS_VERDICTS`, as it was
    c2 = list(_c2_records().values())
    c0 = (_load(BASE / "oneshot_sciml_scheme_matched.json") or {}).get("results", [])
    return (sum(1 for r in c2 if _R.compute_verdict(r) in old),
            sum(1 for r in c2 if _C._pipeline_pass(r)),
            sum(1 for r in c2 if (r.get("verify") or {}).get("passed")),
            sum(1 for r in c0 if _C._oneshot_pass(r)),
            sorted(r["id"] for r in c2 if _R.compute_verdict(r) == "UNDERCLAIM"))


def _underclaim_is_a_verified_pass():
    """No `UNDERCLAIM` record fails the harness, and no baseline record can be one
    (§10i). The first is why the widening is safe; the second is why it was needed."""
    import report as _R

    for r in _c2_records().values():
        verdict = _R.compute_verdict(r)
        if verdict == "UNDERCLAIM" and not (r.get("verify") or {}).get("passed"):
            return False
        if _R.compute_oneshot_verdict(r) == "UNDERCLAIM":
            return False
    return True


def _score_cap_incidence():
    """`(entries capped, their verified scores, distinct violation paths)` (§10i).

    The step-4 caveat rests on all seven violations being one finding, so the third
    element is what makes that checkable rather than asserted.
    """
    caps, scores, paths = 0, set(), set()
    for r in _c2_records().values():
        v = r.get("verify") or {}
        if v.get("score_cap"):
            caps += 1
            scores.add(v.get("verified_score"))
            for viol in (v.get("ledger_violations") or []):
                paths.add(viol.get("path"))
    return caps, sorted(scores), sorted(paths)


def _num_paths_provenance():
    """Per entry, `(problems.py, the brief's own call, the formulator's declared
    threshold, the executed count, score_cap)` (§10j).

    The claim this supports is a correlation, so it is derived rather than quoted:
    the cap fires exactly where the declared threshold disagrees with the run, and
    the first two columns agree with the fourth on every entry.
    """
    import re

    import problems as _P

    spec = {p["slug"]: p for p in _P.PROBLEMS}
    out = {}
    for r in _c2_records().values():
        slug = r["slug"]
        v = r.get("verify") or {}
        ps = ROOT / "workspace" / slug / "problem_spec.json"
        pm = ROOT / "workspace" / slug / "problem.md"
        declared = None
        if ps.exists():
            declared = ((json.loads(ps.read_text()).get("evaluation_thresholds") or {})
                        .get("num_paths"))
        brief = None
        if pm.exists():
            m = re.search(r"solve_sde\(num_paths=(\d+)", pm.read_text())
            brief = int(m.group(1)) if m else None
        out[r["id"]] = (spec[slug]["num_paths"], brief, declared,
                        (v.get("run") or {}).get("num_paths"), v.get("score_cap"))
    return out


def _cap_tracks_the_formulator():
    """`(entries where cap <=> declared != ran, total, uncapped ids)` (§10j).

    If this ever stops being unanimous, the cap has started depending on something
    other than the formulator's declared threshold and §10j needs re-deriving.
    """
    prov = _num_paths_provenance()
    agree = sum(1 for _ps, _b, d, ran, cap in prov.values()
                if bool(cap) == (d is not None and d != ran))
    consistent = all(ps == brief == ran
                     for ps, brief, d, ran, cap in prov.values())
    return agree, len(prov), sorted(k for k, v in prov.items() if not v[4]), consistent


def _declared_count_is_absent_from_the_brief():
    """Does any capped entry's brief actually contain its declared count? (§10j)"""
    for eid, (_ps, _brief, declared, _ran, cap) in _num_paths_provenance().items():
        if not cap or declared is None:
            continue
        slug = next(r["slug"] for r in _c2_records().values() if r["id"] == eid)
        pm = ROOT / "workspace" / slug / "problem.md"
        if pm.exists() and str(declared) in pm.read_text():
            return False
    return True


def _reaudited_under_the_fix():
    """`(entries the fix would cap, entries reporting a spec defect, their paths)`
    for the nine recorded runs (§10j rev17).

    Drives the real `audit_run` against each entry's own `problem_spec.json` and
    recorded `run_meta`, so the claim fails if the severity or the attribution moves.
    """
    from verifylib.audit import audit_run

    capped, reported, paths = 0, 0, set()
    for r in _c2_records().values():
        ps = ROOT / "workspace" / r["slug"] / "problem_spec.json"
        if not ps.exists():
            continue
        found = audit_run(json.loads(ps.read_text()),
                          run_meta=(r.get("verify") or {}).get("run"))
        capped += any(f.severity == "error" for f in found)
        sd = [f for f in found if f.severity == "spec"]
        reported += bool(sd)
        paths.update(f.path for f in sd)
    return capped, reported, sorted(paths)


def _pde_output_still_gates():
    """The re-attribution is scoped: a PDE output finding still caps (§10j rev17)."""
    import numpy as np

    from verifylib.audit import audit_run

    spec = {"spatial_variables": ["x"], "domain": {"bounds": {"x": [0.0, 1.0]}},
            "parameters": {"t_final": 0.5}}

    def _run(t_final=0.5):
        return {"numerical_solution": np.zeros(9),
                "grid": {"x": np.linspace(0, 1, 9)}, "t_final": t_final}

    short = audit_run(spec, result=_run(t_final=0.25))
    return (any(f.severity == "error" for f in short)
            and audit_run(spec, result=_run()) == [])


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
        # ~~"the heun stays 8"~~ SS10k: the band that kept it at 8 is fixed, so all three
        # of M01's plans grade 10. The SS10g table keeps the old cell struck through
        # rather than replaced, because 8 is what rev 13 measured.
        ("M01 re-graded: the milstein goes 8 to 10, and SS10k takes the heun to 10 too",
         (_m01_regraded() == [("3-heun-srk", 10, True),
                              ("2-milstein", 10, True),
                              ("1-euler-maruyama", 10, True)]),
         "| 8 | ~~8~~ **10 in \u00a710k** |"),
        # ~~geomean 2.466, vs_c0 0.931, M01 regret 2.132~~ superseded by SS10k. Those
        # were rev 13's measurement and are not re-derivable, because `cli.py evaluate`
        # rewrote the `evaluation.json` files this reads; the doc keeps them as the row
        # SS10g measured, and the live assertion is SS10k's.
        ("and that takes the arm past the C0 one-shot",
         (f"{_study('regraded')[0]['vs_c0']:.3f}" == "0.855"
          and next(r["decided_by"] for r in _study("regraded")[1]
                   if r["id"] == "M01") == "sqi"),
         "| **`regraded` (this fix)** | 5/9 | 3 | **2.466** | **0.931\u00d7 \u2014 better** |"),
        ("only M01 and M02 declare a terminal law, and M02 is unmoved",
         (_terminal_law_specs() == ["sde_sciml_additive", "sde_sciml_linear_ito"]
          and all(v == 9 for v in _m02_regraded_scores())),
         "0.52\u00d7 of the old, **0.786\u00d7** of the new"),
        # --- SS10h: the three ranker defects -------------------------------
        ("the comparable_keys fallback is dead code on all 18 recorded pools",
         # ~~(0, ["M05"], ["M02", "M05"])~~ -- SS10l gave M05's three plans the order
         # study SS24 had skipped, so that pool now shares *three* components and all
         # three of them vary. The fallback is still dead code, which is what the claim
         # is for; what changed is the defect SS10h put in its place -- a shared set of
         # **one** no longer occurs on any of the 18 pools, so `thin:K`'s only live
         # instance is gone and M02's single *varying* component is what remains.
         (_fallback_fires() == (0, [], ["M02"])),
         "The fallback at `ranking.py:559` is **dead code on every pool ever run**"),
        ("abstaining is bit-identical to shipping, so item 1 is a measured no-op",
         ([{k: v for k, v in r.items() if k != "source"}
           for r in _study("abstain", source="evaluation")[1]]
          == [{k: v for k, v in r.items() if k != "source"}
              for r in _study("regraded")[1]]),
         "bit-identical to `regraded` on\n18/18 pools"),
        # ~~2.174x / 3.501x / 5.761~~ were rev 14's numbers. SS10k re-graded two plans
        # these read through `evaluation.json`, so the magnitudes moved and the doc
        # carries both tables. The *conclusion* is what the claim is for, and it holds:
        # still harmful, still monotonic, against a baseline that is itself better.
        ("the thin gate is harmful, monotonically",
         # ~~1.999x / 3.219x / 5.297~~ were rev 18's. SS10l re-graded M05's three
         # plans, which these read through `evaluation.json`, and the movement is under
         # 1% -- which is SS10l's negative result, since `thin` was the variant whose
         # one live instance that pool was.
         (f"{_study('thin:2', source='evaluation')[0]['vs_c0']:.3f}" == "1.989"
          and f"{_study('thin:3', source='evaluation')[0]['vs_c0']:.3f}" == "3.203"
          and f"{_study('thin:2', source='evaluation')[0]['geomean']:.3f}"
          == "5.270"),
         "| `thin:2` (\u00a710l) | 6/9 | 5.270 | **1.989\u00d7** | 6/9 | 2.015 |"),
        ("declaration order is the worst plan on sciml and the best on holdout",
         (_decl_order_stats() == {"sciml": (2.89, 0, 8),
                                  "holdout": (1.56, 5, 1)}),
         "| sciml | 2.89 / 3 | 0/9 | **8/9** | `1-euler-maruyama`, all nine |"),
        # ~~6/9 and 0.846x~~ were rev 14's. SS10k fixes the order *gate*, and this
        # variant is an order *tiebreak*, so it gains more: 7/9 and 0.778x. The half
        # that keeps it unshipped is untouched and that is the point of the claim --
        # the holdout pools are BENCHOP's and SS10k re-graded nothing there, so the
        # collapse to 2/9 at the breadth where the gate fires is the same number.
        ("ordertie wins on sciml and breaks holdout where its gate first fires",
         (_study("ordertie:2", source="evaluation")[0]["picks_best"] == 7
          and f"{_study('ordertie:2', source='evaluation')[0]['vs_c0']:.3f}"
          == "0.774"
          and _study("ordertie:3", holdout=True,
                     source="evaluation")[0]["picks_best"] == 2
          and f"{_study('ordertie:3', holdout=True, source='evaluation')[0]['geomean']:.3f}"
          == "4.247"),
         "| `ordertie:3` (\u00a710l) | 7/9 | 2.051 | 0.774\u00d7 | **2/9** | **4.247** |"),
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
        # ~~`whitelisted` picks 5~~ -- it picks whatever `regraded` picks, which is the
        # whole claim, and SS10k took that to 6. The losslessness itself is unchanged
        # and is what `_record_is_lossless` asserts pool by pool; `truncated` reads the
        # record rather than the workspace, so its two numbers do not move.
        ("and the record is a lossless ranking input, both suites",
         (_record_is_lossless()
          and f"{_study('truncated')[0]['geomean']:.3f}" == "14.023"
          and f"{_study('truncated')[0]['vs_c0']:.3f}" == "5.291"
          and _study("whitelisted")[0]["picks_best"] == 6
          and f"{_study('whitelisted')[0]['vs_c0']:.3f}" == "0.855"),
         "**bit-identical on all 18 pools across both suites**"),
        # --- SS10i: the UNDERCLAIM accounting -------------------------------
        ("the old conjunction read 5/9 against the baseline's 9/9",
         (_verdict_axes()[:4] == (5, 9, 9, 9)),
         "| the old `_PASS_VERDICTS` / `_ONESHOT_PASS_VERDICTS` | **5/9** | **9/9** |"),
        ("and the four it lost are the four underclaims",
         (_verdict_axes()[4] == ["M02", "M04", "M05", "M08"]),
         "every\none of them `UNDERCLAIM`"),
        ("the widening admits nothing the harness failed, and is one-sided",
         _underclaim_is_a_verified_pass(),
         "`UNDERCLAIM`\nis a subset of the verified passes by construction"),
        ("seven entries carry a cap, all graded 10, all one finding",
         (_score_cap_incidence() == (7, [10], ["run_meta.num_paths"])),
         "7 of 9,\nall with `verified_score: 10`, all with one `ledger_violations` entry"),
        # --- SS10j: defect 9 settled ----------------------------------------
        ("problems.py, the brief and the run agree, and the cap tracks the formulator",
         (_cap_tracks_the_formulator() == (9, 9, ["M05", "M09"], True)),
         "**cap \u27fa declared \u2260 executed, 9 of 9**"),
        ("no capped entry's brief contains the count its formulator declared",
         _declared_count_is_absent_from_the_brief(),
         "returns **nothing**\non all seven"),
        ("the seven capped entries were all graded 10 by the harness",
         (_score_cap_incidence() == (7, [10], ["run_meta.num_paths"])),
         "graded **10/10 by the harness independently**"),
        ("the fix caps nothing and reports the defect against the ledger field",
         (_reaudited_under_the_fix()
          == (0, 7, ["evaluation_thresholds.num_paths"])),
         "| entries capped | **7/9** | **0/9** |"),
        ("and the PDE output checks still gate",
         _pde_output_still_gates(),
         "**The PDE output checks are untouched and still gate.**"),
        # --- SS10k: the strong-order gate -----------------------------------
        ("two plans were capped for converging faster than their floor",
         (_overshoot_before() == [("3-heun-srk", 0.9831, 0.5, "two-sided", 8.0),
                                  ("1-euler-maruyama", 0.9514, 0.5, "two-sided", 8.0)]),
         "| M01 `linear_ito` | `3-heun-srk` | 0.9831 | 0.5 | **+0.483** |"),
        ("and both now grade 10 with the order check passing",
         (_overshoot_after() == [("3-heun-srk", 10, True),
                                 ("1-euler-maruyama", 10, True)]),
         "The gate is now one-sided, the overshoot is reported instead of"),
        ("the gate passes an overshoot, flags it, and still fails a shortfall",
         (_gate_is_one_sided() == (True, True, False, False)),
         "is a shortfall and survives\nuntouched"),
        ("the one true positive the gate has is the canary's 0.504 against 1.0",
         ("0.504" in _orders_reason(0.504, 1.0)
          and "does not clear" in _orders_reason(0.504, 1.0)),
         "reads **0.504 against 1.0**"),
        ("and a non-shortfall is no longer rendered as one",
         (all("does not clear" not in _orders_reason(o, f)
              and "do not shrink" in _orders_reason(o, f)
              for o, f in ((0.983, 0.5), (1.0, 1.0), (0.7, 1.0)))),
         "*\"the observed strong order 0.983 does not clear 0.5\"*"),
        ("the component picker is one-sided too, and publishes the lower order",
         (_worst_component_is_one_sided() == (0, 0)
          and _m09_published_order() == (0, 0.5104)),
         "| `1-euler-maruyama` | 0.5104 (pass) | 0.9514 (**failed**) | **0.9514** | "
         "**0.5104** |"),
        ("the arm reaches 0.855x on 6 of 9 pools, with M01 at regret 1.000",
         (f"{_study('regraded')[0]['geomean']:.3f}" == "2.267"
          and f"{_study('regraded')[0]['vs_c0']:.3f}" == "0.855"
          and _study("regraded")[0]["picks_best"] == 6
          and f"{next(r['regret'] for r in _study('regraded')[1] if r['id'] == 'M01'):.3f}"
          == "1.000"
          and next(r["top"] for r in _study("regraded")[1]
                   if r["id"] == "M01") == "3-heun-srk"),
         "| **after (\u00a710k)** | **6/9** | **2/9** | **2.267** | **0.855\u00d7** |"),
        ("and M09 holds only because SS10b excludes a component it alone measures",
         (_m09_keys_exclude_order() == ("accuracy", "structure", "consistency")
          and f"{next(r['regret'] for r in _study('regraded')[1] if r['id'] == 'M09'):.3f}"
          == "1.033"),
         "**`('accuracy', 'structure', 'consistency')`**"),
        # --- SS10l: M05's skipped order study -------------------------------
        ("the \u00a724 cost skip had fired on three plans in the whole corpus",
         (_not_run_census() == (["1-euler-maruyama", "2-milstein", "3-ito-taylor-1p5"],
                                68)),
         "| SDE plans with a `<metrics>` block | **68** | **3** |"),
        ("and it had left M05's pool ranked on one component, with no order and no z",
         (_m05_before() == ([("1-euler-maruyama", "not_run", None, None, 6.0),
                             ("2-milstein", "not_run", None, None, 6.0),
                             ("3-ito-taylor-1p5", "not_run", None, None, 6.0)],
                            ("structure",))),
         "`comparable_keys`: ~~`('structure',)`~~"),
        ("both studies now run, on all three plans, at the measured orders",
         (_m05_after() == ([("1-euler-maruyama", "true", 0.965, 0.2054, 6),
                            ("2-milstein", "true", 0.9981, 0.2038, 6),
                            ("3-ito-taylor-1p5", "true", 1.5851, 0.2035, 6)],
                           ("order", "structure", "consistency"))),
         "**`('order', 'structure', 'consistency')`** \u2014 one"),
        ("and no score moves, which is what Path A's rubric row order guarantees",
         all(row[4] == 6 for row in _m05_after()[0]),
         "Every score is unchanged, as predicted above."),
        ("the raw measurement ranks the pool exactly as the harness does",
         (_m05_order_is_informative_and_discarded()[0]
          == _m05_order_is_informative_and_discarded()[1]
          == ("3-ito-taylor-1p5", "2-milstein", "1-euler-maruyama")),
         "**Identical, all three positions.**"),
        ("and `ranking.order` discards it: the best plan ties the worst at 1.0",
         (_m05_order_is_informative_and_discarded()[2] == (1.0, 0.9981, 1.0)),
         "| `3-ito-taylor-1p5` | 1.5851 | 0.5 | **1.0000** |"),
        ("so `structure` decides the pool by 7.4\u00d7, and the index gap is 0.0056",
         (_m05_structure_outweighs_order() == (0.0056, 7.4)),
         "| **index** | | | **+0.0056** |"),
        ("the arm does not move: 6/9, 2.267, 0.855\u00d7, M05 still at 622.233",
         (f"{_study('regraded')[0]['geomean']:.3f}" == "2.267"
          and f"{_study('regraded')[0]['vs_c0']:.3f}" == "0.855"
          and _study("regraded")[0]["picks_best"] == 6
          and f"{next(r['regret'] for r in _study('regraded')[1] if r['id'] == 'M05'):.3f}"
          == "622.233"
          and next(r["decided_by"] for r in _study("regraded")[1]
                   if r["id"] == "M05") == "sqi"),
         "| **after (\u00a710l)** | 6/9 | 2/9 | 2.267 | 0.855\u00d7 | 622.233 | `sqi` |"),
        ("and `thin` and `ordertie` are still powerless on M05 with the order present",
         (all(f"{next(r['regret'] for r in _study(v, source='evaluation')[1] if r['id'] == 'M05'):.3f}"
              == "622.233" for v in ("thin:2", "thin:3", "ordertie:2", "ordertie:3"))
          and _study("ordertie:2", source="evaluation")[0]["picks_best"] == 7
          and f"{_study('ordertie:2', source='evaluation')[0]['vs_c0']:.3f}" == "0.774"),
         "| `ordertie:2` (\u00a710l) | 7/9 | 2.051 | **0.774\u00d7** | 6/9 | 2.015 |"),
        # --- SS10m: the three routes out of M05 -----------------------------
        ("the two corpora are disjoint on the two fields the routes read",
         (_route_corpus_split() == {"sciml": (9, 8, 2), "holdout": (9, 0, 9)}),
         "| sciml | 9 | **8** | **2** (M08, M09) |"),
        ("route (i) reaches 7 of 9 at 0.419x and leaves the holdout bit-identical",
         (_study("relorder:2")[0]["picks_best"] == 7
          and f"{_study('relorder:2')[0]['geomean']:.3f}" == "1.109"
          and f"{_study('relorder:2')[0]['vs_c0']:.3f}" == "0.419"
          and _study("relorder:2", holdout=True)[0]["picks_best"] == 6
          and f"{_study('relorder:2', holdout=True)[0]['geomean']:.3f}" == "2.015"),
         "| **(i) `relorder:2`** | **7/9** | **1/9** | **1.109** | **0.419\u00d7** | 6/9 | 2.015 |"),
        ("and its whole gain is one pool: it flips exactly one winner",
         (_relorder_flips_one_winner()
          == [("M05", "2-milstein", "3-ito-taylor-1p5")]),
         "**flips exactly one\nwinner**"),
        ("the resolution guard is load-bearing, and M01 is what proves it",
         (_relorder_guard_two_is_load_bearing()
          == ((7, "3-heun-srk", 1.0), (6, "2-milstein", 2.132))),
         "| 0 \u2013 0.185 | 6/9 | 1.207 | 0.455\u00d7 | `2-milstein` \u2717 |"),
        ("its one free parameter is flat across the sweep and degrades at both ends",
         (_relorder_ci_is_flat() == [(0.0, 6, 1.207), (0.2, 7, 1.109),
                                     (2.0, 7, 1.109), (13.0, 7, 1.109),
                                     (15.0, 6, 2.267)]),
         "| **0.2 \u2013 13.0** | **7/9** | **1.109** | **0.419\u00d7** | `3-heun-srk` \u2713 |"),
        ("while the shortfall guard never fires on any of the 18 pools",
         (_relorder_shortfall_guard_is_dead() == (0.2296, 0.4)),
         "18 pools is **0.2296**"),
        ("route (ii) is harmful: it fixes M05 alone and costs four other pools",
         (_study("structadm")[0]["picks_best"] == 4
          and f"{_study('structadm')[0]['vs_c0']:.3f}" == "2.378"
          and f"{_study('structadm')[0]['geomean']:.3f}" == "6.303"
          and f"{next(r['regret'] for r in _study('structadm')[1] if r['id'] == 'M05'):.3f}"
          == "1.000"
          and f"{next(r['regret'] for r in _study('structadm')[1] if r['id'] == 'M02'):.3f}"
          == "4879.117"),
         "| | 6/9 \u00b7 0.855\u00d7 | 4/9 \u00b7 **2.378\u00d7** | worse than Opus answering cold |"),
        ("and composing the two routes is worse than route (i) alone",
         (_study("routes:2")[0]["picks_best"] == 7
          and f"{_study('routes:2')[0]['geomean']:.3f}" == "1.523"
          and f"{_study('routes:2')[0]['vs_c0']:.3f}" == "0.575"),
         "| (i)+(ii) `routes:2` | 7/9 | 1/9 | 1.523 | 0.575\u00d7 | 6/9 | 2.015 |"),
        ("route (iii) agrees at every path count instead of separating the pool",
         (_m05_compare_agrees_at_every_path_count()
          == [(1000, "INCONCLUSIVE", "INCONCLUSIVE"),
              (20000, "AGREE", "AGREE"), (200000, "AGREE", "AGREE")]),
         "| 200 000 | **AGREE** | **AGREE** |"),
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
