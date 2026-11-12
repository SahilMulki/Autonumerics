"""Check the numbers quoted in plan-sciml-benchmark.md §3f/§9a/§10a–10d against the data.

A plan that records measured findings in prose is only as good as the prose matching the
artifacts, and prose drifts silently: a table edited after a re-run, a figure rounded
twice, a count that was true of an earlier sweep. This re-derives each quoted number from
`comparators.json`, the two baseline result files and **both** step-10 results files, and
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


def _cost(doc):
    return sum(float((r.get("run") or {}).get("cost_usd") or 0) for r in doc["results"])


def main():
    plan = PLAN.read_text()
    comp, probe, ks = _load(COMP), _load(PROBE), _load(KS)
    rest = _load(REST)
    A = _load(BASE / "oneshot_sciml_scheme_matched.json")
    B = _load(BASE / "oneshot_sciml_unrestricted.json")
    if not all((comp, probe, A, B, ks, rest)):
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
    by = {r["id"]: r for r in probe["results"] + rest["results"]}
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
        ("defects 9 and 10 now stand at seven of eight completed entries",
         (len(capped) == 7 and "M05" not in capped
          and sum(1 for r in by.values()
                  if r["run"]["status"] == "completed") == 8),
         "**seven of eight**"),
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
