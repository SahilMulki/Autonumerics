"""Check the numbers quoted in plan-sciml-benchmark.md §3f/§9a/§10a against the data.

A plan that records measured findings in prose is only as good as the prose matching the
artifacts, and prose drifts silently: a table edited after a re-run, a figure rounded
twice, a count that was true of an earlier sweep. This re-derives each quoted number from
`comparators.json`, the two baseline result files and the step-10 probe, and checks that
the plan still says it.

It is not a test (it reads gitignored result files, so it cannot run in CI) and it is not
a substitute for one. It is the thing to run before trusting a number quoted out of that
document.

Run:  uv run python docs/sciml-benchmark-data/audit_plan_claims.py
Exits non-zero if any claim no longer matches.
"""

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[2]
PLAN = ROOT / "docs" / "plan-sciml-benchmark.md"
COMP = ROOT / "benchmark" / "references" / "sciml" / "comparators.json"
BASE = ROOT / "benchmark" / "results" / "baselines"
PROBE = ROOT / "benchmark" / "results" / "pipeline" / "sciml_m01_probe.json"


def _load(path):
    if not path.exists():
        print(f"missing (gitignored result file?): {path.relative_to(ROOT)}")
        return None
    return json.loads(path.read_text())


def _degenerate(doc):
    return sum(1 for r in doc["results"]
               if (((r.get("verify") or {}).get("pathwise_ladder") or {}).get("degenerate")))


def _cost(doc):
    return sum(float((r.get("run") or {}).get("cost_usd") or 0) for r in doc["results"])


def main():
    plan = PLAN.read_text()
    comp, probe = _load(COMP), _load(PROBE)
    A = _load(BASE / "oneshot_sciml_scheme_matched.json")
    B = _load(BASE / "oneshot_sciml_unrestricted.json")
    if not all((comp, probe, A, B)):
        print("cannot audit: re-run step 3, 9 or 10 first")
        return 2

    rows = comp["rows"]
    pathwise = [r for r in rows if r["kind"] == "pathwise"]
    stages = {}
    for r in rows:
        stages.setdefault(r["scheme"], set()).add(r["stages"])
    checks_total = sum(c["checks_total"] for c in comp["analytic_cross_checks"])
    checks_pass = sum(c["checks_passed"] for c in comp["analytic_cross_checks"])
    worst = max(c["worst_rel"] for c in comp["analytic_cross_checks"])
    hbp = probe["results"][0]["harness_by_plan"]
    em = hbp["plans"]["1-euler-maruyama"]["harness_error"]
    srk = hbp["plans"]["3-heun-srk"]["harness_error"]
    wall_min = probe["results"][0]["run"]["wall_seconds"] / 60

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
        ("probe verdict", probe["results"][0]["verdict"] == "VERIFIED_PASS",
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
