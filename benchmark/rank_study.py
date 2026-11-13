"""Re-rank finished pipeline pools offline, and score a candidate ranker against
the harness.

The file `verifylib/ranking.py:55` and `docs/plan-blind-evaluator.md` §5 have both
promised for a while: a way to measure a ranking change against recorded runs
instead of asserting it. It costs nothing to run -- every plan's independently
measured error is already in the results files -- so a ranker change can be scored
before it is shipped.

**What it measures.** For each finished pool it takes the ranker's input (the plans'
`<metrics>` blocks, which is what the conductor sees) and the harness's independent
verdict (`harness_by_plan.plans[*].harness_error`, the plan's own re-measured error
as a fraction of its own gate, which is what the conductor does *not* see). Regret is
the ratio between them:

    regret = harness_error(the plan the ranker puts first)
             / harness_error(the best plan in the pool)

1.0 means the ranker chose the best available plan. A regret of 622 means it shipped
a plan 622x worse than one it had already written -- which is `sde_sciml_nltest`, and
is why this file exists.

**Why regret and not agreement.** `benchmark/results/benchop-step10/rankcheck.py` --
this file's one-off predecessor -- printed whether the ranker's top agreed with the
shipped plan. That answers a compliance question (does the conductor obey its ranker)
and not a quality one. Both matter and they are different: a ranker can agree with
every shipped plan and still be choosing badly.

**Variants.** `--variant` selects the ranker under test. `current` is what ships.
The others are the candidates `plan-sciml-benchmark.md` §10b-10e recorded without
fixing, each isolated so its effect is attributable:

* `abstain`  -- `comparable_keys` returns nothing where no component is shared by the
  whole pool, instead of falling back to all four. The fallback is what let
  `sde_sciml_additive` be decided on the fourth significant figure of
  `invariant_worst_headroom` (10c).
* `floor:E`  -- `abstain`, plus two indices within `E` rank as tied, so the keys below
  decide. Fitted here rather than asserted, which is the whole point.
* `discriminating` -- `abstain`, plus a component must *vary* across the pool to be
  comparable. A component every plan scores identically carries no information and
  still takes its share of the weight, diluting the ones that do discriminate.
* `noscore` -- the index above `score` in the key. **A diagnostic, not a proposal:**
  `score` is a claim about what was established and must not be traded against a
  weighted mean (`rank_key`'s own docstring). It is here to answer one question --
  how many of the ranker's misses are the *scorer's* misses -- and the answer is
  worth having before any ranking weight is touched.
* `truncated` -- `current`, but reading the *record's* per-plan `kernel` summary
  instead of `SOLUTION.md`. Not a candidate either: it measures how much the
  recorder's field truncation (10e) costs anything that re-ranks from the record.

A variant that ties falls through to `iter` and then to declaration order, exactly as
the live ranker does, so "abstaining" is never a silent coin flip -- it is the pool's
own ordering, and the report says when that is what decided.

Run:  uv run python benchmark/rank_study.py
      uv run python benchmark/rank_study.py --variant floor:0.05 --sweep
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import statistics
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from verifylib import ranking  # noqa: E402

#: The nine sciml pools, across the three files step 10 wrote (§10a, 10d, 10e). The
#: last wins: it supersedes the first file's `RUN_ERROR` row for the same slug.
SCIML = ("benchmark/results/pipeline/sciml_probe.json",
         "benchmark/results/pipeline/sciml_rest/sciml_probe_rest.json",
         "benchmark/results/pipeline/sciml_m09/sciml_probe_m09.json")

#: The C0 one-shot baseline, for the axis the whole comparison is about: whether
#: the pipeline's shipped answer beats Opus answering the same problem cold. The
#: baseline ran at the *same* dt, path count and evaluation count on every entry, so
#: the errors are comparable without a work correction (plan-sciml-benchmark.md
#: SS10f). Only the scheme-matched arm: the unrestricted arm transcribes.
C0 = "benchmark/results/baselines/oneshot_sciml_scheme_matched.json"


def c0_errors():
    """``{id: harness error as a fraction of its own gate}`` for the baseline."""
    path = ROOT / C0
    if not path.exists():
        return {}
    out = {}
    for r in json.loads(path.read_text())["results"]:
        v = r.get("verify") or {}
        if v.get("status") != "ok":
            continue
        if v.get("pathwise_rel_err") is not None and v.get("pathwise_tol_used"):
            out[r["id"]] = v["pathwise_rel_err"] / v["pathwise_tol_used"]
            continue
        errs, tols = v.get("functional_errs") or {}, v.get("functional_tol_used") or {}
        if errs:
            out[r["id"]] = max(e / (tols.get(k) or v["functional_tol"])
                               for k, e in errs.items())
    return out


#: Held out. These pools are older than the sciml ones and their `workspace/`
#: directories have been overwritten since, so most of them can only be read from
#: the record's truncated summary -- the report says which, per pool, because a
#: floor fitted on nine pools and shipped corpus-wide needs the caveat visible.
HOLDOUT = ("benchmark/results/pipeline/benchop_sonnet.json",
           "benchmark/results/pipeline/benchop_sonnet_other9_run2.json",
           "benchmark/results/pipeline/no_closed_form_v2.json")


def parse_metrics(text):
    """The `<metrics>` block as a dict, with the same coercions the kernel uses."""
    match = re.search(r"<metrics>(.*?)</metrics>", text, re.S)
    if not match:
        return None
    out = {}
    for line in match.group(1).strip().splitlines():
        key, _, value = line.partition(":")
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        if value in ("true", "false"):
            out[key] = value == "true"
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def _same_run(summary, parsed):
    """Is this `SOLUTION.md` the one that produced this record's numbers?

    `workspace/` is gitignored and every pipeline run rewrites it, so a plan
    directory with the right name is not evidence of the right run. The hash the
    kernel scored is, where both sides carry one; where either does not, there is
    nothing to contradict and the block is taken.
    """
    mine, theirs = summary.get("solver_sha256"), parsed.get("solver_sha256")
    return mine is None or theirs is None or mine == theirs


def pools(paths, *, truncated=False):
    """One entry per problem: its plans, each plan's ranker input and its harness
    error. Plans the harness could not grade are kept and ranked -- excluding them
    would hand the ranker an answer it does not have -- but they cannot be `best`."""
    records = {}
    for rel in paths:
        path = ROOT / rel
        if not path.exists():
            continue
        for record in json.loads(path.read_text())["results"]:
            records[record["id"]] = (record, rel)

    out = []
    for pid, (record, rel) in sorted(records.items()):
        by_plan = (record.get("harness_by_plan") or {}).get("plans") or {}
        if not by_plan:
            continue
        plans, sources = [], set()
        for name, entry in by_plan.items():
            summary, source = entry.get("kernel") or {}, "summary"
            if not truncated:
                solution = (ROOT / "workspace" / record["slug"] / "plans" / name
                            / "SOLUTION.md")
                if solution.exists():
                    parsed = parse_metrics(solution.read_text(encoding="utf-8"))
                    # A metrics block is only this pool's if the solver it scored is
                    # the one on disk. Without that check an overwritten workspace
                    # silently supplies another run's numbers.
                    if parsed and _same_run(summary, parsed):
                        summary, source = parsed, "metrics"
            sources.add(source)
            error = entry.get("harness_error")
            plans.append({
                "name": name,
                # `rank_key` reads score/provenance/iter off the entry and the rest
                # off `kernel.summary`, so the shape has to match a live record's.
                "entry": {"kernel": {"summary": summary,
                                     "score": summary.get("score"),
                                     "provenance": summary.get("provenance")},
                          "iter": entry.get("iter")},
                "summary": summary,
                "error": error if isinstance(error, (int, float)) else None,
                "shipped": bool(entry.get("is_winner")),
            })
        graded = [p for p in plans if p["error"] is not None]
        if not graded:
            continue
        out.append({
            "id": pid, "slug": record["slug"], "file": rel, "plans": plans,
            "best": min(graded, key=lambda p: p["error"]),
            "margin": (record.get("harness_by_plan") or {}).get(
                "harness_error_margin"),
            "source": "metrics" if sources == {"metrics"}
                      else ("summary" if sources == {"summary"} else "mixed"),
        })
    return out


def _keys(variant, tol, summaries):
    if variant in ("current", "truncated", "noscore"):
        return ranking.comparable_keys(tol, summaries)
    # `abstain`, `floor:` and `discriminating` share this: no fallback. Where the
    # pool shares no component there is no comparison to make, and the index says so.
    if len(summaries) < 2:
        return ranking.CERT_COMPONENTS
    parts = [ranking.components(s, tol, summaries) for s in summaries]
    keys = [k for k in ranking.CERT_COMPONENTS
            if all(p.get(k) is not None for p in parts)]
    if variant == "discriminating":
        # Identical to ~1e-12 is identical: these are floats off a rubric, and a
        # component that separates nothing should not dilute one that does.
        keys = [k for k in keys
                if max(p[k] for p in parts) - min(p[k] for p in parts) > 1e-12]
    return tuple(keys)


def key_of(plan, variant, summaries, floor):
    """`rank_key`, with the variant's key restriction and resolution floor."""
    summary = plan["summary"]
    tol = ranking._tol_of(summary, summary)
    keys = _keys(variant, tol, summaries)
    index = (ranking.sqi(summary, tol, summaries, keys=keys)[0] if keys else None)
    if index is not None and floor:
        # Quantising is how a floor becomes a *tie* rather than a nudge: two indices
        # in the same bucket compare equal and the keys below decide. Rounding half
        # up would still split a pair that straddles a boundary, which is why the
        # report also counts how often the floor is what decided.
        index = math.floor(index / floor) * floor
    score = ranking._num(plan["entry"]["kernel"].get("score")) or 0.0
    quality = -(index if index is not None else -1.0)
    provenance = ranking.provenance_rank(plan["entry"]["kernel"].get("provenance"))
    iters = ranking._num(plan["entry"].get("iter")) or 0.0
    if variant == "noscore":
        return (0 if summary else 1, provenance, quality, -score, iters)
    return (0 if summary else 1, -score, provenance, quality, iters)


def rank(pool, variant, floor):
    summaries = [p["summary"] for p in pool["plans"] if p["summary"]]
    ordered = sorted(pool["plans"], key=lambda p: key_of(p, variant, summaries, floor))
    return ordered


def study(pool_list, variant, floor, baseline=None):
    baseline = {} if baseline is None else baseline
    rows = []
    for pool in pool_list:
        ordered = rank(pool, variant, floor)
        top, best = ordered[0], pool["best"]
        error = top["error"]
        regret = (error / best["error"]) if (error and best["error"]) else None
        # The flag `run.py` sets: a difference the harness's own noise margin cannot
        # explain. A pick inside the margin is not a wrong pick.
        margin = pool["margin"]
        outside = (regret is not None and margin
                   and (error - best["error"]) > margin)
        rows.append({
            "id": pool["id"], "slug": pool["slug"], "source": pool["source"],
            "top": top["name"], "best": best["name"],
            "shipped": next((p["name"] for p in pool["plans"] if p["shipped"]), None),
            "err": error, "best_err": best["error"], "regret": regret,
            "picks_best": top["name"] == best["name"],
            "c0": baseline.get(pool["id"]),
            "vs_c0": ((error / baseline[pool["id"]])
                      if (error and baseline.get(pool["id"])) else None),
            "outside_margin": bool(outside),
            "decided_by": decided(ordered, variant, floor),
        })
    return rows


def decided(ordered, variant, floor):
    """Which key separated the top two, by the same rule `ranking.decided_by` uses."""
    if len(ordered) < 2:
        return "only-plan"
    summaries = [p["summary"] for p in ordered if p["summary"]]
    a = key_of(ordered[0], variant, summaries, floor)
    b = key_of(ordered[1], variant, summaries, floor)
    names = (("provenance", "sqi", "score", "iter") if variant == "noscore"
             else ranking.RANK_KEYS)
    for name, left, right in zip(names, a[1:], b[1:], strict=True):
        if left != right:
            return name
    return "tie"


def summarise(rows):
    regrets = [r["regret"] for r in rows if r["regret"]]
    vs = [r["vs_c0"] for r in rows if r.get("vs_c0")]
    return {
        "n": len(rows),
        "picks_best": sum(1 for r in rows if r["picks_best"]),
        "outside_margin": sum(1 for r in rows if r["outside_margin"]),
        "beats_c0": sum(1 for v in vs if v < 0.999),
        "vs_c0": statistics.geometric_mean(vs) if vs else None,
        "geomean": statistics.geometric_mean(regrets) if regrets else None,
        "p90": (sorted(regrets)[min(len(regrets) - 1, int(0.9 * len(regrets)))]
                if regrets else None),
        "max": max(regrets) if regrets else None,
    }


def show(label, rows):
    print(f"\n=== {label}")
    print(f"{'':5} {'src':<8} {'ranker top':<34} {'best in pool':<34} "
          f"{'regret':>9} {'vs C0':>8} {'decided_by':<12} {'>margin':>7}")
    for r in sorted(rows, key=lambda r: -(r["regret"] or 0)):
        print(f"{r['id']:<5} {r['source']:<8} {r['top'][:34]:<34} "
              f"{r['best'][:34]:<34} "
              f"{'-' if r['regret'] is None else format(r['regret'], '9.3f')} "
              f"{'-' if r.get('vs_c0') is None else format(r['vs_c0'], '8.3f')} "
              f"{r['decided_by']:<12} {'YES' if r['outside_margin'] else '':>7}")
    s = summarise(rows)
    print(f"      picks the best plan on {s['picks_best']}/{s['n']}; "
          f"outside the harness margin on {s['outside_margin']}/{s['n']}; "
          f"regret geomean {s['geomean']:.3f}, P90 {s['p90']:.3f}, "
          f"max {s['max']:.1f}")
    if s["vs_c0"]:
        print(f"      against the C0 one-shot: beats it on {s['beats_c0']}/{s['n']}, "
              f"geomean {s['vs_c0']:.3f} "
              f"({'better' if s['vs_c0'] < 1 else 'worse'} than Opus cold)")
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--variant", default="current",
                    help="current | abstain | discriminating | floor:E | "
                         "noscore | truncated")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the floor over a decade and print the table")
    ap.add_argument("--holdout", action="store_true",
                    help="also score the held-out BENCHOP / no-closed-form pools")
    ap.add_argument("--json", metavar="PATH", help="write the rows to a JSON file")
    args = ap.parse_args(argv)

    variant, floor = args.variant, 0.0
    if variant.startswith("floor:"):
        floor = float(variant.split(":", 1)[1])
        variant = "floor"

    base = c0_errors()
    sciml = pools(SCIML, truncated=(args.variant == "truncated"))
    sets = [("sciml, 9 pools", sciml)]
    if args.holdout:
        sets.append(("holdout", pools(HOLDOUT,
                                      truncated=(args.variant == "truncated"))))

    out = {}
    for label, pool_list in sets:
        if not pool_list:
            print(f"\n=== {label}: no pools found (gitignored results?)")
            continue
        rows = study(pool_list, variant, floor,
                     base if label.startswith("sciml") else None)
        out[label] = {"rows": rows, "summary": show(f"{label} -- {args.variant}", rows)}

    if args.sweep:
        for label, pool_list in sets:
            if not pool_list:
                continue
            print(f"\n=== floor sweep ({label}; 'abstain' is floor 0)")
            print(f"{'floor':>8} {'picks best':>11} {'>margin':>8} {'geomean':>9} "
                  f"{'P90':>8} {'max':>9} {'ties':>6}")
            for eps in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15,
                        0.2, 0.3):
                rows = study(pool_list, "floor" if eps else "abstain", eps,
                             base if label.startswith("sciml") else None)
                s = summarise(rows)
                ties = sum(1 for r in rows if r["decided_by"] in ("iter", "tie"))
                print(f"{eps:8.3f} {s['picks_best']:>6}/{s['n']:<4} "
                      f"{s['outside_margin']:>8} {s['geomean']:9.3f} "
                      f"{s['p90']:8.3f} {s['max']:9.1f} {ties:>6}")

    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
