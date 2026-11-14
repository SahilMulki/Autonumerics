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
* `thin:K`   -- `abstain`, plus the index must have **K** components that actually
  separate the pool before it may decide it. This is the defect `abstain` was
  supposed to be: the all-four fallback turns out never to fire (SS10h -- `shared` is
  non-empty on all 18 recorded pools), while a shared set of *one* is real and is how
  `sde_sciml_nltest` was decided. Measured **harmful** and recorded, not shipped.
* `ordertie:K` -- `thin:K`, but ties fall through to the higher `observed_order`
  instead of to declaration order. The one candidate here that is not fitted to a
  suite, since a measured convergence order is a property of the method and a
  position in a list is not. Also not shipped, and the reason is worth more than the
  variant: an **exact** sampler has no order to measure, so the closed-form plans
  that are the best plan in 7 of 9 holdout pools sort last.
* `relorder:C` -- `regraded`, but the `order` **component** becomes pool-relative
  wherever its usual reading carries no information. SS10l measured the hole: `order`
  scores a *shortfall against a floor*, and on a pool where every plan clears its own
  floor every plan scores 1.0 -- M05's three read 0.9650 / 0.9981 / 1.5851, in perfect
  rank agreement with the harness, and the component mapped them to 1.000 / 0.998 /
  1.000, tying the pool's best plan with its worst. Two guards keep this from becoming
  "higher order is better", which SS19 is explicit is wrong ("an order far above the
  theoretical one is noise, not a bonus"): the pool-relative reading fires **only**
  where no plan is short of its own floor by more than the gate's own `STRONG_BAND`,
  and **only** where the *top two* measured orders are more than `C` standard errors
  apart. So the noise case is decided by the data rather than by a rule of thumb --
  M01's top two read 0.98647 +/- 0.0096 and 0.98312 +/- 0.0150, a gap of 0.0033
  against a 2-sigma bar of 0.0357, and this abstains there. It had better: the lower
  of those two is that pool's **best** plan by the harness's own measurement.
* `structadm` -- `regraded`, plus `cost_admissible`'s rule applied to **`structure`**.
  That rule is already argued in `ranking.py` for `cost`: a component subordinate to
  accuracy may contribute only where every plan in the pool *had* its accuracy
  measured, because "a component promoted by the absence of the one it is subordinate
  to is not breaking a tie, it is deciding". SS10l measured `structure` in exactly that
  position on M05 -- `accuracy` unavailable on all three plans, `invariant_worst_headroom`
  anti-correlated with quality, and deciding the pool by 7.4x over the one component
  pointing the other way. This is a transplant of an existing rule, not a new one.
* `routes:C`  -- `relorder:C` and `structadm` together. SS10l named them as two of the
  three routes out of M05 and they are not independent: one restores a discriminating
  component, the other removes a misleading one.
* `whitelisted` -- `regraded`, then cut down to `run.RECORDED_METRICS`. This is how a
  change to the **recorder** is priced: it reads the full block and throws away
  exactly what the record throws away, so a whitelist change can be measured without
  spending a pipeline run on it. `truncated` measures the record as it *was* written;
  this measures it as it *will be*.
* `truncated` -- `current`, but reading the *record's* per-plan `kernel` summary
  instead of `SOLUTION.md`. Not a candidate either: it measures how much the
  recorder's field truncation (10e) costs anything that re-ranks from the record.
* `regraded` -- `current`, reading each plan's `evaluation.json`, which is what
  `verifylib/cli.py evaluate` rewrites. That is the kernel as it stands *now*, so
  this is how a change to the **scorer** is priced: re-grade the plans, then
  re-rank. SS10g uses it to measure the terminal-law bar fix.

A variant that ties falls through to `iter` and then to declaration order, exactly as
the live ranker does, so "abstaining" is never a silent coin flip -- it is the pool's
own ordering, and the report says when that is what decided.

`--source` overrides the block a variant's grades are read from, which is what makes
a candidate priceable against the *re-graded* scorer rather than the one that shipped:
`--variant abstain --source evaluation` is `abstain` crossed with SS10g's fix. Without
it each variant is locked to one source and the two effects cannot be separated.

Run:  uv run python benchmark/rank_study.py
      uv run python benchmark/rank_study.py --variant floor:0.05 --sweep
      uv run python benchmark/rank_study.py --variant thin:2 --source evaluation
      uv run python benchmark/rank_study.py --variant whitelisted --holdout
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

from benchmark.run import RECORDED_METRICS  # noqa: E402
from verifylib import ranking  # noqa: E402
from verifylib.kernel.sde.crn import STRONG_BAND  # noqa: E402

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


#: Where a plan's ranker input comes from. `metrics` is what the conductor reads and
#: is the default. `record` is the results file's truncated copy. `evaluation` is the
#: kernel's own output file, which `cli.py evaluate` rewrites -- so re-grading a pool
#: and reading `evaluation` prices a change to the scorer.
SOURCES = ("metrics", "record", "evaluation")


def pools(paths, *, truncated=False, source="metrics"):
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
        want = "record" if truncated else source
        plans, sources = [], set()
        for name, entry in by_plan.items():
            summary, source_used = entry.get("kernel") or {}, "summary"
            plan_dir = ROOT / "workspace" / record["slug"] / "plans" / name
            if want == "metrics":
                solution = plan_dir / "SOLUTION.md"
                if solution.exists():
                    parsed = parse_metrics(solution.read_text(encoding="utf-8"))
                    # A metrics block is only this pool's if the solver it scored is
                    # the one on disk. Without that check an overwritten workspace
                    # silently supplies another run's numbers.
                    if parsed and _same_run(summary, parsed):
                        summary, source_used = parsed, "metrics"
            elif want == "evaluation":
                graded = plan_dir / "evaluation.json"
                if graded.exists():
                    doc = json.loads(graded.read_text())
                    # `cli.py evaluate` writes score/provenance/certified at the top
                    # level and the ranked components under `summary`; `rank_key`
                    # wants them flattened as a `<metrics>` block carries them.
                    fresh = {**(doc.get("summary") or {}),
                             "score": doc.get("score"),
                             "provenance": doc.get("provenance"),
                             "certified": doc.get("certified")}
                    if _same_run(summary, fresh):
                        summary, source_used = fresh, "regraded"
            sources.add(source_used)
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
            "source": sources.pop() if len(sources) == 1 else "mixed",
        })
    return out


def _restrict_to_record(pool_list):
    """Each plan's summary cut down to `run.RECORDED_METRICS` (SS10h).

    `truncated` measures the record **as written** -- by the whitelist that was live
    when the run happened. This measures the record as it *will be* written, so a
    change to the whitelist can be priced before a pipeline run is spent on it. It
    reads the evaluation block and then throws away exactly what the recorder throws
    away, which is why it imports the real tuple instead of restating it.
    """
    for pool in pool_list:
        for plan in pool["plans"]:
            if plan["summary"]:
                plan["summary"] = {k: v for k, v in plan["summary"].items()
                                   if k in RECORDED_METRICS}
    return pool_list


#: Default standard-error multiplier for `relorder`'s resolution test. The kernel's own
#: `ci_mult` is 2.0 on every spec in both suites, so this is that convention and not a
#: new constant to tune.
REL_ORDER_CI = 2.0


def _rel_order(summary, summaries, ci=REL_ORDER_CI):
    """Route (i) of SS10l: `ranking.order`, made pool-relative where it is inert.

    Returns the shipped reading unchanged in every case but one. The exception is a
    pool in which **no** plan is short of its own floor -- there `ranking.order` is
    1.0 for everyone by construction, so it is carrying no information at all -- and
    in which the spread of measured orders clears `ci` standard errors of its own two
    endpoints. Only then is the order read against the **best rate anyone demonstrated
    on this problem** instead of against the floor, reusing the same
    `shortfall / max(target, 0.5)` shape the floor reading uses.

    The two guards are the whole design. Without the first, this would overrule a real
    shortfall with a pool comparison; without the second it would be the "higher order
    is better" rule SS19 forbids, and **M01** is the case that shows why -- its top two
    orders are 0.98647 and 0.98312, 0.0033 apart against a 2-sigma bar of 0.0357, and
    the *lower* one is the pool's best plan. Firing there would have ranked a pair
    separated by a tenth of its own error bar, in the wrong direction.
    """
    base = ranking.order(summary)
    if base is None:
        return None
    bases = [ranking.order(s) for s in summaries]
    pairs = [(ranking._num(s.get("observed_order")),
              ranking._num(s.get("observed_order_se")),
              ranking._num(s.get("order_floor"))) for s in summaries]
    mine = ranking._num(summary.get("observed_order"))
    if (any(b is None for b in bases) or mine is None
            or any(v is None or e is None or f is None for v, e, f in pairs)):
        return base

    # Guard 1: nobody is short of their own floor. ~~`min(bases) == 1.0`~~ -- that was
    # a bug, not a stricter rule: `base` saturates at 1.0 only for a plan at or above
    # its floor *exactly*, so a 0.2% miss blocked the whole pool. M05's `2-milstein`
    # reads 0.99805 against a floor of 1.0 and was excluded on 0.00195, as were M01,
    # M03, M04 and M08 on misses of 1-5%. The kernel's own threshold for a shortfall
    # that counts is `crn.STRONG_BAND`, which is what `crn.guards` passed every one of
    # these plans by, so that is the number to import rather than invent.
    if any(max(0.0, f - v) > STRONG_BAND for v, _, f in pairs):
        return base

    # Guard 2: the two plans this rule would reorder are distinguishable. Endpoints
    # are the wrong pair to test -- on M01 the spread 0.4804..0.98647 is resolved many
    # times over while the *top two* (0.98647 and 0.98312, standard errors 0.0096 and
    # 0.0150) are 0.0034 apart and are not, so an endpoint test would have ranked a
    # pair separated by a fifth of its own error bar. `ranking.accuracy` already makes
    # this move for the error ("snapped up to its own uncertainty ... turns two moment
    # errors inside the same Monte-Carlo noise into a genuine tie instead of one more
    # tiebreaker to read the seed off"); this is the same move for the order.
    top = sorted(pairs, key=lambda r: -r[0])
    (o1, e1, _), (o2, e2, _) = top[0], top[1]
    if o1 - o2 <= ci * math.sqrt(e1 ** 2 + e2 ** 2):
        return base

    hi = o1
    return ranking._clip01(1.0 - max(0.0, hi - mine) / max(hi, 0.5))


def _index(summary, tol, summaries, keys, variant, ci=REL_ORDER_CI):
    """`ranking.sqi`'s weighted mean, with a variant's component substitution.

    Kept here rather than in `ranking.py` for the reason this whole file exists: a
    candidate rule is priced before it is shipped, so nothing the conductor reads
    changes while it is being measured.
    """
    if not keys:
        return None
    if variant not in ("relorder", "routes"):
        return ranking.sqi(summary, tol, summaries, keys=keys)[0]
    parts = ranking.components(summary, tol, summaries)
    parts["order"] = _rel_order(summary, summaries, ci)
    measured = {k: v for k, v in parts.items() if v is not None and k in keys}
    if not measured:
        return None
    total = sum(ranking.WEIGHTS[k] for k in measured)
    return sum(ranking.WEIGHTS[k] * v for k, v in measured.items()) / total


def _keys(variant, tol, summaries, breadth=0):
    if variant in ("structadm", "routes"):
        # Route (ii) of SS10l. `cost_admissible` is `ranking.py`'s own rule, imported
        # rather than restated: `structure` may contribute only where **every** plan in
        # the pool had its accuracy measured. On a pool where it did not, a weak
        # structural diagnostic is not breaking a tie, it is deciding.
        keys = ranking.comparable_keys(tol, summaries)
        if not ranking.cost_admissible(tol, summaries):
            keys = tuple(k for k in keys if k != "structure")
        return keys
    if variant in ("current", "truncated", "regraded", "noscore", "relorder",
                   "whitelisted"):
        return ranking.comparable_keys(tol, summaries)
    # `abstain`, `floor:`, `thin:`, `ordertie:` and `discriminating` share this: no
    # fallback. Where the
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
    if breadth:
        # `thin:K` (SS10h). The *empty* shared set this file was built to price turns
        # out never to occur -- on all 18 recorded pools `comparable_keys` restricted
        # to something non-empty and the fallback was dead code. What does occur is a
        # shared set of **one**: M05 ranked on ('structure',) alone and shipped a plan
        # 622x worse. So the quantity to price is not emptiness but *breadth* -- how
        # many components have to separate the pool before a weighted mean over them
        # is allowed to decide it.
        varying = [k for k in keys
                   if max(p[k] for p in parts) - min(p[k] for p in parts) > 1e-12]
        if len(varying) < breadth:
            return ()
    return tuple(keys)


def key_of(plan, variant, summaries, floor, breadth=0, ci=REL_ORDER_CI):
    """`rank_key`, with the variant's key restriction and resolution floor."""
    summary = plan["summary"]
    tol = ranking._tol_of(summary, summary)
    keys = _keys(variant, tol, summaries, breadth)
    index = _index(summary, tol, summaries, keys, variant, ci)
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
    if variant == "ordertie":
        # SS10h. Every abstention route -- `abstain`, `floor:E`, `thin:K` -- falls
        # through to **declaration order**, and that is measured to be the thing
        # that hurts: the first-declared plan is the pool's worst on 8 of 9 sciml
        # pools and its best on 5 of 9 holdout pools. So the question is not whether
        # to abstain but what to abstain *to*. `observed_order` is a property of the
        # method rather than a position in a list, so preferring the higher measured
        # order is a prior that cannot be suite-fitted the way declaration order is.
        order = ranking._num(summary.get("observed_order"))
        return (0 if summary else 1, -score, provenance, quality, iters,
                -(order if order is not None else -1.0))
    return (0 if summary else 1, -score, provenance, quality, iters)


def rank(pool, variant, floor, breadth=0, ci=REL_ORDER_CI):
    summaries = [p["summary"] for p in pool["plans"] if p["summary"]]
    ordered = sorted(pool["plans"],
                     key=lambda p: key_of(p, variant, summaries, floor, breadth, ci))
    return ordered


def study(pool_list, variant, floor, baseline=None, breadth=0, ci=REL_ORDER_CI):
    baseline = {} if baseline is None else baseline
    rows = []
    for pool in pool_list:
        ordered = rank(pool, variant, floor, breadth, ci)
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
            "decided_by": decided(ordered, variant, floor, breadth, ci),
        })
    return rows


def decided(ordered, variant, floor, breadth=0, ci=REL_ORDER_CI):
    """Which key separated the top two, by the same rule `ranking.decided_by` uses."""
    if len(ordered) < 2:
        return "only-plan"
    summaries = [p["summary"] for p in ordered if p["summary"]]
    a = key_of(ordered[0], variant, summaries, floor, breadth, ci)
    b = key_of(ordered[1], variant, summaries, floor, breadth, ci)
    names = (("provenance", "sqi", "score", "iter") if variant == "noscore"
             else (*ranking.RANK_KEYS, "observed_order") if variant == "ordertie"
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
                         "thin:K | ordertie:K | noscore | truncated | regraded | "
                         "relorder:C | structadm | routes:C | whitelisted")
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the floor over a decade and print the table")
    ap.add_argument("--source", choices=SOURCES,
                    help="override which block the plans' grades are read from, so "
                         "a key-restriction variant can be priced against the "
                         "*re-graded* scorer rather than the one that shipped "
                         "(SS10h). Default: the variant's own source.")
    ap.add_argument("--holdout", action="store_true",
                    help="also score the held-out BENCHOP / no-closed-form pools")
    ap.add_argument("--json", metavar="PATH", help="write the rows to a JSON file")
    args = ap.parse_args(argv)

    variant, floor, breadth = args.variant, 0.0, 0
    if variant.startswith("floor:"):
        floor = float(variant.split(":", 1)[1])
        variant = "floor"
    if variant.startswith("thin:"):
        breadth = int(variant.split(":", 1)[1])
        variant = "abstain"
    if variant.startswith("ordertie:"):
        breadth = int(variant.split(":", 1)[1])
        variant = "ordertie"
    ci = REL_ORDER_CI
    for name in ("relorder", "routes"):
        if variant.startswith(f"{name}:"):
            ci = float(variant.split(":", 1)[1])
            variant = name

    base = c0_errors()
    # SS10l's three routes are candidate *ranking* rules measured against the scorer
    # that ships today, so they read `evaluation.json` like `regraded` does. Pricing
    # them against the pre-SS10g/SS10k blocks would conflate two changes.
    src = args.source or ("evaluation" if variant in (
        "regraded", "relorder", "structadm", "routes") else "metrics")
    cut = _restrict_to_record if args.variant == "whitelisted" else (lambda x: x)
    if args.variant == "whitelisted":
        src = args.source or "evaluation"
    sciml = cut(pools(SCIML, truncated=(args.variant == "truncated"), source=src))
    sets = [("sciml, 9 pools", sciml)]
    if args.holdout:
        sets.append(("holdout", cut(pools(
            HOLDOUT, truncated=(args.variant == "truncated"), source=src))))

    out = {}
    for label, pool_list in sets:
        if not pool_list:
            print(f"\n=== {label}: no pools found (gitignored results?)")
            continue
        rows = study(pool_list, variant, floor,
                     base if label.startswith("sciml") else None, breadth, ci)
        default_src = ("evaluation" if variant in (
            "regraded", "relorder", "structadm", "routes") else "metrics")
        tag = (args.variant if src == default_src
               else f"{args.variant}+{src}")
        out[label] = {"rows": rows, "summary": show(f"{label} -- {tag}", rows)}

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
