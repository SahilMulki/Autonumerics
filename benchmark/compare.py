"""Head-to-head report: the AutoNumerics pipeline vs the one-shot / single-agent
baselines, on identical problems and identical (independent, sandboxed) grading.

Consumes:
  * one or more pipeline result files (run.py's results.json) -- each becomes a
    column, e.g. the default Opus+Sonnet pipeline and a Sonnet-pinned pipeline;
  * the one-shot result file (oneshot.py's oneshot_results.json) -- split into a
    column per (condition, knowledge-setting, model), e.g. C0-naked@opus.

and renders results/COMPARISON.md:
  * headline independently-verified pass rate per system, overall / by type / by
    tier / on the hard discriminator subset;
  * an accuracy-vs-cost table (pass rate beside median tokens / cost / wall time),
    so the pipeline's higher accuracy is always shown against its higher cost;
  * the discriminator table -- per hard problem, each system's pass@1 -- which is
    where "the structure wins" has to show up.

Everything keys off problems.py, so it is robust to partial data: columns with no
data for a bucket render "-", and problems added later appear automatically. Pass
on BOTH sides means the same thing -- an independently-confirmed pass
(VERIFIED_PASS / STABLE_PASS for the pipeline; PASS / STABLE_PASS for a baseline) --
so the comparison is apples-to-apples.

Examples
--------
    # default: pipeline results.json vs oneshot_results.json
    uv run python benchmark/compare.py

    # name several pipeline configs explicitly (path=label)
    uv run python benchmark/compare.py \
        --pipeline results/pipeline_opus_sonnet.json=C2a:opus+sonnet \
        --pipeline results/pipeline_sonnet.json=C2b:sonnet
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(HERE, "results")

sys.path.insert(0, HERE)
import problems as P  # noqa: E402, I001
import report as R  # noqa: E402

DEFAULT_PIPELINE = os.path.join(RESULTS_DIR, "results.json")
DEFAULT_ONESHOT = os.path.join(RESULTS_DIR, "oneshot_results.json")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "COMPARISON.md")


# --- loading ----------------------------------------------------------------


def _load(path):
    with open(path) as fh:
        payload = json.load(fh)
    if isinstance(payload, dict):
        return payload.get("results", []), payload.get("config", {})
    return payload, {}


def _pipeline_pass(rec):
    return R.compute_verdict(rec) in R._PASS_VERDICTS


def _oneshot_pass(rec):
    v = rec.get("verdict") or R.compute_oneshot_verdict(rec)
    return R.oneshot_passed(v)


class Column:
    """One system variant: a pipeline config, or a one-shot (cond, manuals, model)."""

    def __init__(self, key, label, kind):
        self.key = key
        self.label = label
        self.kind = kind              # "pipeline" | "oneshot"
        self.by_slug = {}             # slug -> rec (pipeline) | [recs] (oneshot)

    def pass_at_1(self, slug):
        """Fraction of this column's reps that independently passed for ``slug``;
        None if the column has no data for it."""
        if slug not in self.by_slug:
            return None
        if self.kind == "pipeline":
            return 1.0 if _pipeline_pass(self.by_slug[slug]) else 0.0
        recs = self.by_slug[slug]
        if not recs:
            return None
        return sum(1.0 for r in recs if _oneshot_pass(r)) / len(recs)

    def checkable(self, slug):
        """True if ``slug`` is independently checkable in this column (excludes the
        no-ground-truth problems, which cannot count toward a pass rate)."""
        if slug not in self.by_slug:
            return False
        rec = self.by_slug[slug]
        rec0 = rec if self.kind == "pipeline" else (rec[0] if rec else None)
        if rec0 is None:
            return False
        return bool(rec0.get("has_ground_truth")) or \
            rec0.get("ground_truth_kind") == "stability"

    def costs(self, kind):
        """Flattened list of a numeric run field over all reps/problems."""
        out = []
        recs = ([self.by_slug[s]] if self.kind == "pipeline" else self.by_slug[s]
                for s in self.by_slug)
        for group in recs:
            for r in group:
                v = (r.get("run") or {}).get(kind)
                if isinstance(v, (int, float)):
                    out.append(float(v))
        return out


def pipeline_columns(specs):
    cols = []
    for spec in specs:
        path, _, label = spec.partition("=")
        path = path.strip()
        if not os.path.exists(path):
            print(f"warning: pipeline file not found, skipping: {path}", file=sys.stderr)
            continue
        results, _cfg = _load(path)
        label = label.strip() or os.path.splitext(os.path.basename(path))[0]
        col = Column(f"pipe:{label}", label, "pipeline")
        for rec in results:
            col.by_slug[rec["slug"]] = rec
        cols.append(col)
    return cols


def oneshot_columns(path, model_filter=None):
    if not os.path.exists(path):
        return []
    results, _cfg = _load(path)
    groups = {}
    for rec in results:
        model = rec.get("model", "?")
        if model_filter and model != model_filter:
            continue
        gkey = (model, rec.get("condition", "?"), rec.get("manuals", "?"))
        groups.setdefault(gkey, Column(
            f"os:{model}:{gkey[1]}:{gkey[2]}",
            f"{gkey[1].upper()}-{gkey[2]}@{model}", "oneshot"))
        groups[gkey].by_slug.setdefault(rec["slug"], []).append(rec)
    # Order: C0 before C1, naked before manual, grouped by model.
    order = {"c0": 0, "c1": 1, "naked": 0, "manual": 1}
    return [groups[k] for k in sorted(
        groups, key=lambda k: (k[0], order.get(k[1], 9), order.get(k[2], 9)))]


# --- aggregation ------------------------------------------------------------


def bucket_rate(col, slugs):
    """(rate, n_covered) over the independently-checkable slugs in ``slugs``."""
    vals = [col.pass_at_1(s) for s in slugs if col.checkable(s)]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, 0
    return sum(vals) / len(vals), len(vals)


def _fmt_rate(rate, n):
    return "-" if rate is None else f"{rate:.0%} ({n})"


def _median(xs):
    return statistics.median(xs) if xs else None


# --- report -----------------------------------------------------------------


def generate(columns, out_path):
    problems = list(P.PROBLEMS)
    all_slugs = [p["slug"] for p in problems]
    sde = [p["slug"] for p in problems if p["type"] == "sde"]
    pde = [p["slug"] for p in problems if p["type"] == "pde"]
    disc = [p["slug"] for p in P.discriminator_problems()]
    tiers = {t: [p["slug"] for p in problems if p["tier"] == t] for t in (1, 2, 3)}

    L = []
    ap = L.append
    ap("# Pipeline vs. Baseline — Head-to-Head\n")
    ap("_Independently-verified pass rate; identical problems, identical sandboxed "
       "grading. A pass is an independently-confirmed pass on both sides._\n")

    if not columns:
        ap("\n**No data yet.** Run `oneshot.py` (and `run.py` for the pipeline), then "
           "re-run `compare.py`.\n")
        _write(out_path, L)
        return

    ap("## Systems compared\n")
    for c in columns:
        covered = sum(1 for s in all_slugs if c.checkable(s))
        kind = "pipeline" if c.kind == "pipeline" else "baseline"
        ap(f"- **{c.label}** ({kind}) — data for {covered} checkable problem(s)")
    ap("")

    # ---- headline pass-rate table ----------------------------------------
    ap("## Independently-verified pass rate\n")
    headers = ["System", "Overall", "SDE", "PDE", "Tier 1", "Tier 2", "Tier 3",
               "Discriminators"]
    rows = []
    for c in columns:
        r_all = bucket_rate(c, all_slugs)
        rows.append([
            c.label,
            _fmt_rate(*r_all),
            _fmt_rate(*bucket_rate(c, sde)),
            _fmt_rate(*bucket_rate(c, pde)),
            _fmt_rate(*bucket_rate(c, tiers[1])),
            _fmt_rate(*bucket_rate(c, tiers[2])),
            _fmt_rate(*bucket_rate(c, tiers[3])),
            _fmt_rate(*bucket_rate(c, disc)),
        ])
    ap(_table(headers, rows))
    ap("\n> Cells show mean pass@1 with the number of independently-checkable "
       "problems covered in parentheses. `-` = no data for that bucket yet.\n")

    # ---- accuracy vs cost -------------------------------------------------
    ap("## Accuracy vs. cost\n")
    headers = ["System", "Overall pass", "Median wall", "Median cost", "Median turns",
               "Total cost"]
    rows = []
    for c in columns:
        rate, n = bucket_rate(c, all_slugs)
        wall = _median(c.costs("wall_s") or c.costs("wall_seconds"))
        costs = c.costs("cost_usd")
        turns = _median(c.costs("num_turns"))
        rows.append([
            c.label,
            _fmt_rate(rate, n),
            R._fmt_secs(wall) if wall is not None else "-",
            f"${_median(costs):.3f}" if costs else "-",
            f"{turns:.0f}" if turns is not None else "-",
            f"${sum(costs):.2f}" if costs else "-",
        ])
    ap(_table(headers, rows))
    ap("\n> Pipeline runs record wall time only; one-shot runs also record token "
       "cost. The point of this table is the asymmetry: read any accuracy gap "
       "against the cost gap.\n")

    # ---- discriminator table (the money table) ---------------------------
    ap("## Discriminator subset (per problem)\n")
    if not disc:
        ap("No problems are flagged as discriminators yet.\n")
    else:
        headers = ["Problem", "Tier", *[c.label for c in columns]]
        rows = []
        for slug in disc:
            prob = P.by_slug(slug)
            cells = []
            for c in columns:
                p1 = c.pass_at_1(slug)
                cells.append("-" if p1 is None else f"{p1:.0%}")
            rows.append([slug, f"T{prob['tier']}", *cells])
        ap(_table(headers, rows))
        ap("\n> Per-cell pass@1 on each hard problem. The thesis predicts pipeline "
           "columns stay high here while the baseline columns fall off.\n")

    # ---- full per-problem table ------------------------------------------
    ap("## All problems (per problem)\n")
    headers = ["ID", "Problem", "Type", "Tier", *[c.label for c in columns]]
    rows = []
    for p in problems:
        slug = p["slug"]
        cells = []
        for c in columns:
            p1 = c.pass_at_1(slug)
            cells.append("-" if p1 is None else f"{p1:.0%}")
        rows.append([p["id"], slug, p["type"].upper(), p["tier"], *cells])
    ap(_table(headers, rows))
    ap("")

    _write(out_path, L)


def _table(headers, rows):
    out = ["| " + " | ".join(str(h) for h in headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _write(out_path, lines):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        fh.write("\n".join(lines))
    print(f"wrote {os.path.relpath(out_path, REPO_ROOT)}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the pipeline-vs-baseline comparison.",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pipeline", action="append", default=None,
                    help="pipeline results file, as PATH or PATH=LABEL (repeatable). "
                         "Default: results/results.json=pipeline")
    ap.add_argument("--oneshot", default=DEFAULT_ONESHOT, help="one-shot results file")
    ap.add_argument("--model", default=None,
                    help="restrict one-shot columns to this model (default: all present)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output markdown path")
    args = ap.parse_args(argv)

    specs = args.pipeline if args.pipeline else [f"{DEFAULT_PIPELINE}=pipeline"]
    columns = pipeline_columns(specs) + oneshot_columns(args.oneshot, args.model)
    generate(columns, args.out)


if __name__ == "__main__":
    main()
