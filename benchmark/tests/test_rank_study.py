"""`benchmark/rank_study.py` -- the offline re-ranker.

The study itself reads gitignored result files, so what is testable here is its
*reasoning*: that it feeds the ranker the plans' own metrics rather than the
record's truncated copy, that it refuses a metrics block belonging to a different
run, that each variant restricts the key set the way its docstring claims, and that
the floor turns a small index gap into a tie rather than a nudge. Those are the four
things a wrong conclusion in plan-sciml-benchmark.md SS10f would have come from.
"""

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmark import rank_study  # noqa: E402


def _plan(name, summary, iters=0):
    return {"name": name, "summary": summary,
            "entry": {"kernel": {"summary": summary,
                                 "score": summary.get("score"),
                                 "provenance": summary.get("provenance")},
                      "iter": iters},
            "error": None, "shipped": False}


def test_parse_metrics_coerces_the_way_the_kernel_does():
    text = ("noise\n<metrics>\nscore: 10\ncertified: true\n"
            "order_check: unresolved\nobserved_order: 0.9514\n"
            "provenance: analytic\n</metrics>\ntail\n")
    block = rank_study.parse_metrics(text)
    assert block == {"score": 10.0, "certified": True,
                     "order_check": "unresolved", "observed_order": 0.9514,
                     "provenance": "analytic"}
    # A bool must not come back as a float: `ranking._num` rejects bools on
    # purpose, and `order()` tests `order_check` for stringhood.
    assert block["certified"] is True
    assert rank_study.parse_metrics("no block here") is None


def test_a_metrics_block_is_refused_when_its_solver_is_not_the_one_scored():
    """The guard that stops an overwritten `workspace/` supplying another run's
    numbers. `workspace/` is gitignored and every pipeline run rewrites it, so a
    plan directory with the right name is not evidence of the right run -- this is
    the sixth place in the repo that has had to say so."""
    assert rank_study._same_run({"solver_sha256": "abc"}, {"solver_sha256": "abc"})
    assert not rank_study._same_run({"solver_sha256": "abc"},
                                    {"solver_sha256": "def"})
    # No hash recorded on either side is not evidence of a mismatch.
    assert rank_study._same_run({}, {"solver_sha256": "abc"})
    assert rank_study._same_run({"solver_sha256": "abc"}, {})


def _pool_sharing_nothing():
    """Two plans whose measured components do not overlap at all, which is what
    makes `comparable_keys` fall back to all four keys."""
    a = {"score": 10, "provenance": "analytic", "kind": "sde",
         "order_check": "true", "observed_order": 1.0, "design_order": 1.0,
         "order_floor": 1.0}
    b = {"score": 10, "provenance": "analytic", "kind": "sde",
         "invariant_worst_headroom": 0.5}
    return [a, b]


def test_abstain_drops_the_fallback_that_current_keeps():
    from verifylib import ranking
    summaries = _pool_sharing_nothing()
    tol = ranking._tol_of(summaries[0], summaries[0])
    # The live ranker: nothing is shared, so it ranks on everything anyway.
    assert rank_study._keys("current", tol, summaries) == ranking.CERT_COMPONENTS
    # The candidate: nothing shared means no comparison.
    assert rank_study._keys("abstain", tol, summaries) == ()


def test_discriminating_drops_a_component_the_whole_pool_scores_alike():
    from verifylib import ranking
    same = {"score": 10, "provenance": "analytic", "kind": "sde",
            "order_check": "true", "observed_order": 1.0, "design_order": 1.0,
            "order_floor": 0.5}
    a = dict(same, invariant_worst_headroom=0.9)
    b = dict(same, invariant_worst_headroom=0.1)
    tol = ranking._tol_of(a, a)
    keys = rank_study._keys("abstain", tol, [a, b])
    assert "order" in keys and "structure" in keys
    # `order` is 1.000 for both -- it separates nothing and still takes 0.25 of the
    # weight, which is the dilution this variant exists to test.
    assert rank_study._keys("discriminating", tol, [a, b]) == ("structure",)


def test_the_floor_makes_a_tie_rather_than_a_nudge():
    """A floor that only shifted the numbers would still split any pair; the study's
    claim is that two indices inside the floor compare *equal*, so the keys below
    decide. That is why the sweep's harm is attributable to the fallback."""
    a = {"score": 10, "provenance": "analytic", "kind": "sde",
         "invariant_worst_headroom": 0.7460}
    b = {"score": 10, "provenance": "analytic", "kind": "sde",
         "invariant_worst_headroom": 0.7443}
    pool = [_plan("1-a", a, iters=1), _plan("2-b", b, iters=0)]
    summaries = [a, b]
    sharp = [rank_study.key_of(p, "abstain", summaries, 0.0) for p in pool]
    assert sharp[0][3] != sharp[1][3], "the raw indices must differ"
    assert sorted(pool, key=lambda p: rank_study.key_of(
        p, "abstain", summaries, 0.0))[0]["name"] == "1-a"
    coarse = [rank_study.key_of(p, "floor", summaries, 0.05) for p in pool]
    assert coarse[0][3] == coarse[1][3], "inside the floor they must tie"
    # Tied on the index, `iter` decides -- the fallback the sweep measures.
    assert sorted(pool, key=lambda p: rank_study.key_of(
        p, "floor", summaries, 0.05))[0]["name"] == "2-b"


def test_noscore_moves_the_index_above_score_and_nothing_else():
    """The diagnostic variant. It is not a proposal, and the test says so: `score`
    stays in the key, below the index, rather than being dropped."""
    worse_but_scored_higher = {"score": 10, "provenance": "analytic", "kind": "sde",
                               "invariant_worst_headroom": 0.1}
    better_but_scored_lower = {"score": 8, "provenance": "analytic", "kind": "sde",
                               "invariant_worst_headroom": 0.9}
    pool = [_plan("1-high-score", worse_but_scored_higher),
            _plan("2-high-index", better_but_scored_lower)]
    summaries = [p["summary"] for p in pool]
    first = sorted(pool, key=lambda p: rank_study.key_of(p, "current", summaries, 0))
    assert first[0]["name"] == "1-high-score"
    flipped = sorted(pool, key=lambda p: rank_study.key_of(p, "noscore", summaries, 0))
    assert flipped[0]["name"] == "2-high-index"
    # `score` is still consulted, just later: two equal indices still order by it.
    tie_a = dict(worse_but_scored_higher, invariant_worst_headroom=0.5)
    tie_b = dict(better_but_scored_lower, invariant_worst_headroom=0.5)
    tied = [_plan("1-ten", tie_a), _plan("2-eight", tie_b)]
    sums = [p["summary"] for p in tied]
    assert sorted(tied, key=lambda p: rank_study.key_of(
        p, "noscore", sums, 0))[0]["name"] == "1-ten"


def test_regret_is_measured_against_the_harness_and_not_the_ranker():
    """The study's definition: the ranker's pick divided by the *best measured*
    plan. A pool where the ranker picks the best reads exactly 1.0, and a pool it
    gets wrong reads the true ratio -- not a rank distance, which would hide how
    much the miss cost."""
    good = {"score": 10, "provenance": "analytic", "kind": "sde",
            "invariant_worst_headroom": 0.9}
    bad = {"score": 10, "provenance": "analytic", "kind": "sde",
           "invariant_worst_headroom": 0.1}
    plans = [_plan("1-picked", good), _plan("2-better", bad)]
    plans[0]["error"], plans[1]["error"] = 0.5, 0.01
    pool = {"id": "X", "slug": "x", "file": "f", "plans": plans,
            "best": plans[1], "margin": 0.001, "source": "metrics"}
    row, = rank_study.study([pool], "current", 0.0)
    assert row["top"] == "1-picked" and row["best"] == "2-better"
    assert row["regret"] == 50.0
    assert row["outside_margin"] is True
    assert row["picks_best"] is False


def test_the_source_decides_which_file_the_ranker_is_fed(tmp_path, monkeypatch):
    """Three sources, three different answers, and §10f/§10g both depend on the
    distinction: `metrics` is what the conductor reads, `record` is the results
    file's truncated copy (which costs 4.5x in regret, §10e/§10f), and `evaluation`
    is the kernel's own output file — the one `cli.py evaluate` rewrites, so reading
    it is how a change to the *scorer* gets priced (§10g).
    """
    import json

    slug = "sde_demo"
    plan = tmp_path / "workspace" / slug / "plans" / "1-only"
    plan.mkdir(parents=True)
    (plan / "SOLUTION.md").write_text(
        "<metrics>\nscore: 8\nprovenance: analytic\n"
        "solver_sha256: abc\ninvariant_worst_headroom: 0.5\n</metrics>\n")
    (plan / "evaluation.json").write_text(json.dumps(
        {"score": 10, "provenance": "analytic", "certified": True,
         "summary": {"solver_sha256": "abc", "invariant_worst_headroom": 0.5}}))
    results = tmp_path / "r.json"
    results.write_text(json.dumps({"results": [{
        "id": "X", "slug": slug,
        "harness_by_plan": {"plans": {"1-only": {
            "kernel": {"score": 3, "provenance": "analytic",
                       "solver_sha256": "abc"},
            "harness_error": 0.5, "is_winner": True}}}}]}))
    monkeypatch.setattr(rank_study, "ROOT", tmp_path)

    seen = {}
    for source in rank_study.SOURCES:
        pool, = rank_study.pools(
            ["r.json"], truncated=(source == "record"), source=source)
        seen[source] = (pool["source"], pool["plans"][0]["summary"]["score"])
    assert seen == {"metrics": ("metrics", 8.0),
                    "record": ("summary", 3),
                    "evaluation": ("regraded", 10)}, seen


def test_a_regraded_plan_is_refused_when_its_solver_is_not_the_one_scored():
    """The `solver_sha256` guard applies to `evaluation.json` too, and has to: a
    re-grade run against a solver that has since changed would price a fix on the
    wrong code.
    """
    assert rank_study._same_run({"solver_sha256": "abc"}, {"solver_sha256": "abc"})
    assert not rank_study._same_run({"solver_sha256": "abc"},
                                    {"solver_sha256": "zzz"})


def test_the_record_carries_every_field_the_ranker_reads():
    """A census over `verifylib/ranking.py`, not a frozen list (sciml SS10h).

    The per-plan record is a *ranking input*: `rank_study`'s `truncated` variant
    re-ranks finished pools from it, and so would anything else that wanted to
    without paying for a pipeline run. So every field the ranker reads off a
    summary has to survive `run.solution_metrics`' whitelist, and the way to keep
    that true is to ask `ranking.py` what it reads rather than to restate the answer
    here.

    This is the test that found `dynkin_bar`. Dropping it neither abstained nor
    failed: `consistency()` reads it with a hardcoded fallback of 2.43, the pools
    carry 2.4284 / 2.5310 / 2.8049, and the pools inspected by hand happened to
    carry the one that matches the fallback. Only a census sees that.
    """
    import re

    from benchmark import run

    src = (ROOT / "verifylib" / "ranking.py").read_text(encoding="utf-8")
    # `[A-Za-z_0-9]`, not `[a-z_0-9]`: `estimated_rel_error_T` is a real field name
    # and a lowercase-only class skips it. Nothing capitalised is read off a summary
    # today, so this is a blind spot in a test whose entire job is the ones added
    # later.
    read = set(re.findall(r'summary\.get\("([A-Za-z_0-9]+)"', src))
    assert read, "the scan found no summary reads at all -- it has stopped working"
    missing = sorted(read - set(run.RECORDED_METRICS))
    assert not missing, (
        f"`ranking.py` reads {missing} off a plan summary and "
        "`run.RECORDED_METRICS` does not carry them, so anything that re-ranks "
        "from the record scores those components as absent -- or, where the reader "
        "has a fallback, as something quietly wrong. Add them to the whitelist.")


def test_dropping_the_error_bar_fabricates_an_accuracy_instead_of_abstaining():
    """Why the whitelist is a correctness concern and not a reporting one (SS10h).

    `ranking.accuracy()` abstains under F16 where the reported error sits inside the
    plan's own error bar, and it learns the bar from `uncertainty()` =
    `mc_se_rel * ci_mult`. Those two fields going missing does not make the check
    return "unknown" -- `noise_limited()` returns **False**, accuracy falls through
    to `headroom_fraction`, and the record ends up holding a confident number for a
    quantity the kernel declined to locate.

    Measured consequence, M05: the cut block scored three unresolved plans at 0.141
    / 0.008 / 0.046, had only that component to rank on, and put the 0.141 first --
    `1-euler-maruyama`, the worst plan in the pool by 651x.
    """
    from verifylib import ranking

    full = {"estimated_rel_error": 1e-4, "mc_se_rel": 1e-3, "ci_mult": 2.0}
    assert ranking.noise_limited(full), "1e-4 is inside a 2e-3 bar"
    assert ranking.accuracy(full, 1e-2) is None, "an unresolved error is not located"

    cut = {k: v for k, v in full.items() if k in ("estimated_rel_error",)}
    assert not ranking.noise_limited(cut), "no bar reads as resolved, not as unknown"
    assert ranking.accuracy(cut, 1e-2) is not None, (
        "this is the defect being guarded: without the bar, accuracy invents a value")

    from benchmark import run
    for field in ("mc_se_rel", "ci_mult"):
        assert field in run.RECORDED_METRICS, field


def test_the_record_whitelist_loses_nothing_the_ranker_would_have_used():
    """End to end on a synthetic pool: cutting a summary to the whitelist leaves
    every ranking component bit-identical (SS10h).

    On the nine real sciml pools this is what took a re-rank from the record from
    3 of 9 and a regret geomean of 14.0 to the conductor's own 5 of 9 and 2.47 --
    bit-identical rows, both suites.
    """
    from benchmark import run
    from verifylib import ranking

    full = {"score": 10, "provenance": "analytic", "solver_sha256": "abc",
            "estimated_rel_error": 2e-3, "mc_se_rel": 1e-3, "ci_mult": 2.0,
            "observed_order": 0.98, "design_order": 1.0, "order_floor": 1.0,
            "order_check": "true", "scheme_family": "milstein",
            "invariant_worst_headroom": 0.42, "dynkin_z": 1.1, "dynkin_bar": 2.53,
            "solve_time_s": 3.0, "notes_for_a_human": "not a ranking input"}
    cut = {k: v for k, v in full.items() if k in run.RECORDED_METRICS}

    assert "notes_for_a_human" not in cut, "the whitelist is still a whitelist"
    assert (ranking.components(cut, 1e-2, [cut, cut])
            == ranking.components(full, 1e-2, [full, full]))


def test_the_thin_gate_abstains_only_below_the_breadth_it_is_given():
    """`thin:K` drops the index where fewer than K components separate the pool.

    Recorded as measured-harmful rather than shipped (SS10h): abstaining lands on
    declaration order, and that is the pool's worst plan on 8 of 9 sciml pools. The
    gate itself still has to do what it says, or the measurement that retired it
    would have been of something else.
    """
    from verifylib import ranking

    # `structure` separates these two; nothing else does.
    a = {"invariant_worst_headroom": 0.9, "dynkin_z": 1.0, "dynkin_bar": 2.43}
    b = {"invariant_worst_headroom": 0.1, "dynkin_z": 1.0, "dynkin_bar": 2.43}
    pool = [a, b]
    varying = [k for k in ranking.CERT_COMPONENTS
               if len({ranking.components(s, 1e-2, pool).get(k) for s in pool}) > 1]
    assert varying == ["structure"], varying

    assert rank_study._keys("abstain", 1e-2, pool, 0), "no gate: the keys stand"
    assert rank_study._keys("abstain", 1e-2, pool, 1), "breadth 1 is satisfied"
    assert rank_study._keys("abstain", 1e-2, pool, 2) == (), "breadth 2 is not"


def test_ordertie_prefers_the_higher_measured_order_and_says_why_it_was_not_shipped():
    """`ordertie:K` replaces declaration order as the fall-through.

    It is the one non-fitted candidate -- `observed_order` is a property of the
    method, not a position in a list -- and it is still not shipped, because an
    **exact** sampler has no convergence order to measure. `observed_order` is None
    for the closed-form plans that are the best plan in 7 of 9 holdout pools, so the
    tiebreak sorts them last: holdout falls to 2 of 9 at K=3, where the gate first
    fires there. This pins both halves of that.
    """
    hi = {"observed_order": 1.0, "score": 10, "provenance": "analytic"}
    lo = {"observed_order": 0.5, "score": 10, "provenance": "analytic"}
    exact = {"score": 10, "provenance": "analytic"}  # a closed form: no order at all

    def key(s):
        return rank_study.key_of(_plan("p", s), "ordertie", [hi, lo], 0.0)

    assert key(hi) < key(lo), "a higher measured order wins the tiebreak"
    assert key(lo) < key(exact), (
        "and this is the defect that kept it unshipped: an exact sampler has no "
        "order, so it loses to a discretisation scheme with a measured one")
