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
