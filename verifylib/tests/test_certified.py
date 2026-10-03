"""``certified`` -- the field that lets the score change without moving the gate.

``benchmark/report.py``'s VERIFIED_PASS and the conductor's winner early-exit both
asked "is ``score == 10``". That spelling ties two behaviours to one integer at the
moment the score starts resolving *quality* below the tolerance line. ``certified``
carries the original question -- did every gate this evidence path requires come
back clean -- so the score is free to move and neither behaviour does.

The whole point is that it is **inert today**: these tests pin that, and they are
what a later change to the score has to keep passing.
"""
import json
import os

import pytest

from benchmark import report as R
from verifylib.kernel.score import certify
from verifylib.tests.test_kernel_score import CLEAN_A, CLEAN_B, score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(REPO, "benchmark", "results")


# --- the kernel side ---------------------------------------------------------

@pytest.mark.parametrize("base", [CLEAN_A, CLEAN_B], ids=["path_a", "path_b"])
@pytest.mark.parametrize("kw", [
    dict(tier_b=True, d1_outcome="clean", operator_validated=True),   # 10, two breaks
    dict(tier_b=True, d1_outcome="unavailable"),                      # 9, one break
    dict(tier_b="unavailable", d1_outcome="unavailable"),             # 8, Tier C
    dict(gate_violation=True),                                        # 3, gated
    dict(crashed=True),                                               # 1
    dict(any_test_ran=False),                                         # 2
], ids=["two_breaks", "one_break", "tier_c", "gated", "crash", "nothing_ran"])
def test_certified_equals_score_ten_across_every_certification_row(base, kw):
    got = score(base, **kw)
    assert got["certified"] is (got["score"] >= 10), (
        f"certified {got['certified']} against score {got['score']} "
        f"(bound by {got['bound_by']})")


def test_a_cap_that_lowers_the_score_below_ten_also_clears_certified():
    """A cap is the one thing that can decertify an otherwise-clean run, and it has
    to take both fields with it -- an agent that can stop a wrong 10 from shipping
    but not a wrong VERIFIED_PASS has only half the authority §2c gives it."""
    clean = score(CLEAN_A, tier_b=True, d1_outcome="clean", operator_validated=True)
    assert (clean["score"], clean["certified"]) == (10, True)
    capped = score(CLEAN_A, tier_b=True, d1_outcome="clean", operator_validated=True,
                   agent_cap={"score": 4, "reason": "SOLUTION.md describes ETDRK4; "
                                                    "solver.py integrates with forward Euler"})
    assert (capped["score"], capped["certified"]) == (4, False)
    assert capped["bound_by"] == "agent_cap"


def test_certified_is_a_plain_bool_so_it_survives_json_and_the_metrics_block():
    got = certify({**CLEAN_A, "tier_b": True})
    assert type(got["certified"]) is bool
    assert json.loads(json.dumps(got))["certified"] == got["certified"]


# --- the benchmark side ------------------------------------------------------

def _rec(**kw):
    base = {"has_ground_truth": True, "ground_truth_kind": "exact",
            "run": {"status": "completed"},
            "verify": {"status": "ok", "passed": True},
            "pipeline": {"best_score": 10}}
    base.update(kw)
    return base


def test_compute_verdict_prefers_certified_over_the_score():
    """The split only earns its keep if the verdict follows `certified`. A record
    scoring 9 because its accuracy headroom is thin, on a run where every gate came
    back clean, is a VERIFIED_PASS -- not the UNDERCLAIM `best_score >= 10` reads."""
    thin = _rec(pipeline={"best_score": 9, "certified": True})
    assert R.compute_verdict(thin) == "VERIFIED_PASS"
    # ...and the converse: a 10 that did not certify is not a pass.
    assert R.compute_verdict(
        _rec(pipeline={"best_score": 10, "certified": False},
             verify={"status": "ok", "passed": False})) == "FAIL"


def test_frozen_records_fall_back_to_the_old_test_unchanged():
    """Every record written before the split carries no `certified`, and must be
    read exactly as it was. This is the inertness claim, run against the real files
    rather than a fixture."""
    seen = 0
    for name in os.listdir(RESULTS):
        if not name.endswith(".json") or ".bak" in name or "selection" in name:
            continue
        try:
            with open(os.path.join(RESULTS, name)) as fh:
                payload = json.load(fh)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        for rec in (payload.get("results") if isinstance(payload, dict) else None) or []:
            if "pipeline" not in rec:
                continue        # a baseline unit: no self-score to read
            assert "certified" not in (rec.get("pipeline") or {})
            expected = (rec["pipeline"].get("best_score") or 0) >= 10
            assert R._pipeline_claimed_pass(rec) is expected, f"{name}:{rec['slug']}"
            seen += 1
    assert seen > 50, f"only {seen} frozen pipeline records checked"


# --- the band: where in it quality places a plan ------------------------------

def test_quality_places_a_plan_inside_its_certification_band():
    """`ceiling()` says which band the evidence opens; quality says whether the run
    earned the top of it. A plan that squeaked under the tolerance and one three
    decades inside it both certify, and only the second scores 10."""
    clean = dict(CLEAN_A, tier_b=True, d1_outcome="clean", operator_validated=True)
    top = score(clean, headroom_fraction=6.7 / 6)
    bottom = score(clean, headroom_fraction=1.99 / 6, headroom_decades=1.99)
    assert (top["score"], top["certified"]) == (10, True)
    assert (bottom["score"], bottom["certified"]) == (9, True)
    assert bottom["bound_by"] == "quality"
    assert "decades inside the tolerance" in bottom["reason"]


def test_placement_reads_accuracy_alone_not_the_composite_index():
    """The correction that mattered. Comparing the *composite* index against a bar
    makes the effective accuracy requirement move with which other checks happened to
    run: where only accuracy and structure were measurable, accuracy carried 64% of the
    weight and a 0.75 bar demanded ~3.6 decades -- and the canary suite's deliberately
    **honest** control solver scored 9. Placement reads decades of headroom directly,
    so the bar means what it says whatever else ran."""
    from verifylib.kernel.score import PLACEMENT_FRACTION, place
    assert abs(PLACEMENT_FRACTION - 1 / 3) < 1e-12
    assert place(10, 2.45 / 6)[0] == 10    # pde_heat_1d FTCS
    assert place(10, 2.59 / 6)[0] == 10    # pde_heat_1d Crank-Nicolson
    assert place(10, 1.99 / 6)[0] == 9     # pde_heat_2d, all three second-order plans
    # ...and the honest SDE canary, whose ceiling is 0.90 decades rather than 6: it
    # sits at 0.896, i.e. 100% of what 50 000 paths allow, and reaches the top.
    assert place(10, 0.896 / 0.90)[0] == 10


def test_unmeasurable_quality_places_at_the_top_never_the_floor():
    """Missing information is not a defect. The rule that a skipped check never
    counts against a plan holds here exactly as it does inside the index -- and in
    the certifying bands it is unreachable anyway, since they require `converged`,
    which requires the finite error the accuracy component reads."""
    clean = dict(CLEAN_A, tier_b=True, d1_outcome="clean", operator_validated=True)
    assert score(clean, headroom_fraction=None)["score"] == 10


@pytest.mark.parametrize("fraction", [None, 0.0, 0.1, 0.33, 1 / 3, 0.5, 1.0, 2.0])
def test_quality_never_raises_a_score_and_never_moves_certified(fraction):
    """The monotonicity the whole design rests on: placement can only lower the
    score, so no plan scores better under the band than it did before it, and
    `certified` is settled off the other caps."""
    for base in (CLEAN_A, CLEAN_B):
        for kw in ({"tier_b": True, "d1_outcome": "clean", "operator_validated": True},
                   {"tier_b": True, "d1_outcome": "unavailable"},
                   {"tier_b": "unavailable", "d1_outcome": "unavailable"},
                   {"gate_violation": True}, {"crashed": True}):
            without = score(base, **kw)
            with_q = score(base, **kw, headroom_fraction=fraction)
            assert with_q["score"] <= without["score"]
            assert with_q["certified"] is without["certified"]


def test_a_findings_band_has_no_width():
    """3, 2 and 1 are findings -- a gate was violated, nothing ran, it crashed -- and
    a finding does not have a quality."""
    from verifylib.kernel.score import BANDS, place
    for band in (3, 2, 1):
        assert band not in BANDS
        assert place(band, 0.0) == (band, None)


def test_the_tier_c_band_is_two_wide():
    """Self-convergence spans 6-8: it is the widest claim with the least evidence
    behind it, so it is where quality has the most to say."""
    from verifylib.kernel.score import place
    assert place(8, 1 / 3)[0] == 8        # the full bar
    assert place(8, 1 / 6)[0] == 7        # one bar in
    assert place(8, 0.0)[0] == 6          # at the tolerance line


def test_placement_only_refines_a_result_the_rubric_put_at_the_top_of_its_band():
    """The rows below a band's top are already accuracy-graded -- §12's 7 is
    "converging, not yet accurate enough" -- so placing them again demotes one
    shortfall twice. Measured: a plan whose graded-grid error sits above the tolerance
    takes 7 from the rubric and was then placed at 6 by a headroom of zero, for the
    same reason, stated twice."""
    # Tier C (band 6-8) with `converged: False`: rubric 7, below the band top.
    below = score(CLEAN_B, tier_b="unavailable", d1_outcome="unavailable",
                  converged=False, headroom_fraction=0.0)
    assert below["score"] == 7 and below["bound_by"] == "rubric"
    # The same evidence with the rubric at the band top *is* placed.
    at_top = score(CLEAN_B, tier_b="unavailable", d1_outcome="unavailable",
                   headroom_fraction=0.0)
    assert at_top["score"] == 6 and at_top["bound_by"] == "quality"
