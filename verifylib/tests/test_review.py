"""Criterion 11: every review passes scoped numeric binding, including those whose
"Feedback for solver" sections contain CFL arithmetic.

Calibration measured over 206 archived reviews (83 from the current pipeline, 123
from the pre-metrics-block archive): 0 false positives.
"""
import glob
import os

import pytest

from verifylib.review import (
    bound_claims,
    bound_section,
    check_review,
    grid_ladder,
    parse_metrics,
    printed_tolerance,
)

from .conftest import REPO, read_json, read_text

METRICS = """<metrics>
provenance: analytic
estimated_rel_error: 1.968e-04
error_is_estimate: false
observed_order: 3.999
order_floor: 1.0
wall_time_s: 0.3
</metrics>
"""

SPEC = {"evaluation_thresholds": {"rel_l2_err_max": 0.01, "grid_N": 64},
        "parameters": {"a": 1.0}}


def _solution(error="1.9675e-04", order="3.999", feedback=True):
    body = f"""{METRICS}
<review score=10>

**Score: 10/10**

### Numerical Accuracy
- provenance:              analytic (measured against the closed form)
- error (l2):              {error} at N=128  (PASS)  -- N=64: 3.1467e-03
- tolerance:               0.01
- observed_order:          {order}  (primary 'u', min 1.0; ok)
- invariants:              mass drift 1.57e-16 (tol 1e-8, ok)

### Feedback for solver
- CFL violated: dt=0.01, dx=0.1 gives dt/dx**2=1.0 > 0.5 stability limit -- halve dt
- Dirichlet BC not re-imposed after each step -- u(0,t) drifts by 4e-3

</review>
"""
    return body if feedback else body.split("### Feedback")[0] + "\n</review>\n"


def test_a_faithful_review_passes():
    assert check_review(_solution(), SPEC) == []


def test_cfl_arithmetic_in_feedback_is_not_bound():
    """The prompt *instructs* the evaluator to write those four numbers. Binding
    that section rejects the review the prompt asked for."""
    with_feedback = check_review(_solution(feedback=True), SPEC)
    without = check_review(_solution(feedback=False), SPEC)
    assert with_feedback == without == []


def test_transcription_drift_is_caught():
    """The review headline disagrees with the metrics block above it."""
    findings = check_review(_solution(error="9.1e-03"), SPEC)
    assert len(findings) == 1 and findings[0].severity == "warning"
    assert "error" in findings[0].message


def test_drifted_order_is_caught():
    assert check_review(_solution(order="1.850"), SPEC)


def test_correct_rounding_is_not_drift():
    """A review prints 0.000026 for a value carried as 2.556e-05, and the metrics
    block prints 1.968e-04 for one the review gives as 1.9675e-04. Both are correct
    rounding; whichever side rounded harder sets the slack."""
    assert check_review(_solution(error="1.97e-04"), SPEC) == []
    assert check_review(_solution(error="0.000197"), SPEC) == []


def test_non_numeric_claim_is_not_mined_for_a_number():
    """"observed_order: inf -- both grid errors are below the 1e-9 floor" claims
    `inf`, not 1e-9."""
    text = "### Numerical Accuracy\n- observed_order: inf -- both errors below the 1e-9 floor\n"
    assert bound_claims(text) == []


def test_a_stub_review_claims_nothing():
    stub = "<review score=0>\n\nAwaiting solver.\n\n</review>"
    assert check_review(stub, SPEC) == []


def test_measurements_without_a_metrics_block_are_flagged():
    findings = check_review(_solution().split("</metrics>")[1], SPEC)
    assert findings and "<metrics>" in findings[0].path


# --- §14 C3: both metrics shapes, from day one -------------------------------

def test_blind_evaluator_metrics_shape_is_accepted():
    """plan-blind-evaluator §7 changes values to {"value": x, "oracle_derived": bool}.
    Cheap if written this way now; a rewrite if not."""
    scalar = parse_metrics("<metrics>\nestimated_rel_error: 1.968e-04\n</metrics>")
    structured = parse_metrics(
        '<metrics>\nestimated_rel_error: {"value": 1.968e-04, "oracle_derived": true}\n</metrics>')
    assert scalar["estimated_rel_error"] == structured["estimated_rel_error"]


# --- helpers -----------------------------------------------------------------

def test_printed_tolerance_is_half_a_unit_in_the_last_place():
    assert printed_tolerance("0.000026") == pytest.approx(5e-7, rel=1e-6)
    assert printed_tolerance("1.968e-04") == pytest.approx(5e-8, rel=1e-6)


def test_grid_ladder():
    assert grid_ladder({"evaluation_thresholds": {"grid_N": 64}}) == [64.0, 127.0, 253.0]
    assert grid_ladder({"evaluation_thresholds": {"grid_N": 64},
                        "boundary_conditions": {"type": "periodic"}}) == [64.0, 128.0, 256.0]


def test_section_scoping():
    assert "CFL" not in bound_section(_solution())
    assert "provenance" in bound_section(_solution())


# --- the calibration itself --------------------------------------------------

def test_no_false_positives_on_archived_reviews():
    """The measurement the design rests on. An earlier version, binding every number
    in the section rather than the claim on each keyed line, flagged 28%."""
    paths = sorted(glob.glob(os.path.join(REPO, "workspace", "*", "plans", "*",
                                          "SOLUTION.md")))
    if not paths:
        pytest.skip("no archived reviews staged")
    flagged = []
    for path in paths:
        text = read_text(path)
        if "<review" not in text:
            continue
        spec_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(path))),
                                 "problem_spec.json")
        spec = read_json(spec_path) if os.path.exists(spec_path) else {}
        for finding in check_review(text, spec):
            if "<metrics>" not in finding.path:
                flagged.append((path, finding.message))
    assert not flagged, f"{len(flagged)} of {len(paths)} reviews false-flagged: {flagged[:3]}"
