"""The Dynkin check's asymptotic self-consistency test (2026-11-18).

Three of the four Dynkin failures on the notable suite were plans the external
harness passes, and all three failed the same way: a limit read by Richardson
extrapolation off three levels that were not yet in one asymptotic regime. The fix
adds a fourth level and asks whether two overlapping windows of the ladder reach the
*same* limit before the leftover is called a generator violation.

The cases below are the measured ones, reproduced through designed residual
sequences so the test is a statement about the rule rather than about a solver:

* a residual that is ``O(dt^2)`` with a surviving higher-order term — BAOAB's ``q``,
  whose limit went 6.48e-7 -> 4.77e-8 across the windows at a 1.5e-8 standard error;
* a ``dt``-independent offset — §20's documented ``-0.111``, the real violation the
  check exists to catch, which must keep its verdict;
* a slow-but-shrinking residual, which was already an abstention and stays one.

Residuals are built per path, and the noise is **shared across levels**, which is
what CRN does (§18): the paired difference of two limits is then far tighter than
either limit's own standard error, and that is the statistic the test reads.
"""
import numpy as np
import pytest

from verifylib.kernel.sde import driver, dynkin

SPEC = {"parameters": {}}
PHIS = ["X"]
PHI0 = [0.0]
PATHS = 20000


def _runs(residuals):
    """One ``runs`` entry per level from designed per-path residuals.

    ``per_path_residual`` computes ``phi(X_T) - E[phi(X_0)] - integral(L phi)``, so
    handing it ``terminal_paths = x`` and ``L0 = x - r`` makes the residual exactly
    ``r`` with no dependence on how ``phi`` is parsed.
    """
    out = []
    for r in residuals:
        x = np.ones(r.size)
        out.append({"result": {"terminal_paths": x, "path_integrals": {"L0": x - r}}})
    return out


def _levels(fn, n_levels, noise_sd, seed=0):
    """``fn(dt) -> mean residual`` sampled down a halving ladder, coarse to fine."""
    rng = np.random.default_rng(seed)
    shared = rng.standard_normal(PATHS)          # CRN: one path set, every level
    return [np.full(PATHS, fn(1.0 / 2 ** k)) + noise_sd * shared
            for k in range(n_levels)]


# --- the false failures the fix exists to stop -------------------------------

def _second_order_with_tail(dt):
    """BAOAB's ``q`` residual: the O(dt^2) term the extrapolation removes, plus a
    surviving higher-order term that it does not."""
    return -9.377e-3 * dt ** 2 + 2.0e-3 * dt ** 3


def test_three_levels_fail_a_scheme_whose_limit_is_still_moving():
    """The behaviour before the fix, pinned so the fix is measured against it."""
    outcome, detail = dynkin.check(
        SPEC, PHIS, _runs(_levels(_second_order_with_tail, 3, 1e-9)), PHI0)
    assert outcome is False
    assert abs(detail["z"][0]) > 10
    assert detail["window_agreement"] == {}, "three levels cannot run the test"


def test_a_fourth_level_abstains_where_the_limit_is_still_moving():
    outcome, detail = dynkin.check(
        SPEC, PHIS, _runs(_levels(_second_order_with_tail, 4, 1e-9)), PHI0)
    assert outcome == "unresolved"
    w = detail["window_agreement"]["X"]
    assert w["agrees"] is False
    assert abs(w["limit_fine"]) < abs(w["limit_coarse"]), "the limit is shrinking"
    assert w["shift_in_se"] > 5
    assert "not in one asymptotic regime" in detail["reason"]
    assert detail["unresolved_functions"] == ["X"]


def test_the_abstention_is_not_a_pass():
    """`"unresolved"` is a non-verdict everywhere in the driver, and must not be
    cast to True by this fix: a plan forfeits the Dynkin circularity break."""
    outcome, _ = dynkin.check(
        SPEC, PHIS, _runs(_levels(_second_order_with_tail, 4, 1e-9)), PHI0)
    assert outcome is not True
    assert not isinstance(outcome, bool)


# --- the real violation, which must keep its verdict ------------------------

def test_a_dt_independent_offset_still_fails_on_four_levels():
    """§20's -0.111 case. The residual does not move with dt, so both windows
    extrapolate to the same offset, the windows agree, and the failure stands."""
    outcome, detail = dynkin.check(
        SPEC, PHIS, _runs(_levels(lambda dt: -0.111, 4, 5e-3)), PHI0)
    assert outcome is False
    assert detail["window_agreement"]["X"]["agrees"] is True
    assert detail["extrapolated"][0] == pytest.approx(-0.111, abs=2e-3)


def test_a_violation_with_a_small_dt_dependence_still_fails():
    """A sign error does not have to be perfectly dt-independent to be caught: the
    offset dominates, both windows see it, and the verdict holds.

    This is the case that found the roundoff bug. ``-0.111 + 0.01 dt`` extrapolates
    to ``-0.111`` from *either* window analytically, so the shift and its paired
    standard error both collapse to float64 noise -- 1.5e-17 against 1.4e-18, which
    read as 10.8 standard errors of disagreement and abstained on a real violation
    until :data:`dynkin.WINDOW_ROUNDOFF_REL` was added.
    """
    outcome, detail = dynkin.check(
        SPEC, PHIS, _runs(_levels(lambda dt: -0.111 + 0.01 * dt, 4, 5e-3)), PHI0)
    assert outcome is False
    w = detail["window_agreement"]["X"]
    assert w["agrees"] is True
    assert w["at_roundoff"] is True, "agreement is on the roundoff floor, not the s.e."
    assert w["shift_in_se"] > 2.43, "the paired s.e. alone would have abstained"
    assert abs(w["shift"]) < w["roundoff_floor"]


def test_the_roundoff_floor_scales_with_the_ladder_not_the_limit():
    """A limit that is a cancellation of much larger residuals carries their
    roundoff, so the floor is referenced to the residuals the extrapolation was
    formed from. Passing the limit's own magnitude instead would put the floor
    ~16 orders of magnitude too low."""
    fine = np.full(PATHS, 1e-17)       # two limits that cancelled to roundoff
    coarse = np.full(PATHS, 3e-17)
    tight, _ = dynkin.windows_agree(fine, coarse, 2.43)
    loose, detail = dynkin.windows_agree(fine, coarse, 2.43, scale=0.5)
    assert tight is False, "referenced to the limits, roundoff reads as disagreement"
    assert loose is True
    assert detail["roundoff_floor"] == pytest.approx(0.5e-12)


# --- the pre-existing abstention is untouched -------------------------------

def test_a_slow_shrinking_residual_still_abstains():
    """The CIR Feller-violated case: order ~0.07, shrinking. It abstained before the
    fix through `residual_order`'s own branch, and still does."""
    outcome, detail = dynkin.check(
        SPEC, PHIS, _runs(_levels(lambda dt: 0.0289 * dt ** 0.07, 4, 1e-6)), PHI0)
    assert outcome == "unresolved"
    assert detail["order_detail"][0].get("unresolved") is True


# --- windows_agree in isolation ---------------------------------------------

def test_windows_agree_reads_a_paired_standard_error():
    """The two limits share their Brownian paths, so a common per-path offset cancels
    in the difference. Comparing unpaired standard errors would miss a real shift
    buried under a wide common component -- which is the BAOAB case exactly."""
    rng = np.random.default_rng(1)
    common = rng.standard_normal(PATHS)          # wide, and shared
    fine = 1e-8 + common
    coarse = 1e-6 + common
    agrees, detail = dynkin.windows_agree(fine, coarse, 2.43)
    assert agrees is False
    assert detail["shift"] == pytest.approx(-9.9e-7, rel=1e-3)
    assert detail["se_shift"] == pytest.approx(0.0, abs=1e-12)
    unpaired = float(np.std(fine, ddof=1) / np.sqrt(PATHS))
    assert unpaired > 100 * max(detail["se_shift"], 1e-300), \
        "the unpaired bar would have called this agreement"


def test_windows_agree_on_identical_limits():
    rng = np.random.default_rng(2)
    noise = rng.standard_normal(PATHS)
    agrees, detail = dynkin.windows_agree(1.0 + noise, 1.0 + noise, 2.43)
    assert agrees is True
    assert detail["shift"] == pytest.approx(0.0, abs=1e-12)


# --- the driver's escalation policy -----------------------------------------

def _stub(monkeypatch, outcomes):
    """Replace `_dynkin_once` with a counter that returns `outcomes` in order."""
    calls = []

    def fake(spec, plan_dir, solve, num_paths, dt, T, seed, ci_mult, levels=3,
             m=None, max_bytes=None, sub_dt=None):
        calls.append({"levels": levels, "seed": seed})
        return outcomes[len(calls) - 1]

    monkeypatch.setattr(driver, "_dynkin_once", fake)
    return calls


def _call(**kw):
    return driver._dynkin({}, "plans/1", None, 1000, 0.04, 1.0, 42, 2.0, **kw)


def test_a_failure_escalates_to_four_levels(monkeypatch):
    calls = _stub(monkeypatch, [(False, {"z": [9.0]}), (True, {"z": [0.4]})])
    outcome, detail = _call(levels=3)
    assert [c["levels"] for c in calls] == [3, 4]
    assert outcome is True, "the four-level verdict is the one reported"
    assert detail["confirmation"]["ran"] is True
    assert detail["confirmation"]["verdict_on_short_ladder"] is False


def test_an_escalation_that_confirms_the_failure_keeps_it(monkeypatch):
    calls = _stub(monkeypatch, [(False, {}), (False, {"z": [300.0]})])
    outcome, detail = _call(levels=3)
    assert len(calls) == 2
    assert outcome is False
    assert detail["z"] == [300.0], "the detail is the four-level run's own"


def test_a_pass_does_not_pay_for_a_fourth_level(monkeypatch):
    calls = _stub(monkeypatch, [(True, {})])
    outcome, _ = _call(levels=3)
    assert [c["levels"] for c in calls] == [3]
    assert outcome is True


@pytest.mark.parametrize("skipped", [True, "unresolved", "unavailable"])
def test_only_a_false_escalates(monkeypatch, skipped):
    calls = _stub(monkeypatch, [(skipped, {})])
    outcome, _ = _call(levels=3)
    assert len(calls) == 1, f"{skipped!r} is not a failure to confirm"
    assert outcome == skipped


def test_a_ladder_already_long_enough_does_not_escalate(monkeypatch):
    calls = _stub(monkeypatch, [(False, {})])
    _call(levels=4)
    assert [c["levels"] for c in calls] == [4], "check ran the window test itself"


def test_a_failure_over_the_byte_budget_stands_and_says_so(monkeypatch):
    calls = _stub(monkeypatch, [(False, {"z": [9.0]})])
    outcome, detail = _call(levels=3, max_bytes=1)
    assert len(calls) == 1
    assert outcome is False, "the verdict is not downgraded by an unaffordable test"
    assert detail["confirmation"]["ran"] is False
    assert "untested" in detail["confirmation"]["reason"]


def test_the_escalated_run_shares_the_graded_seed(monkeypatch):
    """Not a new seed: the replication wrapper owns seed variation (D9), and an
    escalation that also changed the seed would confound the two."""
    calls = _stub(monkeypatch, [(False, {}), (False, {})])
    _call(levels=3)
    assert {c["seed"] for c in calls} == {42}
