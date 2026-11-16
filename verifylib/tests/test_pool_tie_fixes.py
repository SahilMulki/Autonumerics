"""The second N01 failure (2026-11-16): a pool the kernel could not separate.

plan-notable-sde.md §8d. With sub-increments passed, the kernel measured the strong
order the problem turns on -- 0.49 for Euler, 1.01 for the Levy-area Milstein -- and
the ranker still tied the pool and shipped the Euler control, because:

1. S is integrated exactly by every scheme, its ladder sat at roundoff, it
   abstained, and the abstention vetoed the measured U component's verdict;
2. one 2.3-sigma draw on U's zero mean, shared by every plan through the graded
   call's common increments, failed the bare 2-sigma absolute mean test on both
   plans (and the formulator's martingale invariant, the same test again);
3. the symbolic initial condition `["S_0", "U_0"]` crashed Dynkin once the moments
   passed and it ran.
"""
import json
import pathlib

import numpy as np
import pytest

from verifylib import ranking
from verifylib.kernel import run
from verifylib.kernel.sde import constraints, crn, driver, dynkin

from .test_sub_increments import _EULER, _STRICT_MILSTEIN, _clark_cameron_spec, _plan

# --- 1. an exactly integrated component does not veto ------------------------------


def test_roundoff_is_told_apart_from_noise():
    assert crn.at_roundoff({"roundoff_floor": 2e-12, "s0": 7e-16, "s1": 6e-16})
    assert not crn.at_roundoff({"roundoff_floor": 2e-12, "s0": 3e-2, "s1": 1.5e-2})
    assert not crn.at_roundoff({})


@pytest.fixture(scope="module")
def pool(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("tie")
    spec = _clark_cameron_spec()
    # The N01 formulator's martingale, the zero-mean test a second time.
    spec["verification"]["invariants"] = [
        {"name": "martingale", "gate": False, "expr": "X[:, 1]",
         "reference_value_at_T": 0.0}]
    euler = run(spec, _plan(tmp, "1-euler", _EULER, "scheme_family: euler-maruyama"), {})
    milstein = run(spec, _plan(tmp, "2-milstein", _STRICT_MILSTEIN,
                               "scheme_family: milstein\n"
                               "levy_area_treatment: sub-increments"), {})
    return euler, milstein


def test_the_measured_component_carries_the_order_verdict(pool):
    for m in pool:
        orders = m["detail"]["orders_ok"]
        assert m["checks"]["orders_ok"] is True, orders
        assert orders["exact_components"] == [0]
        assert orders["component"] == 1


def test_every_component_at_roundoff_still_abstains():
    """The excuse needs a measured component beside it."""
    exact = {"orders_ok": "unresolved", "abstained": "roundoff", "component": 0,
             "strong_order": None,
             "strong_detail": {"roundoff_floor": 1e-12, "s0": 1e-16, "s1": 1e-16}}
    per = [exact, {**exact, "component": 1}]
    others = [e for e in per if not crn.at_roundoff(e["strong_detail"])]
    assert others == []          # so `_orders` falls back to binding on all of them


def test_the_ranker_now_ships_the_levy_area_milstein(pool):
    entries = []
    for name, m in zip(("1-euler", "2-milstein"), pool, strict=True):
        s = {**m["summary"], "score": m["score"], "provenance": m["provenance"],
             "certified": m["certified"], "kind": "sde"}
        entries.append({"plan": name, "kernel": {"score": m["score"],
                                                 "provenance": m["provenance"],
                                                 "certified": m["certified"],
                                                 "summary": s}})
    summaries = [e["kernel"]["summary"] for e in entries]
    entries.sort(key=lambda e: ranking.rank_key(e, summaries))
    assert entries[0]["plan"] == "2-milstein"
    tol = ranking._tol_of(summaries[0], {})
    assert "order" in ranking.comparable_keys(tol, summaries)


# --- 2. the zero-mean test is held at the family-wise level ----------------------


def test_a_two_point_three_sigma_zero_mean_passes_at_the_family_level():
    """N01's numbers: 0.0429 against se 0.0187."""
    assert constraints.compare_moment(0.0429, 0.0187, 0.0, 0.05, near_zero=0.01)[0] is False
    ok, detail = constraints.compare_moment(0.0429, 0.0187, 0.0, 0.05, near_zero=0.01,
                                            abs_ci_mult=2.608)
    assert ok is True and detail["ci_mult_used"] == pytest.approx(2.608)


def test_a_real_mean_bias_still_fails_the_family_level():
    assert constraints.compare_moment(0.08, 0.0187, 0.0, 0.05, near_zero=0.01,
                                      abs_ci_mult=2.608)[0] is False


def test_the_relative_branch_keeps_its_own_multiplier():
    ok, detail = constraints.compare_moment(2.019, 0.0328, 2.0, 0.05, near_zero=0.01,
                                            abs_ci_mult=9.0)
    assert ok is True and "ci_mult_used" not in detail


def test_the_family_counts_the_moments_and_the_martingales(pool):
    fw = pool[0]["detail"]["moments_ok"]["family_wise"]
    assert fw["tests"] == 5                      # 2 means + 2 variances + 1 martingale
    assert fw["multiplier"] == pytest.approx(dynkin.family_wise_multiplier(2.0, 5))
    assert all(m["checks"]["moments_ok"] is True for m in pool)
    assert all(m["checks"]["invariants_ok"] is True for m in pool)


def test_a_declared_martingale_ci_mult_is_never_overwritten():
    verification = {"invariants": [
        {"name": "martingale", "expr": "X"},
        {"name": "martingale", "expr": "X", "ci_mult": 3.5},
        {"name": "finite", "gate": True}]}
    snapshot = json.dumps(verification, sort_keys=True)
    indices = driver._undeclared_ci_martingales(verification)
    assert indices == [0]
    out = driver._with_family_ci(verification, 2.6, indices)
    assert out["invariants"][0]["ci_mult"] == 2.6
    assert out["invariants"][1]["ci_mult"] == 3.5
    assert json.dumps(verification, sort_keys=True) == snapshot   # not mutated


# --- 3. a symbolic initial condition ---------------------------------------------


def test_a_named_initial_condition_resolves_against_the_parameters():
    params = {"S_0": 1.0, "U_0": 0.0}
    spec = {"state_dimension": 2, "parameters": params,
            "initial_condition": {"law": "deterministic", "value": ["S_0", "U_0"]}}
    assert dynkin._initial_state(spec, spec["initial_condition"], params) == [1.0, 0.0]


def test_an_unresolvable_initial_condition_is_none_not_a_crash():
    spec = {"initial_condition": {"law": "deterministic", "value": ["S_0", "q*2"]}}
    assert dynkin._initial_state(spec, spec["initial_condition"], {"S_0": 1.0}) is None


def test_dynkin_runs_on_the_n01_initial_condition(tmp_path):
    spec = _clark_cameron_spec()
    spec["parameters"] = {"S_0": 1.0, "U_0": 0.0, "mu": 1.0}
    spec["initial_condition"] = {"law": "deterministic", "value": ["S_0", "U_0"]}
    for key in ("mean_X", "mean_Y"):
        spec["analytic_moments"][key] = spec["analytic_moments"][key].replace("X_0", "S_0")
    m = run(spec, _plan(tmp_path, "m", _STRICT_MILSTEIN, "scheme_family: milstein\n"
                                                        "levy_area_treatment: x"), {})
    assert m.get("crash") is None, m.get("crash")
    assert m["checks"]["dynkin_ok"] in (True, False), m["detail"].get("dynkin_ok")


def test_the_dynkin_skip_names_the_finding_that_caused_it():
    src = pathlib.Path(driver.__file__).read_text()
    assert '"the moments failed against the exact reference"' in src
    assert np.isfinite(dynkin.family_wise_multiplier(2.0, 5))
