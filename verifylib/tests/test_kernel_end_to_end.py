"""The kernel on real artifacts: reproduction, reconciliation, and the CLI.

Criterion 4's gate is that the kernel reproduces each problem's metrics on frozen
artifacts *and* that the four Kuramoto-Sivashinsky plans either reconcile or are
individually flagged. Rev 1's version of that gate -- "existing problems reproduce
their current scores" -- is too weak to keep, because the current scores include a
plan reporting ``observed_order: 0.003`` and scoring 10. Reproducing that
faithfully would be a failure, not a pass.

Plan §12a (3) makes the achievable half explicit: reproduce every metric whose
inputs exist, and enumerate the plans where they do not, naming the missing input.
"""
import glob
import json
import os
import subprocess
import sys

import pytest

from verifylib.kernel import run

from .conftest import REPO, WORKSPACE, requires


def _plans(slug):
    return sorted(p for p in glob.glob(os.path.join(WORKSPACE, slug, "plans", "*"))
                  if os.path.isdir(p) and os.path.exists(os.path.join(p, "solver.py")))


def test_the_four_ks_evaluators_reconcile_onto_one_number():
    """Phase 1's real product. Three regenerated evaluators reported observed orders
    of 10.467, 10.925 and 0.003 for this problem and all three scored 10. The
    kernel gives every plan the same score for the same stated reason -- and it is
    not 10, because an observed order of ~10 on a scheme whose theoretical order is
    4 fails the ``p_sane`` guard, which is what those evaluators were meant to
    apply."""
    spec = requires("pde_kuramoto_sivashinsky")
    plans = _plans("pde_kuramoto_sivashinsky")
    if len(plans) < 3:
        pytest.skip("KS plans not staged")
    results = {os.path.basename(p): run(spec, p, {}) for p in plans}

    scores = {name: m["score"] for name, m in results.items()}
    assert len(set(scores.values())) == 1, f"plans disagree: {scores}"
    assert set(scores.values()) == {4}, scores
    for name, m in results.items():
        richardson = m["detail"]["richardson"]
        # All three plans are spectral, so `p_sane` is the *lower* bound and they
        # clear it. What catches them is that the ladder itself is not one: at
        # t = 50 the coarse and medium solves differ by more than the whole field.
        assert richardson["p_sane_branch"] == "spectral", name
        assert richardson["p_sane"] is True, name
        assert richardson["coarse_resolved"] is False, name
        assert "asymptotic guards fail" in m["certification"]["reason"], name
        # And the flagged reason is the same one, not three different ones.
        assert m["provenance"] == "self_convergence", name


def test_a_plan_with_no_solver_is_reported_rather_than_skipped():
    """The fourth KS plan was never implemented. That is a fact about the run, and
    it has to come back as one."""
    spec = requires("pde_kuramoto_sivashinsky")
    empty = [p for p in glob.glob(os.path.join(WORKSPACE, "pde_kuramoto_sivashinsky",
                                               "plans", "*"))
             if os.path.isdir(p) and not os.path.exists(os.path.join(p, "solver.py"))]
    if not empty:
        pytest.skip("every KS plan has a solver")
    m = run(spec, empty[0], {})
    assert m["score"] == 1 and m["crash"]["reason"] == "no_solve_fn"


@pytest.mark.parametrize("slug,plan,expect", [
    ("pde_heat_1d", "3-fd4-rk4", {"score": 10, "provenance": "analytic", "order": 4.0}),
    ("pde_heat_1d", "2-crank-nicolson", {"score": 10, "provenance": "analytic",
                                         "order": 2.0}),
    ("pde_poisson_2d", "2-fd4-compact-mehrstellen", {"score": 10,
                                                     "provenance": "analytic"}),
])
def test_the_kernel_reproduces_a_correct_plans_measurement(slug, plan, expect):
    spec = requires(slug)
    directory = os.path.join(WORKSPACE, slug, "plans", plan)
    if not os.path.isdir(directory):
        pytest.skip(f"{slug}/{plan} not staged")
    m = run(spec, directory, {})
    assert m["score"] == expect["score"], m["certification"]["reason"]
    assert m["provenance"] == expect["provenance"]
    if "order" in expect:
        assert m["summary"]["observed_order"] == pytest.approx(expect["order"], abs=0.15)


def test_a_graded_mesh_makes_d1_abstain_rather_than_report_a_wrong_number():
    """Shishkin and Bakhvalov meshes have spacing ratios of 115:1 and 485:1 --
    exactly right for a boundary layer, and exactly what makes a fixed-h stencil
    meaningless. Measured: before this, D1 reported ``stalled`` and the score
    dropped from 10 to 3 on two correct solvers."""
    spec = requires("pde_convection_diffusion_bl")
    graded = [p for p in _plans("pde_convection_diffusion_bl")
              if os.path.basename(p).split("-", 1)[1] in ("shishkin-hybrid",
                                                          "bakhvalov-central")]
    if not graded:
        pytest.skip("boundary-layer plans not staged")
    for directory in graded:
        m = run(spec, directory, {})
        assert m["summary"]["d1_outcome"] == "unavailable", os.path.basename(directory)
        assert "non-uniform mesh" in m["d1"]["reason"]
        assert m["score"] == 10, m["certification"]["reason"]


def test_the_cli_is_the_entry_point_and_emits_a_pasteable_metrics_block():
    """Criterion 1: no evaluator writes an ``evaluate.py``. "No evaluate.py contains
    numerical logic" is not a checkable criterion; "no evaluate.py exists" is."""
    directory = os.path.join(WORKSPACE, "pde_heat_1d", "plans", "3-fd4-rk4")
    if not os.path.isdir(directory):
        pytest.skip("pde_heat_1d not staged")
    cli = os.path.join(REPO, "verifylib", "cli.py")
    got = subprocess.run([sys.executable, cli, "evaluate", directory],
                         capture_output=True, text=True, check=False, cwd=REPO)
    assert got.returncode == 0, got.stderr
    assert got.stdout.startswith("<metrics>")
    assert "score: 10" in got.stdout and "provenance: analytic" in got.stdout
    assert "bound by:" in got.stdout

    as_json = subprocess.run([sys.executable, cli, "evaluate", directory, "--json"],
                             capture_output=True, text=True, check=False, cwd=REPO)
    metrics = json.loads(as_json.stdout)
    assert metrics["score"] == 10
    # Criterion 12a(1): every configured value carries where it came from.
    assert all(set(v) == {"value", "source"} for v in metrics["config"].values())
    assert metrics["config"]["grid_N"]["source"] == "spec"


def test_the_config_records_the_source_of_every_value():
    spec = requires("pde_heat_1d")
    directory = os.path.join(WORKSPACE, "pde_heat_1d", "plans", "1-fd-explicit-ftcs")
    if not os.path.isdir(directory):
        pytest.skip("pde_heat_1d not staged")
    m = run(spec, directory, {"grid_N": 17})
    sources = {k: v["source"] for k, v in m["config"].items()}
    assert sources["grid_N"] == "spec", "a spec value outranks an agent selection"
    assert sources["timeout_s"] == "kernel_default"
    assert m["config"]["theoretical_order"]["source"] in ("spec", "kernel_default")


# --- §7a: the two-level probe path, and what it costs -------------------------

CANARY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "canaries", "kernel")


def _canary_plan(tmp_path):
    import shutil
    shutil.copy(os.path.join(CANARY, "honest.py"), tmp_path / "honest.py")
    shutil.copy(os.path.join(CANARY, "honest.py"), tmp_path / "solver.py")
    with open(os.path.join(CANARY, "spec.json")) as fh:
        return json.load(fh)


def test_the_two_level_path_is_taken_only_when_every_precondition_holds(tmp_path):
    """§7a. Rev 1 quoted the saving and not the preconditions, and got the error
    direction wrong besides: a ``p_probe`` that is too high *understates* the error,
    which is the direction that turns a fail into a pass."""
    spec = _canary_plan(tmp_path)

    pinned = run(spec, str(tmp_path), {})
    assert pinned["ladder"]["levels"] == 3, "the spec pins refinement_levels"

    trick = json.loads(json.dumps(spec))
    trick["evaluation_thresholds"].pop("refinement_levels")
    got = run(trick, str(tmp_path), {})
    assert got["ladder"]["levels"] == 2
    assert got["config"]["order_from"]["value"] == "mms_probe"
    assert got["detail"]["richardson"]["fs"] == 3.0, "an imported order needs Roache's 3.0"
    assert got["summary"]["asymptotic_source"] == "probe"
    assert got["score"] == pinned["score"], "the saving must not move the answer"

    shock = json.loads(json.dumps(trick))
    shock["verification"]["structural_facts"] = {"shock": "declared"}
    assert run(shock, str(tmp_path), {})["ladder"]["levels"] == 3

    waived = json.loads(json.dumps(trick))
    waived["evaluation_thresholds"]["order_check"] = False
    assert run(waived, str(tmp_path), {})["ladder"]["levels"] == 3


def test_a_value_decided_during_the_run_still_reaches_the_config_record(tmp_path):
    """§12a (1)'s whole point. ``new_metrics`` snapshots the config at creation, and
    several values -- whether the probe path was taken, which Stage-2 checks ran --
    are only decided later. A snapshot taken before they are set records nothing."""
    spec = _canary_plan(tmp_path)
    spec["evaluation_thresholds"].pop("refinement_levels")
    config = run(spec, str(tmp_path), {})["config"]
    for key in ("order_from", "run_temporal", "run_tier_b", "refinement_levels"):
        assert key in config and config[key]["value"] is not None, key


# --- §8a: Tier C at the chaotic reference horizon -----------------------------

def _chaotic_ks():
    spec = requires("pde_kuramoto_sivashinsky")
    spec = json.loads(json.dumps(spec))
    spec["chaotic"] = True
    spec["chaotic_T_ref"] = 5.0
    spec["requirements"].append({
        "id": "R20", "kind": "other", "status": "mapped",
        "quote": "This equation is chaotic and has no closed-form solution",
        "spec_path": "chaotic, chaotic_T_ref"})
    return spec


def test_the_chaotic_waiver_moves_tier_c_rather_than_only_relabelling_it():
    """§8a's second bullet. Before this, ``chaotic: true`` changed ``order_check``
    from ``waived_spec`` to ``waived_chaotic`` and left the score at 4 -- because
    ``order_ok`` is never reached: ``asymptotic`` fails first, and the waiver did
    not touch it.

    At ``t = 50`` the coarse and medium KS solves differ by 406% of the field; at
    ``t = 5`` by 0.2%. Only the second is a ladder.
    """
    plan = os.path.join(WORKSPACE, "pde_kuramoto_sivashinsky", "plans", "1-etdrk4-spectral")
    if not os.path.isdir(plan):
        pytest.skip("KS not staged")

    staged = run(requires("pde_kuramoto_sivashinsky"), plan, {})
    assert staged["score"] == 4
    assert staged["detail"]["richardson"]["coarse_resolved"] is False

    shifted = run(_chaotic_ks(), plan, {})
    assert shifted["summary"]["error_horizon"] == 5.0
    assert shifted["detail"]["richardson"]["coarse_resolved"] is True
    assert shifted["detail"]["richardson"]["asymptotic"] is True
    assert shifted["score"] == 9 and shifted["provenance"] == "manufactured_partial"
    assert any("chaotic reference horizon" in n for n in shifted["notes"])

    # The shifted horizon has to reach the *block*, not just the JSON: the block is
    # what the conductor ranks on and what REPORT.md quotes, and an error measured
    # at t = 5 must not be read as the error at t = 50.
    from verifylib.kernel.metrics import render_block
    block = render_block(shifted)
    assert "error_horizon: 5.0" in block
    assert "order_check: waived_chaotic" in block


def test_the_two_level_saving_is_refused_on_a_chaotic_problem():
    """§7a imports the order from an MMS probe run at full T, while a chaotic
    problem's Tier C is measured at ``chaotic_T_ref``. Mixing an order from one
    horizon with differences from another is not justified, and the fallback that
    buys a third level would have appended a shifted-horizon run to the full-T
    ladder that D1 and D2 read."""
    plan = os.path.join(WORKSPACE, "pde_kuramoto_sivashinsky", "plans", "1-etdrk4-spectral")
    if not os.path.isdir(plan):
        pytest.skip("KS not staged")
    spec = _chaotic_ks()
    spec["evaluation_thresholds"].pop("refinement_levels")   # would otherwise qualify
    got = run(spec, plan, {})
    assert got["ladder"]["levels"] == 3
    assert got["config"]["order_from"]["value"] == "ladder"
    assert got["summary"]["error_horizon"] == 5.0


def test_a_solver_that_ignores_the_horizon_override_is_caught_not_believed():
    """A solver that hard-codes ``t_final`` runs cleanly, returns ``status: ok``,
    and hands back the full-T field. Measured: two of the three KS solvers do
    exactly that, so the shifted ladder would have measured convergence at t = 50
    while reporting it as t = 5. Manual §7 already says such a solver cannot be
    certified above Tier C; this is what makes it detectable.
    """
    plan = os.path.join(WORKSPACE, "pde_kuramoto_sivashinsky", "plans",
                        "2-imex-sbdf3-spectral")
    if not os.path.isdir(plan):
        pytest.skip("KS not staged")
    got = run(_chaotic_ks(), plan, {})
    assert "error_horizon" not in got["summary"], "it must not claim the short horizon"
    assert "ignored override" in got["ladder"]["chaotic_T_ref_unavailable"]
    assert got["score"] == 4, "and it stays on the full-T ladder, which does not converge"
