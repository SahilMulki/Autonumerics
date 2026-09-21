"""F1, F2 and F4 from docs/no-closed-form-findings.md, on canaries and frozen artifacts.

F1: the chaotic waiver waives the *order*, not the *accuracy*. Tier C at
``chaotic_T_ref`` certifies a horizon the problem does not ask about; the relative
difference between the two finest full-T solves is the accuracy statistic at the
horizon it does ask about, and ``converged`` reads it.

F2: cross-plan agreement, measured by ``cli.py compare`` and read back by the
kernel against both solver hashes, is the second circularity break a plan whose
D1 could not run can actually earn -- and disagreement caps *both* plans.

F4: ``evaluation_thresholds.graded_N`` names the ladder level being certified.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from verifylib.kernel import agreement, run
from verifylib.kernel.metrics import render_block
from verifylib.schema import check_spec

from .conftest import REPO, WORKSPACE, requires
from .test_kernel_canaries import CANARY, _spec

CLI = os.path.join(REPO, "verifylib", "cli.py")


def _chaotic_spec(**edits):
    spec = _spec(**edits)
    spec["chaotic"], spec["chaotic_T_ref"] = True, 0.1
    spec["requirements"].append({
        "id": "R4", "kind": "other", "status": "mapped", "spec_path": "chaotic, chaotic_T_ref",
        "quote": "trajectories separate past the Lyapunov time"})
    return spec


def _unshadowable_spec():
    spec = _chaotic_spec()
    spec["unshadowable"] = True
    spec["requirements"].append({
        "id": "R5", "kind": "other", "status": "mapped", "spec_path": "unshadowable",
        "quote": "no resolution shadows the trajectory to the reporting horizon"})
    return spec


def _plan(tmp_path, solver, name="plan"):
    plan = tmp_path / name
    plan.mkdir(exist_ok=True)
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "honest.py")
    shutil.copy(os.path.join(CANARY, solver), plan / "solver.py")
    return str(plan)


# --- F1 -------------------------------------------------------------------------

def test_a_solver_accurate_at_t_ref_and_wrong_at_t_does_not_certify(tmp_path):
    """The KS shape on the reaction-diffusion canary: the reference-horizon GCI
    is 1e-4, the full-T pair difference is 3%, and the plan is 'converging, not
    accurate enough' rather than certified."""
    m = run(_chaotic_spec(), _plan(tmp_path, "grows_with_t_and_dx.py"), {})
    s = m["summary"]
    assert s["error_horizon"] == 0.1
    assert s["estimated_rel_error"] < 0.01, s
    assert s["estimated_rel_error_T"] > 0.01, s
    assert m["checks"]["converged"] is False
    assert m["detail"]["richardson"]["d21_T"] == pytest.approx(s["estimated_rel_error_T"])
    assert m["score"] == 7, m["certification"]
    assert "not yet accurate enough" in m["certification"]["reason"]
    assert any("under-resolved at t_final" in n for n in m["notes"]), m["notes"]
    block = render_block(m)
    assert "estimated_rel_error_T:" in block and "error_horizon: 0.1" in block


def test_the_honest_solver_still_certifies_under_the_chaotic_waiver(tmp_path):
    """The negative control: resolved grids agree at T, so the statistic passes."""
    m = run(_chaotic_spec(), _plan(tmp_path, "honest.py"), {})
    assert m["checks"]["converged"] is True
    assert m["summary"]["estimated_rel_error_T"] < 0.01
    assert m["score"] == 10, m["certification"]


def test_a_declared_unshadowable_horizon_is_named_and_capped_not_sent_to_refine(tmp_path):
    """[rev2]: the outcome is declared, not inferred. With the flag the same
    solver comes back ``unshadowable`` at 8; the feedback says 'do not refine'."""
    m = run(_unshadowable_spec(), _plan(tmp_path, "grows_with_t_and_dx.py"), {})
    assert m["checks"]["converged"] == "unshadowable"
    assert m["score"] == 8, m["certification"]
    assert "unshadowable" in m["certification"]["reason"]
    assert "refine" in m["certification"]["reason"]
    assert any("unshadowable" in n for n in m["notes"])


def test_unshadowable_does_not_loosen_a_solver_that_agrees_at_t(tmp_path):
    """The flag changes the diagnosis of a failing pair difference, not the
    pass: resolved grids that agree at T still certify."""
    m = run(_unshadowable_spec(), _plan(tmp_path, "honest.py"), {})
    assert m["checks"]["converged"] is True
    assert m["score"] == 10


def test_unshadowable_is_gated_like_chaotic():
    spec = _spec()
    spec["unshadowable"] = True
    found = [f for f in check_spec(spec) if f.path in ("unshadowable", "requirements")]
    assert any("chaotic: true" in f.message for f in found)
    assert any("ledger" in f.message for f in found)
    assert not [f for f in check_spec(_unshadowable_spec()) if f.severity == "error"]


# --- F4 -------------------------------------------------------------------------

def test_graded_n_certifies_the_named_level_not_the_top_of_the_ladder(tmp_path):
    """A first-order perturbation of 0.9 dx: the top-level GCI clears 1% and the
    pair difference at the graded (base) grid does not."""
    plan = tmp_path / "p"
    plan.mkdir()
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "honest.py")
    (plan / "solver.py").write_text("""
import numpy as np
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    if override and ("source" in override or "ic" in override):
        return out
    dx = float(out["grid"]["x"][1] - out["grid"]["x"][0])
    u = np.asarray(out["numerical_solution"]) * (1.0 + 0.9 * dx)
    out["numerical_solution"], out["fields"] = u, {"u": u}
    for s in out["snapshots"]:
        s["fields"] = {"u": np.asarray(s["fields"]["u"]) * (1.0 + 0.9 * dx)}
    return out
""")
    spec = _spec(**{"evaluation_thresholds.min_spatial_order": 0.5})
    top = run(spec, str(plan), {})
    assert top["checks"]["converged"] is True, top["detail"]["converged"]
    assert "graded_N" not in top["summary"]

    spec["evaluation_thresholds"]["graded_N"] = 33
    graded = run(spec, str(plan), {})
    s = graded["summary"]
    assert s["graded_N"] == 33
    assert s["estimated_rel_error"] < 0.01 < s["estimated_rel_error_graded"], s
    assert graded["checks"]["converged"] is False
    assert graded["score"] == 7
    assert any("graded grid N=33" in n for n in graded["notes"])
    assert "graded_N: 33" in render_block(graded)
    assert "estimated_rel_error_graded:" in render_block(graded)


def test_graded_n_at_the_top_level_changes_nothing(tmp_path):
    spec = _spec()
    spec["evaluation_thresholds"]["graded_N"] = 129
    m = run(spec, _plan(tmp_path, "honest.py"), {})
    assert m["score"] == 10
    assert m["summary"]["estimated_rel_error_graded"] == m["summary"]["estimated_rel_error"]


def test_graded_n_is_validated():
    spec = _spec()
    spec["evaluation_thresholds"]["graded_N"] = "128"
    assert any(f.path == "evaluation_thresholds.graded_N" for f in check_spec(spec))


# --- F2 -------------------------------------------------------------------------

NO_SNAPSHOTS = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    out.pop("snapshots")
    return out
"""

SCALED = """
import numpy as np
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    u = np.asarray(out["numerical_solution"]) * 1.05
    out["numerical_solution"], out["fields"] = u, {"u": u}
    return out
"""


def _workspace(tmp_path, plans):
    """``plans`` is ``{name: (solver_text, scheme_family)}`` under one spec."""
    ws = tmp_path / "ws"
    (ws / "plans").mkdir(parents=True)
    (ws / "problem_spec.json").write_text(json.dumps(_spec()))
    out = {}
    for name, (text, family) in plans.items():
        plan = ws / "plans" / name
        plan.mkdir()
        shutil.copy(os.path.join(CANARY, "honest.py"), plan / "honest.py")
        (plan / "solver.py").write_text(text)
        (plan / "SOLUTION.md").write_text(f"---\nscheme_family: {family}\nspatial_order: 2\n---\n")
        out[name] = str(plan)
    return ws, out


def _compare(a, b, *flags):
    proc = subprocess.run([sys.executable, CLI, "compare", a, b, "--json", *flags],
                          capture_output=True, text=True, cwd=REPO, timeout=600)
    assert proc.returncode in (0, 2), proc.stderr
    return proc.returncode, json.loads(proc.stdout)


def test_compare_records_agreement_and_the_kernel_prices_it_as_a_break(tmp_path):
    """Two plans of different declared families, both without snapshots (D1
    unavailable, so 9 on Tier B alone). Agreement recorded by `compare` lifts
    the one the kernel evaluates next to 10 -- measured by the kernel, never
    asserted by an agent."""
    ws, plans = _workspace(tmp_path, {"1-fd": (NO_SNAPSHOTS, "fd"),
                                      "2-fe": (NO_SNAPSHOTS, "fe")})
    before = run(_spec(), plans["1-fd"], {})
    assert before["score"] == 9 and before["checks"]["cross_plan_agreement"] == "unavailable"

    code, entry = _compare(plans["1-fd"], plans["2-fe"])
    assert code == 0 and entry["agree"] is True and entry["different_family"] is True
    assert entry["N"] == 129 and entry["rel_diff"] < 1e-12
    assert os.path.exists(ws / "agreements.json")

    after = run(_spec(), plans["1-fd"], {})
    assert after["checks"]["cross_plan_agreement"] is True
    assert after["score"] == 10 and after["provenance"] == "manufactured"
    assert after["certification"]["agreement_is_break"] is True
    ceiling = next(c for c in after["certification"]["caps"] if c["source"] == "certification")
    assert "different scheme family" in ceiling["reason"]


def test_agreement_within_one_family_is_recorded_but_is_not_a_break(tmp_path):
    _, plans = _workspace(tmp_path, {"1-fd": (NO_SNAPSHOTS, "fd"),
                                     "2-fd": (NO_SNAPSHOTS, "fd")})
    code, entry = _compare(plans["1-fd"], plans["2-fd"])
    assert code == 0 and entry["different_family"] is False
    m = run(_spec(), plans["1-fd"], {})
    assert m["checks"]["cross_plan_agreement"] == "unavailable"
    assert m["score"] == 9


def test_disagreement_caps_both_plans_at_eight(tmp_path):
    """The difference does not say which plan is wrong, so the cap goes on both
    (F2 [rev]), with the measured difference as the reason."""
    _, plans = _workspace(tmp_path, {"1-fd": ("from honest import solve_pde\n", "fd"),
                                     "2-fe": (SCALED, "fe")})
    code, entry = _compare(plans["1-fd"], plans["2-fe"])
    assert code == 2 and entry["agree"] is False
    assert entry["rel_diff"] == pytest.approx(0.05 / 1.025, rel=1e-3)
    for name in plans:
        m = run(_spec(), plans[name], {})
        assert m["checks"]["cross_plan_agreement"] is False
        assert m["score"] <= 8
        assert m["certification"]["bound_by"] == "cross_plan_disagreement" or m["score"] < 8
        assert "Neither plan" in json.dumps(m["certification"]["caps"])


def test_a_comparison_goes_stale_when_either_solver_changes(tmp_path):
    _, plans = _workspace(tmp_path, {"1-fd": (NO_SNAPSHOTS, "fd"),
                                     "2-fe": (NO_SNAPSHOTS, "fe")})
    _compare(plans["1-fd"], plans["2-fe"])
    with open(os.path.join(plans["2-fe"], "solver.py"), "a") as fh:
        fh.write("\n# edited after the comparison\n")
    m = run(_spec(), plans["1-fd"], {})
    assert m["checks"]["cross_plan_agreement"] == "unavailable"
    assert m["agreement"]["stale"] and not m["agreement"]["entries"]
    assert m["score"] == 9


def test_compare_refuses_plans_of_different_workspaces(tmp_path):
    _, a = _workspace(tmp_path / "a", {"1-fd": (NO_SNAPSHOTS, "fd")})
    _, b = _workspace(tmp_path / "b", {"1-fd": (NO_SNAPSHOTS, "fd")})
    proc = subprocess.run([sys.executable, CLI, "compare", a["1-fd"], b["1-fd"]],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 1 and "same workspace" in proc.stderr


def test_for_plan_reads_only_live_entries(tmp_path):
    ws = tmp_path / "ws"
    (ws / "plans" / "a").mkdir(parents=True)
    (ws / "plans" / "b").mkdir(parents=True)
    (ws / "plans" / "a" / "solver.py").write_text("a")
    (ws / "plans" / "b" / "solver.py").write_text("b")
    sha = {"a": "ha", "b": "hb"}
    agreement.record(str(ws), {"a": "a", "b": "b", "rel_diff": 1e-5, "tol": 1e-2,
                               "agree": True, "different_family": True, "N": 65,
                               "family_a": "fd", "family_b": "spectral",
                               "sha_a": "ha", "sha_b": "hb", "interpolated": False})
    got = agreement.for_plan(str(ws / "plans" / "a"),
                             sha_of=lambda p: sha[os.path.basename(p)])
    assert got["agreement"] is True and got["entries"][0]["other"] == "b"
    sha["b"] = "changed"
    got = agreement.for_plan(str(ws / "plans" / "a"),
                             sha_of=lambda p: sha[os.path.basename(p)])
    assert got["agreement"] is None and got["stale"]


# --- the frozen KS artifacts ----------------------------------------------------

KS_PLANS = {"1-spectral-etdrk4": 9, "2-spectral-imex-sbdf3": 10, "3-fd4-imex-cnab2": 7}


@pytest.mark.parametrize("plan, expect", sorted(KS_PLANS.items()))
def test_the_frozen_ks_plans_grade_as_the_harness_does(plan, expect):
    """Findings F1's acceptance check. Before: 9 / (unscored) / **10**, and the
    harness measured the FD4 plan at 7.2e-02 (OVERCLAIM). After: the FD4 plan's
    full-T pair difference is 6.9e-02 and it scores 7; the spectral plans' is
    2.1e-03. Plan 2 reaches 10 because the wrap-node trim (F3 (0)) makes the
    spectral stencil selectable and its D1 is then a measurement; plan 1's
    snapshots are on N-1 nodes while its fields are on N, which is named."""
    directory = os.path.join(WORKSPACE, "pde_kuramoto_sivashinsky", "plans", plan)
    if not os.path.isdir(directory):
        pytest.skip("KS not staged")
    m = run(requires("pde_kuramoto_sivashinsky"), directory, {})
    assert m["score"] == expect, m["certification"]
    d21_T = m["summary"]["estimated_rel_error_T"]
    if plan.startswith("3-"):
        assert 0.06 < d21_T < 0.08, d21_T
        assert m["checks"]["converged"] is False
    else:
        assert 1e-3 < d21_T < 3e-3, d21_T
        assert m["checks"]["converged"] is True
    assert m["summary"]["error_horizon"] == 5.0
    assert m["summary"]["estimated_rel_error"] < 1e-4
    if plan.startswith("2-"):
        assert m["d1"]["stencil_pair"] == "fd8+spectral"
        assert m["d1"]["outcome"] == "clean"
        assert m["d1"]["per_level"][-1]["stats_other"]["stencil"] == "spectral"
    if plan.startswith("1-"):
        assert m["d1"]["outcome"] == "unavailable"
        assert "shape" in m["d1"]["reason"]


def test_np_isfinite_guard_in_richardson_reads_nan_as_not_converged():
    from verifylib.kernel import richardson as rich
    x = np.linspace(0, 1, 33)
    u0 = np.sin(np.pi * x)
    u1 = np.sin(np.pi * np.linspace(0, 1, 65))
    u2 = np.sin(np.pi * np.linspace(0, 1, 129))
    u2[64] = np.nan
    got = rich.from_ladder([u0, u1, u2],
                           [[x], [np.linspace(0, 1, 65)], [np.linspace(0, 1, 129)]], 2.0)
    assert got["asymptotic"] is False and not np.isfinite(got["gci"])
    assert "not finite" in got["reason"]
    same = rich.from_ladder([u0, u0, u0], [[x], [x], [x]], 2.0)
    assert same["identical_levels"] is True and same["asymptotic"] is False
    assert "did not change" in same["reason"]
