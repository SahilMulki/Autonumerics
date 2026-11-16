"""Sub-increments in the kernel, and the two N01 defects found beside them.

plan-notable-sde.md §8c. N01 (Clark-Cameron, 2026-11-15) failed in the pipeline with
three correct plans in its pool. The harness handed the solver ``dW_sub`` and the
kernel never did, so every Levy-area Milstein was evaluated on its ``dW_sub=None``
fallback, measured as the same order-1/2 scheme as the Euler control, and the
four-way tie crowned the control. Beside it, a zero exact mean made every plan's
``estimated_rel_error`` about 4e12, and the ``num_paths`` audit flagged the
formulator that evaluated above the graded count rather than the one that did not.
"""
import copy
import json
import os
import textwrap

import numpy as np
import pytest

from verifylib import audit, schema
from verifylib.kernel import plan_meta, run, sandbox
from verifylib.kernel.sde import constraints, crn

from .test_kernel_canaries import CANARY

# --- the near-zero mean ------------------------------------------------------


def test_an_absolute_mode_mean_carries_no_relative_error():
    """The pass is judged absolutely, and the relative error over the 1e-14 guard
    is no longer reported for `_compare` to fold into the worst error."""
    ok, detail = constraints.compare_moment(-0.0436, 0.053, 0.0, 0.05, near_zero=0.01)
    assert ok is True
    assert detail["mode"] == "absolute"
    assert "rel_error" not in detail
    assert detail["abs_error"] == pytest.approx(0.0436)


def test_a_mean_away_from_zero_is_still_relative():
    ok, detail = constraints.compare_moment(2.019, 0.0328, 2.0, 0.05, near_zero=0.01)
    assert ok is True and "mode" not in detail
    assert detail["rel_error"] == pytest.approx(0.0095)


# --- the schema field ----------------------------------------------------------


@pytest.mark.parametrize("value", [2, 64, 256])
def test_a_whole_sub_increment_count_is_accepted(value):
    assert schema.check_sub_increments({"state_dimension": 2, "sub_increments_per_step": value}) == []


@pytest.mark.parametrize("value", [1, 0, 2.5, True, "256"])
def test_an_unusable_sub_increment_count_is_refused(value):
    found = schema.check_sub_increments({"state_dimension": 2, "sub_increments_per_step": value})
    assert len(found) == 1 and found[0].path == "sub_increments_per_step"


def test_an_absent_sub_increment_count_is_the_common_case():
    assert schema.check_sub_increments({"state_dimension": 2}) == []
    assert schema.check_sub_increments({"state_dimension": 2, "sub_increments_per_step": None}) == []


# --- the payloads ----------------------------------------------------------------


def test_the_sub_step_is_held_fixed_across_the_ladder():
    """dt, dt/2, dt/4 all see the same fine grid, as the harness's ladder does."""
    ladder = crn.ladder_spec(100, 0.04, 1.0, levels=3, seed=1, m=2)
    out, detail = crn.with_sub_increments(ladder, 0.04 / 256)
    fines = {p["Nt_fine"] for p in out}
    assert fines == {6400}
    assert [p["aggregate"] for p in out] == [256, 128, 64]
    assert all(p["substeps"] for p in out)
    assert detail["factor"] == detail["factor_declared"] == 64
    assert "reduced" not in detail and "withheld" not in detail


def test_the_sub_step_grid_is_reduced_to_fit_the_memory_cap():
    ladder = crn.ladder_spec(1000, 0.04, 1.0, levels=3, seed=1, m=2)
    cap = 1000 * 100 * 2 * 8 * 10            # room for ten sub-steps per finest step
    out, detail = crn.with_sub_increments(ladder, 0.04 / 256, max_bytes=cap)
    assert detail["factor"] == 10 and "reduced" in detail
    assert out[-1]["aggregate"] == 10 and out[-1]["Nt_fine"] == 1000


def test_fewer_than_two_sub_steps_withholds_the_channel():
    """One sub-increment per step carries no Levy area; a left-point area sum would
    read zero, which is worse than the solver's dW_sub=None fallback."""
    ladder = crn.ladder_spec(1000, 0.04, 1.0, levels=3, seed=1, m=2)
    out, detail = crn.with_sub_increments(ladder, 0.04 / 256,
                                          max_bytes=1000 * 100 * 2 * 8)
    assert "withheld" in detail
    assert out == ladder and not any(p.get("substeps") for p in out)


def test_no_declared_sub_step_leaves_the_payloads_alone():
    ladder = crn.ladder_spec(10, 0.1, 1.0, levels=3, seed=1)
    assert crn.with_sub_increments(ladder, None) == (ladder, None)


def test_the_child_hands_one_path_at_two_resolutions():
    ladder, _ = crn.with_sub_increments(crn.ladder_spec(7, 0.25, 1.0, levels=3, seed=5,
                                                        m=2), 0.25 / 8)
    built = [sandbox._crn_increments_sub(p) for p in ladder]
    for (dW, dW_sub), p in zip(built, ladder, strict=True):
        assert dW_sub.shape == (7, round(1.0 / p["dt"]), p["aggregate"], 2)
        np.testing.assert_allclose(dW_sub.sum(axis=2), dW, rtol=0, atol=1e-14)
    # Every level is the same fine path, regrouped.
    fine = [s.reshape(7, -1, 2) for _, s in built]
    for other in fine[1:]:
        np.testing.assert_array_equal(fine[0], other)


# --- end to end on Clark-Cameron ---------------------------------------------------

_STRICT_MILSTEIN = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None, dW_sub=None, observables=None):
    if dW is None or dW_sub is None:
        raise RuntimeError(f"kernel passed dW={dW is not None} dW_sub={dW_sub is not None}")
    Nt = dW.shape[1]
    dt = T / Nt
    S = np.ones(num_paths)
    U = np.zeros(num_paths)
    acc = {k: np.zeros(num_paths) for k in (observables or {})}
    for k in range(Nt):
        X0 = np.stack([S, U], axis=1)
        w2 = np.cumsum(dW_sub[:, k, :, 1], axis=1)
        w2p = np.concatenate([np.zeros((num_paths, 1)), w2[:, :-1]], axis=1)
        U = U + S * dW[:, k, 0] + np.sum(w2p * dW_sub[:, k, :, 0], axis=1)
        S = S + dt + dW[:, k, 1]
        X1 = np.stack([S, U], axis=1)
        for name, phi in (observables or {}).items():
            acc[name] += 0.5 * (phi(X0) + phi(X1)) * dt
    X = np.stack([S, U], axis=1)
    out = {"terminal_paths": X, "empirical_mean": X.mean(axis=0).tolist(),
           "empirical_variance": X.var(axis=0, ddof=1).tolist(), "dt": dt}
    if observables:
        out["path_integrals"] = acc
    return out
'''

_EULER = _STRICT_MILSTEIN.replace(
    "    if dW is None or dW_sub is None:\n        raise RuntimeError(f\"kernel passed "
    "dW={dW is not None} dW_sub={dW_sub is not None}\")\n", "").replace(
    "        w2 = np.cumsum(dW_sub[:, k, :, 1], axis=1)\n"
    "        w2p = np.concatenate([np.zeros((num_paths, 1)), w2[:, :-1]], axis=1)\n"
    "        U = U + S * dW[:, k, 0] + np.sum(w2p * dW_sub[:, k, :, 0], axis=1)\n",
    "        U = U + S * dW[:, k, 0]\n")


def _clark_cameron_spec():
    with open(os.path.join(CANARY, "sde_spec_2d.json")) as fh:
        spec = copy.deepcopy(json.load(fh))
    spec.update({
        "id": "canary_clark_cameron", "sde_family": "clark_cameron",
        "equation": "dS = mu dt + dW2;  dU = S dW1",
        "parameters": {"X_0": 1.0, "Y_0": 0.0, "mu": 1.0},
        "drift_expression": ["mu + 0.0*X[:, 0]", "0.0*X[:, 0]"],
        "diffusion_expression": ["1.0 + 0.0*X[:, 0]", "X[:, 0]"],
        "sub_increments_per_step": 64,
        "analytic_moments": {"has_analytic_solution": True, "state_dimension": 2,
                             "mean_X": "1 + mu*t", "mean_Y": "0*t",
                             "variance_X": "t",
                             "variance_Y": "t + 1.5*t**2 + t**3/3"},
    })
    spec["evaluation_thresholds"].update(
        {"num_paths": 4000, "dt": 0.04, "dt0": 0.04, "conv_num_paths": 2000,
         "dynkin_levels": 2})
    spec["verification"] = {
        "expected_provenance": "analytic", "expected_strong_order": 0.5,
        "expected_weak_order": 1.0,
        "generator": {"drift": ["mu + 0.0*X[:, 0]", "0.0*X[:, 0]"],
                      "diffusion_matrix": [["0.0*X[:, 0]", "1.0 + 0.0*X[:, 0]"],
                                           ["X[:, 0]", "0.0*X[:, 0]"]]},
        # Column-form test functions, as the 2-D canary declares: the defaults are
        # scalar-state expressions (`X`, `X**2`) that return two columns here.
        "dynkin_test_functions": ["X[:, 0]", "X[:, 0]*X[:, 1]"]}
    return spec


def _plan(tmp_path, name, source, front):
    d = tmp_path / name
    d.mkdir()
    (d / "solver.py").write_text(textwrap.dedent(source).lstrip("\n"))
    (d / "SOLUTION.md").write_text(f"---\n{front}\n---\n\n# plan\n")
    return str(d)


@pytest.fixture(scope="module")
def clark_cameron(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("cc")
    spec = _clark_cameron_spec()
    milstein = run(spec, _plan(tmp, "milstein", _STRICT_MILSTEIN,
                               "scheme_family: milstein\n"
                               "levy_area_treatment: sub-increments"), {})
    euler = run(spec, _plan(tmp, "euler", _EULER, "scheme_family: euler-maruyama"), {})
    return milstein, euler


def test_the_kernel_hands_dw_sub_to_every_solve(clark_cameron):
    """The strict solver raises if any solve gets dW without dW_sub: the graded
    call, all three CRN levels and the Dynkin levels all have to carry it."""
    milstein, _ = clark_cameron
    assert milstein["checks"]["moments_ok"] is True, milstein["detail"]
    assert milstein["detail"]["orders_ok"].get("sub_increments") is not None
    assert milstein["summary"]["sub_increments_graded"]["factor"] == 64
    dynkin = milstein["detail"].get("dynkin_ok") or {}
    assert "crashed" not in str(dynkin.get("reason", "")), dynkin


def test_the_order_study_now_separates_levy_area_milstein_from_euler(clark_cameron):
    """The N01 regression: both read about 1/2 before the kernel passed dW_sub."""
    milstein, euler = clark_cameron
    assert milstein["detail"]["orders_ok"]["strong_order"] > 0.85
    assert euler["detail"]["orders_ok"]["strong_order"] < 0.65


def test_a_zero_exact_mean_no_longer_inflates_the_estimated_error(clark_cameron):
    """U's exact mean is 0: before the fix both plans reported ~1e12 here."""
    for m in clark_cameron:
        assert m["summary"]["estimated_rel_error"] < 0.2, m["summary"]


def test_a_spec_without_the_field_passes_no_sub_increments(tmp_path):
    spec = _clark_cameron_spec()
    del spec["sub_increments_per_step"]
    m = run(spec, _plan(tmp_path, "strict", _STRICT_MILSTEIN, "scheme_family: other"), {})
    assert m["summary"].get("sub_increments_graded") is None
    assert m["score"] <= 3          # the strict solver crashes without dW_sub


# --- the levy_area_treatment frontmatter -----------------------------------------


def test_plan_meta_returns_the_levy_area_treatment(tmp_path):
    (tmp_path / "SOLUTION.md").write_text(
        "---\nscheme_family: milstein\nlevy_area_treatment: sub-increments\n---\n")
    assert plan_meta.read(str(tmp_path))["levy_area_treatment"] == "sub-increments"


# --- the num_paths audit ---------------------------------------------------------


def _audit(thresholds, harness_paths):
    spec = {"evaluation_thresholds": {"seed": 42, **thresholds},
            "time_interval": {"T0": 0.0, "T": 1.0}}
    return audit.audit_sde_output(spec, {"num_paths": harness_paths, "seed": 42,
                                         "T": 1.0, "dt": 0.04})


def test_evaluating_above_the_graded_count_is_not_a_spec_defect():
    """The N02 unrestricted formulator: 20 000 evaluated, 2 000 graded."""
    assert _audit({"num_paths": 20000, "graded_num_paths": 2000}, 2000) == []


def test_a_wrong_graded_count_is_still_a_spec_defect():
    found = _audit({"num_paths": 20000, "graded_num_paths": 5000}, 2000)
    assert len(found) == 1
    assert found[0].path == "evaluation_thresholds.graded_num_paths"


def test_without_a_graded_count_num_paths_is_compared_as_before():
    found = _audit({"num_paths": 20000}, 2000)
    assert len(found) == 1 and found[0].path == "evaluation_thresholds.num_paths"
    assert _audit({"num_paths": 2000}, 2000) == []
