"""The notable-SDE suite (plan-notable-sde.md, steps 3b, 4 and 6): H-N1, H-N4, the arm
separation on every route, and the entries as authored.

No model calls. The sandbox is used where the thing being verified is what the harness
does with a real solver -- the sub-increment plumbing and the reference truth of N01
cannot be shown with synthetic output -- and everything else is driven through
`verify.py`'s `injected=` path, as the diagnostics and pathwise tests are.
"""
import json
import os
import textwrap
from functools import partial

import numpy as np
import pytest

from .conftest import RUN, P, R, S, V

SUITE = P.notable_problems()
N01 = P.by_slug("sde_clark_cameron")
N02 = P.by_slug("sde_kubo_oscillator")


# --- the suite as authored ------------------------------------------------------

def test_the_suite_is_the_size_step_six_authored():
    assert [p["id"] for p in SUITE] == [f"N0{i}" for i in range(1, 10)]
    assert all(p["type"] == "sde" and p["tier"] == 3 for p in SUITE)
    assert all(S.problem_suite(p) == "notable" for p in SUITE)
    assert "notable" in S.SUITES
    assert len(S.apply_suite_filter(list(P.PROBLEMS), "", suite="notable")) == 9
    assert len(S.apply_suite_filter(list(P.PROBLEMS), "", suite="main")) == 49


def test_the_manifest_counts_five_suites_apart(tmp_path):
    S.write_manifest(list(P.PROBLEMS), path=str(tmp_path / "m.json"))
    with open(tmp_path / "m.json") as fh:
        manifest = json.load(fh)
    assert manifest["n_by_suite"] == {"main": 49, "no-closed-form": 7, "benchop": 10,
                                      "sciml": 9, "notable": 9}
    assert manifest["n_total"] == 84


def test_the_gate_constants_problems_mirrors_are_the_verifiers():
    """problems.py cannot import verify.py (verify imports it), so N08's functional
    tolerances mirror the moment gates by value; this is what keeps them equal."""
    assert P.V_SDE_MEAN_TOL == V.SDE_MEAN_TOL and P.V_SDE_VAR_TOL == V.SDE_VAR_TOL


def test_every_entry_has_an_independent_truth_and_a_calibrated_gate():
    for p in SUITE:
        kind = p.get("ground_truth_kind")
        assert p["has_ground_truth"]
        if kind == "pathwise":
            assert V._pathwise_truth_kind(p) in ("exact", "reference"), p["slug"]
            assert p.get("pathwise_tol") is not None, "no pathwise gate is left at the default"
        else:
            assert callable(p.get("moments")) or p.get("functional_truth"), p["slug"]
        # Every statement says the modelling fact and never the answer.
        for key in ("functional_truth",):
            for v in (p.get(key) or {}).values():
                assert f"{v:.4f}" not in p["description"], (p["slug"], v)


def test_the_two_reference_entries_declare_their_measured_error():
    for slug in ("sde_aitsahalia_rate", "sde_sis_epidemic"):
        p = P.by_slug(slug)
        assert p["ground_truth_kind"] == "reference"
        gt = p["moments"](p["T"])
        for k in ("mean", "variance"):
            assert len(gt[k]) == 2 and gt[k][1] > 0.0
            # ...and the error is far below the gate it widens.
            assert gt[k][1] / abs(gt[k][0]) < 1e-3, (slug, k)


# --- H-N1: a pathwise truth that is a functional of the whole path ----------------

def test_hn1_exactly_one_truth_and_a_measured_error():
    assert V._pathwise_truth_kind(N01) == "reference"
    assert V._pathwise_truth_kind(P.by_slug("sde_sciml_linear_ito")) == "exact"
    both = dict(N01, pathwise_exact=lambda t, W: W)
    assert "both" in V._pathwise_truth_kind(both)
    for bad in (None, 0.0, float("nan")):
        assert "without a measured" in V._pathwise_truth_kind(
            dict(N01, pathwise_reference_error=bad))
    neither = {k: v for k, v in N01.items() if k != "pathwise_reference"}
    assert "neither" in V._pathwise_truth_kind(neither)


def test_hn1_a_reference_without_an_error_is_refused_at_grading_time():
    lv = V.sde_pathwise_crn(N01, N01["num_paths"], V.sde_verify_dt(N01))
    ref = V._sde_pathwise_reference(N01, lv)
    result = {"terminal_paths": ref, "empirical_mean": ref.mean(axis=0).tolist(),
              "empirical_variance": ref.var(axis=0, ddof=1).tolist()}
    meta = {"num_paths": N01["num_paths"], "dt": lv["dt"], "T": 1.0, "seed": 42}
    out = V.verify_sde_pathwise(dict(N01, pathwise_reference_error=None), None,
                                injected=(result, meta))
    assert out["status"] == "error" and "measured" in out["error"]
    ok = V.verify_sde_pathwise(N01, None, injected=(result, meta))
    assert ok["status"] == "ok" and ok["pathwise_truth"] == "reference"
    assert ok["pathwise_reference_error"] == N01["pathwise_reference_error"]
    assert ok["pathwise_rel_err"] == 0.0


def test_hn1_sub_increments_are_the_same_path_in_parent_and_child():
    import runner
    lv = V.sde_pathwise_crn(N01, 16, V.sde_verify_dt(N01))
    assert lv["substeps"] is True and lv["aggregate"] == 256
    dW, sub = V.crn_increments_sub(lv)
    assert sub.shape == (16, 25, 256, 2) and dW.shape == (16, 25, 2)
    assert np.array_equal(dW, V.crn_increments(lv))
    assert np.array_equal(sub.sum(axis=2), dW)
    c_dW, c_sub = runner._crn_increments_sub(lv)
    assert np.array_equal(c_dW, dW) and np.array_equal(c_sub, sub)
    # Every ladder level is the same fine path, re-cut.
    fine = V.crn_increments(dict(lv, aggregate=1))
    for level in V.sde_pathwise_ladder(N01, num_paths=16):
        assert level["substeps"] is True
        d, s = V.crn_increments_sub(level)
        assert np.allclose(s.reshape(16, -1, 2), fine)


def test_hn1_a_problem_without_substeps_has_no_flag():
    for p in P.sciml_problems():
        if p.get("ground_truth_kind") == "pathwise":
            assert "substeps" not in V.sde_pathwise_crn(p, 8, V.sde_verify_dt(p))


_EM_N01 = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None, dW_sub=None):
    Nt = dW.shape[1]
    dt = T / Nt
    S = np.ones(num_paths); U = np.zeros(num_paths)
    for k in range(Nt):
        U = U + S * dW[:, k, 0]
        S = S + dt + dW[:, k, 1]
    X = np.stack([S, U], axis=1)
    return {"terminal_paths": X, "empirical_mean": X.mean(axis=0).tolist(),
            "empirical_variance": X.var(axis=0, ddof=1).tolist()}
'''

_MILSTEIN_AREA_N01 = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None, dW_sub=None):
    Nt = dW.shape[1]
    dt = T / Nt
    S = np.ones(num_paths); U = np.zeros(num_paths)
    for k in range(Nt):
        w2 = np.cumsum(dW_sub[:, k, :, 1], axis=1)
        w2_prev = np.concatenate([np.zeros((num_paths, 1)), w2[:, :-1]], axis=1)
        I21 = np.sum(w2_prev * dW_sub[:, k, :, 0], axis=1)
        U = U + S * dW[:, k, 0] + I21
        S = S + dt + dW[:, k, 1]
    X = np.stack([S, U], axis=1)
    return {"terminal_paths": X, "empirical_mean": X.mean(axis=0).tolist(),
            "empirical_variance": X.var(axis=0, ddof=1).tolist()}
'''

_FINE_GRID_EM_N01 = '''
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None, dW_sub=None):
    fine = dW_sub.reshape(num_paths, -1, 2)
    nf = fine.shape[1]
    h = T / nf
    S = np.ones(num_paths); U = np.zeros(num_paths)
    for j in range(nf):
        U = U + S * fine[:, j, 0]
        S = S + h + fine[:, j, 1]
    X = np.stack([S, U], axis=1)
    return {"terminal_paths": X, "empirical_mean": X.mean(axis=0).tolist(),
            "empirical_variance": X.var(axis=0, ddof=1).tolist()}
'''


def _plan(tmp_path, source, name="plan"):
    d = tmp_path / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "solver.py").write_text(textwrap.dedent(source).lstrip("\n"))
    (d / "SOLUTION.md").write_text("---\nscheme_family: other\n---\n\n# plan\n")
    return str(d)


#: N01 at a quarter of its paths, so the three sandboxed ladder solves stay cheap.
N01_SMALL = dict(N01, num_paths=250)


def _graded(problem, plan_dir):
    return V.verify_problem(problem, os.path.dirname(plan_dir), plan_dir=plan_dir,
                            sandbox=True, work_precision=False)


@pytest.mark.parametrize("source, passes", [(_EM_N01, False), (_MILSTEIN_AREA_N01, True)])
def test_hn1_n01_separates_the_levy_area_from_its_absence_through_the_sandbox(tmp_path, source, passes):
    """§5 as the harness grades it, with the child building the sub-increments: Euler
    fails the 2% gate, Milstein with the Levy area from the sub-grid clears it, and
    neither is degenerate across the ladder (both step at `dt`)."""
    out = _graded(N01_SMALL, _plan(tmp_path, source))
    assert out["status"] == "ok", out
    assert out["pathwise_truth"] == "reference"
    assert out["passed"] is passes, out["pathwise_rel_err"]
    assert out["pathwise_ladder"]["status"] == "ok", out["pathwise_ladder"]
    assert out["pathwise_ladder"]["degenerate"] is False


def test_hn1_integrating_on_the_sub_grid_is_excluded_not_rewarded(tmp_path):
    """The reading the sub-grid makes necessary: a solver that ignores `dt` and runs
    Euler on the fine increments is almost exact and returns bit-identical terminal
    states at every ladder level -- excluded as `pathwise_exact`, as a transcription
    is, in the scheme-matched arm; a pass on its merits in the other."""
    out = _graded(N01_SMALL, _plan(tmp_path, _FINE_GRID_EM_N01))
    assert out["status"] == "ok" and out["passed"] is True
    assert out["pathwise_rel_err"] < 5e-3
    assert out["pathwise_ladder"]["degenerate"] is True
    assert out["pathwise_ladder"]["reason"] == "pathwise_exact"
    rec = {"has_ground_truth": True, "ground_truth_kind": "pathwise",
           "run": {"status": "completed"}, "pipeline": {"best_score": 10}, "verify": out}
    assert R.compute_verdict(dict(rec, arm="scheme-matched")) == "ANALYTIC_TRANSCRIPTION"
    assert R.compute_verdict(dict(rec, arm="unrestricted")) in R._VERIFIED_VERDICTS


def test_hn1_the_brief_states_the_sub_increments_and_the_reference():
    md = S.contract_footer(N01)
    assert "dW_sub: np.ndarray | None = None" in md
    assert "`(1000, 25, 256, 2)`" in md
    assert "reference solution the harness computes on the finer grid" in md
    assert "exact solution that is a function of the Brownian path" not in md
    assert "A discretised scheme at the step `dt` is required" in md
    free = S.contract_footer(N01, arm="unrestricted")
    assert "including integrating on the sub-grid" in free
    for p in P.sciml_problems():
        assert "dW_sub" not in S.contract_footer(p), p["slug"]


def test_hn1_the_manifest_records_the_reference_and_nothing_else_moves():
    e = S.manifest_entry(N01)
    assert e["pathwise_truth"] == "reference" and e["pathwise_substeps"] == 256
    assert e["pathwise_reference_error"] == 2e-3
    for p in P.sciml_problems():
        e2 = S.manifest_entry(p)
        for key in ("pathwise_truth", "pathwise_substeps", "pathwise_reference_error"):
            assert key not in e2, (p["slug"], key)


# --- H-N4: the vacuity table ---------------------------------------------------

def _entry(moments, **extra):
    base = {"slug": "test_vacuity", "type": "sde", "state_dimension": 1, "T": 0.25,
            "num_paths": 1000, "seed": 42, "has_ground_truth": True,
            "ground_truth_kind": "exact", "moments": moments}
    base.update(extra)
    return base


def test_hn4_refuses_n05s_mean_as_the_plan_predicted():
    """The acceptance line: Merton's mean at the published horizon sits 1.26% from
    the initial condition, inside the 5% gate -- a solver returning S0 passes it.
    The table refuses the gate, and records the zero sensitivity to the four jump
    parameters that the plan's §2b found."""
    tab = V.sde_vacuity_table(_entry(partial(P.merton_moments, S0=100.0, mu=0.05, sigma=0.15,
                                             lam=0.1, m=-0.90, delta=0.45)))
    assert tab["refused"] == ["mean"]
    g = tab["gates"]["mean"]
    assert g["ic"]["name"] == "S0" and g["ic"]["rel_distance"] < V.SDE_MEAN_TOL
    assert all(g["sensitivity"][k] == 0.0 for k in ("sigma", "lam", "m", "delta"))
    assert g["sensitivity"]["mu"] > 0.0
    assert "variance" not in tab["refused"]


def test_hn4_refuses_the_symmetric_wright_fisher_mean():
    tab = V.sde_vacuity_table(_entry(partial(P.wright_fisher_moments, a=0.3, b=0.3, sigma=1.0,
                                             X0=0.5), T=1.0))
    assert tab["refused"] == ["mean"]
    assert tab["gates"]["mean"]["ic"]["rel_distance"] == 0.0
    # Breaking the symmetry, or moving the start, lifts the refusal.
    assert V.sde_vacuity_table(_entry(partial(P.wright_fisher_moments, a=0.3, b=0.45,
                                              sigma=1.0, X0=0.5), T=1.0))["refused"] == []


def test_hn4_refuses_a_gate_the_stationary_law_answers():
    stat = P.langevin_stationary(1.0, 4.0, 1.0)
    long = _entry(partial(P.langevin_moments, gamma=1.0, omega=4.0, beta=1.0, q0=1.0, p0=0.0),
                  state_dimension=2, T=50.0, stationary_moments=stat)
    tab = V.sde_vacuity_table(long)
    assert "variance_X" in tab["refused"] and "variance_Y" in tab["refused"]
    assert "stationary value inside the gate" in tab["gates"]["variance_X"]["reasons"]
    short = dict(long, T=1.0)
    assert V.sde_vacuity_table(short)["refused"] == []


def test_hn4_does_not_refuse_a_gate_merely_for_low_sensitivity():
    """The criterion H-N4 first asked for would have refused `sde_gbm`'s mean: a 10%
    change of mu moves e^{mu T} by 1.0%, under the 5% gate. The gate is not vacuous --
    the initial condition fails it by 10.5% -- and the table says so."""
    tab = V.sde_vacuity_table(P.by_slug("sde_gbm"))
    assert tab["refused"] == []
    g = tab["gates"]["mean"]
    assert 0.005 < g["sensitivity"]["mu"] < V.SDE_MEAN_TOL      # the drift moves it 1.0%
    assert g["sensitivity"]["sigma"] == 0.0                        # the volatility not at all
    assert g["ic"]["rel_distance"] > V.SDE_MEAN_TOL                # and it is still not vacuous


def test_hn4_runs_on_every_sde_entry_and_clears_the_whole_new_suite():
    for p in P.sde_problems():
        tab = V.sde_vacuity_table(p)
        assert tab["slug"] == p["slug"]
        if p.get("suite") == "notable" and "gates" in tab and tab["gates"]:
            assert tab["refused"] == [], (p["slug"], tab["refused"])
    # The entries that declare a stationary law are checked against it.
    for slug in ("sde_aitsahalia_rate", "sde_wright_fisher_mutation", "sde_sis_epidemic",
                 "sde_langevin_baoab"):
        tab = V.sde_vacuity_table(P.by_slug(slug))
        assert any("stationary" in g for g in tab["gates"].values()), slug


def test_hn4_a_lambda_truth_is_reported_as_not_introspectable():
    tab = V.sde_vacuity_table(P.by_slug("sde_bm_standard"))
    assert tab["introspectable"] is False and tab["refused"] == []


# --- H-N3 in use: the first SDE structural gates that can fail ---------------------

def test_n02_fails_an_ito_misreading_on_its_structural_gate():
    lv = V.sde_pathwise_crn(N02, N02["num_paths"], V.sde_verify_dt(N02))
    dW = V.crn_increments(lv)
    J = np.array([[0.0, -1.0], [1.0, 0.0]])
    X = np.tile([1.0, 0.0], (N02["num_paths"], 1))
    for k in range(dW.shape[1]):
        X = X + (X @ J.T) * (lv["dt"] + dW[:, k][:, None])     # Stratonovich read as Ito
    meta = {"num_paths": N02["num_paths"], "dt": lv["dt"], "T": 1.0, "seed": 42}
    out = V.verify_sde_pathwise(N02, None, injected=(
        {"terminal_paths": X, "empirical_mean": X.mean(axis=0).tolist(),
         "empirical_variance": X.var(axis=0, ddof=1).tolist()}, meta))
    assert out["passed"] is False
    assert out["violated_constraints"] == ["conserved_norm"]
    gate = next(d for d in out["diagnostics"] if d["name"] == "conserved_norm")
    assert gate["value"] > 1.0                      # e - 1 = 1.72 on the mean square
    assert out["pathwise_rel_err"] > V._pathwise_tol(N02)


def test_the_domain_gates_fail_a_sample_that_left_the_domain():
    for slug, lo, hi in (("sde_aitsahalia_rate", 0.0, np.inf), ("sde_three_halves_vol", 0.0, np.inf),
                         ("sde_wright_fisher_mutation", 0.0, 1.0), ("sde_sis_epidemic", 0.0, 100.0)):
        p = P.by_slug(slug)
        fn = p["diagnostics"][0]["fn"]
        inside = np.full(100, 0.5 if hi <= 1.0 else min(hi, 50.0))
        assert fn(inside, {}, p) == (0.0, True)
        outside = inside.copy()
        outside[3] = lo - 1e-9 if slug != "sde_three_halves_vol" else 0.0
        val, ok = fn(outside, {}, p)
        assert ok is False and val == pytest.approx(0.01)
        val, ok = fn(np.r_[inside, np.nan], {}, p)
        assert ok is False


# --- step 3b: the arms are separate experiments on every route -----------------

def test_the_arm_reaches_the_brief_on_every_notable_route(tmp_path):
    for p in SUITE:
        a = S.contract_footer(p, arm="scheme-matched")
        b = S.contract_footer(p, arm="unrestricted")
        assert a != b, p["slug"]
        assert a == S.contract_footer(p), p["slug"]       # the default is scheme-matched
        if p.get("ground_truth_kind") != "pathwise":
            assert "A discretised time-stepping scheme is required" in a
            assert "including sampling an exact law" in b


def test_the_arm_reaches_the_workspace_path_and_the_default_does_not_move(tmp_path):
    assert S.workspace_name("x") == "x" == S.workspace_name("x", S.DEFAULT_ARM)
    assert S.workspace_name("x", "unrestricted") == "x-unrestricted"
    a = S.write_problem_md(N02, str(tmp_path), arm="scheme-matched")
    b = S.write_problem_md(N02, str(tmp_path), arm="unrestricted")
    assert a == str(tmp_path / N02["slug"] / "problem.md")
    assert b == str(tmp_path / f"{N02['slug']}-unrestricted" / "problem.md")
    with open(a) as fa, open(b) as fb:
        assert fa.read() != fb.read()


def test_the_fence_treats_the_other_arms_workspace_as_a_sibling(tmp_path, monkeypatch):
    """Two arms of one problem must not see each other's plans: the sibling rule
    already fences every other directory under workspace/, and the arm's directory is
    just another one of them -- except its own."""
    root = tmp_path / "workspace"
    for name in ("sde_kubo_oscillator", "sde_kubo_oscillator-unrestricted", "pde_heat_1d"):
        (root / name).mkdir(parents=True)
    monkeypatch.setattr(RUN, "WORKSPACE_ROOT", str(root))
    monkeypatch.setattr(RUN, "archived_workspaces", lambda: [])
    path = RUN.isolated_settings("sde_kubo_oscillator-unrestricted", str(tmp_path / "logs"))
    with open(path) as fh:
        deny = "\n".join(json.load(fh)["permissions"]["deny"])
    assert "workspace/sde_kubo_oscillator/**" in deny
    assert "workspace/pde_heat_1d/**" in deny
    assert "workspace/sde_kubo_oscillator-unrestricted/**" not in deny


def test_the_oneshot_runner_keys_the_arm_too():
    import oneshot
    assert oneshot.unit_key("s", "c0", "naked", "opus", 0) != \
        oneshot.unit_key("s", "c0", "naked", "opus", 0, arm="unrestricted")
    assert oneshot.plan_dir_for(N02, "c0", "naked", "opus", 0) != \
        oneshot.plan_dir_for(N02, "c0", "naked", "opus", 0, arm="unrestricted")


def test_the_report_names_the_fifth_suite():
    text = R.generate_report([], config={})
    assert "`notable`" in text


# --- the frozen suites are untouched ---------------------------------------------

def test_no_problem_outside_the_new_suite_is_changed_by_any_of_this():
    for p in P.PROBLEMS:
        if p.get("suite") == "notable":
            continue
        md = S.contract_footer(p)
        assert "dW_sub" not in md and "time-stepping scheme is required" not in md, p["slug"]
        assert md == S.contract_footer(p, arm="scheme-matched"), p["slug"]
        if p.get("ground_truth_kind") != "pathwise":
            assert md == S.contract_footer(p, arm="unrestricted"), p["slug"]
        e = S.manifest_entry(p)
        for key in ("pathwise_truth", "pathwise_substeps", "pathwise_reference_error",
                    "stationary_moments"):
            assert key not in e, (p["slug"], key)
