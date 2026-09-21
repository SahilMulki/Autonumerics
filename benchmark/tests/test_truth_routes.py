"""The four routes to truth without a closed form, as the harness grades them.

NO_CLOSED_FORM_CANDIDATES.md §1: "no closed form" is not "no ground truth", and a
problem with no independent check reports SELF_ONLY and measures nothing. Every
test here feeds ``verify.py`` synthetic solver output built from the harness's own
truth -- the reference field itself, the exact eigenfunction, the recalled
eigenvalue with no mode behind it -- and pins down the verdict. No solver runs.
"""
import os

import numpy as np
import pytest

from .conftest import P, R, V, grid_for, problem, record, result_from, truth_on

REFERENCE_PROBLEMS = [p for p in P.pde_problems() if p.get("ground_truth_kind") == "reference"]


# --- D1: no route, no problem --------------------------------------------------

def test_every_no_closed_form_problem_has_an_independent_truth_route(ncf_problems):
    """§1 of the candidates file as a test: a route to truth that does not go
    through a formula the pipeline could also write down."""
    assert len(ncf_problems) >= 4
    for p in ncf_problems:
        assert p["has_ground_truth"] is True, p["slug"]
        routes = {"reference": p.get("reference") is not None,
                  "semi_analytic": p.get("analytic") is not None,
                  "functional": bool(p.get("functional_truth"))}
        assert any(routes.values()), f"{p['slug']} has no route to truth: {routes}"
        assert p.get("closed_form") is False


def test_a_no_closed_form_problem_is_never_self_only(ncf_problems):
    for p in ncf_problems:
        rec = record(p, verify={"status": "ok", "passed": True, "rel_l2_err": 1e-4})
        assert R.compute_verdict(rec) == "VERIFIED_PASS", p["slug"]
        rec = record(p, verify={"status": "ok", "passed": False, "rel_l2_err": 0.5})
        assert R.compute_verdict(rec) == "OVERCLAIM", p["slug"]


def test_route_i_never_ships_alone(ncf_problems):
    """§1: structure-only is the weakest route, and a wrong-but-conservative scheme
    passes it. A problem with diagnostics must also carry a field or a functional."""
    for p in ncf_problems:
        if p.get("diagnostics"):
            assert p.get("reference") or p.get("analytic") or p.get("functional_truth"), p["slug"]


# --- D2: the reference artifacts match what problems.py says about them ----------

@pytest.mark.parametrize("prob", REFERENCE_PROBLEMS, ids=lambda p: p["slug"])
def test_a_reference_problem_declares_an_error_matching_its_artifact(prob):
    meta = P._load_reference(prob["slug"])
    assert meta["axes"] == list(prob["axes"])
    assert meta["t_eval"] == pytest.approx(float(prob["t_eval"]))
    assert prob["reference_error"] >= meta["reference_error"] > 0.0, (
        "problems.py must not claim the reference is better than the artifact measured")
    for name, periodic, axis in zip(meta["axes"], meta["periodic"], meta["grids"], strict=True):
        lo, hi = prob["domain"][name]
        assert axis[0] == pytest.approx(lo)
        if periodic:
            assert axis[-1] < hi                      # endpoint-exclusive by construction
        else:
            assert axis[-1] == pytest.approx(hi)


def test_a_reference_without_a_callable_or_a_measured_error_is_refused(tmp_path):
    """A reference whose accuracy nobody established is an answer key, not ground
    truth; ``verify_pde_reference`` refuses to grade against one."""
    base = problem("pde_kuramoto_sivashinsky")
    no_callable = {**base, "reference": None}
    out = V.verify_pde_reference(no_callable, str(tmp_path))
    assert out["status"] == "error" and "no reference callable" in out["error"]
    no_error = {**base, "reference_error": 0.0}
    out = V.verify_pde_reference(no_error, str(tmp_path))
    assert out["status"] == "error" and "reference_error" in out["error"]


# --- D3: the tolerance widening -------------------------------------------------

def test_reference_tolerance_widens_by_the_factor_and_is_recorded():
    """A solver more accurate than the answer key must not fail for being right."""
    tight = {"reference_error": 1e-9}
    assert V._pde_l2_tol(tight) == V.PDE_L2_TOL
    loose = {"reference_error": 0.01}
    assert V._pde_l2_tol(loose) == pytest.approx(V.REFERENCE_TOL_FACTOR * 0.01)
    out = V._fold_reference({}, loose, {"functional_errs": {}})
    assert out["tol_widened"] is True and out["tol_used"] == pytest.approx(0.03)
    out = V._fold_reference({}, tight, {"functional_errs": {}})
    assert out["tol_widened"] is False
    assert "tol_used" not in V._fold_reference({}, {}, {})   # a closed form records nothing


# --- D4 / D5: leaving the reference grid -----------------------------------------

@pytest.mark.parametrize("M", [16, 17])
def test_periodic_interpolation_is_exact_on_and_off_the_nodes(M):
    """The band-limited interpolant is the identity on the reference nodes and
    exact to round-off on a resolved field at any target -- for even ``M`` (the
    Nyquist mode is real and must recombine as a cosine) and odd ``M`` alike."""
    L = 2 * np.pi
    ref = np.linspace(0.0, L, M, endpoint=False)
    f = lambda x: np.cos(3 * x) + 0.5 * np.sin(2 * x) - 0.25 * np.cos(x)  # noqa: E731
    on = P._trig_interp_matrix(ref, ref)
    assert np.allclose(on, np.eye(M), atol=1e-12)
    targets = np.linspace(0.0, L, 41)             # includes the wrap node
    off = P._trig_interp_matrix(ref, targets)
    assert np.allclose(off @ f(ref), f(targets), atol=1e-12)


def test_a_periodic_reference_accepts_both_endpoint_conventions():
    """Errata I5: the solver contract asks for an endpoint-inclusive grid, a
    spectral solve keeps an exclusive one, and both must land on the same field."""
    prob = problem("pde_kuramoto_sivashinsky")
    inclusive = truth_on(prob, grid_for(prob, 65, endpoint=True))["u"]
    exclusive = truth_on(prob, grid_for(prob, 64, endpoint=False))["u"]
    assert np.allclose(inclusive[:-1], exclusive, atol=1e-10)
    assert inclusive[0] == pytest.approx(inclusive[-1])


def test_a_non_periodic_reference_refuses_a_grid_that_does_not_nest(tmp_path, monkeypatch):
    """Authoring checklist item 5: non-periodic axes nest or the reference reports
    that it cannot be evaluated. No shipped reference is non-periodic, so the rule
    is tested on a synthetic artifact."""
    x = np.linspace(0.0, 1.0, 1025)
    np.savez(tmp_path / "synthetic.npz", axis_names=np.array(["x"]),
             field_names=np.array(["u"]), periodic=np.array([False]),
             t_eval=np.array(1.0), reference_error=np.array(1e-10),
             axis_x=x, field_u=np.sin(np.pi * x))
    monkeypatch.setattr(P, "_REFERENCE_DIR", str(tmp_path))
    P._load_reference.cache_clear()
    try:
        evaluate = P.reference_evaluator("synthetic")
        nested = evaluate(1.0, np.linspace(0.0, 1.0, 65))
        assert np.allclose(nested["u"], np.sin(np.pi * np.linspace(0.0, 1.0, 65)))
        with pytest.raises(P.ReferenceUnavailable, match="not a subset"):
            evaluate(1.0, np.linspace(0.0, 1.0, 64))
        with pytest.raises(P.ReferenceUnavailable, match="asked for 2-D"):
            evaluate(1.0, np.zeros((3, 3)), np.zeros((3, 3)))
    finally:
        P._load_reference.cache_clear()


def test_the_shipped_grids_nest_or_interpolate():
    """For every reference problem, ``grid_N`` and ``2*grid_N`` are evaluable --
    by nesting on a non-periodic axis, by interpolation on a periodic one."""
    for prob in REFERENCE_PROBLEMS:
        for N in (prob["grid_N"], 2 * prob["grid_N"]):
            got = truth_on(prob, grid_for(prob, N))
            assert all(np.all(np.isfinite(v)) for v in got.values()), (prob["slug"], N)


def test_a_missing_artifact_names_the_command_that_builds_it():
    with pytest.raises(P.ReferenceUnavailable, match="make_references.py"):
        P._load_reference("pde_no_such_reference")


# --- D6: the reference route grades in both directions ------------------------

def _ks_runs(perturb):
    prob = problem("pde_kuramoto_sivashinsky")
    runs = []
    for N in (prob["grid_N"], 2 * prob["grid_N"]):
        grid = grid_for(prob, N)
        u = truth_on(prob, grid)["u"]
        runs.append((N, result_from({"u": perturb(u, grid["x"], N)}, grid, t_final=prob["t_eval"])))
    return prob, runs


def test_the_reference_field_itself_passes_and_the_initial_condition_fails():
    prob, runs = _ks_runs(lambda u, x, N: u + 1e-6 * np.sin(x))
    out = V._score_pde_runs(prob, runs)
    assert out["status"] == "ok" and out["passed"] is True, out
    assert out["rel_l2_err"] < V._pde_l2_tol(prob)
    assert out["tol_widened"] is False              # 1.2e-9 * 3 is far below 1%

    prob, runs = _ks_runs(lambda u, x, N: np.cos(x / 16) * (1 + np.sin(x / 16)))
    out = V._score_pde_runs(prob, runs)
    assert out["passed"] is False
    assert out["rel_l2_err"] > 0.5


def test_a_solver_that_ignores_n_fails_the_order_check_on_the_reference():
    """Same field at both grids: accurate at neither, and no order to speak of."""
    prob = problem("pde_kuramoto_sivashinsky")
    grid = grid_for(prob, prob["grid_N"])
    u = truth_on(prob, grid)["u"] * 1.05                     # 5% off, everywhere
    runs = [(prob["grid_N"], result_from({"u": u}, grid, t_final=50.0)),
            (2 * prob["grid_N"], result_from({"u": u}, grid, t_final=50.0))]
    out = V._score_pde_runs(prob, runs)
    assert out["passed"] is False and out["order_ok"] is False
    assert out["observed_order"] == pytest.approx(0.0)


def test_a_nan_in_the_field_is_nonfinite_not_a_score():
    prob, runs = _ks_runs(lambda u, x, N: np.where(np.arange(N) == N // 2, np.nan, u))
    out = V._score_pde_runs(prob, runs)
    assert out["status"] == "nonfinite" and out["verified_score"] == 1


# --- D7-D10: the functional route and the memorization trap ---------------------

def _eigen_runs(field_fn, lam):
    prob = problem("pde_schrodinger_eigen_2d")
    grid = grid_for(prob, prob["grid_N"])
    exact = truth_on(prob, grid)["u"]
    return prob, [(prob["grid_N"], result_from({"u": field_fn(exact, grid)}, grid,
                                               functionals={"lambda_1": lam}))]


def test_a_recalled_eigenvalue_with_no_mode_behind_it_fails_on_the_field():
    """§3, the memorization trap: the constant is right to twelve digits and the
    field is noise. The field check is what makes a recalled number worthless."""
    lam = problem("pde_schrodinger_eigen_2d")["functional_truth"]["lambda_1"]
    prob, runs = _eigen_runs(lambda u, g: np.random.default_rng(0).standard_normal(u.shape), lam)
    out = V._score_pde_runs(prob, runs)
    assert out["passed"] is False
    assert out["functional_errs"]["lambda_1"] < 1e-12
    assert out["rel_l2_err"] > 0.5


def test_the_right_mode_with_a_slightly_wrong_eigenvalue_fails_on_the_functional():
    lam = problem("pde_schrodinger_eigen_2d")["functional_truth"]["lambda_1"]
    prob, runs = _eigen_runs(lambda u, g: u, lam * (1 + 2e-3))
    out = V._score_pde_runs(prob, runs)
    assert out["rel_l2_err"] < 1e-10
    assert out["functional_errs"]["lambda_1"] == pytest.approx(2e-3, rel=1e-6)
    assert out["functional_errs"]["lambda_1"] > out["functional_tol"]
    assert out["passed"] is False
    prob, runs = _eigen_runs(lambda u, g: u, lam * (1 + 1e-4))
    assert V._score_pde_runs(prob, runs)["passed"] is True


def test_the_eigen_gauge_makes_sign_and_scale_irrelevant():
    lam = problem("pde_schrodinger_eigen_2d")["functional_truth"]["lambda_1"]
    prob, runs = _eigen_runs(lambda u, g: -3.0 * u, lam)
    out = V._score_pde_runs(prob, runs)
    assert out["passed"] is True and out["rel_l2_err"] < 1e-10
    assert all(d["passed"] for d in out["diagnostics"]), out["diagnostics"]


def test_an_excited_state_is_caught_by_the_nodeless_gate():
    """``which='SM'`` returns the eigenvalue of smallest magnitude, which on a deep
    potential is not the ground state. A mode with a nodal line fails the gate
    before its error is even considered."""
    lam = problem("pde_schrodinger_eigen_2d")["functional_truth"]["lambda_1"]
    prob, runs = _eigen_runs(
        lambda u, g: np.sin(2 * np.pi * g["x"])[:, None] * np.sin(np.pi * g["y"])[None, :], lam)
    out = V._score_pde_runs(prob, runs)
    assert out["constraint_violation"] is True
    assert out["violated_constraints"] == ["ground_state_nodeless"]
    assert R.compute_verdict(record(prob, verify=out)) == "CONSTRAINT_VIOLATION"


def test_a_functional_problem_with_no_functionals_returned_is_a_failure():
    prob, runs = _eigen_runs(lambda u, g: u, 0.0)
    runs[0][1].pop("functionals")
    out = V._score_pde_runs(prob, runs)
    assert out.get("passed") is False, out
    assert out["status"] != "inconclusive"


def test_a_missing_or_nonfinite_functional_is_named():
    prob, runs = _eigen_runs(lambda u, g: u, float("nan"))
    out = V._score_pde_runs(prob, runs)
    assert out["status"] == "nonfinite" and out["verified_score"] == 1
    prob, runs = _eigen_runs(lambda u, g: u, 0.0)
    runs[0][1]["functionals"] = {"lambda_2": 0.0}
    out = V._score_pde_runs(prob, runs)
    # A missing *name* is the same refusal as a missing dict (§7.7): a failure.
    assert out["status"] == "contract" and "lambda_1" in out["error"]
    assert out["passed"] is False
    runs[0][1]["functionals"] = {"lambda_1": "about minus eight"}
    out = V._score_pde_runs(prob, runs)
    assert out["status"] == "inconclusive" and "not a real number" in out["error"]


# --- D11: the matched pairs -----------------------------------------------------

@pytest.mark.parametrize("ncf, twin", [
    ("pde_cahn_hilliard_2d_coarsening", "pde_cahn_hilliard_2d"),
    ("pde_burgers_viscous_1d", "pde_burgers_inviscid"),
])
def test_the_matched_pair_shares_its_operator_family_and_differs_in_the_closed_form(ncf, twin):
    a, b = problem(ncf), problem(twin)
    assert a["dims"] == b["dims"]
    assert a["family"].split("-")[0] == b["family"].split("-")[0]
    assert a.get("closed_form") is False and b.get("closed_form") is not False
    assert b["has_ground_truth"] and a["has_ground_truth"]


def test_the_cahn_hilliard_pair_is_the_controlled_experiment():
    """Same operator, same domain, source deleted: the one variable is the formula."""
    a, b = problem("pde_cahn_hilliard_2d_coarsening"), problem("pde_cahn_hilliard_2d")
    assert a["domain"] == b["domain"] and a["axes"] == b.get("axes", ["x", "y"])
    assert "there is no source term" in a["description"].lower()
    assert "manufactured" not in a["description"].lower()
    assert "source" in b["description"].lower()          # the MMS twin states one


# --- D12: the semi-analytic truths are converged to what they claim --------------

def test_the_cole_hopf_quadrature_is_converged():
    x = np.linspace(0.0, 2 * np.pi, 65)
    coarse = P._burgers_viscous_1d(1.5, x, n_quad=8192)
    fine = P._burgers_viscous_1d(1.5, x, n_quad=16384)
    assert np.max(np.abs(coarse - fine)) < 1e-12
    assert np.max(np.abs(coarse)) < 1.0 and np.max(np.abs(coarse)) > 0.5   # a real layer


def test_the_eigen_expansion_is_converged():
    lam_K, _, C_K = P._eig_ground_state(P._EIG_K)
    lam_more, _, C_more = P._eig_ground_state(P._EIG_K + 12)
    P._eig_ground_state.cache_clear()
    assert abs(lam_K - lam_more) < 1e-10
    assert lam_K == pytest.approx(problem("pde_schrodinger_eigen_2d")["functional_truth"]["lambda_1"],
                                  abs=1e-11)
    K = C_K.shape[0]
    assert np.max(np.abs(C_K - C_more[:K, :K])) < 1e-8


# --- D13: Cahn-Hilliard's structure gates ---------------------------------------

def _ch_run(transform):
    prob = problem("pde_cahn_hilliard_2d_coarsening")
    grid = grid_for(prob, prob["grid_N"])
    u = truth_on(prob, grid)["u"]
    return prob, [(prob["grid_N"], result_from({"u": transform(u)}, grid, t_final=0.1))]


def test_cahn_hilliard_structure_gates_bite():
    prob, runs = _ch_run(lambda u: u)
    assert V._score_pde_runs(prob, runs)["passed"] is True

    prob, runs = _ch_run(lambda u: u + 0.02)                   # mass drift
    out = V._score_pde_runs(prob, runs)
    assert "mass_error" in out["violated_constraints"]
    assert R.compute_verdict(record(prob, verify=out)) == "CONSTRAINT_VIOLATION"

    prob, runs = _ch_run(lambda u: np.full_like(u, np.mean(u)))   # collapsed
    out = V._score_pde_runs(prob, runs)
    assert "nontrivial" in out["violated_constraints"]


# --- D14: the references rebuild to their stored error (slow) -------------------

@pytest.mark.slow
@pytest.mark.parametrize("slug", [p["slug"] for p in REFERENCE_PROBLEMS])
def test_make_references_check_reproduces_the_stored_error(slug):
    import make_references as M
    err, ok = M.build(slug, write=False)
    assert ok, f"{slug}: rebuilt reference drifted from the stored artifact"
    assert err <= problem(slug)["reference_error"] * 1.5


def test_reference_artifacts_are_denied_to_the_pipeline():
    import fnmatch
    import json

    with open(os.path.join(os.path.dirname(V.HERE), "benchmark", "pipeline-settings.json")) as fh:
        deny = json.load(fh)["permissions"]["deny"]
    reads = [d[len("Read("):-1] for d in deny if d.startswith("Read(")]
    for prob in REFERENCE_PROBLEMS:
        rel = f"./benchmark/references/{prob['slug']}.npz"
        assert any(fnmatch.fnmatch(rel, g) for g in reads), (rel, reads)
