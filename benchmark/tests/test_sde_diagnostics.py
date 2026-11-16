"""H-N3: structure diagnostics, and hard structural gates, on the SDE routes.

The PDE side has had hard gates since the systems extension; the SDE side ran no
diagnostics at all, so an SDE problem could not state a structural fact a scheme is
obliged to respect. These tests pin the mechanism and, in the last section, the
*calibration* -- because the thing H-N3 had to avoid was shipping a gate that
reports and never fails, which is what `plan-sciml-benchmark.md` §10d measured on
M08: a declared invariant violated by all three plans, and the problem still graded
10 with `passed: true`.

No model calls, no workspace, no solver import: every route is driven through
`verify.py`'s `injected=` path with synthetic solver output, which is the same path
the sandboxed one-shot runner uses.
"""
import numpy as np
import pytest

from .conftest import P, R, V

A = SIG = 1.0
T = 1.0
DT = 0.04            # == V.sde_verify_dt(_kubo()) : min(0.04, 0.04/1.0)
NT = 25
X0 = np.array([1.0, 0.0])
R0 = 1.0
N = 2000
SEED = 42

#: The gate, from docs/notable-sde-data/kubo_invariant_calibration.py: the geometric
#: mean of the worst correct scheme (5.133e-2) and the one-decision-wrong one
#: (1.771), measured at this exact configuration. 34.5x separation.
GATE = 0.30


def _kubo(*, diagnostics=None, kind="exact", **extra):
    """A Kubo-oscillator-shaped problem entry: 2-D state, 1 noise channel, exact
    moments, and whichever diagnostics a test wants to drive."""
    e = np.exp(-SIG ** 2 * T / 2.0)
    e2 = np.exp(-2.0 * SIG ** 2 * T)
    # E[X1] = e^{-s^2T/2} cos(aT), E[X2] = e^{-s^2T/2} sin(aT); E|X|^2 = 1 exactly,
    # and E[X1^2] = (1 + e2 cos 2aT)/2 by the half-angle identity.
    m1, m2 = e * np.cos(A * T), e * np.sin(A * T)
    s1 = (1.0 + e2 * np.cos(2 * A * T)) / 2.0
    s2 = (1.0 - e2 * np.cos(2 * A * T)) / 2.0
    prob = {
        "id": "T01", "slug": "sde_test_kubo", "type": "sde", "tier": 2,
        "family": "kubo_oscillator", "title": "Kubo oscillator (test)",
        "challenge": "test fixture", "state_dimension": 2, "noise_dimension": 1,
        "T": T, "dt": DT, "num_paths": N, "seed": SEED,
        "has_ground_truth": True, "ground_truth_kind": kind,
        "moments": lambda t: {"mean_X": m1, "variance_X": s1 - m1 ** 2,
                              "mean_Y": m2, "variance_Y": s2 - m2 ** 2},
    }
    if diagnostics is not None:
        prob["diagnostics"] = diagnostics
    prob.update(extra)
    return prob


def _increments(n=N, nt=NT, seed=SEED):
    return np.random.default_rng(seed).standard_normal((n, nt)) * np.sqrt(DT)


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack([X0[0] * c - X0[1] * s, X0[0] * s + X0[1] * c], axis=1)


def exact_rotation(dW):
    return _rot(A * T + SIG * dW.sum(axis=1))


def heun(dW):
    """Explicit Stratonovich Heun: right equation, amplitude conserved to O(dt)."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        h = A * DT + SIG * dW[:, k]
        p0 = np.stack([-x[:, 1] * h, x[:, 0] * h], axis=1)
        xp = x + p0
        p1 = np.stack([-xp[:, 1] * h, xp[:, 0] * h], axis=1)
        x = x + 0.5 * (p0 + p1)
    return x


def euler_converted(dW):
    """Correct: Stratonovich converted to Ito, then Euler-Maruyama."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        dw, x1, x2 = dW[:, k], x[:, 0], x[:, 1]
        x = np.stack([x1 + (-A * x2 - 0.5 * SIG ** 2 * x1) * DT - SIG * x2 * dw,
                      x2 + (A * x1 - 0.5 * SIG ** 2 * x2) * DT + SIG * x1 * dw], axis=1)
    return x


def euler_misread(dW):
    """Wrong by exactly one decision: the conversion term omitted."""
    x = np.tile(X0, (dW.shape[0], 1))
    for k in range(dW.shape[1]):
        dw, x1, x2 = dW[:, k], x[:, 0], x[:, 1]
        x = np.stack([x1 + (-A * x2) * DT - SIG * x2 * dw,
                      x2 + (A * x1) * DT + SIG * x1 * dw], axis=1)
    return x


def _injected(paths, n=N):
    """`(result, run_meta)` of the shape `runner.run_solver` returns."""
    paths = np.asarray(paths, dtype=float)
    return ({"terminal_paths": paths,
             "empirical_mean": float(paths[:, 0].mean()),
             "empirical_variance": float(paths[:, 0].var(ddof=1))},
            {"num_paths": int(n), "dt": DT, "T": T, "seed": SEED})


def _gate(tol=GATE):
    return [{"name": "amplitude_conserved",
             "fn": P._sde_diag_conserved_norm(R0, tol), "gate": True}]


def _gate_and_info(tol=GATE):
    return _gate(tol) + [{"name": "amplitude_spread",
                          "fn": P._sde_diag_amplitude_spread(R0), "gate": False}]


# --- A. the mechanism -------------------------------------------------------

def test_a_passing_gate_leaves_the_verdict_alone():
    out = V.verify_sde(_kubo(diagnostics=_gate()), None,
                       injected=_injected(exact_rotation(_increments())))
    assert out["status"] == "ok"
    assert "constraint_violation" not in out
    assert out["passed"] is True
    assert [d["name"] for d in out["diagnostics"]] == ["amplitude_conserved"]
    assert out["diagnostics"][0]["passed"] is True
    assert out["diagnostics"][0]["value"] == pytest.approx(0.0, abs=1e-12)


def test_a_failed_gate_is_a_constraint_violation_and_fails_the_run():
    """The whole point of H-N3: the gate *bites*."""
    out = V.verify_sde(_kubo(diagnostics=_gate()), None,
                       injected=_injected(euler_misread(_increments())))
    assert out["constraint_violation"] is True
    assert out["violated_constraints"] == ["amplitude_conserved"]
    assert out["passed"] is False
    assert out["diagnostics"][0]["value"] > GATE


def test_a_non_gate_diagnostic_reports_and_cannot_fail_anything():
    """Driven with a *correct* scheme on purpose: the misread also fails the moment
    gates, so it could not tell a diagnostic that gated from one that did not."""
    diags = [{"name": "amplitude_spread",
              "fn": P._sde_diag_amplitude_spread(R0, tol=1e-9), "gate": False}]
    out = V.verify_sde(_kubo(diagnostics=diags), None,
                       injected=_injected(heun(_increments())))
    assert out["diagnostics"][0]["passed"] is False      # it did fail its own tol
    assert "constraint_violation" not in out             # and gated nothing
    assert out["passed"] is True                         # the moments decide


def test_the_score_is_not_bounded_by_a_violation_and_the_verdict_carries_it():
    """The PDE precedent, pinned so it is a decision rather than an oversight:
    `verified_score` stays the accuracy earned, and the *verdict* is where the
    structural failure shows up."""
    prob = _kubo(diagnostics=_gate())
    out = V.verify_sde(prob, None, injected=_injected(euler_misread(_increments())))
    assert isinstance(out["verified_score"], int)
    rec = {"id": prob["id"], "slug": prob["slug"], "type": "sde", "tier": 2,
           "family": prob["family"], "title": prob["title"],
           "challenge": prob["challenge"], "has_ground_truth": True,
           "ground_truth_kind": "exact",
           "run": {"status": "completed", "returncode": 0, "timed_out": False,
                   "wall_seconds": 1.0, "log": None},
           "pipeline": {"phase": "done", "best_plan": "1-p", "best_score": 10,
                        "n_plans": 1, "total_iters": 1, "plans": {}},
           "verify": out}
    assert R.compute_verdict(rec) == "CONSTRAINT_VIOLATION"
    assert R.compute_oneshot_verdict(rec) == "CONSTRAINT_VIOLATION"


def test_a_raising_diagnostic_is_a_failed_gate_rather_than_a_crash():
    def boom(paths, result, problem):
        raise ValueError("nope")
    out = V.verify_sde(_kubo(diagnostics=[{"name": "boom", "fn": boom, "gate": True}]),
                       None, injected=_injected(exact_rotation(_increments())))
    d = out["diagnostics"][0]
    assert np.isnan(d["value"]) and d["passed"] is False
    assert "ValueError: nope" in d["error"]
    assert out["constraint_violation"] is True


def test_a_gate_that_cannot_be_evaluated_has_not_been_cleared():
    """A sample the gate cannot be evaluated on must not thereby clear it.

    The reachable case, and it is worth more than the mechanism: a solver that
    returns **three** components for a 2-D problem is scored `ok`/`passed: True` by
    the moment route, because that route reads the first `d` columns and never asks
    whether there are more (measured: `verify_sde` returns passed=True on an (n, 3)
    sample for `state_dimension: 2`). The diagnostic cannot form `|X|` from a sample
    whose width disagrees with the declared dimension, so it reports that it could
    not, and the gate is not cleared.
    """
    x = heun(_increments())
    wide = np.concatenate([x, np.zeros((x.shape[0], 1))], axis=1)   # (n, 3) for d = 2
    result, run_meta = _injected(x)
    result["terminal_paths"] = wide
    out = V.verify_sde(_kubo(diagnostics=_gate()), None, injected=(result, run_meta))
    assert out["constraint_violation"] is True
    assert out["passed"] is False
    assert "unreadable" in out["diagnostics"][0]["error"]
    # ... and without the gate the same output is a clean pass, which is the finding.
    clean = V.verify_sde(_kubo(), None, injected=(result, run_meta))
    assert clean["status"] == "ok" and clean["passed"] is True


def test_an_unevaluable_diagnostic_fails_every_declared_check():
    """The unit-level statement, so the branch is pinned independently of which
    route happens to reach it."""
    prob = _kubo(diagnostics=_gate_and_info())
    diags = V._run_sde_diagnostics(prob, None, {})
    assert [d["name"] for d in diags] == ["amplitude_conserved", "amplitude_spread"]
    assert all(d["passed"] is False for d in diags)
    assert all(np.isnan(d["value"]) for d in diags)
    assert all("unreadable" in d["error"] for d in diags)


def test_both_orientations_of_the_sample_are_read_the_same_way():
    x = euler_misread(_increments())
    a = V.verify_sde(_kubo(diagnostics=_gate()), None, injected=_injected(x))
    result, run_meta = _injected(x)
    result["terminal_paths"] = np.ascontiguousarray(x.T)   # (d, n)
    b = V.verify_sde(_kubo(diagnostics=_gate()), None, injected=(result, run_meta))
    assert a["diagnostics"][0]["value"] == pytest.approx(b["diagnostics"][0]["value"])


def test_orient_paths_leaves_a_square_sample_alone():
    """Nothing can disambiguate (k, k), so the rule must not guess."""
    sq = np.arange(9.0).reshape(3, 3)
    assert V._orient_paths(sq, 3) is sq
    tall = np.zeros((7, 2))
    assert V._orient_paths(tall, 7).shape == (7, 2)
    assert V._orient_paths(np.zeros((2, 7)), 7).shape == (7, 2)


# --- B. every SDE route folds them ------------------------------------------

def test_the_reference_route_gates():
    prob = _kubo(diagnostics=_gate(), kind="reference")
    m = prob["moments"](T)
    prob["moments"] = lambda t: {k: (v, abs(v) * 1e-3 + 1e-6) for k, v in m.items()}
    out = V.verify_sde_reference(prob, None,
                                 injected=_injected(euler_misread(_increments())))
    assert out["gt_kind"] == "reference" and out["constraint_violation"] is True


def test_the_stability_route_gates():
    prob = _kubo(diagnostics=_gate(), kind="stability",
                 has_ground_truth=False, stability_check={"type": "finite"})
    out = V.verify_sde_stability(prob, None,
                                 injected=_injected(euler_misread(_increments())))
    assert out["gt_kind"] == "stability"
    assert out["finite"] is True              # it did stay finite ...
    assert out["constraint_violation"] is True  # ... and still broke the invariant
    assert out["passed"] is False


def test_the_pathwise_route_gates():
    prob = _kubo(diagnostics=_gate(), kind="pathwise",
                 pathwise_exact=lambda t, W: _rot(A * float(t) + SIG * np.asarray(W)),
                 pathwise_tol=0.10, pathwise_fine_steps=2500)
    dW = V.crn_increments(V.sde_pathwise_crn(prob, N, DT))
    out = V.verify_sde_pathwise(prob, None, injected=_injected(euler_misread(dW)))
    assert out["gt_kind"] == "pathwise" and out["constraint_violation"] is True
    out = V.verify_sde_pathwise(prob, None, injected=_injected(exact_rotation(dW)))
    assert "constraint_violation" not in out and out["passed"] is True


# --- C. the calibration, both ways ------------------------------------------
#
# docs/notable-sde-data/kubo_invariant_calibration.py measures these at 50 000
# paths; the numbers here are the same quantity at 2 000, which is why the
# assertions are inequalities against the gate rather than equalities.

@pytest.mark.parametrize("scheme", [exact_rotation, heun, euler_converted],
                         ids=["exact-rotation", "stratonovich-heun", "euler+conversion"])
def test_every_correct_scheme_clears_the_gate(scheme):
    out = V.verify_sde(_kubo(diagnostics=_gate_and_info()), None,
                       injected=_injected(scheme(_increments())))
    assert "constraint_violation" not in out, out["diagnostics"]
    assert out["diagnostics"][0]["value"] < GATE


def test_the_one_decision_wrong_scheme_fails_the_gate():
    out = V.verify_sde(_kubo(diagnostics=_gate_and_info()), None,
                       injected=_injected(euler_misread(_increments())))
    assert out["diagnostics"][0]["value"] > GATE
    assert out["constraint_violation"] is True


def test_the_gated_statistic_is_the_sharper_of_the_two():
    """Why the gate is on the mean square and not the per-path drift: measured, the
    per-path RMS charges a correct scheme for its own O(dt) variance. If this ever
    inverts, the choice in `_sde_diag_conserved_norm` should be revisited."""
    dW = _increments()
    gated = P._sde_diag_conserved_norm(R0, GATE)
    spread = P._sde_diag_amplitude_spread(R0)

    def separation(diag):
        """wrong / worst-correct, the ratio the repo's calibration rule reads."""
        val = lambda x: diag(x, None, None)[0]   # noqa: E731
        return val(euler_misread(dW)) / max(val(euler_converted(dW)), val(heun(dW)))

    sep_gated, sep_spread = separation(gated), separation(spread)
    # Measured at 2 000 paths: 32.3x against 4.78x. At the 50 000 of the calibration
    # script: 34.5x against 4.7x. The bound is 5x rather than the measured 6.8x so
    # that sampling noise cannot flip it, and 20x on the gate itself because that is
    # the claim the gate rests on.
    assert sep_spread > 1.0
    assert sep_gated > 5.0 * sep_spread
    assert sep_gated > 20.0


# --- D. the change is inert on everything that already ships ----------------

def test_only_the_notable_suite_declares_sde_diagnostics():
    """So H-N3 cannot move a measured baseline on any of the four older suites. ~~No
    SDE problem declares one~~ -- the first ones did, in plan-notable-sde.md step 6,
    and this is the place that says so deliberately: N02's conserved norm, and the
    domain gates of N03, N04, N06 and N08. Every other SDE entry still declares none."""
    declared = [p["slug"] for p in P.PROBLEMS
                if p["type"] == "sde" and p.get("diagnostics")]
    assert declared == ["sde_kubo_oscillator", "sde_aitsahalia_rate",
                        "sde_wright_fisher_mutation", "sde_sis_epidemic",
                        "sde_three_halves_vol"]
    assert all(P.by_slug(s).get("suite") == "notable" for s in declared)


def test_a_structural_gate_never_ships_alone_on_an_sde_problem():
    """§1 of plan-notable-sde.md, and route I of the no-closed-form catalogue: a
    wrong-but-conservative scheme passes a structure check, so it can never be the
    only one."""
    for p in P.PROBLEMS:
        if p["type"] != "sde" or not p.get("diagnostics"):
            continue
        assert (p.get("moments") or p.get("functional_truth")
                or p.get("pathwise_exact")), p["slug"]
