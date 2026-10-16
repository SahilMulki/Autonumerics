"""Criterion 11: a hanging or memory-hungry solver yields ``crashed``, not a hung
session.

The kernel imports and executes solver code up a three-level ladder -- in 3-D the
top level is 32x the base -- so every failure mode has to come back as data. The
measured precedent is on the guardrails side: an unbounded Stop hook fired 8 times
and the run died at max-turns. A hung top-level solve burns the same budget with
less to show for it.
"""
import numpy as np
import pytest

from verifylib.kernel import sandbox

SOLVER = """
import numpy as np

def solve_pde(N, override=None):
    ov = override or {}
    x = np.linspace(0.0, 1.0, N)
    u = ov["ic"]((x,)) if "ic" in ov else np.sin(np.pi * x)
    u = np.asarray(u, dtype=float) * float(ov.get("params", {}).get("scale", 1.0))
    return {"numerical_solution": u, "grid": {"x": x}, "t_final": 1.0, "dt": 0.01,
            "invariant_trace": {"energy": np.ones(4)},
            "snapshots": [{"t": 0.98, "fields": {"u": u * 0.9}},
                          {"t": 0.99, "fields": {"u": u * 0.95}},
                          {"t": 1.00, "fields": {"u": u}}]}
"""


def _write(tmp_path, text):
    (tmp_path / "solver.py").write_text(text)
    return str(tmp_path)


def test_the_whole_return_contract_survives_the_process_boundary(tmp_path):
    """`benchmark/runner.py` extracts the field, the grid and t_final and drops
    everything else -- including the three keys the kernel exists to consume."""
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9})
    assert got["status"] == "ok"
    result = got["result"]
    assert set(result) >= {"grid", "fields", "t_final", "dt", "invariant_trace",
                           "snapshots"}
    assert list(result["invariant_trace"]) == ["energy"]
    assert [s["t"] for s in result["snapshots"]] == [0.98, 0.99, 1.0]
    assert result["snapshots"][-1]["fields"]["u"].shape == (9,)


def test_an_override_reaches_the_solver_as_a_rebuilt_callable(tmp_path):
    """Callables cannot cross a process boundary, so ``ic`` travels as the spec
    expression it is built from and is rebuilt in the child."""
    plan = _write(tmp_path, SOLVER)
    payload = {"ic": {"kind": "expr", "exprs": {"u": "np.cos(np.pi*x)"},
                      "params": {}, "spatial_variables": ["x"], "multi": False,
                      "fixed_t": 0.0}}
    got = sandbox.run(plan, "pde", {"N": 9}, override=payload)
    assert got["status"] == "ok"
    assert np.allclose(got["result"]["fields"]["u"], np.cos(np.pi * np.linspace(0, 1, 9)))
    scaled = sandbox.run(plan, "pde", {"N": 9}, override={"params": {"scale": 2.0}})
    assert np.allclose(scaled["result"]["fields"]["u"],
                       2.0 * np.sin(np.pi * np.linspace(0, 1, 9)))


def test_an_array_override_travels_through_the_npz(tmp_path):
    """The route ``translation_invariance`` takes: a shifted initial condition is
    genuinely an array, not an expression."""
    want = np.arange(9, dtype=float)
    got = sandbox.run(_write(tmp_path, SOLVER), "pde", {"N": 9},
                      override={"ic": {"kind": "array", "key": "ov_ic"}},
                      in_arrays={"ov_ic": want})
    assert got["status"] == "ok"
    assert np.allclose(got["result"]["fields"]["u"], want)


@pytest.mark.parametrize("body,reason", [
    ("", "no_solve_fn"),
    ("def solve_pde(N, override=None):\n    raise RuntimeError('boom')\n", "exception"),
    ("def solve_pde(N, override=None):\n    return 42\n", "bad_schema"),
    ("def solve_pde(N, override=None):\n    return {'numerical_solution': [1.0]}\n",
     "bad_schema"),
    ("import pandas\ndef solve_pde(N, override=None):\n    return {}\n", "blocked_import"),
])
def test_every_failure_mode_comes_back_as_a_reason(tmp_path, body, reason):
    got = sandbox.run(_write(tmp_path, body), "pde", {"N": 9})
    assert got["status"] == "crashed"
    assert got["reason"] == reason, got.get("error")
    assert got["reason"] in sandbox.CRASH_REASONS


def test_a_hanging_solver_is_killed_rather_than_waited_on(tmp_path):
    plan = _write(tmp_path, "import time\n"
                            "def solve_pde(N, override=None):\n"
                            "    time.sleep(60)\n")
    got = sandbox.run(plan, "pde", {"N": 9}, timeout_s=2)
    assert got["status"] == "crashed" and got["reason"] == "timeout"
    assert got["wall_s"] < 20, "the timeout must bound the wait, not merely report it"


def test_a_missing_solver_is_reported_not_raised(tmp_path):
    got = sandbox.run(str(tmp_path), "pde", {"N": 9})
    assert got["status"] == "crashed" and got["reason"] == "no_solve_fn"


def test_a_leaking_solver_never_runs(tmp_path):
    """The leakage scan happens *before* import, so an answer-key read cannot
    execute even once."""
    plan = _write(tmp_path, "import json\n"
                            "def solve_pde(N, override=None):\n"
                            "    return json.load(open('benchmark/results/exact.json'))\n")
    got = sandbox.run(plan, "pde", {"N": 9})
    assert got["status"] == "crashed" and got["reason"] == "leakage"


def test_the_import_policy_matches_what_the_solver_agents_are_granted():
    """Library parity, so the kernel measures the same code the pipeline runs:
    solver-pde.md grants scipy.sparse, solver-sde.md grants numpy only."""
    assert "scipy" in sandbox.deny_imports("sde")
    assert "scipy" not in sandbox.deny_imports("pde")
    assert "sympy" in sandbox.deny_imports("pde")


def test_the_crn_increments_are_rebuilt_identically_at_every_level():
    """The finest level is 160 MB at 50 000 paths; it is rebuilt from a seed rather
    than shipped through a file, so every level must be an exact coarsening."""
    fine = sandbox._crn_increments({"seed": 3, "num_paths": 5, "Nt_fine": 8, "T": 1.0,
                                    "aggregate": 1, "m": None})
    coarse = sandbox._crn_increments({"seed": 3, "num_paths": 5, "Nt_fine": 8, "T": 1.0,
                                      "aggregate": 2, "m": None})
    assert fine.shape == (5, 8) and coarse.shape == (5, 4)
    assert np.allclose(coarse, fine.reshape(5, 4, 2).sum(axis=2))
    assert np.allclose(fine.sum(axis=1), coarse.sum(axis=1)), "same Brownian path"


@pytest.mark.parametrize("m", [1, 2, 3])
def test_the_crn_ladder_round_trips_at_the_declared_noise_dimension(m):
    """Plan §4c: the test whose absence let Finding 2 survive.
    ``test_the_crn_increments_are_rebuilt_identically_at_every_level`` above only
    ever passed ``{"m": None}``, so nothing exercised the rank-3 branch that
    ``solver-sde.md`` promises for ``m`` noise sources -- and neither
    ``ladder_spec`` call site passed ``m`` at all. The real
    ``sde_gbm_2d_high_corr`` plan indexes ``dW[:, n, 0]`` and ``dW[:, n, 1]`` and
    raised ``IndexError: too many indices for array``, reported as "the CRN level
    at dt=0.02 crashed"."""
    from verifylib.kernel.sde import crn

    levels = crn.ladder_spec(5, 0.25, 1.0, levels=3, seed=3, m=m)
    assert [lv["m"] for lv in levels] == [m, m, m]
    built = [sandbox._crn_increments(lv) for lv in levels]
    assert [b.shape for b in built] == [(5, 4, m), (5, 8, m), (5, 16, m)]
    # Still one Brownian path, coarsened: the whole point of §18.
    for coarse, fine in zip(built, built[1:], strict=False):
        assert np.allclose(coarse, fine.reshape(5, coarse.shape[1], 2, m).sum(axis=2))
    assert np.allclose(built[0].sum(axis=1), built[-1].sum(axis=1))


def test_a_single_channel_still_takes_the_rank_two_shape():
    """``m=None`` and ``m`` absent both mean one channel, and both must keep the
    ``(num_paths, Nt)`` shape every scalar solver in ``workspace/`` indexes. Giving
    a correct 1-noise solver a rank-3 array is the inverse false positive [REV 19]
    warns about -- both oscillators are 2 states driven by 1 channel."""
    from verifylib.kernel.sde import crn

    for m in (None, 1):
        level = crn.ladder_spec(4, 0.5, 1.0, levels=1, seed=1, m=m)[0]
        shape = sandbox._crn_increments(level).shape
        assert shape == ((4, 2) if m is None else (4, 2, 1)), (m, shape)
    assert crn.ladder_spec(4, 0.5, 1.0, levels=1, seed=1)[0]["m"] is None


# --- the multi-dimensional generator (plan §3 2b) -----------------------------
#
# [REV 61]. ``_generator_nd`` is the newest and most delicate numerics in the SDE
# change -- a central-difference Hessian at ``h = 1e-4 * (1 + |X|)``, chosen at
# ``eps^(1/4)`` -- and until this block its only guard was one live plan's Dynkin
# ``z``. A step-size regression would have surfaced as a shifted ``dynkin_z`` on
# ``sde_gbm_2d_high_corr`` rather than as a red test, and the shift would have been
# read as a property of the solver.
#
# Every case below compares against a generator differentiated **by hand**, so the
# test knows the answer independently of the code under test.

def _nd(drift, matrix, expr, X, params=None, d=None):
    """``_generator_nd`` applied the way the sandbox child applies it."""
    from verifylib.operator import evaluate

    item = {"kind": "generator_nd", "expr": expr, "drift": list(drift),
            "diffusion_matrix": [list(r) for r in matrix], "params": dict(params or {}),
            "state_dimension": int(d or len(drift))}

    def phi(Y):
        return np.asarray(evaluate(expr, {**item["params"], "X": Y}), dtype=float)

    return sandbox._generator_nd(item, phi, np, evaluate)(np.asarray(X, dtype=float))


#: 2-D GBM with correlated noise, in the Cholesky form `sde_gbm_2d_high_corr`
#: declares: dX_i = mu_i X_i dt + X_i sum_j G_ij dW_j.
GBM2 = {"mu1": 0.07, "mu2": -0.03, "s1": 0.4, "s2": 0.25, "rho": 0.9}
GBM2_DRIFT = ["mu1 * X[:, 0]", "mu2 * X[:, 1]"]
GBM2_G = [["s1 * X[:, 0]", "0.0 * X[:, 0]"],
          ["s2 * rho * X[:, 1]", "s2 * np.sqrt(1.0 - rho**2) * X[:, 1]"]]


def test_the_vector_generator_matches_a_hand_differentiated_cross_term():
    """``phi = XY`` is the sharpest case available: ``d_ii phi = 0`` for both
    components, so the **entire** second-order contribution is the mixed partial.
    An ``L`` that never formed the off-diagonal Hessian term still passes a
    ``phi = X**2`` check and fails this one.

        L(XY) = (mu1 + mu2 + rho*s1*s2) * X * Y
    """
    rng = np.random.default_rng(0)
    X = np.exp(rng.standard_normal((400, 2)) * 0.5)
    got = _nd(GBM2_DRIFT, GBM2_G, "X[:, 0] * X[:, 1]", X, GBM2)
    p = GBM2
    want = (p["mu1"] + p["mu2"] + p["rho"] * p["s1"] * p["s2"]) * X[:, 0] * X[:, 1]
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-7


def test_the_vector_generator_matches_the_diagonal_second_derivative():
    """``L(X**2) = 2*mu1*X**2 + s1**2 * X**2`` -- the diagonal half, which a
    cross-term-only implementation would miss."""
    rng = np.random.default_rng(1)
    X = np.exp(rng.standard_normal((400, 2)) * 0.5)
    got = _nd(GBM2_DRIFT, GBM2_G, "X[:, 0]**2", X, GBM2)
    want = (2.0 * GBM2["mu1"] + GBM2["s1"] ** 2) * X[:, 0] ** 2
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-7


def test_every_pair_of_components_reaches_the_hessian():
    """Three dimensions, so the ``i < j`` loop has three pairs rather than one, and
    ``phi = XYZ`` makes each pair the sole carrier of one term:

        L(XYZ) = f1*YZ + f2*XZ + f3*XY + A12*Z + A13*Y + A23*X

    A loop that covered only ``(0, 1)`` -- the shape a two-component test can never
    distinguish from a correct one -- is off by ``A13*Y + A23*X``.
    """
    params = {"a": 0.3, "b": -0.2, "c": 0.11}
    drift = ["a * X[:, 0]", "b * X[:, 1]", "c * X[:, 2]"]
    # A deliberately dense, non-square G: d = 3 against m = 2. The docstring's claim
    # is that m never appears because G G^T is d x d whatever m is.
    G = [["0.5 + 0.1 * X[:, 0]", "0.2"],
         ["0.3", "0.4 - 0.05 * X[:, 1]"],
         ["0.15", "0.25 + 0.1 * X[:, 2]"]]
    rng = np.random.default_rng(2)
    X = 0.5 + rng.random((300, 3))
    got = _nd(drift, G, "X[:, 0] * X[:, 1] * X[:, 2]", X, params)

    from verifylib.operator import evaluate
    Gm = np.stack([np.stack([np.asarray(evaluate(e, {**params, "X": X}), dtype=float)
                             * np.ones(len(X)) for e in row], axis=1) for row in G],
                  axis=1)
    A = np.einsum("nim,njm->nij", Gm, Gm)
    x, y, z = X[:, 0], X[:, 1], X[:, 2]
    want = (params["a"] * x * y * z + params["b"] * x * y * z + params["c"] * x * y * z
            + A[:, 0, 1] * z + A[:, 0, 2] * y + A[:, 1, 2] * x)
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-6


def test_the_step_size_holds_on_a_function_with_large_higher_derivatives():
    """The step-size regression guard. ``h = 1e-4 * (1 + |X|)`` sits at ``eps^(1/4)``:
    the second difference loses ``h**2`` to truncation and gains ``eps/h**2`` to
    cancellation, and the two meet there. ``phi = exp(X + Y)`` has every derivative
    equal to itself, so a step one or two decades either side of the balance point
    shows up here and nowhere else in this file.

        L(exp(X+Y)) = (f1 + f2 + 0.5*(A11 + 2*A12 + A22)) * exp(X+Y)

    Measured on this case by substituting the constant into the module source:

        h0     1e-2      1e-3      1e-4      1e-5      1e-6
        rel    1.6e-4    1.6e-6    1.8e-8    1.2e-6    1.0e-4

    The balance point is where the shipped constant sits, and the ``1e-7`` bar below
    sits between it and its nearest neighbours, so **one decade in either direction
    fails this test**. None of those errors is large next to the Monte Carlo standard
    error at the path counts in use, which is exactly why only an assertion catches
    a regression here.
    """
    params = {"m1": 0.2, "m2": -0.1, "g11": 0.5, "g12": 0.2, "g21": 0.1, "g22": 0.4}
    drift = ["m1 + 0.0 * X[:, 0]", "m2 + 0.0 * X[:, 1]"]
    G = [["g11 + 0.0 * X[:, 0]", "g12 + 0.0 * X[:, 0]"],
         ["g21 + 0.0 * X[:, 1]", "g22 + 0.0 * X[:, 1]"]]
    rng = np.random.default_rng(3)
    X = rng.standard_normal((300, 2)) * 0.8
    got = _nd(drift, G, "np.exp(X[:, 0] + X[:, 1])", X, params)
    Gm = np.array([[params["g11"], params["g12"]], [params["g21"], params["g22"]]])
    A = Gm @ Gm.T
    rate = (params["m1"] + params["m2"]
            + 0.5 * (A[0, 0] + 2.0 * A[0, 1] + A[1, 1]))
    want = rate * np.exp(X[:, 0] + X[:, 1])
    assert np.max(np.abs(got - want) / np.abs(want)) < 1e-7


def test_the_noise_dimension_does_not_change_the_generator():
    """The ``m``-independence the docstring claims, asserted. Two ``G`` matrices with
    the same ``G G^T`` and different widths -- 2 and 13, the live range -- must give
    the same ``L phi`` to roundoff. This is what makes ``m13``'s 13 channels cost
    what two cost."""
    rng = np.random.default_rng(4)
    X = 0.5 + rng.random((200, 2))
    drift, phi = ["0.1", "0.1"], "X[:, 0]**2 + X[:, 1]**2"
    narrow = [["0.6", "0.0"], ["0.0", "0.6"]]
    # 13 channels carrying 0.6/sqrt(13) each, the first component on the first 13
    # and the second on a disjoint 13: G G^T = 0.36 * I either way.
    chan = f"{0.6 / np.sqrt(13):.17g}"
    wide = [[chan] * 13 + ["0.0"] * 13, ["0.0"] * 13 + [chan] * 13]
    a = _nd(drift, narrow, phi, X, d=2)
    b = _nd(drift, wide, phi, X, d=2)
    assert np.allclose(a, b, rtol=1e-9)
    # ...and both are the hand-differentiated answer: L(X^2+Y^2) = 2*0.1*(X+Y) + 0.72
    want = 2.0 * 0.1 * (X[:, 0] + X[:, 1]) + 0.36 * 2.0
    assert np.max(np.abs(a - want)) < 1e-7
