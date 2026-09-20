"""Malformed plans and broken solver contracts: what the kernel does with them.

The §9c canaries seed *numerical* defects into a correct solver. These seed
*contract* defects -- a snapshot without a time, a grid that ignores ``N``, a
field with a NaN in it, a mask that does not evaluate -- and pin down how each is
reported. The standing rules being tested are the manual's: a skipped check is
never a pass, a crash is a property of the code and gets a named reason, and the
kernel must never quietly measure the wrong thing and call it a verdict.

Several of these are ``xfail(strict=True)``. Each names a behaviour the plan
states and the code does not yet enforce, measured before the test was written;
``strict`` means the fix flips the test rather than silently passing it.

Every solver here is ``canaries/kernel/honest.py`` with one thing changed, written
beside it so the sandbox's ``sys.path`` rule lets it ``from honest import ...``.
"""
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from verifylib.kernel import plan_meta, run

from .conftest import REPO
from .test_kernel_canaries import CANARY, _spec

# --- the seeded contract defects --------------------------------------------

IGNORES_N = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    # Whatever the ladder asks for, the answer is on the same 33-point grid.
    return _honest(33, override)
"""

IGNORES_N_NO_SNAPSHOTS = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(33, override)
    out.pop("snapshots")
    return out
"""

SNAPSHOT_WITHOUT_T = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    for snap in out["snapshots"]:
        del snap["t"]
    return out
"""

DUPLICATE_SNAPSHOT_TIMES = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    for snap in out["snapshots"]:
        snap["t"] = out["t_final"]
    return out
"""

SNAPSHOTS_END_EARLY = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    for snap in out["snapshots"]:
        snap["t"] = snap["t"] - 0.1      # the stored states are not the final one
    return out
"""

TWO_SNAPSHOTS = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    out["snapshots"] = out["snapshots"][-2:]
    return out
"""

NAN_IN_DOMAIN = """
import numpy as np
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    out = _honest(N, override)
    u = np.array(out["numerical_solution"])
    u[N // 2] = np.nan
    out["numerical_solution"] = u
    return out
"""

IGNORES_SOURCE_OVERRIDE = """
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    ov = dict(override or {})
    ov.pop("source", None)       # the MMS probe's manufactured source never arrives
    return _honest(N, ov)
"""

WRONG_INITIAL_STATE = """
import numpy as np
from honest import solve_pde as _honest

def solve_pde(N, override=None):
    ov = dict(override or {})
    if "ic" not in ov:
        # Starts from twice the declared initial condition. Linear problem, so
        # the field is exactly right up to a factor of two -- and the residual is
        # identically satisfied by it.
        ov["ic"] = lambda coords: 2.0 * np.sin(np.pi * coords[0])
    return _honest(N, ov)
"""


def _plan(tmp_path, solver_text):
    shutil.copy(os.path.join(CANARY, "honest.py"), tmp_path / "honest.py")
    (tmp_path / "solver.py").write_text(solver_text)
    return str(tmp_path)


def _run(tmp_path, solver_text, spec=None, config=None):
    return run(spec or _spec(), _plan(tmp_path, solver_text), config or {})


# --- the ladder ---------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "a solver that returns the same grid at every N makes d10 = d21 = 0, which the "
    "ladder reads as 'converged to round-off' (order inf, converged True). Measured: "
    "with no snapshots this scores 9 manufactured_partial. The ladder must notice "
    "that the grid did not change with N and name it"))
def test_a_solver_that_ignores_n_is_not_read_as_converged_to_roundoff(tmp_path):
    """Manual §7: a solver that ignores its inputs cannot be certified. Identical
    grids at three ladder levels are not machine-precision agreement."""
    m = _run(tmp_path, IGNORES_N_NO_SNAPSHOTS)
    assert m["checks"]["converged"] is not True, m["detail"]["richardson"]
    assert m["score"] < 9
    text = json.dumps(m, default=str).lower()
    assert "n" in text and ("ignore" in text or "did not change" in text or "same grid" in text)


def test_a_solver_that_ignores_n_with_snapshots_is_sunk_by_d1(tmp_path):
    """The one thing that catches it today: the residual cannot fall between two
    identical grids, so D1 reports ``stalled`` and gates. Recorded so the xfail
    above is not mistaken for 'the kernel is blind to this'."""
    m = _run(tmp_path, IGNORES_N)
    assert m["d1"]["outcome"] == "stalled"
    assert m["d1"]["p_res"] == pytest.approx(0.0)
    assert m["score"] == 3


# --- snapshots ----------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "sandbox._dump_pde checks `'fields' in snap` but reads snap['t'] unguarded; the "
    "KeyError is reported as reason 'exception', not 'bad_schema'"))
def test_a_snapshot_missing_its_time_is_a_contract_violation(tmp_path):
    """The contract is ``{'t': float, 'fields': {...}}``. A snapshot without ``t``
    is a schema violation and the crash reason must say so, so the solver agent is
    told to fix its return value rather than to debug a KeyError."""
    m = _run(tmp_path, SNAPSHOT_WITHOUT_T)
    assert m["checks"]["crashed"] is True
    assert m["crash"]["reason"] == "bad_schema", m["crash"]


@pytest.mark.xfail(strict=True, reason=(
    "duplicate snapshot times reach fornberg_weights outside any try/except, divide "
    "by zero, and produce NaN medians that the stencil-insensitivity guard then "
    "reports as 'unresolved (nan vs nan)'. Measured: score 9. D1 should be "
    "`unavailable` with a reason naming the times"))
def test_snapshots_with_duplicate_times_make_d1_unavailable_not_nan(tmp_path):
    m = _run(tmp_path, DUPLICATE_SNAPSHOT_TIMES)
    assert m["d1"]["outcome"] == "unavailable", m["d1"]
    assert "t" in str(m["d1"].get("reason", "")).lower()
    for level in m["d1"].get("per_level", []):
        assert np.isfinite(level["median"])


@pytest.mark.xfail(strict=True, reason=(
    "snapshots whose last t is not t_final are used as if they were the final "
    "state; the residual is formed at the wrong instant and reports clean. "
    "Measured: score 10 with the snapshots shifted by 0.1"))
def test_snapshots_that_end_before_t_final_are_named(tmp_path):
    """§5c: the snapshots are 'the last K states, ending at t_final'. A solver
    that returns states from somewhere else in the run is not returning what D1
    differences, and the mismatch must be visible in the metrics."""
    m = _run(tmp_path, SNAPSHOTS_END_EARLY)
    assert m["d1"]["outcome"] in ("unavailable", "stalled"), m["d1"]
    assert "t_final" in json.dumps(m["d1"], default=str)


def test_two_snapshots_run_and_record_a_first_order_time_term(tmp_path):
    """§5c: ``K = 2`` runs but records its own truncation floor. The outcome is
    still a verdict -- a first-order time term is a measurement, not a skip."""
    m = _run(tmp_path, TWO_SNAPSHOTS)
    assert m["d1"]["outcome"] in ("clean", "slow", "stalled")
    assert all(level["snapshot_order"] == 1 for level in m["d1"]["per_level"])


def test_three_snapshots_record_a_second_order_time_term(tmp_path):
    """The control for the case above."""
    m = _run(tmp_path, "from honest import solve_pde\n")
    assert all(level["snapshot_order"] == 2 for level in m["d1"]["per_level"])


# --- the field ----------------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "a NaN inside the domain makes d10 = d21 = NaN, and `not_at_roundoff` reads "
    "NaN as 'converged to machine precision' (converged True, order inf). It is "
    "only sunk today by the positivity gate and the nontrivial guard noticing the "
    "NaN. The SDE driver crashes on non-finite output; the PDE driver should too"))
def test_a_nan_in_the_field_is_a_crash_not_a_round_off_convergence(tmp_path):
    m = _run(tmp_path, NAN_IN_DOMAIN)
    assert m["checks"].get("converged") is not True
    assert m["checks"].get("crashed") is True or m["score"] <= 2


def test_a_nan_in_the_field_never_certifies_today(tmp_path):
    """Whatever catches it, it must not pass. The gate layer does today."""
    m = _run(tmp_path, NAN_IN_DOMAIN)
    assert m["score"] <= 3


# --- overrides ----------------------------------------------------------------

def test_a_solver_that_ignores_the_source_override_cannot_reach_tier_b_by_mms(tmp_path):
    """Manual §7: the MMS probe hands the solver a manufactured source term. A
    solver that drops it solves the homogeneous problem and compares badly against
    the probe's exact solution -- so the MMS route must not certify it. (The
    degenerate limit, which only overrides ``params``, may still pass: that is the
    point of having two routes.)"""
    m = _run(tmp_path, IGNORES_SOURCE_OVERRIDE)
    tier_b = m["tier_b"]
    assert tier_b.get("route") != "mms", tier_b
    mms = tier_b.get("mms") or tier_b
    assert mms.get("outcome") is not True


@pytest.mark.xfail(strict=True, reason=(
    "plan §5f names three trivial-attractor guards, all gating; residual.ic_consistency "
    "exists but driver._run_d1 never calls it. Measured: a solver that starts from "
    "2*sin(pi x) scores 10"))
def test_ic_consistency_gates_a_solver_that_starts_from_the_wrong_state(tmp_path):
    """The operator is linear, so twice the solution is a solution: the residual is
    identically satisfied, the ladder converges, and MMS (which supplies its own
    IC) passes. Only 'did the run start where it should' sees it."""
    m = _run(tmp_path, WRONG_INITIAL_STATE)
    assert m["d1"].get("ic_consistent") is False, m["d1"]
    assert m["score"] <= 3


# --- the domain mask ----------------------------------------------------------

@pytest.mark.xfail(strict=True, reason=(
    "driver._domain_mask returns None on any exception, so D1 runs unmasked -- "
    "plan §5e: 'never quietly difference across a hole'. Measured: an unevaluable "
    "mask expression leaves D1 clean and the score at 10"))
def test_an_unevaluable_domain_mask_makes_d1_unavailable_not_unmasked(tmp_path):
    spec = _spec(**{"evaluation_thresholds.domain_mask": "np.no_such_function(x)"})
    m = _run(tmp_path, "from honest import solve_pde\n", spec=spec)
    assert m["d1"]["outcome"] == "unavailable", m["d1"]
    assert "mask" in str(m["d1"].get("reason", "")).lower()


@pytest.mark.xfail(strict=True, reason=(
    "driver.evaluate_pde builds the domain mask on the finest grid only, but "
    "_run_d1 differences the top two grids; at the coarser level the mask has the "
    "wrong shape and D1 reports 'unavailable: operands could not be broadcast "
    "together with shapes (65,) (129,)'. Every masked-domain problem loses D1"))
def test_an_evaluable_mask_is_applied_at_every_level_d1_reads(tmp_path):
    """The control for the case above: a mask that evaluates narrows the interior
    and D1 still runs -- on both grids the slope test needs."""
    spec = _spec(**{"evaluation_thresholds.domain_mask": "(x > 0.2) & (x < 0.8)"})
    m = _run(tmp_path, "from honest import solve_pde\n", spec=spec)
    assert m["d1"]["outcome"] == "clean", m["d1"]
    full = _run(tmp_path, "from honest import solve_pde\n")
    assert m["d1"]["per_level"][-1]["stats"]["n"] < full["d1"]["per_level"][-1]["stats"]["n"]


# --- the plan's frontmatter ---------------------------------------------------

@pytest.mark.parametrize("text, family, order", [
    ("---\nscheme_family: banana\nspatial_order: 2\n---\n", None, 2),
    ("---\nscheme_family: fd\nspatial_order: two\n---\n", "fd", None),
    ("---\nscheme_family: FD\nspatial_order: 4.0\n---\n", "fd", 4),
    ("﻿---\r\nscheme_family: spectral\r\nspatial_order: 8\r\n---\r\nbody", "spectral", 8),
    ("no frontmatter at all\nscheme: finite-difference-x\n", None, None),
    ("---\nscheme: finite-difference-x\n---\n", "fd", None),
    ("---\nscheme_family:\n---\n", None, None),
])
def test_malformed_frontmatter_falls_back_rather_than_crashing(tmp_path, text, family, order):
    """An unknown family is ``None``, not an error; a non-numeric order is ``None``;
    a BOM and CRLF are tolerated; a ``scheme:`` line *outside* the block is never
    read (§3b: parse the frontmatter, never grep), and one inside it is only ever
    an inference."""
    (tmp_path / "SOLUTION.md").write_text(text, encoding="utf-8")
    got = plan_meta.read(str(tmp_path))
    assert got["scheme_family"] == family
    assert got["spatial_order"] == order


def test_an_invalid_family_declaration_is_recorded_as_a_default(tmp_path):
    """Through the driver: the stencil pair falls back to the kernel default and
    the config record says so, rather than treating 'banana' as a declaration."""
    plan = _plan(tmp_path, "from honest import solve_pde\n")
    (tmp_path / "SOLUTION.md").write_text("---\nscheme_family: banana\n---\n")
    m = run(_spec(), plan, {})
    assert m["plan"]["scheme_family"] is None
    assert m["d1"]["stencil_choice"] == "default"
    assert m["config"]["scheme_family"]["source"] == "kernel_default"


def test_a_missing_solution_md_is_not_a_crash(tmp_path):
    m = _run(tmp_path, "from honest import solve_pde\n")
    assert m["plan"]["declared"] is False
    assert m["plan"]["scheme_family_source"] == "unknown"
    assert m["checks"].get("crashed") is not True
    assert m["score"] == 10


# --- the CLI ------------------------------------------------------------------

CLI = os.path.join(REPO, "verifylib", "cli.py")


def test_cli_evaluate_with_no_spec_above_the_plan_exits_one_with_a_message(tmp_path):
    (tmp_path / "solver.py").write_text("def solve_pde(N, override=None): return {}\n")
    proc = subprocess.run([sys.executable, CLI, "evaluate", str(tmp_path)],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode == 1
    assert "problem_spec.json" in proc.stderr
    assert "Traceback" not in proc.stderr


def test_cli_evaluate_with_a_bad_config_override_exits_one(tmp_path):
    plan = tmp_path / "plans" / "p"
    plan.mkdir(parents=True)
    (tmp_path / "problem_spec.json").write_text(json.dumps(_spec()))
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "solver.py")
    proc = subprocess.run([sys.executable, CLI, "evaluate", str(plan)],
                          capture_output=True, text=True, cwd=REPO,
                          env={**os.environ, "VERIFYLIB_EVAL_CONFIG": "{not json"})
    assert proc.returncode == 1
    assert "VERIFYLIB_EVAL_CONFIG" in proc.stderr


def test_cli_evaluate_without_a_plan_directory_argument_exits_nonzero():
    proc = subprocess.run([sys.executable, CLI, "evaluate"],
                          capture_output=True, text=True, cwd=REPO)
    assert proc.returncode != 0
    assert "plan directory" in proc.stderr


def test_cli_json_output_carries_score_provenance_and_certification(tmp_path):
    plan = tmp_path / "plans" / "p"
    plan.mkdir(parents=True)
    (tmp_path / "problem_spec.json").write_text(json.dumps(_spec()))
    shutil.copy(os.path.join(CANARY, "honest.py"), plan / "solver.py")
    proc = subprocess.run([sys.executable, CLI, "evaluate", str(plan), "--json"],
                          capture_output=True, text=True, cwd=REPO, timeout=300)
    assert proc.returncode == 0, proc.stderr
    m = json.loads(proc.stdout)                 # numpy types were serialised
    assert m["score"] == 10 and m["provenance"] == "manufactured"
    assert m["certification"]["bound_by"] in ("rubric", "certification")
    assert isinstance(m["ladder"]["Ns"], list)
