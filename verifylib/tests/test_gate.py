"""Criteria 12-15: the enforcement model.

Every guard runs in two places -- as a hook for in-loop correction, and as a gate
that does not depend on an agent cooperating. These tests exercise the gate path
*with the hooks absent*, which is criterion 13's point: the out-of-band path has to
catch the same file on its own.
"""
import json
import os
import subprocess
import sys

import pytest

from verifylib import gate
from verifylib.cli import hook_post_tool_use, hook_pre_tool_use, hook_stop, hook_targets

from .conftest import FIXTURES, REPO, read_text

BAD_SPEC = {
    "spatial_variables": ["x"],
    "analytic_solution": {"expression": "def f(x):\n    return x\n"},
    "verification": {"operator": None},
    "requirements": [{"id": "R1", "kind": "equation", "quote": "u_t = u_xx",
                      "status": "mapped", "spec_path": "governing_equation"}],
}


def _write(tmp_path, name, text, subdir=""):
    directory = tmp_path / subdir if subdir else tmp_path
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text)
    return str(path)


# --- dispatch ----------------------------------------------------------------

def test_gate_dispatches_on_filename(tmp_path):
    spec = _write(tmp_path, "problem_spec.json", json.dumps(BAD_SPEC))
    assert gate.check_path(spec)
    unrelated = _write(tmp_path, "notes.md", "benchmark/problems.py")
    assert gate.check_path(unrelated) == []


def test_missing_file_is_not_an_error(tmp_path):
    assert gate.check_path(str(tmp_path / "problem_spec.json")) == []


def test_malformed_json_is_reported_not_raised(tmp_path):
    path = _write(tmp_path, "problem_spec.json", "{not json")
    findings = gate.check_path(path)
    assert findings and findings[0].check == "schema"


def test_archived_leak_artifacts_are_caught(tmp_path):
    """Both real incidents, through the gate rather than the scanner directly."""
    spec = _write(tmp_path, "problem_spec.json",
                  read_text(os.path.join(FIXTURES, "leak_heston_spec.json")))
    assert gate.errors(gate.check_path(spec))
    solver = _write(tmp_path, "evaluate.py",
                    read_text(os.path.join(FIXTURES, "leak_quintic_evaluate.py")))
    assert gate.errors(gate.check_path(solver))


def test_a_real_spec_passes_the_gate():
    """The guard must be silent on correct work, or it gets switched off."""
    path = os.path.join(REPO, "workspace", "pde_heat_1d", "problem_spec.json")
    if not os.path.exists(path):
        pytest.skip("workspace not staged")
    assert gate.errors(gate.check_path(path)) == []


# --- criterion 12: a Bash heredoc must fire the hook -------------------------

def test_bash_redirect_targets_are_seen():
    """`cat > problem_spec.json` fires no Write|Edit hook at all, and the file
    lands unvalidated. The matcher has to include Bash, and the redirect target
    has to be parsed out of tool_input.command."""
    payload = {"tool_name": "Bash",
               "tool_input": {"command": "cat > workspace/p/problem_spec.json <<'EOF'\n{}\nEOF"}}
    assert hook_targets(payload) == ["workspace/p/problem_spec.json"]


def test_write_target_is_seen():
    payload = {"tool_name": "Write", "tool_input": {"file_path": "/tmp/problem_spec.json"}}
    assert hook_targets(payload) == ["/tmp/problem_spec.json"]


def test_heredoc_write_fires_the_post_hook(tmp_path):
    path = _write(tmp_path, "problem_spec.json", json.dumps(BAD_SPEC))
    payload = {"tool_name": "Bash", "cwd": str(tmp_path),
               "tool_input": {"command": f"cat > {path} <<'EOF'\n{{}}\nEOF"}}
    assert hook_post_tool_use(payload) == 2


def test_post_hook_passes_a_clean_write(tmp_path):
    path = _write(tmp_path, "solver.py", "import numpy as np\n\ndef solve_pde(N):\n    return {}\n")
    payload = {"tool_name": "Write", "cwd": str(tmp_path), "tool_input": {"file_path": path}}
    assert hook_post_tool_use(payload) == 0


# --- PreToolUse blocks, and only where blocking is meaningful ----------------

def test_pre_hook_blocks_a_leaking_solver_before_it_exists(tmp_path):
    payload = {"tool_name": "Write", "cwd": str(tmp_path),
               "tool_input": {"file_path": str(tmp_path / "solver.py"),
                              "content": "import problems\n"}}
    assert hook_pre_tool_use(payload) == 2
    assert not (tmp_path / "solver.py").exists(), "the hook must not create the file"


def test_pre_hook_ignores_a_spec_write():
    """PreToolUse on an Edit sees only the diff, so a validator that must parse the
    finished spec would have to apply the edit itself. Specs go to PostToolUse and
    the gate instead."""
    payload = {"tool_name": "Write",
               "tool_input": {"file_path": "problem_spec.json", "content": "{"}}
    assert hook_pre_tool_use(payload) == 0


# --- criterion 15: the Stop gate terminates ----------------------------------

def test_stop_gate_blocks_then_gives_up_with_a_recorded_reason(tmp_path):
    """Measured, an unbounded Stop hook fired 8 times and the run died at
    max-turns. The bail-out is mandatory, not defensive."""
    workspace = tmp_path / "workspace" / "prob"
    workspace.mkdir(parents=True)
    (workspace / "problem_spec.json").write_text(json.dumps(BAD_SPEC))
    payload = {"cwd": str(tmp_path)}

    codes = [hook_stop(payload) for _ in range(gate.MAX_STOP_ATTEMPTS)]
    assert codes[:-1] == [2] * (gate.MAX_STOP_ATTEMPTS - 1), "should block while retrying"
    assert codes[-1] == 0, "must let the session end rather than run to max-turns"

    marker = tmp_path / "workspace" / "prob" / gate.UNSATISFIED_MARKER
    assert marker.exists(), "an unfixable run must terminate with the reason on disk"
    assert "expression" in marker.read_text()


def test_stop_gate_scopes_to_the_problem_the_session_is_working_on(tmp_path):
    """The staged benchmark workspace holds 29 problem directories from previous
    runs, three of which carry pre-existing findings. Sweeping all of them would
    block a session working on a fourth, for something it did not do and cannot
    fix. Both the pipeline and a user's own run work one problem at a time."""
    workspace = tmp_path / "workspace"
    stale = workspace / "old_problem"
    stale.mkdir(parents=True)
    (stale / "problem_spec.json").write_text(json.dumps(BAD_SPEC))
    os.utime(stale / "problem_spec.json", (1, 1))

    current = workspace / "current_problem"
    current.mkdir()
    (current / "problem_spec.json").write_text(json.dumps(
        {**BAD_SPEC, "analytic_solution": {"expression": "np.sin(x)"},
         "verification": {"operator": None, "operator_note": "steady state"}}))

    assert gate.active_problem(str(workspace)) == str(current)
    assert hook_stop({"cwd": str(tmp_path)}) == 0, "a stale problem must not block this session"


def test_active_problem_falls_back_to_a_bare_directory(tmp_path):
    """A scratch directory holding one problem_spec.json is a perfectly good
    workspace -- that is criterion 14's shape."""
    (tmp_path / "problem_spec.json").write_text(json.dumps(BAD_SPEC))
    assert gate.active_problem(str(tmp_path)) == str(tmp_path)


def test_stop_gate_passes_a_clean_workspace(tmp_path):
    (tmp_path / "workspace").mkdir()
    assert hook_stop({"cwd": str(tmp_path)}) == 0


def test_a_different_finding_resets_the_counter(tmp_path):
    """The agent fixed one thing and broke another. That is progress, not a loop."""
    workspace = tmp_path / "workspace" / "prob"
    workspace.mkdir(parents=True)
    spec = workspace / "problem_spec.json"
    spec.write_text(json.dumps(BAD_SPEC))
    payload = {"cwd": str(tmp_path)}
    assert hook_stop(payload) == 2
    assert hook_stop(payload) == 2
    spec.write_text("{not json")           # a different finding
    assert hook_stop(payload) == 2, "the counter should have reset"


def test_hook_errors_never_kill_the_session():
    """A guard that crashes must not take the run with it."""
    from verifylib.cli import main
    assert main(["hook", "stop"]) in (0, 2)


# --- criterion 13/14: the gate stands alone ----------------------------------

def test_workspace_gate_finds_what_the_write_hooks_would_have(tmp_path):
    """Criterion 13: the same file, caught with the write hooks disabled."""
    problem = tmp_path / "prob"
    problem.mkdir()
    (problem / "problem_spec.json").write_text(json.dumps(BAD_SPEC))
    (problem / "solver.py").write_text("from benchmark import verify\n")
    findings = gate.check_workspace(str(tmp_path))
    checks = {f.check for f in gate.errors(findings)}
    assert {"expression", "leakage"} <= checks, checks


def test_cli_runs_from_any_directory_as_a_script(tmp_path):
    """Criterion 14 in miniature, and a regression test for a real landmine:
    run as a script, sys.path[0] is verifylib/, whose operator.py shadows the
    stdlib `operator` module and breaks `import json` inside the interpreter."""
    spec = tmp_path / "problem_spec.json"
    spec.write_text(json.dumps(BAD_SPEC))
    result = subprocess.run(
        [sys.executable, os.path.join(REPO, "verifylib", "cli.py"), "check-spec", str(spec)],
        capture_output=True, text=True, cwd=str(tmp_path), timeout=120,
    )
    assert "Traceback" not in result.stderr, result.stderr
    assert result.returncode == 2, result.stdout + result.stderr
    assert "expression" in result.stdout


def test_plugin_hooks_json_is_wired_to_the_cli():
    """The hooks ship with the plugin, not with benchmark/pipeline-settings.json --
    otherwise every guard only ever fires on benchmark runs."""
    config = json.loads(read_text(os.path.join(REPO, "hooks", "hooks.json")))["hooks"]
    assert set(config) == {"PreToolUse", "PostToolUse", "Stop"}
    for event, entries in config.items():
        for entry in entries:
            command = entry["hooks"][0]["command"]
            assert "${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py" in command
            if event != "Stop":
                # A Bash heredoc walks straight past a Write|Edit matcher.
                assert "Bash" in entry["matcher"], f"{event} matcher misses Bash"


# --- the plugin runs on whatever interpreter the shell resolves --------------

def test_ast_only_guards_survive_an_interpreter_without_numpy(tmp_path):
    """Hooks are executed by whatever ``python3`` the shell resolves, not by the
    repo's own interpreter, and nothing guarantees that one has numpy. Measured:
    with ``PATH=/usr/bin:/bin`` on macOS it does not.

    A hard numpy import would take the whole guard down -- including the checks
    that need nothing but ``ast``, which are the ones that catch both archived
    incidents. Those must still run, and the numeric check must say plainly that
    it did not.
    """
    shim = tmp_path / "noshim"
    shim.mkdir()
    (shim / "numpy.py").write_text("raise ImportError('numpy disabled for this test')\n")
    (shim / "scipy.py").write_text("raise ImportError('scipy disabled for this test')\n")

    env = {**os.environ, "PYTHONPATH": str(shim)}
    env.pop("PYTHONHOME", None)

    def run(name, text):
        path = tmp_path / name
        path.write_text(text)
        return subprocess.run(
            [sys.executable, os.path.join(REPO, "verifylib", "cli.py"), "check-spec", str(path)],
            capture_output=True, text=True, cwd=str(tmp_path), env=env, timeout=120,
        )

    # The archived leak: both guards that caught the real incidents still fire.
    leak = run("problem_spec.json", read_text(os.path.join(FIXTURES, "leak_heston_spec.json")))
    assert "Traceback" not in leak.stderr, leak.stderr
    assert leak.returncode == 2, leak.stdout + leak.stderr
    assert "expression" in leak.stdout and "leakage" in leak.stdout

    # A spec with nothing for the AST guards to say: the numeric check reports its
    # own absence rather than being silently skipped.
    clean = run("clean_spec.json", json.dumps({
        **BAD_SPEC,
        "analytic_solution": {"expression": "np.sin(np.pi * x)"},
        "verification": {"operator": None, "operator_note": "steady state"},
    }))
    assert "Traceback" not in clean.stderr, clean.stderr
    assert "reference check skipped" in clean.stdout, clean.stdout
