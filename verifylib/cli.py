#!/usr/bin/env python3
"""The single entry point. Invoked by absolute path so it works from any cwd.

    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py check-spec <path>
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook post-tool-use     # payload on stdin
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook pre-tool-use
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook stop
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py gate <workspace_dir> [--json]

Exit codes follow the hook contract measured in ``tests/hookprobe/FINDINGS.md``:
2 is the blocking/feedback code the agent sees on stderr, 0 is pass. Only
``PreToolUse`` and ``Stop`` actually block on 2; ``PostToolUse`` exit 2 delivers
stderr to the agent but leaves the file on disk, which is why nothing depends on
it alone.
"""

from __future__ import annotations

import os
import sys

# Run as a script, sys.path[0] is *this* directory -- which contains operator.py and
# would therefore shadow the stdlib `operator` module for the whole interpreter.
# collections imports operator, functools imports collections, re imports functools,
# json imports re: the failure surfaces as a circular-import error inside stdlib
# json, with nothing pointing at the real cause. Drop this directory and add the
# package root before importing anything else. `sys` and `os` are safe to import
# first; neither reaches operator.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import json  # noqa: E402
import re  # noqa: E402

from verifylib import gate  # noqa: E402
from verifylib.findings import errors, render  # noqa: E402

BLOCK = 2
PASS = 0

#: `matcher` is tool-name only -- a path glob never fires -- so path filtering
#: happens here, and the matcher must include Bash or a heredoc walks past it.
_REDIRECT_RE = re.compile(r">>?\s*([^\s|&;<>]+)")
_HEREDOC_RE = re.compile(r"<<-?\s*'?\"?(\w+)")


def _targets_from_bash(command: str) -> list[str]:
    """Files a Bash command writes to. ``cat > problem_spec.json`` fires no
    Write|Edit hook at all, and the file lands unvalidated."""
    return [m.group(1) for m in _REDIRECT_RE.finditer(command or "")]


def hook_targets(payload: dict) -> list[str]:
    tool = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if tool == "Bash":
        return _targets_from_bash(tool_input.get("command", ""))
    path = tool_input.get("file_path") or tool_input.get("path")
    return [path] if path else []


def _resolve(path, payload):
    if os.path.isabs(path):
        return path
    root = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return os.path.join(root, path)


def _read_payload():
    try:
        return json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}


# --- hook entry points -------------------------------------------------------

def hook_post_tool_use(payload) -> int:
    """Feedback after a write. Cannot block (measured), so this is an optimisation
    over the out-of-band gate, never a replacement for it."""
    found = []
    for target in hook_targets(payload):
        path = _resolve(target, payload)
        # with_reference=False: the schema half is instant; the residual check
        # evaluates a formula on up to 1025^d points and belongs at the gate.
        found.extend(gate.check_path(path, with_reference=False))
    if not errors(found):
        return PASS
    print(render(errors(found)), file=sys.stderr)
    return BLOCK


def hook_pre_tool_use(payload) -> int:
    """The only hook that actually blocks. Used where blocking matters: leakage
    into ``solver.py`` / ``evaluate.py``, where a ``Write`` carries the whole
    content and can be checked before it exists.

    An ``Edit`` carries only ``old_string``/``new_string``, so the finished file
    cannot be reconstructed here -- those fall through to the gate rather than
    being validated against a fragment.
    """
    tool_input = payload.get("tool_input") or {}
    content = tool_input.get("content")
    found = []
    for target in hook_targets(payload):
        if os.path.basename(target) not in gate.SOLVER_FILES:
            continue
        if isinstance(content, str):
            found.extend(gate.check_solver_text(content, path=target))
        else:
            found.extend(gate.check_path(_resolve(target, payload), with_reference=False))
    if not errors(found):
        return PASS
    print(render(errors(found)), file=sys.stderr)
    return BLOCK


def hook_stop(payload) -> int:
    """The gate that works everywhere -- including a user's own problem, where
    ``benchmark/verify.py`` does not exist.

    Bounded on purpose. Measured, an unbounded Stop hook fired 8 times and killed
    the run at max-turns, so the attempt counter is mandatory rather than
    defensive: on the 3rd firing for the same finding it records the reason and
    lets the session end.
    """
    root = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    workspace = os.path.join(root, "workspace")
    target = gate.active_problem(workspace if os.path.isdir(workspace) else root)

    found = errors(gate.check_workspace(target))
    if not found:
        gate.clear_stop_attempts(target)
        return PASS

    attempt = gate.record_stop_attempt(target, gate.fingerprint(found))
    if attempt >= gate.MAX_STOP_ATTEMPTS:
        marker = gate.write_unsatisfied_marker(target, found)
        print(f"{gate.UNSATISFIED_MARKER}: {len(found)} finding(s) still outstanding "
              f"after {attempt} attempts. Recorded in {marker}; ending the session.",
              file=sys.stderr)
        gate.clear_stop_attempts(target)
        return PASS
    print(f"Guardrail findings must be fixed before this session ends "
          f"(attempt {attempt} of {gate.MAX_STOP_ATTEMPTS}):\n{render(found)}",
          file=sys.stderr)
    return BLOCK


HOOKS = {"post-tool-use": hook_post_tool_use,
         "pre-tool-use": hook_pre_tool_use,
         "stop": hook_stop}


# --- direct entry points -----------------------------------------------------

def _report(found, as_json):
    if as_json:
        print(json.dumps([f.as_dict() for f in found], indent=2))
    elif found:
        print(render(found))
    else:
        print("ok: no findings")
    return BLOCK if errors(found) else PASS


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    command, args = argv[0], argv[1:]
    as_json = "--json" in args
    args = [a for a in args if not a.startswith("--")]

    if command == "hook":
        if not args or args[0] not in HOOKS:
            print(f"hook takes one of {sorted(HOOKS)}", file=sys.stderr)
            return 1
        try:
            return HOOKS[args[0]](_read_payload())
        except Exception as exc:  # noqa: BLE001
            # A guard that crashes must not take the session with it.
            print(f"verifylib hook error ({type(exc).__name__}: {exc})", file=sys.stderr)
            return PASS

    if command == "check-spec":
        return _report(gate.check_spec_file(args[0]), as_json)
    if command == "check-solver":
        return _report(gate.check_solver_file(args[0]), as_json)
    if command == "check-review":
        return _report(gate.check_review_file(args[0]), as_json)
    if command in ("gate", "check-workspace"):
        return _report(gate.check_workspace(args[0] if args else "workspace"), as_json)

    print(f"unknown command {command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
