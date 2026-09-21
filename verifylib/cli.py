#!/usr/bin/env python3
"""The single entry point. Invoked by absolute path so it works from any cwd.

    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py check-spec <path>
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook post-tool-use     # payload on stdin
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook pre-tool-use
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook stop
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py gate <workspace_dir> [--json]
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py evaluate <plan_dir> [--json]
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py compare <plan_a> <plan_b> [--N=<int>] [--no-record] [--json]

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


def _evaluate(plan_dir, as_json) -> int:
    """Run the numerical kernel on one plan directory.

    This is the entry point that replaces a generated ``evaluate.py``. It is a CLI
    subcommand rather than a ``verifylib.run(...)`` call inside a generated script
    for three reasons: ``import verifylib`` only resolves inside this repo, while on
    a user's own problem the package lives at ``${CLAUDE_PLUGIN_ROOT}``; this module
    already does the ``sys.path`` surgery that stops ``verifylib/operator.py``
    shadowing the stdlib ``operator`` module for the whole interpreter, which the
    kernel would hit the moment it imported ``functools``; and "no ``evaluate.py``
    contains numerical logic" is not a checkable criterion while "no ``evaluate.py``
    exists" is.
    """
    from verifylib import gate as _gate
    from verifylib.kernel import run

    plan_dir = os.path.abspath(plan_dir)
    spec = _gate._sibling_spec(os.path.join(plan_dir, "SOLUTION.md"))
    if spec is None:
        print(f"no problem_spec.json above {plan_dir}", file=sys.stderr)
        return 1

    config = {}
    override = os.environ.get("VERIFYLIB_EVAL_CONFIG")
    if override:
        try:
            config = json.loads(override)
        except json.JSONDecodeError as exc:
            print(f"VERIFYLIB_EVAL_CONFIG is not valid JSON: {exc}", file=sys.stderr)
            return 1

    metrics = run(spec, plan_dir, config)
    if as_json:
        print(json.dumps(metrics, indent=2, default=_jsonable))
    else:
        from verifylib.kernel.metrics import render_block
        print(render_block(metrics))
        print()
        print(f"bound by: {metrics['certification']['bound_by']} -- "
              f"{metrics['certification']['reason']}")
        for name, outcome in sorted(metrics.get("checks", {}).items()):
            print(f"  {name:26s} {outcome}")
        for line in metrics.get("notes", []):
            print(f"  note: {line}")
    # 0 whatever the score: a low score is a measurement, not a tool failure. Only a
    # kernel that could not run at all is an error, and that raises.
    return PASS


def _compare(plan_a, plan_b, *, N=None, no_record=False, as_json=False) -> int:
    """Findings F2: difference two plans' finest full-T fields on a common grid.

    Solves both at the top ladder level (or ``--N``), restricts onto the common
    grid, and reports the relative difference against the spec's
    ``rel_l2_err_max``. Unless ``--no-record`` it writes the result to
    ``<workspace>/agreements.json`` with both solver hashes, which the kernel
    reads back on the next ``evaluate`` of either plan: agreement between
    different scheme families is priced as a circularity break, disagreement
    caps both plans at 8. Exit 0 on agreement, 2 on disagreement, 1 on a crash.
    """
    from verifylib import gate as _gate
    from verifylib.kernel import agreement, ladder, plan_meta, sandbox
    from verifylib.kernel.driver import solver_sha256
    from verifylib.kernel.metrics import Config

    plan_a, plan_b = os.path.abspath(plan_a), os.path.abspath(plan_b)
    if agreement.workspace_of(plan_a) != agreement.workspace_of(plan_b):
        print("compare takes two plans of the same workspace", file=sys.stderr)
        return 1
    spec = _gate._sibling_spec(os.path.join(plan_a, "SOLUTION.md"))
    if spec is None:
        print(f"no problem_spec.json above {plan_a}", file=sys.stderr)
        return 1
    thresholds = spec.get("evaluation_thresholds") or {}
    cfg = Config()
    grid_N = int(cfg.pick("grid_N", None, thresholds.get("grid_N"), 64))
    tol = float(cfg.pick("rel_l2_err_max", None, thresholds.get("rel_l2_err_max"), 0.01))
    metric = cfg.pick("metric", None, thresholds.get("metric"), "l2")
    levels = int(cfg.pick("refinement_levels", None, thresholds.get("refinement_levels"), 3))
    axis_names = list(thresholds.get("axes") or spec.get("spatial_variables") or ["x", "y", "z"])
    primary = thresholds.get("primary_field")
    bounds = ((spec.get("domain") or {}).get("bounds") or {})

    def solve(plan, n):
        return sandbox.run(plan, "pde", {"N": int(n)})

    graded = thresholds.get("graded_N")
    if N is None and isinstance(graded, (int, float)) and not isinstance(graded, bool) \
            and graded > 0:
        # F4: the grid the statement grades at is the grid the answer is judged on.
        N = int(graded)
    if N is None:
        # Otherwise the ladder's top level, detected the way the driver detects it.
        probe = solve(plan_a, grid_N)
        if probe["status"] != "ok":
            print(f"{os.path.basename(plan_a)} crashed at N={grid_N}: {probe.get('error')}",
                  file=sys.stderr)
            return 1
        axes = ladder.axes_of(probe["result"], axis_names)
        exclusive = (ladder.detect_periodic(axes[0], bounds[axis_names[0]])
                     if axis_names[0] in bounds else False)
        N = ladder.build_ladder(grid_N, exclusive, levels=levels)[-1]
    runs = {}
    for plan in (plan_a, plan_b):
        got = solve(plan, N)
        if got["status"] != "ok":
            print(f"{os.path.basename(plan)} crashed at N={N}: {got.get('reason')}: "
                  f"{got.get('error')}", file=sys.stderr)
            return 1
        runs[plan] = got["result"]
    ua = ladder.primary_of(runs[plan_a], primary)
    ub = ladder.primary_of(runs[plan_b], primary)
    axes_a = ladder.axes_of(runs[plan_a], axis_names)
    axes_b = ladder.axes_of(runs[plan_b], axis_names)
    interpolated = False
    if ua.shape == ub.shape and all(len(x) == len(y) and abs(float(x[-1]) - float(y[-1])) < 1e-9
                                   for x, y in zip(axes_a, axes_b, strict=False)):
        a_on, b_on = ua, ub
    else:
        # Restrict the finer onto the coarser; interpolate only if they do not nest,
        # and say so -- an interpolated agreement is not a circularity break.
        coarse_first = ua.size <= ub.size
        u_c, ax_c = (ua, axes_a) if coarse_first else (ub, axes_b)
        u_f, ax_f = (ub, axes_b) if coarse_first else (ua, axes_a)
        f_on_c = ladder.restrict(u_f, ax_f, ax_c)
        if f_on_c is None:
            f_on_c, interpolated = ladder.interpolate_to(u_f, ax_f, ax_c), True
        a_on, b_on = (u_c, f_on_c) if coarse_first else (f_on_c, u_c)
    import numpy as np
    diff = np.asarray(a_on, dtype=float) - np.asarray(b_on, dtype=float)
    if metric == "l1":
        rel = float(np.mean(np.abs(diff)) / (0.5 * (np.mean(np.abs(a_on))
                                                    + np.mean(np.abs(b_on))) + 1e-14))
    else:
        rel = float(np.sqrt(np.mean(diff ** 2))
                    / (0.5 * (ladder.rms(a_on) + ladder.rms(b_on)) + 1e-14))
    meta_a, meta_b = plan_meta.read(plan_a), plan_meta.read(plan_b)
    fam_a, fam_b = meta_a.get("scheme_family"), meta_b.get("scheme_family")
    entry = {"a": os.path.basename(plan_a), "b": os.path.basename(plan_b), "N": int(N),
             "t_final": runs[plan_a].get("t_final"), "rel_diff": rel, "tol": tol,
             "metric": metric, "agree": bool(np.isfinite(rel) and rel < tol),
             "family_a": fam_a, "family_b": fam_b,
             "different_family": bool(fam_a and fam_b and fam_a != fam_b),
             "sha_a": solver_sha256(plan_a), "sha_b": solver_sha256(plan_b),
             "interpolated": interpolated}
    if not no_record:
        agreement.record(agreement.workspace_of(plan_a), entry)
        entry["recorded_in"] = agreement.path_for(plan_a)
    if as_json:
        print(json.dumps(entry, indent=2, default=_jsonable))
    else:
        verdict = "AGREE" if entry["agree"] else "DISAGREE"
        print(f"{verdict}: {entry['a']} vs {entry['b']} differ by {rel:.3e} at N={N} "
              f"(tol {tol:g}, t_final {entry['t_final']}); families {fam_a} / {fam_b}"
              f"{' -- different, counts as a circularity break' if entry['different_family'] and entry['agree'] else ''}"
              f"{' -- interpolated, not a break' if interpolated else ''}")
        if not no_record:
            print(f"recorded in {entry['recorded_in']}")
    return PASS if entry["agree"] else BLOCK


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except ImportError:
        pass
    return str(value)


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    command, args = argv[0], argv[1:]
    as_json = "--json" in args
    flags = [a for a in args if a.startswith("--")]
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
    if command == "evaluate":
        if not args:
            print("evaluate takes a plan directory", file=sys.stderr)
            return 1
        return _evaluate(args[0], as_json)
    if command == "compare":
        if len(args) != 2:
            print("compare takes two plan directories", file=sys.stderr)
            return 1
        N = next((int(f.split("=", 1)[1]) for f in flags if f.startswith("--N=")), None)
        return _compare(args[0], args[1], N=N, no_record="--no-record" in flags,
                        as_json=as_json)

    print(f"unknown command {command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
