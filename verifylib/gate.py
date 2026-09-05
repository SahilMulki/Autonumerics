"""Composition: the guards, assembled into the checks the hooks and gates call.

Every guard runs in **two** places (§3). As a hook, for fast in-loop correction;
and as a gate that does not depend on an agent cooperating. On the benchmark that
gate is ``benchmark/verify.py``; on a user's own problem it is the ``Stop`` hook.
The hook-on-write is an optimisation, not the enforcement.

That split exists because of a measured failure, not a hypothetical one. When a
hook contradicts an instruction, the agent stops, leaves the invalid file on disk,
and escalates to the user. Interactively that is correct. In a headless
``claude -p`` run **nobody answers**: the subagent returns, the file is invalid,
and the conductor never learns. So nothing here may be reachable only from a hook.
"""

from __future__ import annotations

import contextlib
import json
import os

from . import leakage, review, schema
from .findings import Finding, error, errors, warning

#: Files each check owns, matched on basename.
SPEC_FILE = "problem_spec.json"
SOLVER_FILES = ("solver.py", "evaluate.py")
REVIEW_FILE = "SOLUTION.md"

#: Written beside an offending file when the Stop gate gives up (§3d). A run that
#: cannot be fixed terminates with a recorded reason instead of burning the turn
#: budget: measured, an unbounded Stop hook fired 8 times and died at max-turns.
UNSATISFIED_MARKER = "GUARDRAIL_UNSATISFIED"
MAX_STOP_ATTEMPTS = 3


def _load(path):
    with open(path) as fh:
        return fh.read()


# --- per-artifact checks -----------------------------------------------------

def check_spec_text(text, *, path="", with_reference=True) -> list[Finding]:
    """Schema plus, optionally, the reference check.

    ``with_reference=False`` is the hook's mode: the schema half is instant, while
    the reference check evaluates a formula on a 1025-point grid and is better run
    at the gate than on every keystroke.
    """
    try:
        spec = json.loads(text)
    except json.JSONDecodeError as exc:
        return [error("schema", path, f"not valid JSON: {exc}")]
    if not isinstance(spec, dict):
        return [error("schema", path, "problem_spec.json must be a JSON object")]

    found = schema.check_spec(spec)
    found.extend(leakage.citation_findings(text, path=path))
    if with_reference and not errors(found):
        # A spec that fails schema cannot be evaluated meaningfully; reporting a
        # residual on it would bury the real finding under a derived one.
        #
        # Imported here, not at module scope, so that an interpreter without numpy
        # still runs every AST-only guard above instead of failing to import at all.
        try:
            from . import reference
            found.extend(reference.reference_findings(spec))
        except ImportError as exc:  # ModuleNotFoundError included
            found.append(warning("reference", "analytic_solution",
                                 f"reference check skipped: {exc}. The schema and leakage "
                                 f"guards ran; the numeric check did not"))
        except Exception as exc:  # noqa: BLE001 -- a guard must never crash the run
            found.append(warning("reference", "analytic_solution",
                                 f"reference check could not run: {type(exc).__name__}: {exc}"))
    return found


def check_spec_file(path, *, with_reference=True) -> list[Finding]:
    return check_spec_text(_load(path), path=path, with_reference=with_reference)


def check_solver_text(text, *, path="") -> list[Finding]:
    return leakage.leakage_findings(text, path=path)


def check_solver_file(path) -> list[Finding]:
    return check_solver_text(_load(path), path=path)


def check_review_file(path, spec=None) -> list[Finding]:
    text = _load(path)
    return [*review.check_review(text, spec), *leakage.citation_findings(text, path=path)]


def check_path(path, *, with_reference=True) -> list[Finding]:
    """Dispatch on filename. Anything else is not ours and yields nothing."""
    name = os.path.basename(path)
    if not os.path.exists(path):
        return []
    if name == SPEC_FILE:
        return check_spec_file(path, with_reference=with_reference)
    if name in SOLVER_FILES:
        return check_solver_file(path)
    if name == REVIEW_FILE:
        return check_review_file(path, _sibling_spec(path))
    return []


def _sibling_spec(path):
    """The ``problem_spec.json`` governing a plan directory, if it is there.

    Plans live at ``<workspace>/<slug>/plans/<plan>/SOLUTION.md`` and the spec at
    ``<workspace>/<slug>/problem_spec.json``.
    """
    directory = os.path.dirname(os.path.abspath(path))
    for _ in range(4):
        candidate = os.path.join(directory, SPEC_FILE)
        if os.path.exists(candidate):
            try:
                return json.loads(_load(candidate))
            except json.JSONDecodeError:
                return None
        parent = os.path.dirname(directory)
        if parent == directory:
            break
        directory = parent
    return None


# --- the whole-workspace gate ------------------------------------------------

def check_workspace(root, *, with_reference=True) -> list[Finding]:
    """Every guard over one problem directory.

    This is what the ``Stop`` hook and ``benchmark/verify.py`` both run, and it is
    what makes the battery work on a problem outside ``benchmark/`` -- a user's own
    ``problem.md`` in a scratch directory gets the same treatment.
    """
    found: list[Finding] = []
    owned = (SPEC_FILE, REVIEW_FILE, *SOLVER_FILES)
    for directory, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for filename in filenames:
            if filename in owned:
                found.extend(check_path(os.path.join(directory, filename),
                                        with_reference=with_reference))
    return found


def active_problem(root):
    """The problem directory this session is actually working on.

    A whole-``workspace/`` sweep is the wrong scope for a session gate. The staged
    benchmark workspace holds 29 problem directories from previous runs, three of
    which carry pre-existing findings; gating on all of them would block a session
    working on a fourth, for something it did not do and cannot fix. Both the
    pipeline (``/conductor workspace/{slug}/problem.md``) and a user running
    Autonumerics on their own problem work **one problem at a time**, so the
    session's problem is the one whose files were most recently touched.

    Returns ``root`` itself when it holds no problem subdirectories -- a scratch
    directory with a bare ``problem_spec.json`` is a perfectly good workspace.
    """
    candidates = []
    for name in os.listdir(root) if os.path.isdir(root) else []:
        directory = os.path.join(root, name)
        if not os.path.isdir(directory) or name.startswith("."):
            continue
        newest = 0.0
        for sub, dirnames, filenames in os.walk(directory):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                with contextlib.suppress(OSError):
                    newest = max(newest, os.path.getmtime(os.path.join(sub, filename)))
        if newest:
            candidates.append((newest, directory))
    return max(candidates)[1] if candidates else root


# --- the Stop-gate attempt counter (§3d) -------------------------------------

def _counter_path(root):
    return os.path.join(root, ".verifylib_stop_attempts")


def record_stop_attempt(root, fingerprint) -> int:
    """Count consecutive Stop firings for the *same* finding set.

    A different finding set resets the count: the agent fixed something and broke
    something else, which is progress, not a loop.
    """
    path = _counter_path(root)
    previous = {}
    if os.path.exists(path):
        try:
            previous = json.loads(_load(path))
        except json.JSONDecodeError:
            previous = {}
    count = previous.get("count", 0) + 1 if previous.get("fingerprint") == fingerprint else 1
    with open(path, "w") as fh:
        json.dump({"fingerprint": fingerprint, "count": count}, fh)
    return count


def clear_stop_attempts(root):
    path = _counter_path(root)
    if os.path.exists(path):
        os.remove(path)


def fingerprint(findings) -> str:
    return "|".join(sorted(f"{f.check}:{f.path}" for f in findings))


def write_unsatisfied_marker(root, findings) -> str:
    """Let the session end, with the reason on disk **beside the offending file**.

    The conductor reads ``workspace/{slug}/GUARDRAIL_UNSATISFIED`` and halts at
    ``phase: blocked``, so an unfixable run terminates with a recorded cause rather
    than being silently accepted. Writing it beside the file rather than at the
    workspace root is what keeps one problem's unfixable finding from blocking the
    report of an unrelated one.
    """
    located = next((f.path for f in findings if f.path and os.path.exists(f.path)), None)
    if located:
        root = os.path.dirname(os.path.abspath(located))
    path = os.path.join(root, UNSATISFIED_MARKER)
    with open(path, "w") as fh:
        fh.write("The session ended with guardrail findings still outstanding after "
                 f"{MAX_STOP_ATTEMPTS} attempts to fix them.\n\n")
        for finding in findings:
            fh.write(finding.render() + "\n")
    return path
