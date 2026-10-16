"""Two scans for answer-key leakage. They catch different things; run both.

Measured against the archived incidents in ``tests/fixtures/``:

  * ``pde_heston_2d/problem_spec.json`` -- caught by the CITATION scan
    (``benchmark/``, ``problems.py``, ``_heston_call``). The AST scan is blind to
    it: the transcribed price function is pure numpy with no import, no ``open``,
    no prohibited subscript.
  * ``sde_quintic.../evaluate.py`` -- caught by the CITATION scan. The leak is a
    module docstring, and executable-access scanning excludes comments by design.

So the spec is the chokepoint: the Heston leak entered at ``problem_spec.json``
and propagated by hand-transcription into ``evaluate.py``, where no text scan can
see it. Block it at the spec and the copy never happens.

The AST scan is kept for a hole the citation scan does not cover -- a file opened
at runtime -- but it must stay narrow. The rule originally proposed (flag bare
``.get()`` / ``open()`` / ``getattr``) fires on **124 of 137** real pipeline files,
every one a false positive.
"""

from __future__ import annotations

import ast
import re

from .findings import Finding, error, warning

CITATION_PATTERNS = (
    r"benchmark/",
    r"\bproblems\.py\b",
    r"\bverify\.py\b",
    r"\bmanifest\.json\b",
    r"\brunner\.py\b",
    r"\bvalidate_ground_truth\b",
    r"verify_(?:sde|pde)_\w+",
    r"_heston_call\b",
)

#: The one file that legitimately names the protected paths: project_manual.md's
#: "Never read benchmark/" section. A rule has to name what it protects.
CITATION_ALLOWLIST = ("references/project_manual.md",)

PROHIBITED_MODULES = frozenset({"benchmark", "problems", "verify", "manifest", "runner"})
_READERS = frozenset({"open", "Path", "read_text", "load", "loadtxt", "genfromtxt"})


def citation_scan(text: str, *, path: str = "") -> list[str]:
    """Find references to the answer key in prose. Comments count here."""
    if any(path.endswith(a) for a in CITATION_ALLOWLIST):
        return []
    hits = {m.group(0) for p in CITATION_PATTERNS for m in re.finditer(p, text)}
    return sorted(hits)


def oracle_access_scan(source: str) -> list[str]:
    """Find *executable* access to the answer key. Comments are not access.

    Narrowed deliberately: a prohibited import, or a reader call on a string
    literal naming ``benchmark``. Measured 0 false positives over 137 pipeline
    files, with both genuine patterns still caught.
    """
    hits: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in getattr(node, "names", [])]
            if getattr(node, "module", None):
                names.append(node.module)
            for nm in names:
                if nm and nm.split(".")[0] in PROHIBITED_MODULES:
                    hits.append(f"import {nm}")
        elif isinstance(node, ast.Call):
            fn = node.func
            name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
            if name in _READERS:
                for arg in [*node.args, *(k.value for k in node.keywords)]:
                    if (isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                            and "benchmark" in arg.value):
                        hits.append(f"{name}({arg.value!r})")
    return sorted(set(hits))


# --- as findings -------------------------------------------------------------

def citation_findings(text: str, *, path: str = "") -> list[Finding]:
    """Answer-key knowledge arriving as prose. Comments count here: the quintic
    leak was a module docstring."""
    hits = citation_scan(text, path=path)
    if not hits:
        return []
    return [error("leakage", path,
                  f"names the answer key in prose: {hits}. Nothing in the pipeline may "
                  f"read or cite anything under benchmark/ -- see references/project_manual.md")]


def leakage_findings(source: str, *, path: str = "") -> list[Finding]:
    """Both scans over one Python artifact. They catch different things, and the
    Heston incident proves it: its evaluate.py carried the transcribed price
    function with no citation at all, and its spec carried the citation with no
    executable access."""
    found = citation_findings(source, path=path)
    try:
        hits = oracle_access_scan(source)
    except SyntaxError as exc:
        return [*found, error("leakage", path, f"could not be parsed to scan: {exc}")]
    if hits:
        found.append(error("leakage", path,
                           f"executable access to the answer key: {hits}"))
    return found


# --- the state-clip scan (plan §3 5b) ----------------------------------------

#: Calls that truncate a value. Bare and dotted spellings both, because
#: ``from numpy import maximum`` is legal and the scan must not be defeated by an
#: import style.
#:
#: ``where`` is deliberately absent: ``np.where(cond, a, b)`` is a *select*, not a
#: truncation, and including it flagged ``pde_mhd_2d``'s ``full = where(..., full,
#: ...)`` -- a field assembled from two branches, not a state clipped to its own
#: support. The scan is worth exactly as much as its precision.
_CLIPPERS = frozenset({"maximum", "minimum", "clip", "abs", "absolute"})


def state_clip_scan(source: str) -> list[str]:
    """Find a state variable clipped **to itself**, as a statement.

    ``verification_manual.md`` §21 and ``agents/solver-sde.md`` both promise this
    check and neither had one, which is worse than having no deterrent: the solver
    agents are told it exists.

    Narrow by construction, and the narrowing is the whole check.
    ``agents/solver-sde.md:80`` *instructs* the solver to write
    ``np.sqrt(np.maximum(X, 0.0))`` for CIR, and ``sde_cir_feller_violated`` is live
    with two plans. Full-truncation Euler clips **inside** the coefficient and keeps
    the scheme honest; clipping the *state* after the update is what hides the
    defect. So:

    * only an assignment whose **target name also appears in the clip's arguments**
      -- ``X = np.maximum(X, 0)``, ``X[:] = np.clip(X, 0, None)``, ``X *=`` a clip of
      itself -- or a bare ``np.clip(..., out=X)`` statement;
    * never a clip appearing as a sub-expression of another call, which is where
      every endorsed use sits.

    That is also what ``sde_cir_feller_violated``'s own spec note asks for, in as
    many words: *"a bare np.maximum(X, 0) applied only to the RETURNED array would
    hide the defect; the guard must be inside the sqrt of the step itself"*.

    Measured over every solver in ``workspace/``: 0 hits, including on both CIR
    plans and all three tamed Ginzburg-Landau plans.
    """
    def clipper(call):
        fn = call.func
        name = fn.id if isinstance(fn, ast.Name) else getattr(fn, "attr", None)
        return name if name in _CLIPPERS else None

    def names_in(call):
        return {n.id for n in ast.walk(call) if isinstance(n, ast.Name)}

    def target_name(target):
        while isinstance(target, (ast.Subscript, ast.Attribute)):
            target = target.value
        return target.id if isinstance(target, ast.Name) else None

    hits = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, (ast.Assign, ast.AugAssign)):
            value = node.value
            if not isinstance(value, ast.Call):
                continue
            name = clipper(value)
            if name is None:
                continue
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                state = target_name(target)
                if state and state in names_in(value):
                    hits.append(f"{state} = {name}(... {state} ...) at line {node.lineno}")
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            name = clipper(node.value)
            out = [k for k in node.value.keywords if k.arg == "out"]
            if name and out:
                state = target_name(out[0].value)
                if state and state in names_in(node.value):
                    hits.append(f"{name}(..., out={state}) at line {node.value.lineno}")
    return sorted(set(hits))


def state_clip_findings(source: str, *, path: str = "") -> list[Finding]:
    """A **warning**, not an error. Where the clip sits is the whole question and a
    scan cannot always answer it -- a reflecting scheme legitimately maps the state
    back into its support -- so this surfaces the line and the reviewer decides. An
    error here would hard-fail a correct reflecting solver, which is the one failure
    mode this repo treats as invisible."""
    try:
        hits = state_clip_scan(source)
    except SyntaxError:
        return []
    if not hits:
        return []
    return [warning("state_clip", path,
                    f"the state is clipped to itself as a statement: {hits}. "
                    f"Full truncation clips inside the coefficient (np.sqrt(np.maximum("
                    f"X, 0))) and keeps the scheme honest; clipping the state after the "
                    f"update, or on the returned array, hides a negative excursion "
                    f"rather than preventing it (verification_manual.md section 21). If "
                    f"this is a reflecting scheme, say so in SOLUTION.md")]
