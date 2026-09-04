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

from .findings import Finding, error

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
