"""The one shape every guard reports in.

A finding is deliberately flat and stringly-typed: it is rendered into hook
stderr, into ``benchmark/verify.py`` output, and into JSON, and none of those
want a rich object. ``severity`` carries the gate/warn distinction of §5b --
``error`` blocks, ``warning`` is surfaced and recorded but never caps a score.

``spec`` is a third severity and a different kind of claim from either. An ``error``
and a ``warning`` are both findings *about the solver*, differing only in how
reliably they can be read off it. A ``spec`` finding is a real, confirmed
disagreement that the solver had no part in -- typically between what the
formulator's ledger declares and what the harness actually runs -- so it is reported
in full and never touches the solver's score. It exists because conflating the two
cost seven of nine SciML entries three points each: see
``audit.audit_sde_output`` and plan-sciml-benchmark.md SS10j.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ERROR = "error"
WARNING = "warning"
#: A defect in the spec or the harness, not in the solver. Reported, never capping.
SPEC = "spec"


@dataclass(frozen=True)
class Finding:
    severity: str  # ERROR | WARNING | SPEC
    check: str  # short slug, e.g. "expression", "ledger", "leakage"
    path: str  # dotted spec path, or a file-relative location
    message: str

    def render(self) -> str:
        where = f" [{self.path}]" if self.path else ""
        return f"{self.severity}: {self.check}{where}: {self.message}"

    def as_dict(self) -> dict:
        return asdict(self)


def error(check: str, path: str, message: str) -> Finding:
    return Finding(ERROR, check, path, message)


def warning(check: str, path: str, message: str) -> Finding:
    return Finding(WARNING, check, path, message)


def spec_defect(check: str, path: str, message: str) -> Finding:
    """A confirmed defect the solver is not answerable for. Never caps a score.

    Not a `warning`: a warning is a finding about the solver that may be a false
    positive, and is hedged for that reason. This is the opposite -- the finding is
    certain and the solver is simply the wrong party. `path` should name where the
    defect actually is (the ledger field, say), not where it was noticed.
    """
    return Finding(SPEC, check, path, message)


def errors(findings) -> list[Finding]:
    return [f for f in findings if f.severity == ERROR]


def render(findings) -> str:
    return "\n".join(f.render() for f in findings)
