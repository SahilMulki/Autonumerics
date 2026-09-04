"""The one shape every guard reports in.

A finding is deliberately flat and stringly-typed: it is rendered into hook
stderr, into ``benchmark/verify.py`` output, and into JSON, and none of those
want a rich object. ``severity`` is the whole gate/warn distinction of §5b --
``error`` blocks, ``warning`` is surfaced and recorded but never caps a score.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

ERROR = "error"
WARNING = "warning"


@dataclass(frozen=True)
class Finding:
    severity: str  # ERROR | WARNING
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


def errors(findings) -> list[Finding]:
    return [f for f in findings if f.severity == ERROR]


def render(findings) -> str:
    return "\n".join(f.render() for f in findings)
