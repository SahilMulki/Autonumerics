"""The emitted metrics dict: its shape, its skip vocabulary, and its config record.

Two things here are load-bearing beyond "somewhere to put numbers".

**The skip vocabulary is closed.** A check that did not run must be
distinguishable from one that ran and passed, at every consumer, without anyone
having to know which check it was. ``verification_manual.md`` §25 states the rule
negatively -- "a missing reference reported as a solver crash" is the first thing
that must never happen -- and the positive form is a fixed set of strings that are
never truthy and never render as a pass. :data:`SKIPS` is that set.

**Every configured value carries its source.** The evaluator agent selects the
ladder's base ``N``, which Stage-2 checks run this cycle, and which tolerances
apply (manual §24). Each of those moves the outcome and none of it used to be
recorded, so a misconfiguration was only ever visible as an odd result. Recording
``{"value": ..., "source": "spec" | "agent" | "kernel_default"}`` is cheap while
this module is being written and a rewrite once consumers exist -- plan §12a (1).
"""

from __future__ import annotations

#: A check produced no verdict. Never truthy, never a pass, never a failure.
#:
#: ``not_run``      -- skipped for cost (manual §24); by construction only when it
#:                     could not have changed the answer.
#: ``unavailable``  -- the inputs do not exist (no operator declaration, no
#:                     degenerate limit, no ``snapshots`` from the solver).
#: ``unresolved``   -- the *check* is the limit, not the solver: two kernel
#:                     stencils disagree, so the residual measures the kernel
#:                     (plan §5b).
#: ``not_reported`` -- declared by the spec, but the solver did not return what it
#:                     needs (a missing ``invariant_trace``). Caps at 9; manual §8.
#: ``waived_chaotic`` -- waived because trajectories separate past the Lyapunov
#:                     time, so the check carries no information (plan §8a).
#: ``faulty_probe`` -- an MMS probe whose own source term does not match its exact
#:                     solution. The formulator's defect, so it is skipped rather
#:                     than failed (manual §5).
SKIPS = frozenset({
    "not_run", "unavailable", "unresolved", "not_reported", "waived_chaotic",
    "faulty_probe",
})

#: Sources a configured value can have, in descending authority.
#:
#: ``plan`` and ``inferred`` are separate from ``spec`` because they come from a
#: different artifact and carry different weight: ``spec`` is ``problem_spec.json``,
#: ``plan`` is a field the plan-creator declared in ``SOLUTION.md``'s frontmatter,
#: and ``inferred`` is the kernel reading that plan's *free-text* ``scheme:`` and
#: guessing. The last of those selects which asymptotic guard applies, so a reader
#: has to be able to tell it apart from something a person stated.
CONFIG_SOURCES = ("spec", "plan", "agent", "inferred", "kernel_default")


def is_skip(value) -> bool:
    """True when ``value`` is a skip marker rather than a verdict."""
    return isinstance(value, str) and value in SKIPS


def passed(value) -> bool:
    """A check counts as passing only when it actually ran and said so.

    ``passed("unavailable")`` is False and so is ``passed(None)``. This is the one
    function that keeps a skip from reading as evidence, so nothing else in the
    kernel may test a check's outcome with a bare truth test.
    """
    return value is True


def failed(value) -> bool:
    """A check counts as failing only when it ran and said so."""
    return value is False


class Config:
    """The evaluator's selections, each with the source that supplied it."""

    def __init__(self):
        self._values: dict[str, dict] = {}

    def set(self, name, value, source):
        if source not in CONFIG_SOURCES:
            raise ValueError(f"config source {source!r} not one of {CONFIG_SOURCES}")
        self._values[name] = {"value": value, "source": source}
        return value

    def pick(self, name, agent, spec_value, default):
        """Resolve one value by precedence and record where it came from.

        Spec beats agent beats kernel default. The spec is the problem's own
        statement of what it needs; the agent is choosing within that; the default
        is what nobody chose. An agent that could override the spec would be able
        to relax a threshold the formulator set, which is the shape Layer 2 of the
        guardrails plan exists to stop.
        """
        if spec_value is not None:
            return self.set(name, spec_value, "spec")
        if agent is not None and name in agent:
            return self.set(name, agent[name], "agent")
        return self.set(name, default, "kernel_default")

    def get(self, name, default=None):
        entry = self._values.get(name)
        return default if entry is None else entry["value"]

    def as_dict(self):
        return dict(self._values)


def new_metrics(kind, path, config):
    """A metrics dict with every mandatory key present and nothing decided yet."""
    from . import SCHEMA_VERSION
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,                 # "pde" | "sde"
        "path": path,                 # "A" (a closed form is claimed) | "B"
        "score": None,                # filled by score.py, never by an agent
        "provenance": None,
        "agent_cap": None,            # {"score": int, "reason": str}, downward only
        "config": config.as_dict() if isinstance(config, Config) else dict(config),
        "checks": {},                 # name -> True | False | <skip>
        "notes": [],                  # human-readable, never parsed
    }


def record(metrics, name, outcome, **detail):
    """Record one check's verdict plus its diagnostics.

    ``outcome`` is ``True``, ``False`` or a member of :data:`SKIPS`. Anything else
    is a bug in the caller and is rejected here rather than silently becoming a
    truthy pass three layers downstream.
    """
    if not (outcome is True or outcome is False or is_skip(outcome)):
        raise ValueError(
            f"check {name!r} reported {outcome!r}; a check reports True, False, or "
            f"one of the skip markers {sorted(SKIPS)}")
    metrics["checks"][name] = outcome
    if detail:
        metrics.setdefault("detail", {})[name] = detail
    return outcome


def note(metrics, text):
    metrics["notes"].append(text)


#: The keys ``verification_manual.md`` §2 defines for the ``<metrics>`` block in
#: ``SOLUTION.md``, in the order they are rendered. Everything else in the metrics
#: dict is available as JSON; this is the flat view the conductor and
#: ``review.py`` parse.
BLOCK_KEYS = (
    "score", "provenance", "estimated_rel_error", "error_is_estimate",
    # error_horizon appears only on a chaotic problem, where Tier C is measured at
    # `chaotic_T_ref` rather than at t_final. It has to reach the *block*, not just
    # the JSON: the block is what the conductor ranks on and what REPORT.md quotes,
    # and an error measured at t = 5 must not be read as the error at t = 50.
    "error_horizon",
    "observed_order", "order_floor", "invariants_ok", "constraints_ok",
    "d1_outcome", "d1_slope", "operator_validated", "reference_outcome",
    "order_check", "asymptotic_source", "mc_se_rel", "resolved",
    "resolution_evidence", "agent_cap", "wall_time_s",
)


def _fmt(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            return str(value)
        return f"{value:.3e}" if 0 < abs(value) < 1e-2 or abs(value) >= 1e4 else f"{value:.4f}"
    if isinstance(value, dict):
        import json
        return json.dumps(value, sort_keys=True)
    return str(value)


def render_block(metrics) -> str:
    """The ``<metrics>`` block, rendered from the kernel's own dict.

    The evaluator agent copies this verbatim. It does not retype it: ``review.py``
    requires the review's ``Score: N/10`` headline to equal ``metrics.score``
    exactly, and a headline can only be exact against a number that was not
    transcribed twice.
    """
    lines = ["<metrics>"]
    flat = flatten(metrics)
    for key in BLOCK_KEYS:
        if key not in flat:
            continue
        text = _fmt(flat[key])
        if text is not None:
            lines.append(f"{key}: {text}")
    lines.append("</metrics>")
    return "\n".join(lines)


def finalize(metrics, config):
    """Refresh the config record just before emitting.

    :func:`new_metrics` snapshots the config at creation, but several values are
    only decided *during* the run -- whether the two-level probe path was taken,
    which Stage-2 checks ran, how many levels the ladder ended up with. Those are
    exactly the values §12a (1) exists to record, and a snapshot taken before they
    are set records ``None`` for every one of them.
    """
    metrics["config"] = config.as_dict() if isinstance(config, Config) else dict(config)
    return metrics


def flatten(metrics) -> dict:
    """The block-level view: top-level scalars plus the promoted check outcomes."""
    flat = {k: v for k, v in metrics.items()
            if not isinstance(v, (dict, list)) or k == "agent_cap"}
    flat.update({k: v for k, v in metrics.get("checks", {}).items()})
    flat.update(metrics.get("summary") or {})
    return flat
