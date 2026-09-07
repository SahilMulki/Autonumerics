"""The numerical verification kernel: every measurement, in fixed code.

``docs/plan-no-closed-form.md`` §2. The evaluator agent used to write a fresh
``evaluate.py`` on every cycle -- nested ladders, Richardson, GCI guards,
asymptotic tests, MMS probe validation -- and nothing re-checked its arithmetic.
Measured consequence, on ``workspace/pde_kuramoto_sivashinsky``: four
independently regenerated evaluators reported observed orders of 10.467, 10.925
and 0.003 for the same problem, and all four scored 10.

So the split is: this package does every numerical check **and emits the score**;
the agent selects and configures the checks, then diagnoses. The scoring rubrics
(``verification_manual.md`` §12 and §23, and the plan's §4c certification table)
are decision tables with no judgement in them, so handing their booleans to an LLM
adds no information and adds a drift surface. The agent keeps one authority and it
is asymmetric: it may cap the score **downward** with a recorded reason, never
raise it (:mod:`verifylib.kernel.score`).

Submodules are imported lazily, inside :func:`run`. ``verifylib.schema`` imports
the invariant registry out of :mod:`verifylib.kernel.invariants` to validate a
gated invariant name, and ``schema`` must keep working on an interpreter without
numpy (guardrails E32) -- so importing this package must not pull numpy in.
"""

from __future__ import annotations

__all__ = ["run", "SCHEMA_VERSION"]

#: Bumped when the emitted metrics dict changes shape. Consumers (``review.py``'s
#: score binding, ``conductor.md``'s ranking) key off names, not position, so this
#: is for diagnosis rather than dispatch.
SCHEMA_VERSION = 1


def run(spec, plan_dir, config=None):
    """Evaluate one plan directory. Returns the metrics dict (plan §2, §9a).

    ``spec`` is the parsed ``problem_spec.json``; ``plan_dir`` holds ``solver.py``
    and ``SOLUTION.md``; ``config`` is the evaluator agent's selections, each of
    which is recorded in ``metrics["config"]`` **with its source** so a
    misconfiguration is visible directly rather than inferred from an odd result
    (plan §12a item 1).
    """
    from . import schema_bridge  # noqa: F401  -- keeps the lazy contract explicit
    if schema_bridge.is_sde(spec):
        from .sde.driver import evaluate_sde
        return evaluate_sde(spec, plan_dir, config or {})
    from .driver import evaluate_pde
    return evaluate_pde(spec, plan_dir, config or {})
