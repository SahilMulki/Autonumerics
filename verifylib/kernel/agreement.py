"""Cross-plan agreement: the second circularity break Path B can earn without D1.

Findings F2. On Path A the winner early-exit is a scheduling economy -- the 10
was measured against a closed form. On Path B the 10 rests on the kernel's own
estimate, and the strongest cheap evidence about the *answer* is whether an
independently written plan of a different method family agrees with it. That is
the standard the harness uses to certify a Route-R reference ("two independent
method families agreeing far below the problem's tolerance"), and the 2026-09-19
KS run had that evidence in hand and threw it away: the FD4 plan certified at 10
differed from the spectral plan at 9 by 7.6% at the grid the problem names.

``cli.py compare <planA> <planB>`` measures the difference and records it here,
in ``<workspace>/agreements.json``. The driver reads the record back on the next
evaluation **against the current hashes of both solver files**, so an entry that
predates an edit to either plan is stale and ignored, and nothing an agent writes
can raise a score: the kernel measured the agreement, the kernel prices it.

Two consequences are priced separately in ``score.py``:

* agreement between plans of *different* ``scheme_family`` within ``tol`` is a
  circularity break (``agreement_is_break``), which a resolved spectral solver --
  whose D1 is structurally ``unresolved`` -- can actually earn;
* disagreement above ``tol`` caps **both** plans at 8. The difference does not
  say which plan is wrong, and capping the higher score would be a guess dressed
  as a rule.

What it does not break: both plans read the same formulator's operator, so a
mis-transcribed equation passes unanimously. The harness is the only
spec-independent check, and REPORT.md must not imply otherwise.
"""

from __future__ import annotations

import json
import os
import time

AGREEMENTS_FILE = "agreements.json"


def workspace_of(plan_dir):
    """``workspace/<slug>`` for ``workspace/<slug>/plans/<plan>``."""
    return os.path.dirname(os.path.dirname(os.path.abspath(plan_dir)))


def path_for(plan_dir):
    return os.path.join(workspace_of(plan_dir), AGREEMENTS_FILE)


def load(workspace):
    path = os.path.join(workspace, AGREEMENTS_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path) as fh:
            data = json.load(fh)
    except (OSError, json.JSONDecodeError):
        return []
    return [e for e in (data if isinstance(data, list) else []) if isinstance(e, dict)]


def record(workspace, entry):
    """Add ``entry``, replacing any earlier one for the same pair. Returns the
    list as written."""
    key = frozenset((entry["a"], entry["b"]))
    kept = [e for e in load(workspace) if frozenset((e.get("a"), e.get("b"))) != key]
    kept.append({**entry, "recorded_at": time.time()})
    with open(os.path.join(workspace, AGREEMENTS_FILE), "w") as fh:
        json.dump(kept, fh, indent=2)
    return kept


def for_plan(plan_dir, *, sha_of):
    """What the record says about ``plan_dir`` **today**.

    ``sha_of(plan_dir) -> hash | None`` is injected so this module stays free of
    the driver. Returns ``{"agreement": True | None, "disagreement": entry | None,
    "entries": [...], "stale": [...]}``. An entry is live only when both solver
    files still hash to what was compared.
    """
    workspace = workspace_of(plan_dir)
    me = os.path.basename(os.path.abspath(plan_dir))
    my_sha = sha_of(plan_dir)
    live, stale, agreement, disagreement = [], [], None, None
    for entry in load(workspace):
        if me not in (entry.get("a"), entry.get("b")):
            continue
        other = entry["b"] if entry.get("a") == me else entry["a"]
        my_key, other_key = ("sha_a", "sha_b") if entry.get("a") == me else ("sha_b", "sha_a")
        other_sha = sha_of(os.path.join(workspace, "plans", other))
        if my_sha is None or entry.get(my_key) != my_sha or entry.get(other_key) != other_sha:
            stale.append({**entry, "why": "a solver.py changed since the comparison"})
            continue
        view = {"other": other, "other_family": entry.get("family_b" if entry.get("a") == me
                                                          else "family_a"),
                "rel_diff": entry.get("rel_diff"), "tol": entry.get("tol"),
                "N": entry.get("N"), "different_family": entry.get("different_family"),
                "agree": entry.get("agree"), "interpolated": entry.get("interpolated")}
        live.append(view)
        if view["agree"] is True and view["different_family"] and not view["interpolated"]:
            agreement = True
        if view["agree"] is False and (disagreement is None
                                       or (view["rel_diff"] or 0) > (disagreement["rel_diff"] or 0)):
            disagreement = view
    return {"agreement": agreement, "disagreement": disagreement, "entries": live,
            "stale": stale}
