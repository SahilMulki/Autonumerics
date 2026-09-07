"""Layer 5: every number in a review's accuracy claim must trace to a computed one.

Three things this deliberately does **not** do, each because an earlier draft did
and measurement said no:

1. **It does not bind the whole review.** "Feedback for solver" is where the
   evaluator does arithmetic *on purpose* -- ``evaluator-pde.md`` instructs it to
   write "dt=0.01, dx=0.1 gives dt/dx**2=1.0 > 0.5 stability limit", four numbers,
   none of them metrics. Binding that section rejects the review the prompt asked
   for. Only the **Numerical Accuracy** section is bound.
2. **It does not treat ``<metrics>`` as the only backing set.** Parameters,
   evaluation thresholds and the grid sizes actually run are all legitimate
   sources for a number in an accuracy claim.
3. **It does not hard-fail.** Warn, let the caller regenerate once, then pass with
   the discrepancy recorded.

And be honest about what it establishes: the same LLM writes ``<metrics>`` and
``<review>``, so binding one to the other catches **transcription drift** -- the
most common LLM-grader failure, and worth catching -- but not fabrication. That
property arrives only when the metrics are computed by a kernel rather than
written by the grader (§14 C6).
"""

from __future__ import annotations

import re

from .findings import Finding, error, warning

#: The one section whose claims must be traceable.
BOUND_SECTION = "Numerical Accuracy"

#: Within that section, the keyed lines that state a *measurement*. Everything else
#: there is diagnostic context -- per-grid error history, invariant drifts, the
#: tolerance each invariant was judged against -- computed by the evaluator but
#: recorded nowhere the binding can reach. Measured over 206 archived reviews:
#: binding every number in the section flags 28% of them, all on that context.
BOUND_KEYS = (
    "error", "estimated_rel_error", "rel_error", "fine_grid_error", "tolerance",
    "observed_order", "order", "order_floor",
    "variance_rel_err", "mean_rel_err", "mc_se_rel", "wall_time_s",
)

_SECTION_RE = re.compile(r"^#{2,4}\s*(.+?)\s*$", re.M)
#: ``<review score=10>`` and the ``Score: 10/10`` headline inside it.
_REVIEW_ATTR_RE = re.compile(r"<review[^>]*\bscore\s*=\s*[\"']?(\d+)")
_HEADLINE_RE = re.compile(r"^\s*(?:\*\*)?Score:\s*(\d+)\s*/\s*10", re.M)
_METRICS_RE = re.compile(r"<metrics>(.*?)</metrics>", re.S)
_REVIEW_RE = re.compile(r"<review[^>]*>(.*?)</review>", re.S)
#: Numbers, including scientific notation and percentages. Deliberately does not
#: match a bare integer inside a word (``L2``, ``x2``).
_NUMBER_RE = re.compile(r"(?<![\w.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?(?![\w])")

#: Floor on the slack when matching, for a value printed to full precision.
MATCH_RTOL = 1e-9


def parse_metrics(text: str) -> dict:
    """``<metrics>`` as ``{key: value}``.

    Accepts both shapes from day one (§14 C3): a bare scalar today, and
    plan-blind-evaluator's ``{"value": x, "oracle_derived": bool}``. Writing this
    tolerantly now is cheap; retrofitting it later is a rewrite.
    """
    match = _METRICS_RE.search(text or "")
    if not match:
        return {}
    out = {}
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, _, raw = line.partition(":")
        out[key.strip()] = _coerce(raw.strip())
    return out


def metrics_texts(text: str) -> dict:
    """The raw right-hand side of each metrics line, kept so a backing value can be
    matched at the precision *it* was printed to as well."""
    match = _METRICS_RE.search(text or "")
    if not match:
        return {}
    out = {}
    for line in match.group(1).splitlines():
        line = line.split("#", 1)[0].strip()
        if line and ":" in line:
            key, _, raw = line.partition(":")
            out[key.strip()] = raw.strip()
    return out


def _coerce(raw):
    text = raw.strip()
    if text.startswith("{"):
        # {"value": 3.1e-3, "oracle_derived": true} -- take the value, keep the flag
        inner = re.search(r'"?value"?\s*:\s*([-+]?[\d.eE+-]+)', text)
        if inner:
            return _coerce(inner.group(1))
    if text.lower() in ("true", "false"):
        return text.lower() == "true"
    try:
        return float(text)
    except ValueError:
        return text


def metric_values(metrics: dict) -> list[float]:
    return [v for v in metrics.values() if isinstance(v, float)]


def backing_values(metrics: dict, spec=None, grids=(), texts=None) -> list[tuple]:
    """Every number a Numerical Accuracy claim may legitimately cite, each paired
    with the slack its own printed precision allows.

    The metrics block rounds too: it records ``1.968e-04`` for a value the review
    prints as ``1.9675e-04``. Whichever side rounded harder sets the slack.
    """
    texts = texts or {}
    values = [(v, printed_tolerance(texts[k]) if k in texts else 0.0)
              for k, v in metrics.items() if isinstance(v, float)]
    spec = spec or {}
    for block in ("parameters", "evaluation_thresholds"):
        for value in (spec.get(block) or {}).values():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values.append((float(value), 0.0))
    for invariant in ((spec.get("verification") or {}).get("invariants") or []):
        if isinstance(invariant, dict) and isinstance(invariant.get("tol"), (int, float)):
            values.append((float(invariant["tol"]), 0.0))
    values.extend((float(g), 0.0) for g in grids or grid_ladder(spec))
    return values


def grid_ladder(spec, levels=3):
    """The resolutions a PDE run actually visits: N, 2N-1, 4N-3 (or N, 2N, 4N when
    periodic). A review citing the grid it ran on is citing a computed fact."""
    base = (spec.get("evaluation_thresholds") or {}).get("grid_N") if spec else None
    if not isinstance(base, (int, float)):
        return []
    n, out = int(base), []
    periodic = str(((spec.get("boundary_conditions") or {}).get("type") or "")) == "periodic"
    for _ in range(levels):
        out.append(float(n))
        n = 2 * n if periodic else 2 * n - 1
    return out


def bound_section(review_text: str, section=BOUND_SECTION) -> str:
    """The body of one ``###`` section of a review, up to the next heading."""
    headings = list(_SECTION_RE.finditer(review_text or ""))
    for i, match in enumerate(headings):
        if match.group(1).strip().lower() == section.lower():
            end = headings[i + 1].start() if i + 1 < len(headings) else len(review_text)
            return review_text[match.end():end]
    return ""


def printed_tolerance(text: str) -> float:
    """Half a unit in the last printed place.

    A review writes ``0.000026`` for a value carried as ``2.556e-05``; that is
    correct rounding to two significant figures, not drift, and a flat relative
    tolerance cannot tell the two apart at every magnitude. Matching at the
    precision the number was *printed* to accepts the rounding and still rejects a
    genuine disagreement, whatever its scale.
    """
    mantissa, _, exponent = text.lower().partition("e")
    decimals = len(mantissa.partition(".")[2])
    scale = float(exponent) if exponent else 0.0
    return 0.5 * (10.0 ** (-decimals + scale)) * (1 + 1e-9)


def _traces_to(value: float, text: str, backing) -> bool:
    """Small integers are exempt: a section saying "2 grids" or "(1 of 3)" is
    describing the run, not claiming a measurement."""
    if value == int(value) and abs(value) <= 16:
        return True
    slack = max(printed_tolerance(text), MATCH_RTOL * abs(value))
    return any(abs(value - b) <= max(slack, b_slack) for b, b_slack in backing)


def bound_claims(review_text, section=BOUND_SECTION):
    """``(key, first_number)`` for each keyed measurement line in the section.

    The *first* number on the line is the claim; anything after it is context
    ("1.97e-04 at N=128 (PASS) -- N=64: 3.15e-03" claims 1.97e-04 and then recalls
    the ladder it came from). Binding the claim catches the drift that matters --
    a review headline that disagrees with the metrics block above it -- without
    demanding that every diagnostic the evaluator prints be recorded twice.
    """
    out = []
    for line in bound_section(review_text, section).splitlines():
        stripped = line.strip().lstrip("-*").strip()
        key, sep, rest = stripped.partition(":")
        if not sep:
            continue
        key = key.split("(")[0].strip().lower().replace(" ", "_")
        if key not in BOUND_KEYS:
            continue
        # The claim is the *leading* token of the value, not the first number
        # anywhere on the line. "observed_order: inf -- both grid errors are below
        # the 1e-9 floor" claims `inf`, not 1e-9, and has no number to bind.
        head = rest.strip().split()[0].strip(",;()[]") if rest.strip() else ""
        if _NUMBER_RE.fullmatch(head):
            out.append((key, float(head), head))
    return out


def unbound_numbers(review_text, metrics, spec=None, grids=(), section=BOUND_SECTION,
                    texts=None):
    """Claims in the bound section that trace to nothing the pipeline computed."""
    backing = backing_values(metrics, spec, grids, texts)
    return [(key, value) for key, value, text in bound_claims(review_text, section)
            if not _traces_to(value, text, backing)]


def check_score_binding(solution_text) -> list[Finding]:
    """The review's score must equal ``metrics.score`` exactly.

    This is the check that makes a kernel-emitted score **binding rather than
    advisory**, and it is an error rather than a warning for a reason the rest of
    this module does not have: every other binding here compares two numbers an
    LLM wrote about the same measurement, so rounding and phrasing make an exact
    match unreasonable. ``score`` is a small integer the kernel computed and the
    agent is asked only to transcribe. A mismatch is not drift, it is the agent
    substituting its own judgement for the rubric -- which is the failure
    ``workspace/pde_kuramoto_sivashinsky`` recorded three times over.

    Silent on a review whose metrics block carries no ``score``: every SOLUTION.md
    written before the kernel is in exactly that state, and they are not wrong,
    only older.
    """
    metrics = parse_metrics(solution_text)
    computed = metrics.get("score")
    if computed is None:
        return []
    try:
        computed = int(float(computed))
    except (TypeError, ValueError):
        return [error("review", "<metrics>.score",
                      f"score must be an integer, got {computed!r}")]
    match = _REVIEW_RE.search(solution_text or "")
    if not match:
        return []
    found = []
    claimed = _REVIEW_ATTR_RE.search(solution_text or "")
    headline = _HEADLINE_RE.search(match.group(1))
    for label, hit in (("<review score=...>", claimed), ("the `Score: N/10` headline",
                                                         headline)):
        if hit is None:
            found.append(error("review", "<review>",
                               f"{label} is missing, so the kernel's score "
                               f"{computed} is not carried into the review"))
            continue
        stated = int(hit.group(1))
        if stated != computed:
            found.append(error(
                "review", "<review>",
                f"{label} says {stated} but the kernel computed {computed}. The "
                f"evaluator transcribes the score; it does not assign one. To record a "
                f"judgement no check measures, pass an `agent_cap` -- which may only "
                f"move the score downward"))
    return found


def check_review(solution_text, spec=None, grids=()) -> list[Finding]:
    """Warn once. The caller regenerates and, on a second pass, records and moves on."""
    metrics = parse_metrics(solution_text)
    bound = check_score_binding(solution_text)
    review = _REVIEW_RE.search(solution_text or "")
    if not review:
        return bound
    claims = bound_claims(review.group(1))
    if not claims:
        return bound  # a stub review ("Awaiting solver") claims nothing
    if not metrics:
        return [*bound, warning("review", "<metrics>",
                        f"a <review> stating {len(claims)} measurement(s) was written "
                        "with no <metrics> block, so none of them can be traced to a "
                        "computed value")]
    stray = unbound_numbers(review.group(1), metrics, spec, grids,
                            texts=metrics_texts(solution_text))
    if not stray:
        return bound
    return [*bound, warning(
        "review", f"<review>.{BOUND_SECTION}",
        f"{len(stray)} claim(s) in the {BOUND_SECTION} section trace to nothing the "
        f"pipeline computed: {stray[:6]}. The same model writes <metrics> and "
        f"<review>, so this catches transcription drift, not fabrication",
    )]
