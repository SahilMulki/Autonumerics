"""The plan's declared discretization, read from ``SOLUTION.md``'s frontmatter.

D1's two discretization rules (plan §3b) both need an input that did not exist:
rule 1 says difference at *higher order than the solver used*, and rule 2 says use
a *different family*. Rev 1 said to read ``plan.spatial_discretization.scheme``;
there is no such field anywhere in the repo. What exists is a free-text ``scheme:``
in the YAML frontmatter, with 30+ distinct values across ``workspace/``
(``finite-difference-implicit-adi``, ``spectral-imex``,
``finite-difference-implicit (Craig-Sneyd ADI, monotone rotated cross stencil,
...)``), none of which state the spatial order.

So the plan-creators now declare two fields, and this reads them::

    scheme_family: fd | spectral | fv | fe | other
    spatial_order: 2
    exact_sampling_components: 0, 1
    boundary_treatment: brownian-bridge

**Parse the frontmatter block, never grep for ``^scheme:``.** A body line beginning
``scheme:`` already exists in
``pde_convection_diffusion_bl/plans/1-exponential-fitting/SOLUTION.md`` and a grep
picks it up. Where the fields are absent -- every plan written before this landed
-- the caller takes 8th-order FD and records ``stencil_choice: default``.
"""

from __future__ import annotations

import os
import re

#: PDE spatial discretization families, and SDE time-integrator families. One
#: field serves both because one plan is one or the other, never both, and the
#: kernel's two consumers are disjoint: D1 reads the PDE values to pick its
#: differencing pair, and the SDE order study reads the integrator values to pick
#: which of the spec's declared strong orders applies.
PDE_FAMILIES = ("fd", "spectral", "fv", "fe", "other")
#: F6. ``exact-transition`` and ``quadratic-exponential`` are new, and the reason
#: is the agreement gate rather than the order study. Two plans agreeing on a price
#: counts as a circularity break only when they are *different families*; before
#: these existed, an exact non-central chi-squared scheme and Andersen QE both had
#: to declare ``other``, so a genuine cross-family agreement read as a within-family
#: one and counted for nothing. `sde_dupire_local_vol_dnt`'s plan 2 says so in as
#: many words -- it declares ``other`` "precisely so that such an agreement counts".
#:
#: They are deliberately **absent** from :data:`..sde.crn.FAMILY_ORDER_KEY`: naming
#: a family does not create an expected strong order for it, and on this suite the
#: CRN order study reads 0.23-0.31 against declared floors of 0.5 on six plans
#: across three families, so it is in no state to have more requirements hung off
#: it. The vocabulary is the fix; an order bar is not.
SDE_FAMILIES = ("euler-maruyama", "milstein", "tamed-euler-maruyama",
                "tamed-milstein", "implicit", "exact-transition",
                "quadratic-exponential", "other")
FAMILIES = (*PDE_FAMILIES, *SDE_FAMILIES)

#: Prefixes of the free-text ``scheme:`` field that identify a family, used only
#: when ``scheme_family`` is absent -- which is every plan written before it
#: existed.
#:
#: A prefix match rather than a substring search, and measured rather than guessed:
#: the 30 distinct ``scheme:`` values across ``workspace/`` are highly structured
#: (``finite-difference-*`` x22 shapes, ``spectral-*`` x4, ``finite-volume-*``,
#: ``finite-element-*``, ``euler-maruyama``, ``milstein``) and the prefixes below
#: collide on none of them. Inference is recorded separately from a declaration --
#: ``metrics`` carries ``scheme_family_source`` -- because it selects which
#: asymptotic guard applies, and a reader must be able to see that the kernel
#: guessed.
_INFERRED_PREFIX = (
    ("spectral", "spectral"),
    ("finite-difference", "fd"),
    ("finite-volume", "fv"),
    ("finite-element", "fe"),
    ("quadratic-exponential", "quadratic-exponential"),
    ("andersen-qe", "quadratic-exponential"),
    ("qe", "quadratic-exponential"),
    ("exact-transition", "exact-transition"),
    ("exact-conditional", "exact-transition"),
    ("broadie-kaya", "exact-transition"),
    ("non-central-chi", "exact-transition"),
    ("tamed-euler", "tamed-euler-maruyama"),
    ("tamed-milstein", "tamed-milstein"),
    ("euler-maruyama", "euler-maruyama"),
    ("milstein", "milstein"),
)


def infer_family(scheme):
    """The family implied by a free-text ``scheme:``, or ``None``."""
    text = (scheme or "").strip().lower()
    for prefix, family in _INFERRED_PREFIX:
        if text.startswith(prefix):
            return family
    return None

_FRONTMATTER = re.compile(r"\A﻿?---[ \t]*\r?\n(.*?)\r?\n---[ \t]*(?:\r?\n|\Z)", re.S)
_KEY = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")


def parse_frontmatter(text):
    """The leading ``---`` block as ``{key: raw_string}``. ``{}`` when there is none."""
    match = _FRONTMATTER.match(text or "")
    if not match:
        return {}
    out = {}
    for line in match.group(1).splitlines():
        if line[:1] in (" ", "\t", "#") or not line.strip():
            continue  # a continuation or a comment, not a new key
        found = _KEY.match(line)
        if found:
            out[found.group(1)] = found.group(2).strip().strip('"').strip("'")
    return out


def parse_components(raw):
    """``exact_sampling_components`` as a sorted tuple of state indices. (F18)

    The test is **whether the step consumes the Brownian increment the CRN ladder
    supplies, not whether its draw is exact.** A component whose step generates its own
    auxiliary randomness cannot consume that increment -- a non-central chi-squared, a
    Poisson-mixed BESQ(0) draw, Broadie-Kaya, and equally Andersen QE's variance leg,
    which is a *moment-matched approximation* to the non-central chi-squared law and
    not an exact draw from it. The name reads narrower than the rule: B03's QE plan
    read the old wording, correctly concluded it was not doing exact sampling, wrote
    "do not declare `exact_sampling_components` for this plan", and was failed by the
    order study for a near-zero strong order it could not have avoided. The
    pathwise differences across levels then stay O(1) and the measured strong order
    comes back near zero on a scheme that carries *no* discretization error at all.

    ``crn.abstention`` already excuses the sibling case, exact *integration*, and it
    detects it because the differences collapse to roundoff. This one is invisible
    to that test for the opposite reason: the differences are large. So it has to be
    declared, and like ``scheme_family`` it is declared and never inferred -- which
    is also what keeps it honest, because an abstention is not a pass. It costs the
    plan the order check's evidence rather than earning it.
    """
    out = set()
    for piece in re.split(r"[,\s\[\]]+", str(raw or "")):
        if not piece:
            continue
        try:
            value = int(piece)
        except ValueError:
            return ()           # one unparseable entry voids the declaration
        if value < 0:
            return ()
        out.add(value)
    return tuple(sorted(out))


def read(plan_dir):
    """``{"scheme_family", "spatial_order", "scheme", "declared"}`` for a plan dir."""
    path = os.path.join(plan_dir, "SOLUTION.md")
    try:
        with open(path) as fh:
            # 8 KB is far past any real frontmatter block and keeps a 200 KB
            # SOLUTION.md from being read in full for two fields.
            head = fh.read(8192)
    except OSError:
        return {"scheme_family": None, "spatial_order": None, "scheme": None,
                "scheme_family_source": "unknown", "declared": False,
                "boundary_treatment": None,
                "exact_sampling_components": ()}
    front = parse_frontmatter(head)
    family = (front.get("scheme_family") or "").strip().lower() or None
    if family not in FAMILIES:
        family = None
    order = front.get("spatial_order")
    try:
        order = int(float(order)) if order not in (None, "") else None
    except (TypeError, ValueError):
        order = None
    scheme = front.get("scheme")
    inferred = infer_family(scheme) if family is None else None
    return {"scheme_family": family or inferred, "spatial_order": order,
            "scheme": scheme,
            "exact_sampling_components": parse_components(
                front.get("exact_sampling_components")),
            # How this plan handles a boundary the spec declared the answer to be
            # sensitive to: a Brownian-bridge crossing correction, an exact absorption
            # rule, or the word `none` with the plan's argument for why the payoff is
            # insensitive. Declared and never inferred, like `scheme_family`: a
            # crossing treatment is a fact about the scheme that the kernel must trust,
            # and inferring one from a plan's slug would let a plan rename itself into
            # a clean pool report. `pool.check_boundary_facts` is the reader, and until
            # 2026-11-05 `verification.structural_facts.barrier` and
            # `.absorbing_boundary` were declared in live specs and read by nothing.
            "boundary_treatment": (front.get("boundary_treatment") or "").strip()
                                  or None,
            "scheme_family_source": ("declared" if family else
                                     "inferred" if inferred else "unknown"),
            "declared": bool(family is not None or order is not None)}
