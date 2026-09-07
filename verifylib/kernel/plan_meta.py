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
SDE_FAMILIES = ("euler-maruyama", "milstein", "tamed-euler-maruyama",
                "tamed-milstein", "implicit", "other")
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
                "scheme_family_source": "unknown", "declared": False}
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
            "scheme_family_source": ("declared" if family else
                                     "inferred" if inferred else "unknown"),
            "declared": bool(family is not None or order is not None)}
