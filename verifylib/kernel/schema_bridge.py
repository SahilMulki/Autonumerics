"""The one thing the kernel needs from the schema layer, without the import cycle.

``verifylib.schema`` imports the invariant registry from ``kernel.invariants``; a
kernel module importing ``verifylib.schema`` back at module scope closes the loop.
This is the whole of what the kernel needs, and it needs no numpy.
"""

from __future__ import annotations


def is_sde(spec) -> bool:
    """The mirror of ``schema.is_pde``. Kept as one expression so the two cannot
    drift: an SDE is anything ``schema.is_pde`` calls not-a-PDE."""
    from ..schema import is_pde
    return not is_pde(spec)
