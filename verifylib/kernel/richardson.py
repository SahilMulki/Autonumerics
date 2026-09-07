"""Richardson extrapolation, the GCI, and the asymptotic-range guards.

``verification_manual.md`` §4, plus the two-level variant plan §7a defines: when a
Tier-B probe has already measured the order at two grids, the main ladder can run
two levels and plug the probe's order into the GCI instead of buying a third.

That variant is where rev 1 of the plan got the error direction wrong, so the
preconditions are enforced here rather than left to the caller:

* ``p`` is **clamped** to ``theoretical_order``. A ``p`` that is too high
  *understates* the error -- using 4 where the truth is 1 underestimates by 15x --
  and that is precisely the direction that turns a fail into a pass.
* The probe order is measured on a *smooth manufactured* problem while the real
  one may be shock-, layer- or BC-limited, so the caller must establish smoothness
  before offering ``p_probe`` at all (``verification.structural_facts``).
* ``Fs = 3.0``, not 1.25. Roache's larger factor is for a study whose order was
  imported rather than measured on the ladder under test. Keeping 1.25 would
  compound the understatement rather than cover it.
"""

from __future__ import annotations

import numpy as np

from .ladder import rms

#: Roache's grid-convergence-index factor, by where the order came from.
FS_MEASURED = 1.25
FS_IMPORTED = 3.0

#: ``d21`` at or below this multiple of ``rms(u)`` is round-off, not error.
ROUNDOFF_FLOOR = 1e-13

#: The coarse-to-medium difference must be smaller than the solution itself.
#:
#: Measured across every Path-B ladder in ``workspace/``: the three Heston plans sit
#: at 0.005% of the field, the three Kuramoto-Sivashinsky plans at 210-407%. Five
#: orders of separation and nothing in between, so 1.0 is not a tuned threshold --
#: it is the statement that a coarse grid differing from the medium one by more than
#: the whole solution is not approximating the same function, and no ratio formed
#: from it is a convergence rate.
#:
#: Manual §4 lists four guards; this is a fifth. It earns its place because without
#: it the spectral branch below would certify an order of 10.47 built from a ``d10``
#: four times the size of the field.
COARSE_RESOLVED_FRACTION = 1.0

#: Scheme families whose convergence is super-algebraic, so ``p_sane``'s upper bound
#: is the wrong test for them.
SPECTRAL_FAMILIES = ("spectral",)


def observed_order(d10, d21):
    if d10 <= 0.0 or d21 <= 0.0:
        return float("inf") if d21 <= 0.0 < d10 else float("nan")
    return float(np.log2(d10 / d21))


def is_spectral(scheme_family):
    return (scheme_family or "").strip().lower() in SPECTRAL_FAMILIES


def guards(d10, d21, u_ref, order, theoretical_order, *, scheme_family=None):
    """The §4 asymptotic-range guards, each reported separately.

    Separately, because they mean different things to the solver: ``shrinking``
    false says the grids are not yet in the convergent regime, while ``p_sane``
    false says the estimate is noise. Collapsing them into one boolean loses the
    only actionable half.

    **``p_sane`` depends on the scheme family.** §4's ``p < theoretical_order + 1``
    reads a large ``p`` as noise, and for a finite-difference scheme it is. For a
    **spectral** one it is the expected behaviour: a Fourier method converges
    exponentially, so ``log2(d10/d21)`` is not an algebraic order at all and has no
    upper bound. Measured on ``pde_kuramoto_sivashinsky``, whose three plans are all
    spectral: 10.47, 10.92 and 9.98 against a ``theoretical_order`` of 4 -- and at a
    horizon short enough for the ladder to be fully resolved (``d10`` at 0.2% of the
    field) the order rises to **11.54** rather than falling. Applying the algebraic
    guard there fails every correct spectral solver, which is why all three scored
    4. For a spectral scheme the meaningful test is the other direction: it must
    converge at least as fast as the algebraic design order.

    ``theoretical_order`` describes the *equation* -- KS's biharmonic term makes it
    4 -- not the scheme's rate, which is what made the two easy to conflate.
    """
    floor = ROUNDOFF_FLOOR * (rms(u_ref) + 1e-30)
    spectral = is_spectral(scheme_family)
    out = {
        "shrinking": bool(d21 < d10 * 0.95),
        "p_finite": bool(np.isfinite(order) and order > 0),
        "p_sane": bool(order >= float(theoretical_order) if spectral
                       else order < float(theoretical_order) + 1.0),
        "p_sane_branch": "spectral" if spectral else "algebraic",
        "coarse_resolved": bool(d10 < COARSE_RESOLVED_FRACTION * (rms(u_ref) + 1e-30)),
        "not_at_roundoff": bool(d21 > floor),
        "roundoff_floor": float(floor),
    }
    out["asymptotic"] = all(out[k] for k in ("shrinking", "p_finite", "p_sane",
                                             "coarse_resolved", "not_at_roundoff"))
    return out


def estimate(d10, d21, u_star_ref, order, *, fs=FS_MEASURED):
    """Richardson error estimate of the finest solution, and its GCI bound."""
    denominator = (2.0 ** order) - 1.0
    if not np.isfinite(denominator) or denominator <= 0.0:
        return float("nan"), float("nan")
    e_est = d21 / denominator / (rms(u_star_ref) + 1e-14)
    return float(e_est), float(fs * e_est)


def from_ladder(fields, axes, theoretical_order, *, restricted=None,
                scheme_family=None):
    """Three-level Richardson from a nested ladder of the primary field.

    ``fields`` and ``axes`` are coarse-to-fine. ``restricted`` overrides the
    restriction step for a caller that already did it (and knows whether it had to
    interpolate).
    """
    from .ladder import restrict
    u0, u1, u2 = fields
    u1_on_0 = restricted[0] if restricted else restrict(u1, axes[1], axes[0])
    u2_on_1 = restricted[1] if restricted else restrict(u2, axes[2], axes[1])
    if u1_on_0 is None or u2_on_1 is None:
        return None
    d10, d21 = rms(u1_on_0 - u0), rms(u2_on_1 - u1)
    order = observed_order(d10, d21)
    checks = guards(d10, d21, u1, order, theoretical_order,
                    scheme_family=scheme_family)

    if not checks["not_at_roundoff"]:
        # Converged to machine precision. Manual §4 says so explicitly: treat as
        # e_est = 0, order = inf, asymptotic True. Not a failure -- the opposite.
        return {"d10": d10, "d21": d21, "observed_order": float("inf"),
                "e_est": 0.0, "gci": 0.0, "fs": FS_MEASURED,
                "u_star": u2_on_1, "levels": 3, "order_source": "roundoff", **checks,
                "asymptotic": True}

    u_star = u2_on_1 + (u2_on_1 - u1) / ((2.0 ** order) - 1.0) \
        if np.isfinite(order) and order > 0 else u2_on_1
    e_est, gci = estimate(d10, d21, u_star, order, fs=FS_MEASURED)
    return {"d10": d10, "d21": d21, "observed_order": order, "e_est": e_est,
            "gci": gci, "fs": FS_MEASURED, "u_star": u_star, "levels": 3,
            "order_source": "ladder", **checks}


def from_probe(fields, axes, p_probe, theoretical_order, *, restricted=None):
    """The §7a two-level variant. Preconditions are the caller's; the clamp is not.

    Returns ``None`` when the two grids do not nest, so the caller falls back to a
    third level rather than to an interpolated estimate it must not certify off.
    """
    from .ladder import restrict
    u0, u1 = fields[0], fields[1]
    u1_on_0 = restricted[0] if restricted else restrict(u1, axes[1], axes[0])
    if u1_on_0 is None:
        return None
    d10 = rms(u1_on_0 - u0)
    order = min(float(p_probe), float(theoretical_order))
    floor = ROUNDOFF_FLOOR * (rms(u0) + 1e-30)
    if d10 <= floor:
        return {"d10": d10, "d21": None, "observed_order": float("inf"), "e_est": 0.0,
                "gci": 0.0, "fs": FS_IMPORTED, "u_star": u1, "levels": 2,
                "order_source": "probe", "clamped_to_theoretical": False,
                "shrinking": True, "p_finite": True, "p_sane": True,
                "not_at_roundoff": False, "roundoff_floor": float(floor),
                "asymptotic": True}
    # No `coarse_resolved` guard here, and deliberately: on this path the order is
    # imported rather than measured, so a large `d10` is not laundered into a large
    # `p`. It goes straight into `e_est = d10 / (2**p - 1) / rms(u)`, which for any
    # sane `p` is then an error estimate above 30% and fails `converged` on its own.
    # The three-grid path needs the guard precisely because a garbage `d10` there
    # *raises* `p` and so *shrinks* the estimate.
    e_est, gci = estimate(d10, d10, u1, order, fs=FS_IMPORTED)
    return {"d10": d10, "d21": None, "observed_order": order, "e_est": e_est,
            "gci": gci, "fs": FS_IMPORTED, "u_star": u1, "levels": 2,
            "order_source": "probe",
            "clamped_to_theoretical": bool(float(p_probe) > float(theoretical_order)),
            # The three guards need a third grid. On this path the probe's own pair
            # supplies them, and `asymptotic_source: probe` records that in the
            # metrics -- rev 1 removed the third level and left the guards in the
            # tier definition, which is a contradiction, not a saving.
            "shrinking": None, "p_finite": bool(np.isfinite(order) and order > 0),
            "p_sane": bool(order <= float(theoretical_order)),
            "not_at_roundoff": True, "roundoff_floor": float(floor),
            "asymptotic": None}
