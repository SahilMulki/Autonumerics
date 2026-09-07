"""Ladder, Richardson, stencils, metrics and the plan frontmatter.

These are the transcriptions -- ``verification_manual.md`` §3-4 already read as a
reference implementation. What is worth testing is not that the formulas are the
manual's, but the three places the transcription could silently change behaviour:
the stencils' exact cancellation, the round-off branch, and the two-level probe
clamp.
"""
import numpy as np
import pytest

from verifylib.kernel import ladder, metrics, plan_meta
from verifylib.kernel import richardson as rich
from verifylib.kernel.residual import fornberg_weights, time_derivatives
from verifylib.kernel.sde import crn
from verifylib.operator import Stencil, d1, d2, spectral_d

# --- stencils -----------------------------------------------------------------

def test_the_default_stencils_are_unchanged():
    """§9b: the higher orders are *added*, not swapped. reference.TOL, PROBE_N and
    the measured 16-of-20 Tier-A distribution are all calibrated against these."""
    f = np.random.default_rng(0).standard_normal(64)
    h = 0.1
    old_d1 = (-np.roll(f, -2) + 8 * np.roll(f, -1) - 8 * np.roll(f, 1) + np.roll(f, 2)) / (12 * h)
    old_d2 = (-np.roll(f, -2) + 16 * np.roll(f, -1) - 30 * f + 16 * np.roll(f, 1)
              - np.roll(f, 2)) / (12 * h * h)
    assert np.allclose(d1(f, 0, h), old_d1, rtol=1e-13, atol=1e-13)
    assert np.allclose(d2(f, 0, h), old_d2, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("order", [2, 4, 6, 8])
def test_stencils_cancel_exactly_on_piecewise_constant_data(order):
    """The reason the weights are integers over a common denominator.

    ``1/12 - 2/3 + 2/3 - 1/12`` lands at 1.4e-17, not 0. On
    ``pde_burgers_inviscid`` -- whose exact solution is a step function, so every
    derivative is identically zero away from the shock -- that makes the residual
    and its own term-balanced denominator both 1.4e-17 and their ratio 1, so an
    exactly correct formula reports `failed`. Measured: this test fails on the
    float-weight form.
    """
    step = np.where(np.arange(64) < 30, 1.0, 0.0)
    interior = slice(order // 2 + 2, 30 - order // 2 - 1)
    assert np.array_equal(d1(step, 0, 0.1, order=order)[interior], np.zeros(30 - order - 3))
    assert np.array_equal(d2(step, 0, 0.1, order=order)[interior], np.zeros(30 - order - 3))


@pytest.mark.parametrize("order,degree", [(2, 1), (4, 1), (6, 1), (8, 1),
                                          (2, 2), (4, 2), (6, 2), (8, 2)])
def test_each_stencil_achieves_its_declared_order(order, degree):
    fn = d1 if degree == 1 else d2
    errs = []
    for N in (25, 50):
        x = np.linspace(0.3, 1.3, N)
        h = x[1] - x[0]
        u = np.exp(np.sin(3 * x))
        exact = (3 * np.cos(3 * x) * u if degree == 1
                 else (9 * np.cos(3 * x) ** 2 - 9 * np.sin(3 * x)) * u)
        errs.append(float(np.max(np.abs((fn(u, 0, h, order=order) - exact)[6:-6]))))
    measured = np.log2(errs[0] / errs[1])
    assert measured > order - 0.5, f"order {order} measured {measured:.2f}"


def test_spectral_differentiation_is_exact_on_a_periodic_grid():
    L = 2 * np.pi
    for N in (32, 33):
        x = np.linspace(0.0, L, N, endpoint=False)
        h = x[1] - x[0]
        u = np.sin(3 * x) + 0.4 * np.cos(x)
        assert np.max(np.abs(spectral_d(u, 0, h, 1)
                             - (3 * np.cos(3 * x) - 0.4 * np.sin(x)))) < 1e-12
        assert np.max(np.abs(spectral_d(u, 0, h, 2)
                             - (-9 * np.sin(3 * x) - 0.4 * np.cos(x)))) < 1e-11


def test_a_spectral_stencil_falls_back_to_fd8_off_a_periodic_axis():
    """Passing an endpoint-inclusive array to a Fourier derivative puts an extra
    cell in the period and is wrong everywhere, not just at the edge -- so the
    spectral family refuses any axis that was not detected as exclusive."""
    x = np.linspace(0.3, 1.3, 200)
    h = x[1] - x[0]
    u = np.exp(np.sin(3 * x))
    assert np.array_equal(Stencil("spectral", periodic=[False]).d1(u, 0, h),
                          d1(u, 0, h, order=8))


# --- ladder -------------------------------------------------------------------

def test_the_ladder_nests_under_both_endpoint_conventions():
    assert ladder.build_ladder(64, periodic=True) == [64, 128, 256]
    assert ladder.build_ladder(64, periodic=False) == [64, 127, 253]
    coarse, fine = np.linspace(0, 1, 33), np.linspace(0, 1, 65)
    assert ladder.detect_periodic(np.linspace(0, 1, 64, endpoint=False), (0, 1)) is True
    assert ladder.detect_periodic(fine, (0, 1)) is False
    got = ladder.restrict(np.sin(np.pi * fine), [fine], [coarse])
    assert np.allclose(got, np.sin(np.pi * coarse))


def test_restrict_reports_grids_that_do_not_nest():
    assert ladder.restrict(np.zeros(65), [np.linspace(0, 1, 65)],
                           [np.linspace(0, 1, 32)]) is None


# --- Richardson ---------------------------------------------------------------

def _ladder_fields(order):
    axes = [np.linspace(0, 1, n) for n in (33, 65, 129)]
    hs = [1 / 32, 1 / 64, 1 / 128]
    return [np.sin(np.pi * a) + h ** order for a, h in zip(axes, hs, strict=True)], \
        [[a] for a in axes]


def test_richardson_recovers_the_order_it_was_given():
    fields, axes = _ladder_fields(2)
    got = rich.from_ladder(fields, axes, theoretical_order=2.0)
    assert abs(got["observed_order"] - 2.0) < 1e-6
    assert got["asymptotic"] is True and got["fs"] == rich.FS_MEASURED


def test_a_solution_at_roundoff_is_a_pass_not_a_failure():
    """Manual §4: below the floor the scheme has converged to machine precision --
    treat as e_est = 0, order = inf, asymptotic True. The KS spec's own note
    documents a spectral scheme landing here on the 64/127/253 ladder."""
    axes = [[np.linspace(0, 1, n)] for n in (33, 65, 129)]
    fields = [np.sin(np.pi * np.linspace(0, 1, n)) for n in (33, 65, 129)]
    got = rich.from_ladder(fields, axes, theoretical_order=2.0)
    assert got["asymptotic"] is True
    assert got["e_est"] == 0.0 and got["observed_order"] == float("inf")
    assert got["not_at_roundoff"] is False


def test_the_probe_order_is_clamped_and_carries_the_larger_gci_factor():
    """§7a. Rev 1 got the direction wrong: a p_probe that is too high *understates*
    the error -- p=4 where the truth is 1 underestimates by 15x -- which is exactly
    the direction that turns a fail into a pass."""
    fields, axes = _ladder_fields(2)
    got = rich.from_probe(fields[:2], axes[:2], p_probe=4.0, theoretical_order=2.0)
    assert got["observed_order"] == 2.0
    assert got["clamped_to_theoretical"] is True
    assert got["fs"] == rich.FS_IMPORTED == 3.0
    assert got["asymptotic"] is None, "the three-grid guards are not available here"

    loose = rich.from_probe(fields[:2], axes[:2], p_probe=1.0, theoretical_order=2.0)
    assert loose["observed_order"] == 1.0 and loose["clamped_to_theoretical"] is False
    assert loose["e_est"] > got["e_est"], "a lower order must give a *larger* estimate"


# --- the snapshot time term ---------------------------------------------------

def test_fornberg_weights_differentiate_exactly_on_unequal_spacing():
    """The point of Fornberg over a fixed stencil: a solver's final step is
    whatever ``t_final / Nt`` left over, and an adaptive integrator's is not
    uniform at all."""
    for nodes in ([0.8, 0.9, 1.0], [0.0, 0.7, 1.0], [0.1, 0.55, 0.9, 1.0]):
        w = fornberg_weights(nodes[-1], nodes, 2)
        assert abs(sum(w[i, 1] * t ** 2 for i, t in enumerate(nodes)) - 2 * nodes[-1]) < 1e-10
        assert abs(sum(w[i, 2] * t ** 2 for i, t in enumerate(nodes)) - 2.0) < 1e-9


def test_the_time_term_comes_from_snapshots_and_records_its_own_order():
    snaps = [{"t": t, "fields": {"u": np.array([t ** 2, 2 * t ** 2])}}
             for t in (0.8, 0.9, 1.0)]
    derivs, order = time_derivatives(snaps, ["u"])
    assert order == 2
    assert np.allclose(derivs["u"][0], [2.0, 4.0])
    two, order2 = time_derivatives(snaps[1:], ["u"])
    assert order2 == 1, "K = 2 runs, and records its own truncation floor"
    assert time_derivatives([], ["u"]) == (None, 0)


# --- metrics ------------------------------------------------------------------

def test_a_skip_is_never_a_pass():
    """The one function that keeps 'the check did not run' from reading as
    evidence. Manual §25's first rule, in code."""
    for skip in metrics.SKIPS:
        assert metrics.is_skip(skip)
        assert not metrics.passed(skip)
        assert not metrics.failed(skip)
    assert metrics.passed(True) and metrics.failed(False)
    assert not metrics.passed(None) and not metrics.passed(1)


def test_a_check_cannot_report_something_that_is_not_a_verdict():
    cfg = metrics.Config()
    m = metrics.new_metrics("pde", "B", cfg)
    with pytest.raises(ValueError, match="skip markers"):
        metrics.record(m, "converged", "probably")


def test_every_configured_value_carries_its_source():
    """§12a (1): a misconfiguration should be visible directly rather than inferred
    from an odd result."""
    cfg = metrics.Config()
    assert cfg.pick("grid_N", {"grid_N": 32}, 64, 16) == 64
    assert cfg.pick("levels", {"levels": 2}, None, 3) == 2
    assert cfg.pick("metric", {}, None, "l2") == "l2"
    assert {k: v["source"] for k, v in cfg.as_dict().items()} == {
        "grid_N": "spec", "levels": "agent", "metric": "kernel_default"}


def test_a_guessed_value_is_recorded_as_a_guess():
    """``inferred`` is not ``kernel_default`` and not ``spec``. It selects which
    asymptotic guard applies, so a reader has to be able to tell a value the kernel
    guessed from a value someone stated."""
    cfg = metrics.Config()
    cfg.set("scheme_family", "spectral", "inferred")
    cfg.set("theoretical_order", 4.0, "spec")
    assert cfg.as_dict()["scheme_family"] == {"value": "spectral", "source": "inferred"}
    with pytest.raises(ValueError, match="config source"):
        cfg.set("bogus", 1, "vibes")


def test_the_metrics_block_renders_only_what_applies():
    cfg = metrics.Config()
    m = metrics.new_metrics("pde", "B", cfg)
    m.update(score=9, provenance="manufactured_partial")
    m["summary"] = {"estimated_rel_error": 3.1e-6, "observed_order": 1.98}
    metrics.record(m, "invariants_ok", True)
    block = metrics.render_block(m)
    assert "score: 9" in block and "provenance: manufactured_partial" in block
    assert "invariants_ok: true" in block
    assert "mc_se_rel" not in block, "omit fields that do not apply"


# --- the plan frontmatter -----------------------------------------------------

def test_the_frontmatter_parser_ignores_a_body_line_that_looks_like_a_key():
    """§3b: a body line beginning ``scheme:`` already exists in
    ``pde_convection_diffusion_bl/plans/1-exponential-fitting/SOLUTION.md``, and a
    grep for ``^scheme:`` picks it up."""
    text = "---\nid: 1\nscheme_family: spectral\nspatial_order: 8\n---\n\nbody\nscheme: no\n"
    front = plan_meta.parse_frontmatter(text)
    assert front == {"id": "1", "scheme_family": "spectral", "spatial_order": "8"}
    assert plan_meta.parse_frontmatter("no frontmatter\nscheme: nope\n") == {}


def test_an_undeclared_plan_falls_back_rather_than_guessing(tmp_path):
    """``declared`` stays False when only the free-text ``scheme:`` is present --
    D1's stencil choice records ``default`` there. The *family* is still inferred,
    because the asymptotic guard has to pick a branch either way; the two are
    reported separately so neither is mistaken for the other."""
    (tmp_path / "SOLUTION.md").write_text("---\nid: 1\nscheme: spectral-imex\n---\n")
    got = plan_meta.read(str(tmp_path))
    assert got["declared"] is False and got["spatial_order"] is None
    assert got["scheme_family_source"] == "inferred"

    (tmp_path / "SOLUTION.md").write_text(
        "---\nid: 1\nscheme_family: fd\nspatial_order: 6\n---\n")
    got = plan_meta.read(str(tmp_path))
    assert got == {"scheme_family": "fd", "spatial_order": 6, "scheme": None,
                   "scheme_family_source": "declared", "declared": True}


# --- the asymptotic guards depend on the scheme family ------------------------

def test_a_large_p_is_noise_for_a_difference_scheme_and_expected_for_a_spectral_one():
    """§4's ``p < theoretical_order + 1`` reads a large p as noise, and for finite
    differences it is. A Fourier method converges exponentially, so
    ``log2(d10/d21)`` is not an algebraic order at all and has no upper bound.
    Measured on ``pde_kuramoto_sivashinsky``, whose three plans are all spectral:
    10.47, 10.92 and 9.98 against ``theoretical_order`` 4 -- and at a horizon where
    the ladder is fully resolved the order *rises* to 11.54.
    """
    u = np.ones(64)
    algebraic = rich.guards(1.2e-3, 4.2e-7, u, 11.54, 4.0)
    spectral = rich.guards(1.2e-3, 4.2e-7, u, 11.54, 4.0, scheme_family="spectral")
    assert algebraic["p_sane"] is False and algebraic["p_sane_branch"] == "algebraic"
    assert spectral["p_sane"] is True and spectral["p_sane_branch"] == "spectral"
    assert spectral["asymptotic"] is True


def test_a_spectral_scheme_converging_slower_than_its_design_order_still_fails():
    """The spectral branch is a *lower* bound, not the absence of a bound."""
    u = np.ones(64)
    got = rich.guards(1.2e-3, 6.0e-4, u, 1.0, 4.0, scheme_family="spectral")
    assert got["p_sane"] is False


def test_an_unresolved_coarse_grid_is_not_an_order_estimate():
    """``d10`` larger than the solution means the coarse grid is not approximating
    the same function, and no ratio formed from it is a convergence rate. Measured:
    the three Heston plans sit at 0.005% of the field, the three KS plans at
    210-407% -- five orders of separation with nothing in between."""
    u = np.ones(64)
    unresolved = rich.guards(4.98, 3.5e-3, u, 10.47, 4.0, scheme_family="spectral")
    assert unresolved["coarse_resolved"] is False
    assert unresolved["p_sane"] is True, "the spectral branch alone would have passed it"
    assert unresolved["asymptotic"] is False, "so the coarse-grid guard is what catches it"

    resolved = rich.guards(1.2e-3, 4.2e-7, u, 11.54, 4.0, scheme_family="spectral")
    assert resolved["coarse_resolved"] is True and resolved["asymptotic"] is True


def test_the_ordinary_second_order_case_is_untouched():
    u = np.ones(64)
    got = rich.guards(4e-4, 1e-4, u, 2.0, 2.0)
    assert got["asymptotic"] is True and got["coarse_resolved"] is True


# --- inferring the family from the free-text scheme --------------------------

@pytest.mark.parametrize("scheme,family", [
    ("spectral", "spectral"), ("spectral-imex", "spectral"),
    ("finite-difference-implicit-adi", "fd"), ("finite-volume-explicit", "fv"),
    ("finite-element-galerkin-implicit", "fe"), ("euler-maruyama", "euler-maruyama"),
    ("milstein", "milstein"), ("monotone-wide-stencil", None), ("", None),
])
def test_the_family_is_inferred_by_prefix_and_collides_with_nothing_real(scheme, family):
    """A prefix match rather than a substring search, measured against the 30
    distinct ``scheme:`` values in ``workspace/``: no ``finite-difference-*`` value
    contains "spectral" and no ``spectral-*`` value contains "finite"."""
    assert plan_meta.infer_family(scheme) == family


def test_a_declared_family_outranks_an_inferred_one_and_the_source_is_recorded(tmp_path):
    """The inference selects which asymptotic guard applies, so a reader has to be
    able to see that the kernel guessed."""
    (tmp_path / "SOLUTION.md").write_text("---\nid: 1\nscheme: spectral-imex\n---\n")
    got = plan_meta.read(str(tmp_path))
    assert got["scheme_family"] == "spectral" and got["scheme_family_source"] == "inferred"

    (tmp_path / "SOLUTION.md").write_text(
        "---\nid: 1\nscheme: spectral-imex\nscheme_family: fd\n---\n")
    got = plan_meta.read(str(tmp_path))
    assert got["scheme_family"] == "fd" and got["scheme_family_source"] == "declared"


def test_an_inferred_scheme_family_never_narrows_the_order_band():
    """An inference must not make a test stricter -- a false hard-fail on a correct
    solver is invisible as a false positive.

    ``sde_ginzburg_landau_s6/3-log-transform-euler`` is the measured case. Its
    ``scheme:`` reads ``euler-maruyama``, which infers strong order 0.5 -- but it is
    Euler-Maruyama on the *log transform*, where the noise becomes additive and the
    strong order is therefore 1. It measures 1.019, correctly, and the inferred
    two-sided band at 0.5 +/- 0.4 failed it.
    """
    verification = {"expected_strong_order": 0.5, "expected_strong_order_milstein": 1.0}

    _, _, declared = crn.expected_orders(verification, "euler-maruyama", declared=False)
    assert declared is False, "an inference leaves the band one-sided"

    _, _, declared = crn.expected_orders(verification, "euler-maruyama", declared=True)
    assert declared is True, "a stated family narrows it"

    # Selecting among the spec's own declared values is safe either way: every
    # candidate came from the formulator.
    expected, candidates, _ = crn.expected_orders(verification, "milstein", declared=False)
    assert expected == 1.0 and sorted(candidates) == [0.5, 1.0]


def test_the_one_sided_band_accepts_a_faster_scheme_and_still_rejects_a_slower_one():
    detail = {"s0": 2.1e-2, "s1": 1.0e-2}
    faster = crn.guards(1.019, 1.11, 0.5, 1.0, detail, declared=False, candidates=[0.5, 1.0])
    assert faster["strong_ok"] is True and faster["orders_ok"] is True
    slower = crn.guards(0.02, 1.11, 0.5, 1.0, detail, declared=False, candidates=[0.5, 1.0])
    assert slower["strong_ok"] is False
