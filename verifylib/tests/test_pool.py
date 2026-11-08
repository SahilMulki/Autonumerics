"""Layer 6: the pool-level guards, and the comparison that motivated them.

``docs/improving-autonumerics/11-sde-accuracy-fixes.md`` §4. Every other guard in
``verifylib/`` grades one artifact, and the failure that cost the most accuracy on the
BENCHOP suite was a property of the *set*: a pool that did not contain a good solver.
``sde_heston_call_feller`` and ``sde_heston_call_hivar`` are the same problem class
under the same code with the same declared ``terminal_law``, and one pool contains an
exact non-central chi-squared transition while the other contains three discretizations
of it. The difference was which spec's free-text notes happened to mention the scheme.
"""
import glob
import json
import os

from verifylib import pool

# The Heston generator, verbatim from `sde_heston_call_feller`'s spec: rows are state
# components (S, V), columns are independent noise channels.
HESTON = {
    "drift": ["0.0*X[:, 0]", "kappa*(theta - X[:, 1])"],
    "diffusion_matrix": [
        ["np.sqrt(np.maximum(X[:, 1], 0.0))*X[:, 0]", "0.0*X[:, 0]"],
        ["rho*sigma_v*np.sqrt(np.maximum(X[:, 1], 0.0))",
         "sigma_v*np.sqrt(1 - rho**2)*np.sqrt(np.maximum(X[:, 1], 0.0))"],
    ],
}


def _write(tmp_path, spec, plans):
    """A problem directory: one spec, and ``{plan_name: frontmatter_text}``."""
    problem = tmp_path / "sde_probe"
    (problem / "plans").mkdir(parents=True)
    (problem / "problem_spec.json").write_text(json.dumps(spec))
    for name, front in plans.items():
        directory = problem / "plans" / name
        directory.mkdir()
        (directory / "SOLUTION.md").write_text(front)
    return str(problem)


def _plan(*, family=None, exact=None, boundary=None, order=None, error=None):
    lines = ["---"]
    if family:
        lines.append(f"scheme_family: {family}")
    if exact is not None:
        lines.append(f"exact_sampling_components: {exact}")
    if boundary:
        lines.append(f"boundary_treatment: {boundary}")
    lines += ["---", "", "Body.", "", "<metrics>"]
    if order is not None:
        lines.append(f"observed_order: {order}")
    if error is not None:
        lines.append(f"estimated_rel_error: {error}")
    lines += ["</metrics>", ""]
    return "\n".join(lines)


# --- the conditioning detector ------------------------------------------------


def test_the_heston_asset_leg_is_detected_as_conditionally_tractable():
    """The measurement Fix 4's third obligation rests on. The asset's diffusion is
    ``sqrt(max(V,0)) * S`` -- degree one in ``S`` -- and the variance leg never reads
    ``S``, so conditionally on the variance path the asset is a time-changed geometric
    Brownian motion with a closed-form law. Conditional Monte Carlo on that leg removes
    its sampling variance *entirely*, and it appeared in none of the thirty plans."""
    assert pool.conditionally_tractable(HESTON, 2) == (0,)


def test_the_variance_leg_is_not_claimed_to_be_tractable():
    """The false positive that matters. ``np.sqrt(np.maximum(X[:, 1], 0.0))`` has no
    polynomial degree in ``X[:, 1]``, and treating a call as degree 0 would claim a CIR
    variance process has a conditional closed form -- asking a plan for something that
    does not exist."""
    assert 1 not in pool.conditionally_tractable(HESTON, 2)


def test_a_two_way_coupling_is_not_tractable_either_way():
    """Linearity in itself is not enough: the other components' paths have to be
    generatable *first*, so nothing else may read this one."""
    coupled = {
        "drift": ["a*X[:, 1]", "b*X[:, 0]"],
        "diffusion_matrix": [["s*X[:, 0]"], ["s*X[:, 1]"]],
    }
    assert pool.conditionally_tractable(coupled, 2) == ()


def test_a_quadratic_self_dependence_is_not_tractable():
    """Component 0 enters its own drift squared, so fixing component 1's path leaves a
    nonlinear equation with no closed form. Component 1 is excluded by the other
    condition -- component 0 reads it, so its path cannot be generated first."""
    quadratic = {
        "drift": ["a*X[:, 0]**2 + X[:, 1]", "b*X[:, 1]"],
        "diffusion_matrix": [["s*X[:, 0]"], ["s"]],
    }
    assert pool.conditionally_tractable(quadratic, 2) == ()


def test_an_autonomous_component_is_tractable_on_its_own():
    """The complement, asserted because the fixture above got it wrong first: an
    Ornstein-Uhlenbeck leg that nothing else reads is exactly solvable, and the
    detector is right to say so."""
    ou_driving = {
        "drift": ["a*X[:, 0]**2", "b*X[:, 1]"],
        "diffusion_matrix": [["s*X[:, 0]"], ["s"]],
    }
    assert pool.conditionally_tractable(ou_driving, 2) == (1,)


def test_a_scalar_sde_has_nothing_to_condition_on():
    scalar = {"drift": ["a*X[:, 0]"], "diffusion_matrix": [["s*X[:, 0]"]]}
    assert pool.conditionally_tractable(scalar, 1) == ()


def test_an_absent_or_malformed_generator_abstains_rather_than_guessing():
    for bad in (None, {}, {"drift": "x"}, {"drift": ["a"], "diffusion_matrix": [["b"]]}):
        assert pool.conditionally_tractable(bad, 2) == ()


# --- the declared transition law -----------------------------------------------


_LAW_SPEC = {
    "state_dimension": 2,
    "verification": {"terminal_law": [None, {"law": "noncentral_chi2"}]},
}


def test_a_declared_transition_law_no_plan_draws_from_is_a_pool_defect(tmp_path):
    problem = _write(tmp_path, _LAW_SPEC, {
        "1-euler": _plan(family="euler-maruyama"),
        "2-euler-cv": _plan(family="euler-maruyama"),
    })
    found = pool.check_problem(problem)
    assert [f.path for f in found] == ["verification.terminal_law[1]"]
    assert all(f.severity == "warning" for f in found), "a thin pool never blocks"


def test_an_exact_transition_plan_satisfies_the_obligation(tmp_path):
    """`sde_heston_call_hivar`'s shape: the pool that did contain the technique."""
    problem = _write(tmp_path, _LAW_SPEC, {
        "1-exact-chi2": _plan(family="exact-transition", exact="1"),
        "2-euler": _plan(family="euler-maruyama"),
    })
    assert pool.check_problem(problem) == []


def test_quadratic_exponential_does_not_satisfy_it_and_is_told_why(tmp_path):
    """The loophole, and the reason the check cannot key on the component declaration
    alone. Andersen QE declares `exact_sampling_components` *correctly* -- it draws its
    own normals each step and cannot consume the supplied increments -- while being a
    moment-matched approximation to the law rather than a draw from it. Keying on the
    declaration alone cleared `sde_heston_call_feller`'s pool, which is the one pool
    this whole check exists to report."""
    problem = _write(tmp_path, _LAW_SPEC, {
        "1-euler": _plan(family="euler-maruyama"),
        "2-andersen-qe": _plan(family="quadratic-exponential", exact="0, 1"),
    })
    found = pool.check_problem(problem)
    assert len(found) == 1
    assert "2-andersen-qe" in found[0].message
    assert "non-consumption" in found[0].message


# --- the declared boundary ------------------------------------------------------


_BARRIER_SPEC = {
    "state_dimension": 2,
    "verification": {"structural_facts": {
        "barrier": "continuous double barrier at S=50 and S=150; discrete monitoring "
                   "overstates survival with error O(sqrt(dt))"}},
}


def test_a_declared_barrier_no_plan_treats_is_a_pool_defect(tmp_path):
    problem = _write(tmp_path, _BARRIER_SPEC, {
        "1-euler": _plan(family="euler-maruyama"),
        "2-euler-cv": _plan(family="milstein"),
    })
    found = pool.check_problem(problem)
    assert [f.path for f in found] == ["verification.structural_facts.barrier"]
    assert "sqrt(dt)" in found[0].message


def test_one_declared_treatment_satisfies_the_pool(tmp_path):
    problem = _write(tmp_path, _BARRIER_SPEC, {
        "1-euler-bridge": _plan(family="euler-maruyama", boundary="brownian-bridge"),
        "2-euler": _plan(family="milstein"),
    })
    assert pool.check_problem(problem) == []


def test_declaring_none_is_a_treatment_and_silence_is_not(tmp_path):
    """`boundary_treatment: none` is a plan saying it considered the crossing and has
    an argument that the payoff is insensitive to it. That is a position a reader can
    disagree with, which is the whole difference from not mentioning it."""
    problem = _write(tmp_path, _BARRIER_SPEC, {
        "1-euler": _plan(family="euler-maruyama", boundary="none"),
        "2-euler": _plan(family="milstein"),
    })
    assert pool.check_problem(problem) == []


def test_a_structural_fact_that_implies_no_technique_is_not_an_obligation(tmp_path):
    """`kink` and `boundary` describe the problem's shape; `barrier` and
    `absorbing_boundary` say what a correct scheme has to *do*. Only the second kind
    carries an obligation, or every declared fact becomes a warning."""
    spec = {"state_dimension": 2,
            "verification": {"structural_facts": {"kink": "the payoff kinks at K"}}}
    problem = _write(tmp_path, spec, {"1-euler": _plan(family="euler-maruyama")})
    assert pool.check_problem(problem) == []


# --- pool diversity -------------------------------------------------------------


def test_one_scheme_under_two_estimators_is_reported_as_a_two_plan_pool(tmp_path):
    """Four of the ten BENCHOP pools. The CRN ladder takes pathwise differences and a
    variance-reduction wrapper does not change the dynamics, so it cannot change the
    order -- and the pool reads as three plans while being two."""
    spec = {"state_dimension": 2, "verification": {}}
    problem = _write(tmp_path, spec, {
        "1-em-antithetic": _plan(family="euler-maruyama", order=0.4862, error=1.6e-15),
        "2-milstein": _plan(family="milstein", order=1.0021, error=4.0e-06),
        "3-em-control-variate": _plan(family="euler-maruyama", order=0.4862,
                                      error=1.6e-15),
    })
    found = pool.check_problem(problem)
    assert len(found) == 1
    assert found[0].path == "plans/1-em-antithetic + plans/3-em-control-variate"
    assert "two-plan pool wearing three plans" in found[0].message


def test_two_different_families_landing_on_one_rate_is_a_coincidence_not_a_duplicate():
    """An identical order *within* a family is a shared dynamic; across families it is
    a coincidence of the rate, which is a much weaker claim and not worth a warning."""
    plans = [("1-em", {"scheme_family": "euler-maruyama"}, {"observed_order": 1.0}),
             ("2-milstein", {"scheme_family": "milstein"}, {"observed_order": 1.0})]
    assert pool.duplicate_statistics(plans) == []


def test_a_pool_whose_orders_differ_is_not_flagged():
    plans = [("1-em", {"scheme_family": "euler-maruyama"}, {"observed_order": 1.0078}),
             ("2-em", {"scheme_family": "euler-maruyama"}, {"observed_order": 0.9924})]
    assert pool.duplicate_statistics(plans) == []


def test_an_unmeasured_order_is_not_an_identical_one():
    """Two plans that both abstained are not two plans that agree."""
    plans = [("1-em", {"scheme_family": "euler-maruyama"}, {"observed_order": None}),
             ("2-em", {"scheme_family": "euler-maruyama"}, {})]
    assert pool.duplicate_statistics(plans) == []


# --- scoping --------------------------------------------------------------------


def test_a_pde_problem_is_not_reported_as_missing_sde_structure(tmp_path):
    problem = _write(tmp_path, {"verification": {"terminal_law": [{"law": "x"}]}},
                     {"1-fd": _plan(family="fd")})
    assert pool.check_problem(problem) == []


def test_a_problem_with_no_plans_yet_reports_nothing(tmp_path):
    problem = _write(tmp_path, _LAW_SPEC, {})
    assert pool.check_problem(problem) == []


def test_a_directory_that_is_not_a_problem_reports_nothing(tmp_path):
    assert pool.check_problem(str(tmp_path)) == []


# --- the live corpus ------------------------------------------------------------


def test_the_live_corpus_is_reported_exactly_where_the_technique_is_missing():
    """The discrimination this module exists for, asserted against real artifacts.

    **Written against the corpus rather than against a slug**, which is a correction.
    The first version named the two Heston pools directly -- `sde_heston_call_feller`
    is reported for leaving its declared non-central chi-squared transition unused,
    `sde_heston_call_hivar` is not -- and it was **red within the hour**, because a
    `--fresh` re-run of the first problem produced a pool that *does* contain the exact
    draw (12-implementation.md §7). That is the fix working, and the test called it a
    regression.

    `workspace/` is gitignored and turns over; three documents in
    `docs/improving-autonumerics/` have been invalidated by it, and so was this test.
    So the assertion is now the *property*, read off whatever is on disk: a pool with a
    declared `terminal_law` is reported for that component exactly when no plan in it
    declares both the component and `scheme_family: exact-transition`. That holds for
    any corpus, and it is what the fixture tests above check in isolation.
    """
    import pytest

    from .conftest import WORKSPACE

    checked = 0
    for directory in sorted(glob.glob(os.path.join(WORKSPACE, "sde_*"))):
        spec_path = os.path.join(directory, "problem_spec.json")
        if not os.path.isfile(spec_path):
            continue
        with open(spec_path, encoding="utf-8") as fh:
            spec = json.load(fh)
        laws = (spec.get("verification") or {}).get("terminal_law")
        if not isinstance(laws, list):
            continue
        plans = pool._plans(directory)
        if not plans:
            continue
        exact = pool._exact_law(plans)
        reported = {f.path for f in pool.check_terminal_law(spec, plans)}
        for i, entry in enumerate(laws):
            if not isinstance(entry, dict) or not entry.get("law"):
                continue
            checked += 1
            expected = f"verification.terminal_law[{i}]" if not exact.get(i) else None
            if expected:
                assert expected in reported, (
                    f"{os.path.basename(directory)} component {i} declares "
                    f"{entry['law']!r}, no plan draws from it, and it was not reported")
            else:
                assert f"verification.terminal_law[{i}]" not in reported, (
                    f"{os.path.basename(directory)} component {i} is drawn exactly by "
                    f"{exact[i]} and was reported anyway")
    if not checked:
        pytest.skip("no staged SDE spec declares a terminal_law")
