"""Authoring checklist items 2 and 4: the truth lives in ``problems.py`` and the
reference artifacts, never in the pipeline's reach, and ``problem.md`` carries no
formula, no reference value and no benchmark path.
"""
import glob
import json
import os

import pytest

from verifylib.leakage import citation_scan

from .conftest import REPO, P


def _significant(value, digits=6):
    """The leading digits of a number, as they would appear if it were pasted."""
    return f"{abs(float(value)):.{digits - 1}e}".split("e")[0].replace(".", "")


def test_no_closed_form_problem_statements_are_leakage_free(ncf_problems):
    for p in ncf_problems:
        text = p["description"]
        assert citation_scan(text) == [], p["slug"]
        assert "references/" not in text and ".npz" not in text, p["slug"]
        for name, value in (p.get("functional_truth") or {}).items():
            digits = _significant(value)
            assert digits[:5] not in text.replace(".", ""), (
                f"{p['slug']}: problem.md appears to quote {name} = {value}")
        if p.get("reference_error"):
            assert _significant(p["reference_error"])[:4] not in text.replace(".", ""), p["slug"]


def test_no_closed_form_problem_statements_say_so(ncf_problems):
    """The statement must not hand the pipeline a closed form it could grade
    itself against. Each pilot problem says, in its own words, that none exists."""
    for p in ncf_problems:
        text = p["description"].lower()
        assert "exact solution" not in text or "no exact solution" in text, p["slug"]
        assert "u(x, t) =" not in text and "u(x,t) =" not in text, p["slug"]


def test_pipeline_settings_deny_benchmark_wholesale():
    with open(os.path.join(REPO, "benchmark", "pipeline-settings.json")) as fh:
        deny = json.load(fh)["permissions"]["deny"]
    assert "Read(./benchmark/**)" in deny
    assert "Edit(./benchmark/**)" in deny and "Write(./benchmark/**)" in deny
    for cmd in ("cat", "head", "tail", "sed", "awk", "grep", "rg", "python", "python3"):
        assert any(d.startswith(f"Bash({cmd}") and "benchmark/" in d for d in deny), cmd


@pytest.mark.parametrize("prob", P.no_closed_form_problems(), ids=lambda p: p["slug"])
def test_staged_workspace_specs_take_the_route_the_problem_allows(prob):
    """A Route-R problem (harness-owned reference field) has no formula anyone can
    write down, so its spec must claim none and the kernel must take Path B; a
    formulator that 'found' a closed form for a chaotic PDE is the drift this
    catches. A Route-S problem is different by design -- NO_CLOSED_FORM_CANDIDATES
    §4 says Cole-Hopf 'sits on the boundary of the definition' and a solver that
    evaluates the quadrature 'would pass' -- so there the spec is only required to
    be leakage-free, and what the formulator chose is reported, not failed."""
    from verifylib.reference import claimed_fields

    path = os.path.join(REPO, "workspace", prob["slug"], "problem_spec.json")
    if not os.path.exists(path):
        pytest.skip(f"{prob['slug']} not staged in workspace/")
    with open(path) as fh:
        spec = json.load(fh)
    assert citation_scan(json.dumps(spec)) == [], prob["slug"]
    if prob["type"] != "pde":
        assert not (spec.get("analytic_moments") or {}).get("has_analytic_solution")
        return
    claimed = claimed_fields(spec)
    if prob.get("ground_truth_kind") == "reference":
        assert not claimed, f"{prob['slug']}: the spec claims a closed form for a Route-R problem"
    elif claimed:
        print(f"\n{prob['slug']}: the formulator wrote a semi-analytic closed form "
              f"({len(json.dumps(claimed))} chars); the kernel takes Path A here")


def test_no_staged_plan_for_a_no_closed_form_problem_reads_the_reference():
    """Belt and braces on the settings denial: no solver under a no-closed-form
    workspace mentions the answer key."""
    for prob in P.no_closed_form_problems():
        for path in glob.glob(os.path.join(REPO, "workspace", prob["slug"], "plans", "*", "*.py")):
            with open(path) as fh:
                assert citation_scan(fh.read(), path=path) == [], path
