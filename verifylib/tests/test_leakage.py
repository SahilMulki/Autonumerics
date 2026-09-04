"""Acceptance criterion 8: the citation scan catches the archived incidents; the
narrowed AST scan produces no false positives on real pipeline code."""
import glob
import os

from verifylib.leakage import citation_scan, oracle_access_scan

from .conftest import FIXTURES, REPO, WORKSPACE, read_text


def test_citation_scan_catches_heston_spec():
    text = read_text(os.path.join(FIXTURES, "leak_heston_spec.json"))
    hits = citation_scan(text)
    assert "benchmark/" in hits and "problems.py" in hits and "_heston_call" in hits


def test_citation_scan_catches_quintic_docstring():
    text = read_text(os.path.join(FIXTURES, "leak_quintic_evaluate.py"))
    hits = citation_scan(text)
    assert "verify.py" in hits and "verify_sde_stability" in hits


def test_ast_scan_is_blind_to_both_incidents():
    """Documented, not incidental: this is why the citation scan exists."""
    src = read_text(os.path.join(FIXTURES, "leak_quintic_evaluate.py"))
    assert oracle_access_scan(src) == []


def test_ast_scan_catches_executable_access():
    assert oracle_access_scan("import problems") == ["import problems"]
    # reports both names: the module and the prohibited symbol pulled from it
    assert oracle_access_scan("from benchmark import verify") == ["import benchmark", "import verify"]
    hits = oracle_access_scan("d = read_text('../benchmark/problems.py')")
    assert hits and "benchmark" in hits[0]


def test_ast_scan_has_no_false_positives_on_pipeline_code():
    """Measured: the original rule flagged 124/137; the narrowed rule flags 0."""
    files = (glob.glob(os.path.join(WORKSPACE, "*", "plans", "*", "solver.py"))
             + glob.glob(os.path.join(WORKSPACE, "*", "plans", "*", "evaluate.py")))
    flagged = []
    for p in files:
        try:
            if oracle_access_scan(read_text(p)):
                flagged.append(p)
        except SyntaxError:
            continue
    assert flagged == [], f"false positives: {flagged}"


def test_project_manual_is_allowlisted():
    path = os.path.join(REPO, "references", "project_manual.md")
    text = read_text(path)
    assert citation_scan(text) != []                                  # it does name them
    assert citation_scan(text, path="references/project_manual.md") == []
