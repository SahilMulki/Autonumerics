"""The runner, the suite split and the report -- the parts of the harness that
decide *what* gets graded and *how the verdict is read*, none of which touch a
solver. Pure functions, monkeypatched where they would shell out.
"""
import json
import os
import subprocess

import pytest

from .conftest import RUN, P, R, S, problem, record

# --- E1: the suite split ---------------------------------------------------------

def test_apply_suite_filter_partitions_the_suites():
    """The no-closed-form problems run on their own by default, so the baseline
    denominators the C0/C1/C2 rates are quoted against never move."""
    everything = list(P.PROBLEMS)
    main = S.apply_suite_filter(everything, "", no_closed_form=False)
    ncf = S.apply_suite_filter(everything, "", no_closed_form=True)
    assert {p["slug"] for p in main} | {p["slug"] for p in ncf} == {p["slug"] for p in everything}
    assert not ({p["slug"] for p in main} & {p["slug"] for p in ncf})
    assert [p["slug"] for p in ncf] == [p["slug"] for p in P.no_closed_form_problems()]
    assert all(p.get("closed_form") is not False for p in main)
    # An explicit request always wins, whichever suite the problem is in.
    chosen = [problem("pde_kuramoto_sivashinsky"), problem("pde_heat_1d")]
    assert S.apply_suite_filter(chosen, "pde_kuramoto_sivashinsky,pde_heat_1d") == chosen
    assert S.apply_suite_filter(everything, "", include_all=True) == everything


def test_kuramoto_sivashinsky_moved_to_the_no_closed_form_suite():
    """It was the one SELF_ONLY problem in the 50; with a harness reference it is
    graded, and it lives in the suite whose denominators it belongs to."""
    ks = problem("pde_kuramoto_sivashinsky")
    assert ks["closed_form"] is False and ks["ground_truth_kind"] == "reference"
    assert ks["has_ground_truth"] is True


# --- E2: per-run isolation -------------------------------------------------------

def test_isolated_settings_denies_every_sibling_workspace_and_not_its_own(tmp_path, monkeypatch):
    """Found by reading a transcript: a formulator ``cat``ed its matched twin's
    ``problem_spec.json`` and used it as a template. That collapses the matched-pair
    design, so each run sees exactly one workspace."""
    ws = tmp_path / "workspace"
    for slug in ("alpha", "beta", "gamma"):
        (ws / slug).mkdir(parents=True)
    (ws / "not_a_dir.txt").write_text("")
    monkeypatch.setattr(RUN, "WORKSPACE_ROOT", str(ws))
    with open(RUN.PIPELINE_SETTINGS) as fh:
        base = json.load(fh)["permissions"]["deny"]

    path = RUN.isolated_settings("beta", str(tmp_path / "logs"))
    assert path == str(tmp_path / "logs" / "beta.settings.json")
    with open(path) as fh:
        deny = json.load(fh)["permissions"]["deny"]

    assert deny[:len(base)] == base, "the base denials must survive verbatim, in order"
    extra = deny[len(base):]
    assert not any("workspace/beta/" in d for d in extra), "a run must see its own workspace"
    for other in ("alpha", "gamma"):
        assert f"Read(./workspace/{other}/**)" in extra
        assert f"Read(//Users/**/Autonumerics/workspace/{other}/**)" in extra
        assert f"Edit(./workspace/{other}/**)" in extra
        for cmd in RUN._WORKSPACE_READERS:
            assert f"Bash({cmd}:*workspace/{other}/*)" in extra, cmd
    assert not any("not_a_dir" in d for d in extra)


def test_isolated_settings_is_threaded_into_the_conductor_command(tmp_path):
    cmd = RUN.conductor_command("beta", False, None, None, settings=str(tmp_path / "s.json"))
    i = cmd.index("--settings")
    assert cmd[i + 1] == str(tmp_path / "s.json")
    assert RUN.conductor_command("beta", False, None, None)[i + 1] == RUN.PIPELINE_SETTINGS


# --- E3: the usage-limit detector ------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Tier B route: degenerate limit validated the operator", False),
    ("the evaluator's rate limiter is not involved", False),
    ("You have hit your usage limit. Resets at 3pm.", True),
    ("error: rate limit exceeded", True),
    ("Insufficient credit balance", True),
])
def test_looks_like_limit_matches_whole_words_only(tmp_path, text, expected):
    """'degene|rate limit' is the kernel's own vocabulary; a successful transcript
    that names its Tier-B route must not halt the whole queue."""
    log = tmp_path / "run.log"
    log.write_text(text)
    assert RUN._looks_like_limit(str(log)) is expected


def test_looks_like_limit_on_a_missing_log_is_false(tmp_path):
    assert RUN._looks_like_limit(str(tmp_path / "nope.log")) is False


# --- E4: the kernel read never breaks the run -------------------------------------

def test_kernel_metrics_never_raises_and_labels_each_failure(tmp_path, monkeypatch):
    assert RUN.kernel_metrics(None) is None
    assert RUN.kernel_metrics(str(tmp_path / "missing")) is None

    plan = tmp_path / "plan"
    plan.mkdir()

    class Proc:
        def __init__(self, code, out, err=""):
            self.returncode, self.stdout, self.stderr = code, out, err

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc(1, "", "boom " * 200))
    got = RUN.kernel_metrics(str(plan))
    assert got["status"] == "unavailable" and len(got["error"]) <= 400

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc(0, "not json"))
    assert RUN.kernel_metrics(str(plan))["status"] == "error"

    def timeout(*a, **k):
        raise subprocess.TimeoutExpired(cmd="x", timeout=1)
    monkeypatch.setattr(subprocess, "run", timeout)
    assert RUN.kernel_metrics(str(plan), timeout_s=1) == {"status": "timeout", "timeout_s": 1}

    good = json.dumps({"score": 9, "provenance": "manufactured_partial",
                       "certification": {"tier": "B", "bound_by": "certification",
                                         "reason": "one break"}, "agent_cap": None})
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: Proc(0, good))
    got = RUN.kernel_metrics(str(plan))
    assert got == {"status": "ok", "score": 9, "provenance": "manufactured_partial",
                   "tier": "B", "bound_by": "certification", "reason": "one break",
                   "agent_cap": None}


# --- E5: the verdict on the no-closed-form shapes ---------------------------------

@pytest.mark.parametrize("kind, best_score, verify, expected", [
    ("reference", 10, {"status": "ok", "passed": False}, "OVERCLAIM"),
    ("reference", 8, {"status": "ok", "passed": True}, "UNDERCLAIM"),
    ("reference", 10, {"status": "ok", "passed": True}, "VERIFIED_PASS"),
    ("reference", 8, {"status": "ok", "passed": False}, "FAIL"),
    ("reference", 10, {"status": "ok", "passed": False, "constraint_violation": True},
     "CONSTRAINT_VIOLATION"),
    ("reference", 10, {"status": "inconclusive", "error": "x"}, "UNVERIFIED"),
    ("reference", 10, {"status": "error", "error": "x"}, "UNVERIFIED"),
    ("exact", 10, {"status": "ok", "passed": False}, "OVERCLAIM"),
])
def test_compute_verdict_on_the_no_closed_form_shapes(kind, best_score, verify, expected):
    prob = {**problem("pde_kuramoto_sivashinsky"), "ground_truth_kind": kind}
    rec = record(prob, best_score=best_score, verify=verify)
    assert R.compute_verdict(rec) == expected


def test_a_run_that_did_not_complete_is_never_a_verdict_about_the_solver():
    prob = problem("pde_kuramoto_sivashinsky")
    assert R.compute_verdict(record(prob, run_status="timeout")) == "RUN_ERROR"
    assert R.compute_verdict(record(prob, run_status="spec_blocked")) == "SPEC_BLOCKED"


def test_self_only_is_reserved_for_problems_with_no_truth_at_all():
    prob = {**problem("pde_kuramoto_sivashinsky"), "has_ground_truth": False}
    assert R.compute_verdict(record(prob)) == "SELF_ONLY"
    assert not any(not p["has_ground_truth"] for p in P.no_closed_form_problems())


# --- E6: the report renders the cross-table --------------------------------------

def test_report_renders_the_kernel_provenance_cross_table():
    """A ``manufactured`` 10 on an OVERCLAIM row is a rubric bug, and this table
    is the only place the kernel's claim meets an independent verdict."""
    prob = problem("pde_kuramoto_sivashinsky")
    over = record(prob, best_score=10, verify={"status": "ok", "passed": False, "rel_l2_err": 0.3},
                  kernel={"status": "ok", "score": 10, "provenance": "manufactured",
                          "tier": "B", "bound_by": "rubric", "reason": "r", "agent_cap": None})
    under = record(problem("pde_burgers_viscous_1d"), best_score=8,
                   verify={"status": "ok", "passed": True, "rel_l2_err": 1e-4},
                   kernel={"status": "ok", "score": 8, "provenance": "self_convergence",
                           "tier": "C", "bound_by": "certification", "reason": "r",
                           "agent_cap": None})
    text = R.generate_report([over, under], {})
    assert "## Kernel provenance vs the independent verdict" in text
    section = text.split("## Kernel provenance vs the independent verdict")[1]
    assert "manufactured" in section and "OVERCLAIM" in section
    assert "self_convergence" in section and "UNDERCLAIM" in section
    assert "pde_kuramoto_sivashinsky" in section and "pde_burgers_viscous_1d" in section

    plain = R.generate_report([record(prob)], {})
    assert "Kernel provenance vs" not in plain


def test_the_provenance_vocabulary_is_one_set_everywhere():
    """§9d: seven values, read by report.py, compare.py, run.py and the conductor.
    Change them in one pass -- so a test pins that no consumer names a tag the
    kernel does not emit, and that the two documents which spell out the order
    spell out the kernel's."""
    import re

    from verifylib.kernel.score import PROVENANCE_ORDER

    root = os.path.dirname(os.path.dirname(os.path.abspath(R.__file__)))
    tag = re.compile(r"\b([a-z]+_(?:unvalidated|partial|convergence))\b")
    for rel in ("benchmark/report.py", "benchmark/compare.py", "benchmark/run.py",
                "commands/conductor.md", "references/project_manual.md"):
        with open(os.path.join(root, rel)) as fh:
            text = fh.read()
        assert set(tag.findall(text)) <= set(PROVENANCE_ORDER), rel

    ordered = " > ".join(f"`{p}`" for p in PROVENANCE_ORDER)
    for rel in ("commands/conductor.md", "references/project_manual.md"):
        with open(os.path.join(root, rel)) as fh:
            text = re.sub(r"\s+", " ", fh.read())
        assert ordered in text, f"{rel} does not state the kernel's provenance order verbatim"
