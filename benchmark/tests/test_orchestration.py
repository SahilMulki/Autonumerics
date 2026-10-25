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
    """Each suite runs on its own by default, so the baseline denominators the
    C0/C1/C2 rates are quoted against never move. There are three of them since
    step 7 added the externally anchored one, and the partition claim is the same
    claim -- every problem in exactly one suite, and the older two unchanged."""
    everything = list(P.PROBLEMS)
    parts = {name: S.apply_suite_filter(everything, "", suite=name) for name in S.SUITES}
    slugs = {name: {p["slug"] for p in part} for name, part in parts.items()}
    assert set().union(*slugs.values()) == {p["slug"] for p in everything}
    assert sum(map(len, slugs.values())) == len(everything)      # disjoint
    main, ncf = parts["main"], parts["no-closed-form"]
    # `--no-closed-form` is the older spelling and still selects the same suite.
    assert S.apply_suite_filter(everything, "", no_closed_form=True) == ncf
    assert S.apply_suite_filter(everything, "", no_closed_form=False) == main
    assert [p["slug"] for p in ncf] == [p["slug"] for p in P.no_closed_form_problems()]
    assert [p["slug"] for p in parts["benchop"]] == [p["slug"] for p in P.benchop_problems()]
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
    monkeypatch.setattr(RUN, "REPO_ROOT", str(tmp_path))
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


def test_isolated_settings_fences_every_archived_workspace_including_own_slug(tmp_path, monkeypatch):
    """Found by reading a transcript: the porous-medium formulator read
    ``workspace_c2b_sonnet_backup/pde_laplace_2d/problem_spec.json`` with the Read
    tool. The sibling rule is scoped to ``workspace/``, so an archive beside it was
    open -- and an archive holds a previous run's finished spec for the *same*
    problem, which is the one file a fresh run must not see."""
    ws = tmp_path / "workspace"
    (ws / "beta").mkdir(parents=True)
    for archive in ("workspace_old_backup", "workspace_oneshot"):
        (tmp_path / archive / "beta").mkdir(parents=True)
    (tmp_path / "workspace_notes.txt").write_text("")   # not a directory: no fence
    (tmp_path / "workspaces_unrelated").mkdir()          # no underscore prefix: no fence
    monkeypatch.setattr(RUN, "REPO_ROOT", str(tmp_path))
    monkeypatch.setattr(RUN, "WORKSPACE_ROOT", str(ws))

    assert RUN.archived_workspaces() == ["workspace_old_backup", "workspace_oneshot"]
    path = RUN.isolated_settings("beta", str(tmp_path / "logs"))
    with open(path) as fh:
        deny = json.load(fh)["permissions"]["deny"]

    for archive in ("workspace_old_backup", "workspace_oneshot"):
        assert f"Read(./{archive}/**)" in deny
        assert f"Read(//Users/**/Autonumerics/{archive}/**)" in deny
        assert f"Edit(./{archive}/**)" in deny
        for cmd in RUN._WORKSPACE_READERS:
            assert f"Bash({cmd}:*{archive}/*)" in deny, cmd
    assert not any("workspace_notes" in d or "workspaces_unrelated" in d for d in deny)
    assert not any(d.startswith(("Read(./workspace/", "Edit(./workspace/")) for d in deny), \
        "the run's live workspace stays readable; only the archived copy of it is fenced"


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

    # A kernel that reports `certified` carries it through; one that does not (a plan
    # scored before the field existed) omits it rather than recording None, so
    # `report._pipeline_claimed_pass` can tell "did not certify" from "never asked".
    certified = json.loads(good)
    certified["certification"]["certified"] = True
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: Proc(0, json.dumps(certified)))
    assert RUN.kernel_metrics(str(plan))["certified"] is True


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


# --- E7: a non-main sweep cannot land in the standing results file (step 9) -------
#
# The accident `results/README.md` names: a sweep launched with no `--results-json`
# overwrote the live baseline, 44 units where there had been hundreds. What came out
# of it was a *convention*, which is to say a thing a person has to remember at the
# moment they are launching an expensive run. Step 9 made it a refusal.

def test_the_guard_refuses_a_suite_it_would_have_to_merge_into_main():
    for suite in ("benchop", "no-closed-form"):
        chosen = S.apply_suite_filter(list(P.PROBLEMS), "", suite=suite)
        with pytest.raises(SystemExit) as e:
            S.guard_results_path(chosen, None, "results/{suite}.json")
        assert suite in str(e.value) and "--results-json" in str(e.value)


def test_the_guard_names_a_path_that_follows_the_naming_convention():
    chosen = P.benchop_problems()
    with pytest.raises(SystemExit) as e:
        S.guard_results_path(chosen, None, "benchmark/results/pipeline/{suite}.json")
    assert "benchmark/results/pipeline/benchop.json" in str(e.value)
    # The hyphen in a suite name is not a filename this repo writes.
    with pytest.raises(SystemExit) as e:
        S.guard_results_path(S.apply_suite_filter(list(P.PROBLEMS), "", suite="no-closed-form"),
                             None, "benchmark/results/pipeline/{suite}.json")
    assert "pipeline/no_closed_form.json" in str(e.value)


def test_the_guard_leaves_every_main_suite_invocation_alone():
    """Including a one-problem smoke test, which is the invocation the convention
    was most likely to be forgotten on and the one least able to do harm."""
    main = S.apply_suite_filter(list(P.PROBLEMS), "", suite="main")
    assert S.guard_results_path(main, None, "x/{suite}.json") is None
    assert S.guard_results_path(main[:1], None, "x/{suite}.json") is None
    # An explicit path is the whole point: with one, anything may be written.
    assert S.guard_results_path(P.benchop_problems(), "somewhere.json",
                                "x/{suite}.json") is None


def test_a_mixed_selection_is_refused_too():
    """`--all-problems` runs the three suites together, so it has no single
    denominator at all and least of all the main one's."""
    with pytest.raises(SystemExit):
        S.guard_results_path(list(P.PROBLEMS), None, "x/{suite}.json")


def test_the_runner_refuses_before_it_reads_or_writes_anything(capsys):
    """Through `main`, not the helper: the guard has to sit after `--list` (which
    writes nothing and must keep working) and before `load_prior`."""
    with pytest.raises(SystemExit) as e:
        RUN.main(["--suite", "benchop"])
    assert "--results-json" in str(e.value)
    assert RUN.main(["--suite", "benchop", "--list"]) is None
    assert "10 problem(s) selected." in capsys.readouterr().out


# --- E8: the conductor's background tasks are not swept mid-cycle (step 10) -------

def test_the_conductor_is_given_an_unbounded_background_wait(monkeypatch, tmp_path):
    """Headless `claude -p` terminates still-running background tasks 600 s after
    the main thread closes. On this suite one evaluator pass costs minutes, so the
    first BENCHOP run came back with phase `running` and self-score 0 on a problem
    whose plans the harness then graded a verified pass -- a zero that says nothing
    about the pipeline. The run is bounded by `--timeout` instead."""
    seen = {}

    def fake_run(cmd, **kw):
        seen.update(kw)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(RUN, "LOGS_DIR", str(tmp_path))
    monkeypatch.setattr(RUN, "isolated_settings", lambda slug, d: str(tmp_path / "s.json"))
    monkeypatch.setattr(subprocess, "run", fake_run)
    run = RUN.run_conductor("sde_heston_dnt", 3600, True, None, None)

    assert seen["env"]["CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS"] == "0"
    # 0 disables the sweep; any other number would be a new, smaller ceiling.
    assert seen["timeout"] == 3600
    assert run["status"] == "completed"
    # The rest of the environment is inherited, not replaced -- PATH included, or
    # the `claude` the subprocess finds would not be the one on this machine.
    assert set(os.environ) <= set(seen["env"])


def test_the_run_config_says_which_ceiling_the_sweep_was_made_under():
    """A sweep made under the default ceiling can have been cut off mid-cycle, and
    nothing else in the record distinguishes it from one that was not."""
    assert RUN.CONDUCTOR_ENV == {"CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS": "0"}
