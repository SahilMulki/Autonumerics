"""The externally anchored suite (plan-benchop-slv.md step 7).

Ten SDE problems: seven whose answer key a third party published in 2018, and
three shifted twins whose answer key this repo computes, which are section 5's
memorization control.

  M1  every value an entry is graded on comes from the ledger, not from retyping
  M2  the contract the solver is handed states everything it is graded on
  M3  the twins pair with a published sibling and differ in exactly one number
  M4  the settings are the ones the calibration measured, and the two-level check
      can actually run at them
  M5  (step 8) every entry's gate was calibrated at the resolution that entry
      ships with, and separates a correct scheme from a plausible wrong one
  M6  (step 9) what staging writes to `workspace/{slug}/problem.md` is the call
      the verifier will make, carries no value the entry is graded on, and lands
      in a manifest that says which suite it belongs to

The leakage rules for these statements -- no reference value, no benchmark name,
no hint that a twin is a twin -- live in ``test_leakage_ncf.py`` beside the other
suite's, because they are the same two rules.
"""
import json
import os

import pytest

from .conftest import BENCHMARK, P, S, V

LEDGER = os.path.join(BENCHMARK, "references", "benchop_slv.json")
SUITE = P.benchop_problems()


def ledger():
    with open(LEDGER) as fh:
        return json.load(fh)


pytestmark = pytest.mark.skipif(not os.path.exists(LEDGER), reason="ledger not built")


# --- M1: the numbers come from the ledger ---------------------------------------

def test_the_suite_is_the_size_step_seven_authored():
    assert len(SUITE) == 10
    twins = [p for p in SUITE if p.get("shifted_twin_of")]
    assert len(twins) == 3 and len(SUITE) - len(twins) == 7


def test_every_published_entry_is_graded_on_the_ledgers_published_values():
    """Not on a number somebody retyped. The ledger is where the derivation, the
    routes and the agreement live, so an entry that disagrees with it is an entry
    whose truth nothing checked."""
    doc = ledger()["problems"]
    seen = set()
    for prob in SUITE:
        if prob.get("shifted_twin_of"):
            continue
        key = prob["benchop_key"]
        seen.add(key)
        want = doc[key]["published"]
        got = list(prob["functional_truth"].values())
        assert got == pytest.approx(want, rel=0, abs=0), prob["slug"]
    # QLSVeuCall is deliberately absent: its published value is the far-field
    # convention's, not E[(S_T-K)^+], so it cannot grade a sampler at all.
    assert seen == set(doc) - {"qlsv_eucall"}


def test_every_twin_is_graded_on_the_ledgers_computed_value():
    doc = ledger()["shifted"]
    for prob in SUITE:
        key = prob.get("benchop_key")
        if not prob.get("shifted_twin_of"):
            continue
        got = list(prob["functional_truth"].values())
        assert got == pytest.approx(doc[key]["value"], rel=0, abs=0), prob["slug"]


def test_the_declared_reference_uncertainty_is_the_ledgers():
    doc = ledger()
    for prob in SUITE:
        key = prob["benchop_key"]
        src = (doc["shifted"] if prob.get("shifted_twin_of") else doc["problems"])[key]
        declared = prob.get("reference_abs_err") or 0.0
        assert float(declared) == pytest.approx(src["reference_abs_err"], abs=1e-12), \
            prob["slug"]


def test_only_the_three_factor_problem_has_a_published_value_that_is_uncertain():
    """Of the seven published entries, exactly one is not quoted to fifteen digits.
    The twins all carry an uncertainty too, but theirs is this repo's own and is a
    different claim -- which is why they are counted apart."""
    published = [p for p in SUITE if not p.get("shifted_twin_of")]
    loose = sorted(p["slug"] for p in published if p.get("reference_abs_err"))
    assert loose == ["sde_hhw_call_corr"]
    twins = [p for p in SUITE if p.get("shifted_twin_of")]
    assert sum(1 for p in twins if p.get("reference_abs_err")) == 2  # the third is exact


# --- M2: the contract states what the solver is graded on -----------------------

@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_contract_names_every_quantity_and_its_own_gate(prob):
    footer = S.contract_footer(prob)
    tols = V._functional_tols(prob)
    for name in prob["functional_truth"]:
        assert f"`{name}`" in footer, (prob["slug"], name)
    if len(set(tols.values())) > 1:
        for name, tol in tols.items():
            assert f"`{name}` within {tol * 100:g}%" in footer, (prob["slug"], name)
    else:
        assert f"within {next(iter(tols.values())) * 100:g}% (relative)" in footer


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_contract_states_the_second_call_and_the_movement_rule(prob):
    footer = S.contract_footer(prob)
    n_lo = V._sde_level_paths(prob, prob["num_paths"])
    assert f"num_paths={n_lo}" in footer
    assert "is a constant, not an estimate, and\nfails" in footer.replace("  ", " ") \
        or "is a constant, not an estimate" in footer


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_a_barrier_problem_says_the_monitoring_is_continuous(prob):
    """The whole discriminator. Deciding the barrier once per step is biased high
    at O(sqrt(dt)); a solver not told that monitoring is continuous would be failed
    for a reading of the problem the statement allowed."""
    if "dnt" not in prob["slug"]:
        return
    text = prob["description"].lower()
    assert "continuous" in text and "not once per step" in text


# --- M3: the twins ---------------------------------------------------------------

def test_each_twin_names_a_sibling_that_is_in_the_suite():
    slugs = {p["slug"] for p in SUITE}
    for prob in SUITE:
        sib = prob.get("shifted_twin_of")
        if sib:
            assert sib in slugs, prob["slug"]


def test_a_twin_differs_from_its_sibling_in_exactly_one_declared_number():
    """A twin that changed two things would not isolate anything, and one that
    changed the horizon or the payoff would be a different problem rather than the
    same question with a different answer."""
    by_slug = {p["slug"]: p for p in SUITE}
    compared = ("T", "dt", "num_paths", "seed", "state_dimension", "functional_tol")
    for prob in SUITE:
        sib = prob.get("shifted_twin_of")
        if not sib:
            continue
        other = by_slug[sib]
        for k in compared:
            assert prob.get(k) == other.get(k), (prob["slug"], k)
        assert list(prob["functional_truth"]) == list(other["functional_truth"])


def test_a_twins_answer_moves_by_far_more_than_its_gate():
    """Section 5's suggested shift failed this: it moved two of three values by
    *less* than the tolerance, so a recalled constant would have passed the
    control. Measured on the shifts actually used, the movement is 4.8x the gate
    at worst and 34.6x at best; the bar below sits under the measured worst so
    that a later change which erodes a twin fails here rather than quietly
    producing a control that controls nothing."""
    by_slug = {p["slug"]: p for p in SUITE}
    for prob in SUITE:
        sib = prob.get("shifted_twin_of")
        if not sib:
            continue
        tols = V._functional_tols(prob)
        base = by_slug[sib]["functional_truth"]
        for name, value in prob["functional_truth"].items():
            moved = abs(value - base[name]) / abs(base[name])
            assert moved > 4.0 * tols[name], (prob["slug"], name, moved, tols[name])


# --- M4: the settings the calibration measured -----------------------------------

@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_step_the_verifier_uses_is_the_one_the_entry_declares(prob):
    """``sde_verify_dt`` caps the step at 0.04/T. An entry whose declared dt is
    coarser than that cap is graded at a step it never asked for, and the path
    count it was calibrated at would be wrong."""
    assert V.sde_verify_dt(prob) == prob["dt"]


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_two_level_check_has_a_coarser_level_to_run(prob):
    n_lo = V._sde_level_paths(prob, prob["num_paths"])
    assert 2 <= n_lo < prob["num_paths"], prob["slug"]


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_each_entry_is_scored_on_three_numbers_and_nothing_else(prob):
    assert len(prob["functional_truth"]) == 3
    assert prob.get("moments") is None
    assert prob["has_ground_truth"] is True
    assert prob["type"] == "sde" and prob["suite"] == "benchop"


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_manifest_entry_records_the_gate_each_value_is_held_to(prob):
    entry = S.manifest_entry(prob)
    assert entry["functional_truth"] == prob["functional_truth"]
    tols = V._functional_tols(prob)
    # Only when the gates actually differ. Declaring an uncertainty is not the same
    # as being widened by it: on most of these the reference is precise enough that
    # the floor lands below the declared gate and changes nothing.
    if len(set(tols.values())) > 1:
        assert entry["functional_tol_used"] == pytest.approx(tols)
    else:
        assert "functional_tol_used" not in entry


# --- M5: the gate was calibrated, at this resolution, against both halves -------
#
# Step 7 set these gates from what a correct scheme reached. Step 8 measured the
# other half of the repo's rule -- that a plausible wrong scheme must fail -- and
# stored it in the ledger under ``calibration``, because running ten problems at
# their shipped path counts takes half an hour. The tests here hold the stored
# measurement to the entry as it stands today, so a gate edited without re-running
# the calibration is caught rather than silently un-evidenced.

def calibration():
    return ledger().get("calibration", {})


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_every_entry_has_a_calibration_taken_at_its_own_resolution(prob):
    rec = calibration().get(prob["benchop_key"])
    assert rec, f"{prob['slug']}: uv run python benchmark/benchop.py --calibrate"
    assert rec["paths"] == prob["num_paths"]
    assert rec["steps"] == round(prob["T"] / V.sde_verify_dt(prob))
    tols = V._functional_tols(prob)
    assert rec["gate"] == pytest.approx([tols[n] for n in prob["functional_truth"]],
                                        rel=0, abs=0)
    assert rec["labels"] == list(prob["functional_truth"])


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_a_correct_scheme_clears_the_gate_on_every_value(prob):
    rec = calibration()[prob["benchop_key"]]
    right = next(s for s in rec["schemes"] if s["role"] == "correct")
    assert max(right["gate_ratio"]) < 1.0, (prob["slug"], right["name"],
                                            right["gate_ratio"])


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_a_plausible_wrong_scheme_fails(prob):
    """The harness fails a problem when any one reported value misses its gate, so
    that is the bar here. How many of the three miss is recorded in the ledger and
    read in plan-benchop-slv.md; it is not the pass condition."""
    rec = calibration()[prob["benchop_key"]]
    wrong = next(s for s in rec["schemes"] if s["role"] == "wrong")
    assert max(wrong["gate_ratio"]) > 1.0, (prob["slug"], wrong["name"],
                                            wrong["gate_ratio"])


@pytest.mark.parametrize("prob", [p for p in SUITE if p.get("shifted_twin_of")],
                         ids=lambda p: p["slug"])
def test_a_twin_is_calibrated_on_the_same_decision_as_its_sibling(prob):
    """A twin is its sibling with one initial condition moved, so the scheme that
    is right there is right here. If the shift changed which decision the problem
    turns on, the twin would be a different problem and not a control."""
    cal = calibration()
    sib = P.by_slug(prob["shifted_twin_of"])
    mine, theirs = cal[prob["benchop_key"]], cal[sib["benchop_key"]]
    assert mine["decision"] == theirs["decision"]
    assert [s["name"] for s in mine["schemes"]] == [s["name"] for s in theirs["schemes"]]


# --- M6: what staging actually writes (step 9) ----------------------------------
#
# M2 checks `contract_footer`, which is half of the file. The agent is handed the
# other half concatenated to it, by `write_problem_md`, and that join is the only
# artifact the pipeline ever sees. Step 9 found the join untested: every leakage
# rule in `test_leakage_ncf.py` reads `description`, so a value printed by the
# footer -- the half that is generated rather than written -- would have passed
# every check in the repo.

@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_staged_brief_is_the_call_the_verifier_will_make(prob, tmp_path):
    """The step in the brief is the step `sde_verify_dt` imposes, not the one the
    entry declares, and the path counts are both of the two the grade uses. A brief
    that named any other number would be asking for a solver the harness then
    refuses to run at those settings."""
    path = S.write_problem_md(prob, str(tmp_path))
    with open(path) as fh:
        text = fh.read()
    dt = V.sde_verify_dt(prob)
    assert (f"solve_sde(num_paths={prob['num_paths']}, dt={dt:g}, "
            f"T={prob['T']:g}, seed={prob['seed']})") in text, prob["slug"]
    assert f"num_paths={V._sde_level_paths(prob, prob['num_paths'])}" in text
    for name in prob["functional_truth"]:
        assert f"`{name}`" in text, (prob["slug"], name)


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_staged_brief_quotes_no_value_it_is_graded_on(prob, tmp_path):
    with open(S.write_problem_md(prob, str(tmp_path))) as fh:
        flat = fh.read().replace(".", "").replace(",", "")
    for name, value in prob["functional_truth"].items():
        digits = f"{abs(float(value)):.4e}".split("e")[0].replace(".", "")
        assert digits not in flat, f"{prob['slug']}: the staged brief quotes {name}"


@pytest.mark.parametrize("prob", SUITE, ids=lambda p: p["slug"])
def test_the_manifest_records_which_suite_the_entry_is_scored_in(prob):
    """Three suites, three denominators. `closed_form: False` made the second one
    visible in `manifest.json`; until step 9 the third was not, so all ten of these
    read as members of the 49 whose baseline they must never move."""
    assert S.manifest_entry(prob)["suite"] == "benchop"


def test_the_manifest_counts_the_three_suites_apart(tmp_path):
    path = S.write_manifest(list(P.PROBLEMS), str(tmp_path / "manifest.json"))
    with open(path) as fh:
        manifest = json.load(fh)
    assert manifest["n_by_suite"] == {"main": 49, "no-closed-form": 7, "benchop": 10}
    assert sum(manifest["n_by_suite"].values()) == manifest["n_total"]
    assert not any("suite" in e for e in manifest["problems"]
                   if e["slug"] not in {p["slug"] for p in SUITE})
