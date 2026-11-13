"""Thorough benchmark report generator.

Consumes ``benchmark/results/results.json`` (written by run.py) and renders a
Markdown report. The central idea is a *verdict* per problem that cross-checks the
pipeline's self-score against the independent ground-truth verification:

    VERIFIED_PASS  pipeline scored 10/10 AND the independent check confirms it
    OVERCLAIM      pipeline scored 10/10 but the independent check FAILS  (!!)
    UNDERCLAIM     pipeline gave up below 10 but the solution actually passes
    FAIL           neither the pipeline nor the independent check passes
    UNVERIFIED     independent check could not run (error/inconclusive)
    STABLE_PASS    stability problem: a correct scheme stayed finite / in-domain
    BLOWUP         stability problem: the solution diverged or escaped its domain (!!)
    SELF_ONLY      no ground truth exists; only the pipeline self-score is available
    SPEC_BLOCKED   the conductor's requirements gate halted the run: the formulator's
                   ledger did not carry every constraint in problem.md into the spec
    RUN_ERROR      the pipeline run itself failed or timed out

OVERCLAIM (and BLOWUP) are the findings a rigorous benchmark exists to surface: the
pipeline believing it succeeded when it did not. VERIFIED_PASS covers both exact and
reference (discretization-free MC) ground truth.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DEFAULT_RESULTS = os.path.join(HERE, "results", "results.json")
DEFAULT_REPORT = os.path.join(HERE, "results", "REPORT.md")

VERDICT_LABEL = {
    "VERIFIED_PASS": "PASS (verified)",
    "OVERCLAIM": "OVERCLAIM",
    "UNDERCLAIM": "UNDERCLAIM",
    "FAIL": "FAIL",
    "CONSTRAINT_VIOLATION": "CONSTRAINT VIOLATION",
    "UNVERIFIED": "UNVERIFIED",
    "STABLE_PASS": "STABLE (verified)",
    "BLOWUP": "BLOW-UP",
    "SELF_ONLY": "SELF-SCORE ONLY",
    "ANALYTIC_TRANSCRIPTION": "EXCLUDED (analytic transcription)",
    "SPEC_BLOCKED": "SPEC BLOCKED",
    "RUN_ERROR": "RUN ERROR",
}

# Verdicts that count as an independently-confirmed success.
#
# ANALYTIC_TRANSCRIPTION is deliberately in neither this tuple nor the failure list
# below. On a `pathwise` problem the exact solution is a closed form, so a solver can
# *return* it -- exact at every dt, consuming the supplied increments, varying
# correctly with the path count -- and be both perfectly accurate and not a scheme.
# Scoring that as a pass would put a transcription at the top of a comparison between
# discretizations; scoring it as a failure would call a correct answer wrong. It is an
# exclusion: the run is reported, named, and left out of the arm that compares schemes
# (plan-sciml-benchmark.md §0b, H2).
# What the **independent check** confirmed, whatever the pipeline claimed about it.
#
# This replaces `_PASS_VERDICTS`, which was `("VERIFIED_PASS", "STABLE_PASS")` and is
# removed rather than kept beside it, because its name read like "the passes" while it
# was the wrong set for every question about *accuracy* -- and it was being used for
# three of them: the executive summary's "Independently confirmed pass", the
# by-type/by-tier "Independently verified" column, and `compare.py`'s pipeline side.
#
# It was also not internally consistent, which is the clearest sign it was never one
# quantity: `VERIFIED_PASS` carries a self-score conjunct ("the harness passed it AND
# the pipeline claimed 10") and `STABLE_PASS` carries none at all, being returned off
# `ver["passed"]` alone on a stability problem. So the old set mixed a conjunction with
# a non-conjunction and could not have been measuring a single thing.
#
# The asymmetry that makes this a correctness bug rather than a naming one is stated
# fifteen lines down: **a baseline solver has no self-score**, so `_ONESHOT_PASS_VERDICTS`
# cannot carry the second conjunct and `UNDERCLAIM` is a verdict a baseline can never
# receive. Comparing one against the other charges the pipeline for a category the
# thing it is being compared to is structurally incapable of entering. Measured on the
# sciml arm (plan-sciml-benchmark.md SS10i): the pipeline reads **5/9** against the C0
# one-shot's **9/9**, while the harness independently passed **9 of 9** on both sides --
# a 4-entry deficit that is entirely the pipeline's self-score being conservative, which
# is the one direction of error this benchmark exists to *reward*.
#
# `UNDERCLAIM` is a subset of the verified passes by construction (`compute_verdict`
# returns it only where `verified_pass` is true), so adding it here cannot admit an
# unverified run. The exclusions stay excluded: `ANALYTIC_TRANSCRIPTION` and
# `CONSTRAINT_VIOLATION` are returned *before* the pass branches, so neither can reach
# this set.
_VERIFIED_VERDICTS = ("VERIFIED_PASS", "UNDERCLAIM", "STABLE_PASS")
# The order verdicts are listed in the distribution table.
_VERDICT_ORDER = ("VERIFIED_PASS", "STABLE_PASS", "OVERCLAIM", "BLOWUP",
                  "CONSTRAINT_VIOLATION", "UNDERCLAIM", "FAIL", "UNVERIFIED",
                  "SELF_ONLY", "ANALYTIC_TRANSCRIPTION", "SPEC_BLOCKED", "RUN_ERROR")

# --- one-shot / single-agent baseline verdicts ------------------------------
# A baseline solver has NO self-score (the pipeline's OVERCLAIM/UNDERCLAIM/10-of-10
# distinctions do not apply). It either independently passes or it does not, and a
# crash counts as a failure -- never as "unverified", because there is nothing to
# be unsure about. So the taxonomy collapses to: PASS / FAIL / CRASHED /
# STABLE_PASS / BLOWUP, plus NO_GT for the (rare) problem with no independent check
# at all (only Kuramoto-Sivashinsky today), which is simply excluded from the rate.
ONESHOT_VERDICT_LABEL = {
    "PASS": "PASS",
    "FAIL": "FAIL",
    "CONSTRAINT_VIOLATION": "CONSTRAINT VIOLATION",
    "CRASHED": "CRASHED",
    "STABLE_PASS": "STABLE",
    "BLOWUP": "BLOW-UP",
    "NO_GT": "NO GROUND TRUTH",
    "ANALYTIC_TRANSCRIPTION": "EXCLUDED (analytic transcription)",
}
# What counts as an independently-confirmed baseline success. ANALYTIC_TRANSCRIPTION is
# not here and not a failure either, exactly as on the pipeline side.
_ONESHOT_PASS_VERDICTS = ("PASS", "STABLE_PASS")
_ONESHOT_VERDICT_ORDER = ("PASS", "STABLE_PASS", "FAIL", "CONSTRAINT_VIOLATION",
                          "BLOWUP", "CRASHED", "ANALYTIC_TRANSCRIPTION", "NO_GT")


# --- verdict logic ----------------------------------------------------------


def harness_error(ver):
    """How far past its gate a verified result sits, as a multiple of the gate.

    One number for every grading shape, because the three readers of it in this file
    each had their own and one of them had none. ``1.0`` is the pass line, so ``0.4``
    is comfortable and ``1.2`` failed; `None` where nothing gradeable was recorded.

    F10. The per-plan table read ``rel_l2_err``, which **does not exist on a
    functional-graded problem**: every BENCHOP-SLV problem is scored on declared
    option prices, carried in ``functional_errs`` against the per-value gates in
    ``functional_tol_used``, and the whole suite printed ``-`` in the error column.
    Reconstructing those numbers by hand is what made the first analysis of that run
    expensive, and the fix is one function.

    Returns ``(value, label)``.
    """
    ver = ver or {}
    errs = ver.get("functional_errs") or {}
    if errs:
        # Per value, because the gate is widened per value where the reference's own
        # error demands it -- `verify._functional_tols`' rule, read rather than
        # re-derived. The worst functional decides, the way the pass does.
        tols = ver.get("functional_tol_used") or {}
        default = ver.get("functional_tol")
        ratios = [e / tols.get(n, default) for n, e in errs.items()
                  if isinstance(e, (int, float))
                  and isinstance(tols.get(n, default), (int, float))
                  and tols.get(n, default) > 0.0]
        if ratios:
            return max(ratios), "functional"
    if isinstance(ver.get("rel_l2_err"), (int, float)):
        tol = ver.get("rel_l2_tol") or ver.get("tol")
        if isinstance(tol, (int, float)) and tol > 0.0:
            return ver["rel_l2_err"] / tol, "rel_l2"
        return None, "rel_l2"
    if isinstance(ver.get("worst_variance_rel_err"), (int, float)):
        tol = ver.get("variance_rel_err_max")
        if isinstance(tol, (int, float)) and tol > 0.0:
            return ver["worst_variance_rel_err"] / tol, "variance"
    # The pathwise route carries neither a field error nor a moment error, so without
    # this branch every entry in the SciML suite would print `-` in the error column:
    # the F10 defect above, one route later. It is read after `functional_errs`
    # because a pathwise problem may also declare a quantity of interest, and the
    # worst gate decides there exactly as it does everywhere else.
    if isinstance(ver.get("pathwise_rel_err"), (int, float)):
        tol = ver.get("pathwise_tol_used")
        if isinstance(tol, (int, float)) and tol > 0.0:
            return ver["pathwise_rel_err"] / tol, "pathwise"
        return None, "pathwise"
    return None, "none"


def _pipeline_claimed_pass(rec):
    """Did the pipeline claim this problem solved?

    The question OVERCLAIM / UNDERCLAIM turn on is whether every gate the evidence
    path requires came back clean -- which is what the kernel now reports directly as
    ``certified``. It used to be spelled ``best_score >= 10``, and that spelling ties
    the benchmark's pass rate to one integer at the moment the score starts resolving
    *quality* below the tolerance line: a plan three decades inside the tolerance and
    one that squeaked under it both certify, and only one of them should score 10.

    Frozen results carry no ``certified``, so they fall back to the old test. The two
    agree on every record written before the split, which is what
    ``benchmark/tests`` pins.
    """
    pipeline = rec.get("pipeline") or {}
    for source in (pipeline, rec.get("kernel") or {}):
        if isinstance(source.get("certified"), bool):
            return source["certified"]
    return (pipeline.get("best_score") or 0) >= 10


def _is_transcription(ver, arm=None):
    """Did this run carry no discretization error at all, **and does that disqualify
    it**? (H2)

    One reading, used by both verdict functions. The baseline path needs it as much as
    the pipeline path does -- more, in fact: §10's prediction 2 is that the C0 baseline
    is *where* a transcribed answer is most likely, since a single completion with no
    tools is exactly the setting in which writing down the closed form is the obvious
    move. Until step 6 only `compute_verdict` had it, so a transcribing baseline would
    have been recorded as a clean PASS on every problem in the suite.

    **The arm decides whether it is an exclusion at all** -- added when the sciml runs
    were being prepared, because the exclusion was unconditional while `--arm` was a
    label, which made the `unrestricted` arm a second copy of the first experiment.
    In that arm a transcription is the *capability being measured* (§6: does the system
    recognise an exactly solvable SDE?), so it passes on its own merits and is not
    excluded from anything -- there is nothing to exclude it from, since that arm is a
    capability note rather than a comparison against another library's schemes. The
    detection still runs and is still recorded; only the verdict changes.
    """
    if arm == "unrestricted":
        return False
    ladder = (ver or {}).get("pathwise_ladder") or {}
    return ladder.get("status") == "ok" and bool(ladder.get("degenerate"))


def compute_verdict(rec):
    run = rec.get("run", {})
    # The requirements gate firing is a deliberate refusal, not a crash: the pipeline
    # declined to solve a problem its own spec could not represent. Reported apart from
    # RUN_ERROR so a spec regression is never read as flaky infrastructure.
    if run.get("status") == "spec_blocked":
        return "SPEC_BLOCKED"
    if run.get("status") != "completed":
        return "RUN_ERROR"

    kind = rec.get("ground_truth_kind", "exact")
    ver = rec.get("verify", {}) or {}
    vstatus = ver.get("status")

    # Stability problems have no moments (has_ground_truth is False) but are still
    # independently checkable: pass = stayed finite / in-domain.
    if kind == "stability":
        if vstatus != "ok":
            return "UNVERIFIED"
        return "STABLE_PASS" if ver.get("passed") else "BLOWUP"

    if not rec.get("has_ground_truth", False):
        return "SELF_ONLY"

    if vstatus in ("error", "inconclusive"):
        return "UNVERIFIED"

    # Accurate-but-structurally-wrong (div != 0, negative density, locked): surfaced
    # distinctly, like BLOWUP for SDE. A hard-gate constraint failure is reported
    # whatever the pipeline's self-score, because "looks accurate yet violates a
    # required structural constraint" is exactly the finding these problems exist for.
    if ver.get("constraint_violation"):
        return "CONSTRAINT_VIOLATION"

    pipeline_pass = _pipeline_claimed_pass(rec)
    verified_pass = bool(ver.get("passed", False))

    # H2, and the order here was wrong when it shipped in step 5 -- step 6's
    # calibration caught it. The exclusion applies only to a run that would
    # *otherwise have passed*: a solver carrying no discretization error is accurate
    # and is not a scheme, which is a different thing from being wrong.
    #
    # Read before the pass decision, a constant solver came back
    # ANALYTIC_TRANSCRIPTION -- its ladder is degenerate too, since a constant does
    # not change with `dt` either -- and that threw away the fact that it had *also*
    # missed its gate by up to 60x. A wrong solver is a FAIL whatever its ladder says;
    # the exclusion is for the one that is right and uninformative.
    if verified_pass and _is_transcription(ver, rec.get("arm")):
        return "ANALYTIC_TRANSCRIPTION"

    if verified_pass and pipeline_pass:
        return "VERIFIED_PASS"
    if verified_pass and not pipeline_pass:
        return "UNDERCLAIM"
    if pipeline_pass and not verified_pass:
        return "OVERCLAIM"
    return "FAIL"


def compute_oneshot_verdict(rec):
    """Verdict for a baseline (one-shot C0 / single-agent C1) solver.

    Unlike the pipeline, a baseline has no self-score, so this maps purely off the
    independent (sandboxed) verification:

        PASS         exact/reference moments (or PDE L2) within the gate
        FAIL         ran, but out of tolerance / non-finite / wrong output shape
        CRASHED      never produced a scorable run -- no solver, wrong signature,
                     exception, timeout, out-of-memory, or a blocked (scipy-in-SDE)
                     import. Counts as a failure in the pass rate, tracked apart.
        STABLE_PASS  stability problem: stayed finite / in-domain
        BLOWUP       stability problem: diverged / escaped its domain
        NO_GT        no independent check exists (only Kuramoto-Sivashinsky) --
                     excluded from the pass rate
        ANALYTIC_TRANSCRIPTION
                     accurate, and not a scheme: on a `pathwise` problem the solver
                     returned the closed form (or a constant) and carries no
                     discretization error, so it is excluded from the scheme
                     comparison rather than counted as a pass [step 6]

    A crash is deliberately a FAIL, not the pipeline's UNVERIFIED: when the task is
    "produce a working solver", failing to produce one is a loss."""
    kind = rec.get("ground_truth_kind", "exact")
    ver = rec.get("verify", {}) or {}
    vstatus = ver.get("status")

    if kind != "stability" and not rec.get("has_ground_truth", False):
        return "NO_GT"
    if vstatus == "crashed":
        return "CRASHED"
    if kind == "stability":
        if vstatus != "ok":
            return "CRASHED"
        return "STABLE_PASS" if ver.get("passed") else "BLOWUP"
    if ver.get("constraint_violation"):
        return "CONSTRAINT_VIOLATION"   # accurate-looking but broke a hard structural gate
    if vstatus == "ok":
        if not ver.get("passed"):
            return "FAIL"
        return ("ANALYTIC_TRANSCRIPTION"
                if _is_transcription(ver, rec.get("arm")) else "PASS")
    if vstatus in ("nonfinite", "inconclusive", "contract"):
        # It ran but violated the contract (NaN output, or a grid/shape mismatch):
        # the solver's fault, so a FAIL rather than a neutral CRASHED.
        return "FAIL"
    return "CRASHED"  # "error" or anything unexpected: could not be scored


def oneshot_passed(verdict):
    return verdict in _ONESHOT_PASS_VERDICTS


def key_metric(rec):
    """Short human string of the independent-verification metric."""
    ver = rec.get("verify", {}) or {}
    kind = rec.get("ground_truth_kind", "exact")
    if kind == "stability":
        if ver.get("status") != "ok":
            return ver.get("status", "-")
        parts = ["all-finite" if ver.get("finite") else "NON-FINITE"]
        if ver.get("check") == "domain":
            parts.append("in-domain" if ver.get("in_domain") else "ESCAPED-DOMAIN")
        if ver.get("max_abs") is not None:
            parts.append(f"max|X|={ver['max_abs']:.3g}")
        ff = ver.get("finite_fraction")
        if ff is not None and ff < 1.0:
            parts.append(f"finite_frac={ff:.5f}")
        return ", ".join(parts)
    if rec["type"] == "pde":
        if "rel_l2_err" in ver:
            label = "relL1" if ver.get("metric") == "l1" else "relL2"
            s = f"{label}={ver['rel_l2_err']:.2e}"
            if ver.get("order_checked"):
                p = ver.get("observed_order")
                s += f", p={'inf' if p == float('inf') else f'{p:.2f}'}"
            fe = ver.get("field_errs")
            if fe and len(fe) > 1:
                s += " [" + " ".join(f"{k}:{v:.1e}" for k, v in fe.items()) + "]"
            if ver.get("constraint_violation"):
                s += " !CONSTRAINT:" + ",".join(ver.get("violated_constraints", []))
            return s
        return ver.get("status", "-")
    if "worst_variance_rel_err" in ver:
        tag = " (ref)" if kind == "reference" else ""
        fe = ver.get("functional_errs") or {}
        # A problem scored only on a price has no components, so its moment errors
        # are both the 0.0 that "no moment is wrong" puts there -- which would print
        # as a flawless solver. Where there is no moment, the price is the summary.
        if fe and not ver.get("components"):
            body = " ".join(f"{k}:{v:.1e}" for k, v in fe.items())
            lv = (ver.get("functional_levels") or {}).get("status")
            flag = "" if lv == "ok" else " !2-level:" + str(lv or "absent")
            return f"funcErr[{body}]{tag}{flag}"
        s = f"varErr={ver['worst_variance_rel_err']:.1%}, meanErr={ver['worst_mean_rel_err']:.1%}{tag}"
        if fe:
            s += " [" + " ".join(f"{k}:{v:.1e}" for k, v in fe.items()) + "]"
        return s
    return ver.get("status", "-")


def _has_independent_check(rec):
    """True if the problem carries any independent ground truth or stability check."""
    return bool(rec.get("has_ground_truth")) or rec.get("ground_truth_kind") == "stability"


# --- small markdown helpers -------------------------------------------------


def _wp_series(rec):
    """``(axis, points)`` of one record's work-precision series, or ``(None, [])``.

    The series has been collected since the systems extension and rendered by nothing:
    `report.py` and `compare.py` contained no occurrence of `work_precision` at all,
    which the BENCHOP step-8 smoke test recorded as something its step 11 would inherit
    and which stayed true. On the `sciml` suite that is not a cosmetic gap -- error
    against *work* is this suite's primary axis (plan §5), the axis on which a numpy
    solver and compiled Julia can share a plot -- so a sweep that collected five points
    per entry and displayed none of them would have to be re-run to be read.
    """
    wp = ((rec.get("verify") or {}).get("work_precision")) or {}
    pts = [q for q in (wp.get("points") or []) if isinstance(q, dict)]
    return wp.get("axis"), pts


def _wp_slope(points, xkey, ykey):
    """Least-squares log-log slope of `ykey` against `xkey`, or None.

    This is the number a work-precision diagram is read *for*: the exponent relating
    error to work. Reported only from three or more finite, positive, distinct points,
    because two points make a slope out of any pair of numbers and would put a
    confident exponent on a curve that has not been resolved.
    """
    xy = [(float(q[xkey]), float(q[ykey])) for q in points
          if isinstance(q.get(xkey), (int, float)) and isinstance(q.get(ykey), (int, float))
          and q[xkey] > 0 and q[ykey] > 0]
    xy = [(x, y) for x, y in xy if x == x and y == y]
    if len(xy) < 3 or len({x for x, _ in xy}) < 3:
        return None
    xs = [math.log(x) for x, _ in xy]
    ys = [math.log(y) for _, y in xy]
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    den = sum((x - mx) ** 2 for x in xs)
    if den <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / den


def _wp_fmt(point):
    """A formatter bound to one work-precision point; `-` for anything missing.

    A closure over the loop variable would read whichever point the loop finished on
    (ruff B023), which on a five-point series is five copies of the last row.
    """
    def fmt(key, spec="{:g}"):
        v = point.get(key)
        return spec.format(v) if isinstance(v, (int, float)) else "-"
    return fmt


def _table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def _pct(n, d):
    return f"{(100.0 * n / d):.0f}%" if d else "-"


def _fmt_secs(s):
    if s is None:
        return "-"
    s = float(s)
    if s < 90:
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


# --- report sections --------------------------------------------------------


def generate_report(results, config=None):
    config = config or {}
    for rec in results:
        rec["_verdict"] = compute_verdict(rec)

    gt = [r for r in results if r.get("has_ground_truth")]          # exact + reference
    indep = [r for r in results if _has_independent_check(r)]        # + stability
    stability = [r for r in results if r.get("ground_truth_kind") == "stability"]
    completed = [r for r in results if r.get("run", {}).get("status") == "completed"]

    verdict_counts = {}
    for r in results:
        verdict_counts[r["_verdict"]] = verdict_counts.get(r["_verdict"], 0) + 1

    n = len(results)
    pipeline_pass = [r for r in results if (r.get("pipeline", {}).get("best_score") or 0) >= 10]
    independently_passed = [r for r in indep if r["_verdict"] in _VERIFIED_VERDICTS]
    verified_pass = [r for r in gt if r["_verdict"] == "VERIFIED_PASS"]
    overclaims = [r for r in results if r["_verdict"] == "OVERCLAIM"]
    blowups = [r for r in results if r["_verdict"] == "BLOWUP"]

    L = []
    ap = L.append

    # ---- header -----------------------------------------------------------
    ap("# Autonumerics Benchmark Report\n")
    gen = config.get("generated_at") or _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    ap(f"_Generated {gen}_\n")
    if config.get("command_template"):
        ap(f"Pipeline invocation: `{config['command_template']}`\n")
    # Only ever covers the latest invocation (it is overwritten on each run, including
    # --resume), so it is labelled as such rather than as a benchmark-wide total.
    # `total_wall_seconds` is the pre-monotonic key kept for older results.json files.
    elapsed = config.get("invocation_seconds", config.get("total_wall_seconds"))
    if elapsed is not None:
        ap(f"Active time, most recent run invocation: **{_fmt_secs(elapsed)}**\n")

    # ---- executive summary ------------------------------------------------
    ap("## Executive summary\n")
    ap(f"- **Problems:** {n} ({sum(1 for r in results if r['type']=='pde')} PDE, "
       f"{sum(1 for r in results if r['type']=='sde')} SDE); "
       f"{len(gt)} with exact/reference ground truth" +
       (f", {len(stability)} stability checks." if stability else "."))
    ap(f"- **Runs completed:** {len(completed)}/{n}.")
    ap(f"- **Pipeline self-reported pass (10/10):** {len(pipeline_pass)}/{n} "
       f"({_pct(len(pipeline_pass), n)}).")
    ap(f"- **Independently confirmed pass:** {len(independently_passed)}/{len(indep)} "
       f"({_pct(len(independently_passed), len(indep))} of independently-checkable problems).")
    if overclaims or blowups:
        ap(f"- **OVERCLAIMS / BLOW-UPS (pipeline reported success, independent check failed):** "
           f"**{len(overclaims) + len(blowups)}** — see the discrepancies section.")
    else:
        ap("- **OVERCLAIMS / BLOW-UPS:** none — every success the pipeline reported was "
           "independently confirmed.")
    wrong_won = [r for r in results
                 if (r.get("harness_by_plan") or {}).get("wrong_plan_won") is True]
    decided = [r for r in results
               if (r.get("harness_by_plan") or {}).get("wrong_plan_won") is not None]
    if decided:
        ap(f"- **Wrong plan won:** {len(wrong_won)}/{len(decided)} — the winner was less "
           f"accurate than another evaluated plan by more than the reference error "
           f"(see *Every plan, graded*).")
    ap("")

    # ---- verdict distribution --------------------------------------------
    ap("### Verdict distribution\n")
    rows = [[VERDICT_LABEL[v], verdict_counts.get(v, 0), _pct(verdict_counts.get(v, 0), n)]
            for v in _VERDICT_ORDER if verdict_counts.get(v)]
    ap(_table(["Verdict", "Count", "Share"], rows))
    ap("")

    # ---- pass rates by type / tier ---------------------------------------
    ap("### Independently-verified pass rate by type and tier\n")
    rows = []
    for typ in ("pde", "sde"):
        for tier in (1, 2, 3):
            bucket = [r for r in results if r["type"] == typ and r["tier"] == tier]
            indepb = [r for r in bucket if _has_independent_check(r)]
            ipass = sum(1 for r in indepb if r["_verdict"] in _VERIFIED_VERDICTS)
            ppass = sum(1 for r in bucket if (r.get("pipeline", {}).get("best_score") or 0) >= 10)
            rows.append([typ.upper(), f"Tier {tier}", len(bucket),
                         f"{ppass}/{len(bucket)}",
                         f"{ipass}/{len(indepb)}" if indepb else "n/a"])
    ap(_table(["Type", "Tier", "N", "Pipeline pass", "Independently verified"], rows))
    ap("\n> Tier 1 is textbook material the reference manuals cover; Tier 2 needs care "
       "(boundary layers, indefinite operators, multi-D, guards); Tier 3 pushes beyond the "
       "manuals (shocks, fractional operators, stiffness, heavy-tailed sampling, chaos, and "
       "the hard-SDE additions: Feller-violated log-Heston, many-channel stiffness, "
       "superlinear/tamed drift, and blow-up/domain stability). Independently verified counts "
       "both exact/reference moment passes and stability (no blow-up) passes.\n")

    # ---- discrepancies (the important section) ---------------------------
    ap("## Independent-verification discrepancies\n")
    disc = [r for r in results if r["_verdict"] in ("OVERCLAIM", "BLOWUP", "UNDERCLAIM", "UNVERIFIED")]
    if not disc:
        ap("None. Every pipeline self-score agreed with the independent ground-truth check.\n")
    else:
        for r in disc:
            v = r["_verdict"]
            ps = r.get("pipeline", {}).get("best_score")
            ver = r.get("verify", {}) or {}
            ap(f"### {r['id']} {r['slug']} — {VERDICT_LABEL[v]}\n")
            if v == "OVERCLAIM":
                ap(f"Pipeline reported **{ps}/10** on plan `{r.get('pipeline',{}).get('best_plan')}`, "
                   f"but the independent check against the benchmark's own ground truth **fails** "
                   f"({key_metric(r)}). The pipeline's analytic reference or its evaluation is likely wrong.")
            elif v == "BLOWUP":
                ap(f"Pipeline reported **{ps}/10** on plan `{r.get('pipeline',{}).get('best_plan')}`, "
                   f"but re-running the solver shows the solution **diverges or escapes its domain** "
                   f"({key_metric(r)}). A naive scheme blew up where a tamed / boundary-aware one would not.")
            elif v == "UNDERCLAIM":
                ap(f"Pipeline stopped at **{ps}/10**, but the independent check **passes** "
                   f"({key_metric(r)}). The solution was actually good enough; the pipeline "
                   f"under-scored or ran out of iterations.")
            else:  # UNVERIFIED
                ap(f"Pipeline reported **{ps}/10**, but independent verification could not run: "
                   f"`{ver.get('error', ver.get('status'))}`.")
            ap("")

    # ---- full results table ----------------------------------------------
    ap("## Full results\n")
    rows = []
    for r in sorted(results, key=lambda x: x["id"]):
        pl = r.get("pipeline", {})
        rows.append([
            r["id"], r["slug"], r["type"].upper(), r["tier"],
            pl.get("best_score") if pl.get("best_score") is not None else "-",
            pl.get("best_plan") or "-",
            pl.get("total_iters") if pl.get("total_iters") is not None else "-",
            key_metric(r) if _has_independent_check(r) else "no GT",
            VERDICT_LABEL[r["_verdict"]],
            _fmt_secs(r.get("run", {}).get("wall_seconds")),
        ])
    ap(_table(["ID", "Slug", "Type", "Tier", "Pipe", "Best plan", "Iters",
               "Independent metric", "Verdict", "Time"], rows))
    ap("")

    # ---- tier-by-tier detail ---------------------------------------------
    ap("## Tier-by-tier detail\n")
    for typ in ("pde", "sde"):
        ap(f"### {typ.upper()}\n")
        for tier in (1, 2, 3):
            bucket = [r for r in sorted(results, key=lambda x: x["id"])
                      if r["type"] == typ and r["tier"] == tier]
            if not bucket:
                continue
            ap(f"**Tier {tier}**\n")
            for r in bucket:
                ap(f"- `{r['id']} {r['slug']}` ({r['family']}) — **{VERDICT_LABEL[r['_verdict']]}**. "
                   f"{r['challenge']} "
                   + (f"Independent: {key_metric(r)}." if _has_independent_check(r) else
                      "No closed form; pipeline self-score only."))
            ap("")

    # ---- coverage / efficiency -------------------------------------------
    ap("## Scheme coverage and efficiency\n")
    scheme_counts = {}
    for r in verified_pass:
        bp = r.get("pipeline", {}).get("best_plan") or "unknown"
        slug = bp.split("-", 1)[1] if "-" in bp else bp
        scheme_counts[slug] = scheme_counts.get(slug, 0) + 1
    if scheme_counts:
        rows = sorted(scheme_counts.items(), key=lambda kv: -kv[1])
        ap("Winning scheme among verified passes:\n")
        ap(_table(["Scheme (plan slug)", "Times best"], rows))
        ap("")
    iters = [r["pipeline"]["total_iters"] for r in completed
             if r.get("pipeline", {}).get("total_iters") is not None]
    if iters:
        ap(f"- Mean solver-evaluator iterations per problem (summed over plans): "
           f"{sum(iters)/len(iters):.1f}.")
    times = [r["run"]["wall_seconds"] for r in completed if r.get("run", {}).get("wall_seconds")]
    if times:
        ap(f"- Mean wall time per completed run: {_fmt_secs(sum(times)/len(times))}; "
           f"max {_fmt_secs(max(times))}.")
    ap("")

    # ---- work-precision ---------------------------------------------------
    wp_recs = [r for r in results if len(_wp_series(r)[1]) >= 2]
    if wp_recs:
        ap("## Error vs work\n")
        ap("Each row is one solve of the winning plan at a declared resolution. "
           "`evals = paths x steps x stages` is the language-free work axis: `stages` "
           "is the plan's *declared* scheme family's cost per step, so a family that "
           "is not declared counts as 1 and says so. The slope is the least-squares "
           "log-log fit of error against work over the series, reported only where "
           "three or more points resolve it.\n")
        for r in sorted(wp_recs, key=lambda x: x["id"]):
            axis, pts = _wp_series(r)
            fam = {q.get("scheme_family") for q in pts if q.get("scheme_family")}
            src = {q.get("stages_source") for q in pts if q.get("stages_source")}
            slope_e = _wp_slope(pts, "evals", "rel_err")
            slope_dt = _wp_slope(pts, "dt", "rel_err")
            head = f"**{r['id']} {r['slug']}**"
            bits = []
            if fam:
                bits.append("scheme " + "/".join(sorted(fam))
                            + (f" ({'/'.join(sorted(src))})" if src else ""))
            if slope_e is not None:
                bits.append(f"d log err / d log evals = {slope_e:.2f}")
            if slope_dt is not None:
                bits.append(f"observed order in dt = {slope_dt:.2f}")
            ap(head + (" — " + "; ".join(bits) if bits else "") + "\n")
            # A `default` stage count is a 1 the harness guessed, and the direction of
            # that guess is not neutral: it UNDER-states the work of the plan being
            # measured, which flatters it against any comparator costed honestly. The
            # case is real rather than hypothetical -- the kernel's SDE vocabulary
            # (`verifylib/kernel/plan_meta.SDE_FAMILIES`) has no Rossler/SRK entry, so a
            # plan that writes a stochastic Heun or an SRK scheme can only declare
            # `other`, and the first sciml pipeline run wrote exactly that.
            if src and src != {"declared"}:
                ap("> The stage count for this series is a **default of 1**, not a "
                   "declared one: the plan's scheme family is "
                   f"{'/'.join(sorted(str(f) for f in fam)) or 'undeclared'}, which "
                   "carries no cost per step. `evals` is therefore a **lower bound** on "
                   "this plan's work, so any comparison against a scheme whose stages "
                   "are known understates this one's cost.\n")
            cols = ["dt", "steps", "paths", "stages", "evals", "rel err", "solve s",
                    "graded"]
            rows = []
            for q in pts:
                g = _wp_fmt(q)
                rows.append([g("dt"), g("steps"),
                             g("paths") if q.get("paths") else g("num_paths"),
                             g("stages"), g("evals"), g("rel_err", "{:.3e}"),
                             g("solve_s", "{:.3f}"), "*" if q.get("graded") else ""])
            ap(_table(cols, rows))
            ap("")
        skipped = [(r["id"], ((r.get("verify") or {}).get("work_precision") or {})
                    .get("skipped"))
                   for r in results]
        skipped = [(i, s) for i, s in skipped if s]
        if skipped:
            ap("Points recorded but not plotted: "
               + "; ".join(f"{i} ({s})" for i, s in skipped) + "\n")

    # ---- failures detail --------------------------------------------------
    # ANALYTIC_TRANSCRIPTION is not here: it is an exclusion, not a defect of the run
    # (see _VERIFIED_VERDICTS). It appears in the distribution table under its own
    # name.
    fails = [r for r in results if r["_verdict"] in ("FAIL", "OVERCLAIM", "BLOWUP", "RUN_ERROR",
                                                     "UNVERIFIED", "SPEC_BLOCKED")]
    ap("## Failures and errors\n")
    if not fails:
        ap("No failures, overclaims, blow-ups, run errors, or unverifiable results.\n")
    else:
        for r in sorted(fails, key=lambda x: x["id"]):
            run = r.get("run", {})
            ver = r.get("verify", {}) or {}
            detail = []
            if run.get("status") != "completed":
                detail.append(f"run status = {run.get('status')}"
                              + (" (timed out)" if run.get("timed_out") else ""))
            if ver.get("error"):
                detail.append(f"verify: {ver['error']}")
            if "rel_l2_err" in ver:
                detail.append(f"rel L2 = {ver['rel_l2_err']:.3e}")
            margin, label = harness_error(ver)
            if label == "functional" and margin is not None:
                # F10: on a functional-graded problem there is no `rel_l2_err` and
                # the failure line above printed nothing about the error at all.
                worst = max((ver.get("functional_errs") or {}).items(),
                            key=lambda kv: kv[1], default=(None, None))[0]
                detail.append(f"worst functional {worst} at {margin:.3f} of its gate")
            if "worst_variance_rel_err" in ver:
                detail.append(f"var err = {ver['worst_variance_rel_err']:.1%}")
            if ver.get("gt_kind") == "stability":
                s = "all-finite" if ver.get("finite") else "NON-FINITE"
                if ver.get("check") == "domain" and not ver.get("in_domain"):
                    s += ", escaped domain"
                detail.append(s)
            log = run.get("log")
            logstr = f"  \n  log: `{log}`" if log else ""
            ap(f"- `{r['id']} {r['slug']}` — **{VERDICT_LABEL[r['_verdict']]}**: "
               f"{'; '.join(detail) if detail else 'see log'}.{logstr}")
        ap("")

    # ---- kernel provenance vs the independent verdict ---------------------
    # Only rendered when the run recorded it. This is the whole point of the
    # no-closed-form suite: the pipeline's kernel says what its answer *rests on*
    # (a closed form, a manufactured solution, self-convergence alone), and this is
    # the only place that claim meets an independent verdict. A `manufactured` 10
    # sitting on an OVERCLAIM row is a rubric bug; a `self_convergence` 8 on a
    # VERIFIED_PASS row is the rubric being appropriately cautious.
    kerneled = [r for r in results if (r.get("kernel") or {}).get("status") == "ok"]
    if kerneled:
        ap("## Kernel provenance vs the independent verdict\n")
        ap("What the pipeline's own kernel says its score rests on (`provenance`, from "
           "`verifylib/kernel/score.py`), against what the harness's independent check "
           "found. The kernel never sees the ground truth these verdicts are computed "
           "from, so agreement here is evidence and disagreement is a finding.\n")
        provs, verds = {}, []
        for r in kerneled:
            provs.setdefault(r["kernel"].get("provenance") or "none", {})
            provs[r["kernel"]["provenance"] or "none"][r["_verdict"]] = \
                provs[r["kernel"]["provenance"] or "none"].get(r["_verdict"], 0) + 1
            if r["_verdict"] not in verds:
                verds.append(r["_verdict"])
        verds = [v for v in _VERDICT_ORDER if v in verds]
        ap(_table(["kernel provenance", *verds],
                  [[prov, *[str(provs[prov].get(v, 0) or "") for v in verds]]
                   for prov in sorted(provs)]))
        ap("")
        ap(_table(["problem", "kernel score", "provenance", "tier", "independent verdict"],
                  [[r["slug"], str(r["kernel"].get("score")),
                    r["kernel"].get("provenance") or "-", r["kernel"].get("tier") or "-",
                    VERDICT_LABEL.get(r["_verdict"], r["_verdict"])]
                   for r in sorted(kerneled, key=lambda r: r["slug"])]))
        ap("")

    # ---- every plan, not only the winner (findings §11.1) ---------------------
    by_plan = [r for r in results if (r.get("harness_by_plan") or {}).get("plans")]
    if by_plan:
        ap("## Every plan, graded\n")
        wrong = [r for r in by_plan if r["harness_by_plan"].get("wrong_plan_won") is True]
        decided = [r for r in by_plan if r["harness_by_plan"].get("wrong_plan_won") is not None]
        ap(f"The harness grades every plan directory with a `solver.py`, not only the winner. "
           f"**`wrong_plan_won`: {len(wrong)}/{len(decided)}** -- the conductor's winner was "
           f"less accurate than another evaluated plan by more than the reference's own "
           f"error. This is the number the Path-B ranking rules exist to drive to zero, and "
           f"it is invisible in a winner-only record.\n")
        if wrong:
            for r in wrong:
                h = r["harness_by_plan"]
                ap(f"- `{r['slug']}`: winner `{h['winner']}` at {h['winner_harness_err']:.3e}, "
                   f"best `{h['best_plan_by_harness']}` at {h['best_harness_err']:.3e}.")
            ap("")
        # Provenance x verdict over *plans*: ~3x the rows the winner-only table has.
        provs, verds = {}, []
        for r in by_plan:
            for entry in r["harness_by_plan"]["plans"].values():
                prov = (entry.get("kernel") or {}).get("provenance") or "-"
                verdict = entry.get("verdict") or "-"
                provs.setdefault(prov, {})
                provs[prov][verdict] = provs[prov].get(verdict, 0) + 1
                if verdict not in verds:
                    verds.append(verdict)
        verds = [v for v in _VERDICT_ORDER if v in verds] + [v for v in verds
                                                            if v not in _VERDICT_ORDER]
        ap(_table(["kernel provenance (per plan)", *verds],
                  [[prov, *[str(provs[prov].get(v, 0) or "") for v in verds]]
                   for prov in sorted(provs)]))
        ap("")
        rows = []
        for r in sorted(by_plan, key=lambda r: r["slug"]):
            for name, entry in sorted(r["harness_by_plan"]["plans"].items()):
                k, h = entry.get("kernel") or {}, entry.get("harness") or {}
                err = h.get("rel_l2_err")
                margin, _label = harness_error(h)
                drift = h.get("solver_drift")
                rows.append([
                    r["slug"], name + (" *" if entry.get("is_winner") else ""),
                    str(k.get("score", "-")), k.get("provenance") or "-",
                    (f"{err:.3e}" if isinstance(err, (int, float)) else "-")
                    + (" (drifted)" if drift else ""),
                    # The gate-relative number, which is the one a reader compares
                    # across problems: the per-value tolerances on this suite span
                    # 0.002 to 0.015, so two raw errors are not comparable and two
                    # margins are (F10).
                    (f"{margin:.3f}" if isinstance(margin, (int, float)) else "-"),
                    VERDICT_LABEL.get(entry.get("verdict"), entry.get("verdict") or "-")])
        ap(_table(["problem", "plan (* winner)", "kernel score", "provenance",
                   "harness rel err", "harness err / gate", "verdict"], rows))
        ap("")

    # ---- methodology ------------------------------------------------------
    ap("## Methodology and reproducibility\n")
    ap("- **Pipeline pass** = the conductor's best plan reached the terminal self-score of "
       "10/10 (PDE: rel L2 < 1%; SDE: variance rel err < 10% and, unless the mean is near "
       "zero, mean rel err < 5%).")
    ap("- **Independent verification** re-imports the best plan's `solver.py`, re-runs it, and "
       "compares the raw numerical output to ground truth authored and validated in "
       "`benchmark/problems.py` — never to the formulator's extracted solution. SDE moments use "
       "the fixed `(num_paths, dt, seed)` recorded in the manifest; PDE errors use the RMS-based "
       "relative L2 norm from `pde_manual.md`.")
    ap("- **Ground-truth kinds:** *exact* (closed-form / machine-precision moments or analytic "
       "solution); *reference* (no elementary moment formula — the ground truth is a "
       "discretization-free Monte-Carlo of the known exact solution, compared within an "
       "SE-aware tolerance); *stability* (no ground truth — the independent check only confirms "
       "a correct scheme stays finite / in its domain, which is what a naive scheme gets wrong).")
    ap("- **Four suites, four baseline denominators.** `main` (49) is the default; "
       "`no-closed-form` (7), `benchop` (10) and `sciml` (9) each run only when asked "
       "for by `--suite`, because mixing them into the main suite would move the "
       "denominator every published pass rate is quoted against. `benchop` is graded "
       "against 24 published reference values; `sciml` is graded against truth this "
       "repo owns and *compared* against another library's schemes, which is why it "
       "carries the `scheme-matched` / `unrestricted` arm distinction and the "
       "`ANALYTIC_TRANSCRIPTION` exclusion. A run records its arm; two arms are never "
       "merged.")
    ap("- **Problems with no closed form** (`closed_form: False`) run as their own suite "
       "(`--no-closed-form`) and are excluded by default, so the main benchmark's baseline "
       "denominators stay comparable. They are still independently checked, by one of four "
       "routes: a harness-owned high-resolution *reference field* (two independent method "
       "families agreeing far below the pass tolerance, built by `make_references.py` and "
       "recorded as the problem's `reference_error`, which widens the tolerance so the answer "
       "key's own error cannot fail a correct solver); *semi-analytic exact* truth from a "
       "quadrature, series or eigen-expansion; a scalar *functional* (`functional_truth`); and "
       "structural gates. See `benchmark/design/NO_CLOSED_FORM_CANDIDATES.md`.")
    ap(f"- Ground truth for all {len(gt)} exact/reference problems is validated independently "
       "(SDE moments against moment-ODE integration, matrix-exponential / affine systems, and "
       "direct simulation; the Ginzburg-Landau reference anchored to the literature value "
       "E[X_1^2]=0.8114 at sigma=2; PDE solutions against their defining identities).")
    ap("- Runs are **sequential**; each problem has a wall-clock timeout and its full transcript "
       "is saved under `benchmark/results/logs/`.")
    ap("")
    return "\n".join(L)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Render the benchmark REPORT.md from results.json")
    ap.add_argument("--results", default=DEFAULT_RESULTS)
    ap.add_argument("--out", default=DEFAULT_REPORT)
    args = ap.parse_args(argv)

    with open(args.results) as fh:
        payload = json.load(fh)
    results = payload["results"] if isinstance(payload, dict) else payload
    config = payload.get("config", {}) if isinstance(payload, dict) else {}

    md = generate_report(results, config)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        fh.write(md)
    print(f"wrote {os.path.relpath(args.out, REPO_ROOT)}  ({len(results)} problems)")


if __name__ == "__main__":
    main()
