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
    "SPEC_BLOCKED": "SPEC BLOCKED",
    "RUN_ERROR": "RUN ERROR",
}

# Verdicts that count as an independently-confirmed success.
_PASS_VERDICTS = ("VERIFIED_PASS", "STABLE_PASS")
# The order verdicts are listed in the distribution table.
_VERDICT_ORDER = ("VERIFIED_PASS", "STABLE_PASS", "OVERCLAIM", "BLOWUP",
                  "CONSTRAINT_VIOLATION", "UNDERCLAIM", "FAIL", "UNVERIFIED",
                  "SELF_ONLY", "SPEC_BLOCKED", "RUN_ERROR")

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
}
# What counts as an independently-confirmed baseline success.
_ONESHOT_PASS_VERDICTS = ("PASS", "STABLE_PASS")
_ONESHOT_VERDICT_ORDER = ("PASS", "STABLE_PASS", "FAIL", "CONSTRAINT_VIOLATION",
                          "BLOWUP", "CRASHED", "NO_GT")


# --- verdict logic ----------------------------------------------------------


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

    pipeline_pass = (rec.get("pipeline", {}).get("best_score") or 0) >= 10
    verified_pass = bool(ver.get("passed", False))

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
        return "PASS" if ver.get("passed") else "FAIL"
    if vstatus in ("nonfinite", "inconclusive"):
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
        return f"varErr={ver['worst_variance_rel_err']:.1%}, meanErr={ver['worst_mean_rel_err']:.1%}{tag}"
    return ver.get("status", "-")


def _has_independent_check(rec):
    """True if the problem carries any independent ground truth or stability check."""
    return bool(rec.get("has_ground_truth")) or rec.get("ground_truth_kind") == "stability"


# --- small markdown helpers -------------------------------------------------


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
    independently_passed = [r for r in indep if r["_verdict"] in _PASS_VERDICTS]
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
            ipass = sum(1 for r in indepb if r["_verdict"] in _PASS_VERDICTS)
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

    # ---- failures detail --------------------------------------------------
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
