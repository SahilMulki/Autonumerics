"""Sequential benchmark runner for the Autonumerics pipeline.

For each selected problem, in order:
  1. write workspace/{slug}/problem.md
  2. invoke the conductor headlessly:  claude --plugin-dir . -p "/conductor ..."
  3. parse STATE.md for the pipeline's self-score
  4. independently verify the best plan against the benchmark's ground truth
  5. append a result record and rewrite results.json (crash-safe)
Finally, render results/REPORT.md.

Runs are sequential by design (the pipeline already runs up to 3 plans in parallel
internally, and sequential runs keep logs and cost legible).

Examples
--------
    # full 37-problem benchmark (expensive: many Opus/Sonnet calls, hours)
    uv run python benchmark/run.py

    # a cheap subset, or a single smoke test
    uv run python benchmark/run.py --only sde_gbm,pde_heat_1d
    uv run python benchmark/run.py --only pde_heat_1d --timeout 1800

    # continue after an interruption (skip already-completed problems)
    uv run python benchmark/run.py --resume

    # re-verify + re-report without spending anything (no pipeline calls)
    uv run python benchmark/run.py --skip-run
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")  # conductor resolves paths here
RESULTS_DIR = os.path.join(HERE, "results")
LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
RESULTS_JSON = os.path.join(RESULTS_DIR, "results.json")
REPORT_MD = os.path.join(RESULTS_DIR, "REPORT.md")

# Permission settings handed to every pipeline invocation. They deny the agents read
# access to benchmark/ -- the ground truth they are graded against lives there, and a
# formulator or evaluator that reads problems.py or verify.py closes the
# trust-but-verify loop on itself and makes "0 overclaims" meaningless. Deny rules are
# enforced even under --dangerously-skip-permissions, and passing them per-invocation
# (rather than putting them in the repo's .claude/settings.json) keeps ordinary
# development sessions able to work on benchmark/ normally.
PIPELINE_SETTINGS = os.path.join(HERE, "pipeline-settings.json")

sys.path.insert(0, HERE)
import report as R  # noqa: E402, I001
import setup as S  # noqa: E402
import verify as V  # noqa: E402


# --- pipeline invocation ----------------------------------------------------


# Commands the base settings file already denies against benchmark/; the same set is
# denied against every *other* problem's workspace, matched anywhere in the command
# so an absolute path is caught too (the base file's relative-only `cat benchmark/*`
# rules let `cat /Users/.../workspace/<sibling>/problem_spec.json` straight through).
_WORKSPACE_READERS = ("cat", "head", "tail", "less", "more", "sed", "awk", "grep", "rg",
                      "cp", "python", "python3", "uv run python", "uv run python3")


def isolated_settings(slug, log_dir):
    """Write a per-run settings file: the base denials plus a read-denial on every
    other problem's workspace. Returns its path.

    Found by reading a transcript, not by design: the formulator for the coarsening
    Cahn-Hilliard problem `cat`ed its manufactured-solution twin's problem_spec.json
    and used it as a template. Nothing in `pipeline-settings.json` forbade it --
    benchmark/ is denied, workspace/ is not. That is not a ground-truth leak, but it
    collapses the matched-pair design: "same operator, closed form removed" isolates
    one variable only if the pipeline cannot copy from the closed-form sibling. So
    each run sees exactly one workspace: its own.

    The file is generated fresh each run from what is on disk, so a workspace added
    later is covered without touching the base file, and it sits next to the
    transcript so a run can be reproduced with the isolation it actually had."""
    with open(PIPELINE_SETTINGS) as fh:
        settings = json.load(fh)
    deny = list(settings.setdefault("permissions", {}).get("deny", []))
    others = sorted(d for d in os.listdir(WORKSPACE_ROOT)
                    if d != slug and os.path.isdir(os.path.join(WORKSPACE_ROOT, d)))
    for other in others:
        deny.append(f"Read(./workspace/{other}/**)")
        deny.append(f"Read(//Users/**/Autonumerics/workspace/{other}/**)")
        deny.append(f"Edit(./workspace/{other}/**)")
        for cmd in _WORKSPACE_READERS:
            deny.append(f"Bash({cmd}:*workspace/{other}/*)")
    settings["permissions"]["deny"] = deny
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, f"{slug}.settings.json")
    with open(path, "w") as fh:
        json.dump(settings, fh, indent=1)
    return path


def conductor_command(slug, skip_permissions, permission_mode, model, settings=None):
    cmd = ["claude", "--plugin-dir", REPO_ROOT,
           "--settings", settings or PIPELINE_SETTINGS, "-p",
           f"/conductor workspace/{slug}/problem.md"]
    if skip_permissions:
        cmd.append("--dangerously-skip-permissions")
    elif permission_mode:
        cmd += ["--permission-mode", permission_mode]
    if model:
        cmd += ["--model", model]
    return cmd


def run_conductor(slug, timeout, skip_permissions, permission_mode, model):
    """Invoke the conductor headlessly; capture the transcript. Returns a run dict."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    log_path = os.path.join(LOGS_DIR, f"{slug}.log")
    cmd = conductor_command(slug, skip_permissions, permission_mode, model,
                            settings=isolated_settings(slug, LOGS_DIR))

    # Elapsed is measured on the monotonic clock, which is what subprocess's own
    # timeout uses. On macOS it does not advance while the system is asleep, so a
    # laptop sleeping mid-run no longer reports hours of "runtime" that never
    # happened (and that a 40-minute timeout appeared not to catch).
    t0 = time.monotonic()
    # ...and the wall clock beside it (findings §8): a laptop sleeping mid-run was
    # only detectable by diffing the log's mtime (36.7 min) against `wall_seconds`
    # (19.4 min). One `time.time()` field makes the gap visible in the record.
    started_at = time.time()
    timed_out = False
    returncode = None
    try:
        with open(log_path, "w") as log:
            log.write(f"$ {' '.join(cmd)}\n\n")
            log.flush()
            proc = subprocess.run(
                cmd, cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=timeout, check=False,
            )
            returncode = proc.returncode
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = time.monotonic() - t0
    ended_at = time.time()

    status = "completed" if (not timed_out and returncode == 0) else ("timeout" if timed_out else "error")
    return {
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(wall, 1),
        "wall_clock_seconds": round(ended_at - started_at, 1),
        "started_at": _dt.datetime.fromtimestamp(started_at).isoformat(timespec="seconds"),
        "ended_at": _dt.datetime.fromtimestamp(ended_at).isoformat(timespec="seconds"),
        "log": os.path.relpath(log_path, REPO_ROOT),
        "limit_hit": _looks_like_limit(log_path),
    }


# The Claude Pro plan enforces a usage limit that resets ~5 hours after it is hit;
# when exhausted, the CLI exits in ~1s with one of these phrases. Every subsequent
# problem would fail the same way, so the runner stops and tells you to resume after
# the reset rather than burning through the rest of the queue.
_LIMIT_PHRASES = ("spend limit", "usage limit", "rate limit", "session limit",
                  "hit your monthly", "hit your session", "limit reached", "resets at",
                  "· resets", "insufficient credit", "quota")


def _looks_like_limit(log_path):
    try:
        with open(log_path) as fh:
            text = fh.read().lower()
    except OSError:
        return False
    # Whole words only: the kernel's own vocabulary contains "degene|rate limit",
    # and a successful transcript that names its tier-B route must not read as a
    # usage limit -- on a timed-out run that would halt the whole queue.
    return any(re.search(r"\b" + re.escape(phrase) + r"\b", text) for phrase in _LIMIT_PHRASES)


# --- pipeline-state summary -------------------------------------------------


def summarize_pipeline(workspace_dir):
    state = V.parse_state(workspace_dir)
    plans = state.get("plans", {})
    scores = [p.get("score") for p in plans.values() if p.get("score") is not None]
    iters = [p.get("iter") for p in plans.values() if p.get("iter") is not None]
    best_plan = state.get("best_plan")
    best_score = max(scores) if scores else None
    if best_plan and best_plan in plans:
        best_score = plans[best_plan].get("score", best_score)
    elif not best_plan and plans:
        # Conductor didn't finalize STATE (e.g. interrupted before Phase 3); fall
        # back to the highest-scoring plan, matching how verification picks a plan.
        best_plan = max(plans, key=lambda pid: (plans[pid].get("score") or -1,
                                                -(plans[pid].get("iter") or 0)))
    return {
        "phase": state.get("phase"),
        "blocked_reason": state.get("blocked_reason"),
        "problem_type": state.get("problem_type"),
        "best_plan": best_plan,
        "best_score": best_score,
        "n_plans": len(plans),
        "total_iters": sum(iters) if iters else None,
        "plans": plans,
    }


def clear_pipeline_outputs(workspace_dir):
    """Remove prior pipeline artifacts, keeping only problem.md."""
    for name in ("STATE.md", "problem_spec.json", "REPORT.md"):
        p = os.path.join(workspace_dir, name)
        if os.path.exists(p):
            os.remove(p)
    plans = os.path.join(workspace_dir, "plans")
    if os.path.isdir(plans):
        shutil.rmtree(plans)


# --- one problem ------------------------------------------------------------


def process(problem, args):
    slug = problem["slug"]
    workspace_dir = os.path.join(WORKSPACE_ROOT, slug)

    # 1. always (re)write the input problem.md
    S.write_problem_md(problem, WORKSPACE_ROOT)

    # 2. run the pipeline (unless skipped)
    if args.skip_run:
        run = {"status": "completed", "returncode": 0, "timed_out": False,
               "wall_seconds": None, "log": None, "note": "skip-run: used existing STATE"}
    else:
        if args.fresh:
            clear_pipeline_outputs(workspace_dir)
        run = run_conductor(slug, args.timeout, args.skip_permissions,
                            args.permission_mode, args.model)

    # 3. pipeline self-score from STATE.md
    pipeline = summarize_pipeline(workspace_dir)

    # 3a. requirements gate: the conductor halts at phase:blocked when the formulator's
    # requirements ledger did not carry every constraint in problem.md into the spec.
    # That is a real, terminal outcome -- the pipeline correctly refused to solve a
    # problem it could not represent -- so it must not be folded into the "no_progress"
    # guard below, which exists for crashes and would make --resume retry it forever.
    if run.get("status") == "completed" and pipeline.get("phase") == "blocked":
        run["status"] = "spec_blocked"
        run["note"] = ("conductor halted at the requirements gate: "
                       + (pipeline.get("blocked_reason") or "(no reason recorded)"))

    # 3b. no-op guard: a conductor can exit cleanly (returncode 0) yet leave the
    # pipeline at phase:init with no plans -- e.g. a headless run where it paused for
    # user input, or crashed before Phase 1. That is not a real "completed" result;
    # recording it as such makes --resume skip the problem forever. Downgrade it to a
    # non-completed status so --resume retries it on the next pass.
    if run.get("status") == "completed" and (
        pipeline.get("phase") == "init" or not pipeline.get("n_plans")
    ):
        run["status"] = "no_progress"
        run["note"] = (
            f"conductor exited cleanly but left phase={pipeline.get('phase')!r} with "
            f"{pipeline.get('n_plans') or 0} plans -- treated as not completed so "
            "--resume retries it"
        )

    # 4. independent verification
    verify = V.verify_problem(problem, workspace_dir)

    # 5. the pipeline's OWN verdict on itself, computed by the kernel rather than
    #    parsed out of prose an agent wrote about its own work. Recorded beside the
    #    independent one so the two can be cross-tabulated: does `manufactured`
    #    actually correlate with being right, and does `self_convergence` ever hide an
    #    OVERCLAIM? Nothing inside the pipeline can answer that about itself, and on
    #    the no-closed-form suite it is the measurement the suite exists to produce.
    kernel = (None if getattr(args, "skip_kernel", False)
              else kernel_metrics(V.best_plan_dir(workspace_dir)))

    rec = {
        "id": problem["id"], "slug": slug, "type": problem["type"], "tier": problem["tier"],
        "family": problem["family"], "title": problem["title"], "challenge": problem["challenge"],
        "has_ground_truth": problem["has_ground_truth"],
        "ground_truth_kind": problem.get("ground_truth_kind", "exact"),
        "run": run, "pipeline": pipeline, "verify": verify,
    }
    if kernel is not None:
        rec["kernel"] = kernel
    rec["verdict"] = R.compute_verdict(rec)

    # 6. every plan, not only the winner (findings §11.1). The single most
    #    informative measurement in the no-closed-form findings was made by hand
    #    three times -- verify.py on the plans the conductor did *not* pick -- and
    #    each time it produced the finding. `wrong_plan_won` is the number F2 and F3
    #    exist to drive to zero, and it is invisible in a winner-only record.
    if not getattr(args, "skip_plans", False):
        rec["harness_by_plan"] = grade_every_plan(
            problem, workspace_dir, rec, verify, kernel,
            rerun_kernel=not getattr(args, "skip_kernel", False))
    return rec


def grade_every_plan(problem, workspace_dir, rec, winner_verify, winner_kernel, *,
                     rerun_kernel=False):
    """Harness verdict per plan directory with a ``solver.py``, beside the kernel's
    own score for that plan (from its ``SOLUTION.md`` metrics block, and from a
    kernel re-run when ``rerun_kernel``). Respects F6: a plan whose file drifted
    after scoring is graded ``UNVERIFIED`` with the reason, not as the plan.
    """
    plans_root = os.path.join(workspace_dir, "plans")
    if not os.path.isdir(plans_root):
        return None
    winner = (rec.get("pipeline") or {}).get("best_plan")
    winner_dir = V.best_plan_dir(workspace_dir)
    out = {"plans": {}, "winner": winner}
    for name in sorted(os.listdir(plans_root)):
        plan_dir = os.path.join(plans_root, name)
        if not os.path.exists(os.path.join(plan_dir, "solver.py")):
            continue
        is_winner = winner_dir is not None and os.path.samefile(plan_dir, winner_dir)
        harness = (winner_verify if is_winner and winner_verify is not None
                   else V.verify_problem(problem, workspace_dir, plan_dir=plan_dir))
        block = solution_metrics(plan_dir)
        entry = {
            "harness": {k: harness.get(k) for k in (
                "status", "passed", "verified_score", "rel_l2_err", "observed_order",
                "order_ok", "converged", "constraint_violation", "error", "solver_drift",
                "score_cap") if k in harness},
            "kernel": block,
            "is_winner": bool(is_winner),
        }
        if rerun_kernel:
            entry["kernel_rerun"] = (winner_kernel if is_winner and winner_kernel is not None
                                     else kernel_metrics(plan_dir))
        # The same verdict the winner gets, computed on this plan's own scores.
        pipeline_score = (block or {}).get("score")
        entry["verdict"] = R.compute_verdict({
            **{k: rec[k] for k in ("has_ground_truth", "ground_truth_kind")},
            "run": {"status": "completed"},
            "pipeline": {"best_score": pipeline_score},
            "verify": harness,
        })
        out["plans"][name] = entry

    graded = {name: e for name, e in out["plans"].items()
              if (e["harness"].get("status") == "ok"
                  and isinstance(e["harness"].get("rel_l2_err"), (int, float))
                  and e["harness"]["rel_l2_err"] == e["harness"]["rel_l2_err"])}
    if winner in graded and graded:
        best = min(graded, key=lambda n: graded[n]["harness"]["rel_l2_err"])
        margin = float(problem.get("reference_error") or 0.0)
        winner_err = graded[winner]["harness"]["rel_l2_err"]
        best_err = graded[best]["harness"]["rel_l2_err"]
        out.update({"best_plan_by_harness": best, "best_harness_err": best_err,
                    "winner_harness_err": winner_err,
                    "wrong_plan_won": bool(best != winner and winner_err > best_err + margin)})
    else:
        out["wrong_plan_won"] = None
    return out


def solution_metrics(plan_dir):
    """The kernel's own block for a plan, as the conductor ranked it: ``score``,
    ``provenance``, ``estimated_rel_error`` and the ``solver_sha256`` it scored."""
    path = os.path.join(plan_dir, "SOLUTION.md")
    if not os.path.exists(path):
        return None
    try:
        from verifylib.review import parse_metrics
        with open(path, encoding="utf-8") as fh:
            block = parse_metrics(fh.read())
    except Exception as exc:  # noqa: BLE001 -- an unreadable block is "not recorded"
        return {"error": f"{type(exc).__name__}: {exc}"}
    if not block:
        return None
    return {k: block.get(k) for k in (
        "score", "provenance", "estimated_rel_error", "estimated_rel_error_T",
        "error_horizon", "d1_outcome", "reference_outcome", "solver_sha256") if k in block}


def kernel_metrics(plan_dir, timeout_s=900):
    """Read the kernel's own score for the winning plan, or None.

    This re-runs ``verifylib/cli.py evaluate``, which executes the solver in the
    kernel's sandbox -- so it costs a solve and is skippable. It is deliberately a
    *read* of the pipeline's own scoring, never an input to the benchmark verdict:
    ``verify.py`` grades against harness-owned ground truth and must stay independent
    of anything the pipeline produced.

    Never fatal. A plan predating the kernel's contract, a missing spec, or a solver
    the sandbox refuses all yield a recorded reason rather than a failed benchmark
    run."""
    if not plan_dir or not os.path.isdir(plan_dir):
        return None
    cli = os.path.join(REPO_ROOT, "verifylib", "cli.py")
    if not os.path.exists(cli):
        return None
    try:
        proc = subprocess.run([sys.executable, cli, "evaluate", plan_dir, "--json"],
                              capture_output=True, text=True, timeout=timeout_s,
                              cwd=REPO_ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "timeout_s": timeout_s}
    except Exception as exc:  # noqa: BLE001 -- a measurement must not break the run
        return {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {"status": "unavailable",
                "error": (proc.stderr or "").strip()[:400] or f"exit {proc.returncode}"}
    try:
        m = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"status": "error", "error": f"kernel output is not JSON: {exc}"}
    cert = m.get("certification") or {}
    return {"status": "ok", "score": m.get("score"), "provenance": m.get("provenance"),
            "tier": cert.get("tier"), "bound_by": cert.get("bound_by"),
            "reason": cert.get("reason"), "agent_cap": m.get("agent_cap")}


# --- results.json IO --------------------------------------------------------


def load_prior():
    if not os.path.exists(RESULTS_JSON):
        return {}
    try:
        with open(RESULTS_JSON) as fh:
            payload = json.load(fh)
        return {r["slug"]: r for r in payload.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def _json_default(o):
    """Coerce numpy scalars/arrays so any record stays JSON-serializable
    (np.bool_ in particular is not a Python bool and json rejects it)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")


def write_results(records, config):
    """Atomic write: serialize to a temp file, then rename over the target so a
    crash mid-encode can never truncate an existing good results.json."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = RESULTS_JSON + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"config": config, "results": list(records.values())}, fh, indent=2,
                  default=_json_default)
    os.replace(tmp, RESULTS_JSON)


# --- run provenance ---------------------------------------------------------
# agents/*.md are pipeline *inputs*: editing one changes behaviour, so results
# either side of an edit are not strictly comparable. That discontinuity is
# accepted -- the point of changing the prompts is that the new numbers are better
# grounded -- but it has to be legible. A provenance change that is recorded is a
# finding; one that is silent is a confound.

PROVENANCE_DIRS = ("agents", "commands", "references", "verifylib")


def agent_file_hashes():
    """SHA-256 of every prompt and library file the run actually used.

    Taken *inside* the --model override, so a Sonnet-pinned sweep records the
    Sonnet-pinned files rather than the originals restored afterwards.
    """
    import hashlib

    out = {}
    for name in PROVENANCE_DIRS:
        root = os.path.join(REPO_ROOT, name)
        for directory, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".") and d not in ("__pycache__", "tests")]
            for filename in sorted(filenames):
                if filename.startswith(".") or filename.endswith((".pyc", ".bak")):
                    continue
                path = os.path.join(directory, filename)
                with open(path, "rb") as fh:
                    digest = hashlib.sha256(fh.read()).hexdigest()
                out[os.path.relpath(path, REPO_ROOT)] = digest[:16]
    return out


def claude_version():
    try:
        return subprocess.run(["claude", "--version"], capture_output=True, text=True,
                              timeout=30).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- agent-model override ---------------------------------------------------
# Claude Code honors each subagent's own `model:` frontmatter over the session's
# --model flag, so `run.py --model sonnet` on its own pins only the conductor --
# the formulator / plan-creators (model: opus) keep running Opus, silently making
# a "Sonnet-pinned" C2b sweep identical to C2a on its planning agents. To make
# --model actually pin the WHOLE pipeline, we rewrite every agent's `model:`
# frontmatter line to the override for the duration of the run, then restore the
# files to their exact prior bytes. A no-op when --model is absent, so a default
# C2a run never touches the agent files.

AGENTS_DIR = os.path.join(REPO_ROOT, "agents")
# Fixed (not results-relative) so self-heal finds it regardless of --results-json.
_MODEL_BACKUP = os.path.join(HERE, ".agent_model_override.bak.json")
_FM_MODEL_RE = re.compile(r"^(model:[ \t]*)\S.*$", re.M)


def _frontmatter_body_span(text):
    """Char offsets (start, end) of the YAML frontmatter body -- the region between
    the opening `---` line and the next `---` line -- or None if there is no
    frontmatter. Restricting edits to this span avoids touching a `model:` token
    that appears in the agent's prose body."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None
    for i in range(1, len(lines)):
        if lines[i].rstrip("\r\n") == "---":
            return (len(lines[0]), sum(len(ln) for ln in lines[:i]))
    return None


def _rewrite_agent_model(text, model):
    """(text with the frontmatter `model:` line set to `model`, substitutions made).
    Preserves the `model: ` prefix spacing; returns (text, 0) if there is no
    frontmatter `model:` line to change."""
    span = _frontmatter_body_span(text)
    if span is None:
        return text, 0
    s, e = span
    new_body, n = _FM_MODEL_RE.subn(rf"\g<1>{model}", text[s:e])
    return (text[:s] + new_body + text[e:], n) if n else (text, 0)


def restore_agent_models_from_backup():
    """Self-heal: if a prior --model run was killed before its `finally` could restore
    the agent files (SIGKILL, power loss), the sidecar backup still holds their
    original bytes. Restore them and remove the sidecar so the next run -- especially a
    default-model C2a -- is not silently poisoned with a leftover override."""
    if not os.path.exists(_MODEL_BACKUP):
        return
    try:
        with open(_MODEL_BACKUP) as fh:
            saved = json.load(fh)
        for path, original in saved.items():
            with open(path, "w") as fh:
                fh.write(original)
        print(f"note: restored {len(saved)} agent file(s) left overridden by a prior "
              "interrupted --model run", flush=True)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not auto-restore agent models ({exc}); "
              "check `git diff agents/` before trusting a run", file=sys.stderr)
    finally:
        with contextlib.suppress(OSError):
            os.remove(_MODEL_BACKUP)


@contextlib.contextmanager
def override_agent_models(model):
    """Pin every pipeline agent's frontmatter `model:` to `model` for the duration of
    the run, restoring the exact original bytes on exit (including on exceptions).
    A no-op when `model` is falsy, so a default-model run is never touched."""
    if not model:
        yield
        return
    originals = {}  # path -> original text (only files actually changed)
    if os.path.isdir(AGENTS_DIR):
        for name in sorted(os.listdir(AGENTS_DIR)):
            if not name.endswith(".md"):
                continue
            path = os.path.join(AGENTS_DIR, name)
            with open(path) as fh:
                text = fh.read()
            if _rewrite_agent_model(text, model)[1]:
                originals[path] = text
    if originals:
        # Persist originals BEFORE editing so a hard kill mid-run is recoverable.
        with open(_MODEL_BACKUP, "w") as fh:
            json.dump(originals, fh)
        for path, text in originals.items():
            with open(path, "w") as fh:
                fh.write(_rewrite_agent_model(text, model)[0])
        print(f"--model {model}: pinned {len(originals)} pipeline agent(s) to "
              f"'{model}' for this run (originals restored on exit)", flush=True)
    try:
        yield
    finally:
        for path, text in originals.items():
            with open(path, "w") as fh:
                fh.write(text)
        with contextlib.suppress(OSError):
            os.remove(_MODEL_BACKUP)


# --- main -------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated slugs or ids (default: all 37)")
    S.add_suite_arguments(ap)
    ap.add_argument("--skip-kernel", dest="skip_kernel", action="store_true",
                    help="do not record the pipeline's own kernel score/provenance "
                         "(it costs one extra sandboxed solve per problem)")
    ap.add_argument("--skip-plans", dest="skip_plans", action="store_true",
                    help="grade only the winner; skip the per-plan harness_by_plan block "
                         "(one harness solve per plan directory, plus a kernel re-run per "
                         "plan unless --skip-kernel)")
    ap.add_argument("--type", dest="type_filter", choices=("pde", "sde"), default=None,
                    help="restrict to one problem type (for staggering across usage windows)")
    ap.add_argument("--tier", type=int, choices=(1, 2, 3), default=None,
                    help="restrict to one difficulty tier")
    ap.add_argument("--timeout", type=int, default=2400,
                    help="per-problem wall-clock timeout in seconds (default 2400 = 40 min)")
    ap.add_argument("--resume", action="store_true",
                    help="skip problems already completed in results.json")
    ap.add_argument("--fresh", action="store_true",
                    help="clear each workspace's prior pipeline outputs before running")
    ap.add_argument("--skip-run", action="store_true",
                    help="do not invoke the conductor; only parse STATE, verify, and report")
    ap.add_argument("--skip-permissions", dest="skip_permissions", action="store_true", default=True,
                    help="pass --dangerously-skip-permissions (default: on, for unattended runs)")
    ap.add_argument("--no-skip-permissions", dest="skip_permissions", action="store_false",
                    help="do not bypass permissions; use --permission-mode instead")
    ap.add_argument("--permission-mode", default=None,
                    help="claude --permission-mode value when not skipping permissions")
    ap.add_argument("--model", default=None, help="optional top-level model override")
    ap.add_argument("--results-json", default=None,
                    help="path to the results.json to read/write (default: benchmark/results/results.json). "
                         "Use a separate file to keep runs apart -- e.g. a C2b Sonnet sweep vs the C2a run. "
                         "The REPORT.md and per-problem logs are written alongside it unless --report is given.")
    ap.add_argument("--report", default=None,
                    help="path for the rendered REPORT.md (default: REPORT.md next to the results JSON)")
    ap.add_argument("--list", action="store_true", help="print the selection and exit")
    args = ap.parse_args(argv)

    # Self-heal any agent-model override a prior interrupted --model run left applied,
    # before this run reads or edits the agent files (see override_agent_models).
    restore_agent_models_from_backup()

    # Redirect the results/report/logs to a caller-chosen location so parallel or
    # comparison runs never overwrite each other's data.
    global RESULTS_DIR, LOGS_DIR, RESULTS_JSON, REPORT_MD
    if args.results_json:
        RESULTS_JSON = os.path.abspath(args.results_json)
        RESULTS_DIR = os.path.dirname(RESULTS_JSON) or "."
        LOGS_DIR = os.path.join(RESULTS_DIR, "logs")
    if args.report:
        REPORT_MD = os.path.abspath(args.report)
    elif args.results_json:
        REPORT_MD = os.path.join(RESULTS_DIR, "REPORT.md")

    chosen = S.select(args.only)
    chosen = S.apply_suite_filter(chosen, args.only, args.no_closed_form, args.include_all)
    if args.type_filter:
        chosen = [p for p in chosen if p["type"] == args.type_filter]
    if args.tier:
        chosen = [p for p in chosen if p["tier"] == args.tier]
    if not chosen:
        raise SystemExit("no problems match the selection")

    if args.list:
        for p in chosen:
            print(f"{p['id']}  T{p['tier']}  {p['type']}  {p['slug']}")
        print(f"\n{len(chosen)} problem(s) selected.")
        return

    prior = load_prior()
    records = dict(prior)  # slug -> record; preserves earlier runs
    config = {
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "claude_version": claude_version(),
        "command_template": 'claude --plugin-dir . -p "/conductor workspace/{slug}/problem.md"',
        "timeout_seconds": args.timeout,
        "n_selected": len(chosen),
    }

    t_start = time.monotonic()
    # Pin the whole pipeline to --model (if given) for the duration of the loop, not
    # just the conductor -- see override_agent_models. Restored on exit / crash.
    with override_agent_models(args.model):
        config["agent_file_hashes"] = agent_file_hashes()
        write_results(records, config)
        for i, problem in enumerate(chosen, 1):
            slug = problem["slug"]
            if args.resume and slug in prior and prior[slug].get("run", {}).get("status") == "completed":
                print(f"[{i}/{len(chosen)}] {slug}: resume — skipping (already completed)")
                continue

            print(f"[{i}/{len(chosen)}] {slug} ({problem['type']} T{problem['tier']}): running…", flush=True)
            rec = process(problem, args)

            # If we hit the Claude usage limit, this problem did not really run and every
            # remaining one would fail identically. Stop now (don't record a spurious
            # error); the limit resets ~5 hours after it was hit, then --resume continues.
            if rec["run"].get("limit_hit") and rec["run"].get("status") != "completed":
                print("    -> USAGE LIMIT reached. Stopping.\n"
                      "       The Claude Pro limit resets ~5 hours after it was first hit.\n"
                      "       After it resets, continue where you left off with:\n"
                      "           uv run python benchmark/run.py --resume", flush=True)
                break

            records[slug] = rec
            config["invocation_seconds"] = round(time.monotonic() - t_start, 1)
            write_results(records, config)  # crash-safe incremental write

            metric = R.key_metric(rec) if R._has_independent_check(rec) else "no GT"
            print(f"    -> {R.VERDICT_LABEL[rec['verdict']]:<16} pipeline={rec['pipeline'].get('best_score')} "
                  f"{metric}  ({R._fmt_secs(rec['run'].get('wall_seconds'))})", flush=True)

    config["invocation_seconds"] = round(time.monotonic() - t_start, 1)
    write_results(records, config)

    md = R.generate_report(list(records.values()), config)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(REPORT_MD, "w") as fh:
        fh.write(md)

    # Console summary. Two distinct scopes, kept separate: what *this* invocation
    # selected, and the whole benchmark as it now stands in results.json (which also
    # carries problems from earlier invocations, e.g. under --resume).
    chosen_slugs = {p["slug"] for p in chosen}
    sel_done = sum(1 for slug in chosen_slugs
                   if records.get(slug, {}).get("run", {}).get("status") == "completed")
    counts = {}
    for r in records.values():
        counts[r.get("verdict")] = counts.get(r.get("verdict"), 0) + 1

    def _c(v):
        return counts.get(v, 0)

    print(f"\nDone. Selection: {sel_done}/{len(chosen)} completed "
          f"in {R._fmt_secs(config.get('invocation_seconds'))}.")
    print(f"Benchmark ({len(records)} problems): "
          f"verified-pass {_c('VERIFIED_PASS')} | stable-pass {_c('STABLE_PASS')} | "
          f"overclaim {_c('OVERCLAIM')} | blow-up {_c('BLOWUP')} | "
          f"underclaim {_c('UNDERCLAIM')} | fail {_c('FAIL')} | "
          f"unverified {_c('UNVERIFIED')} | self-only {_c('SELF_ONLY')} | "
          f"run-error {_c('RUN_ERROR')}")
    print(f"Report: {os.path.relpath(REPORT_MD, REPO_ROOT)}")
    print(f"Data:   {os.path.relpath(RESULTS_JSON, REPO_ROOT)}")


if __name__ == "__main__":
    main()
