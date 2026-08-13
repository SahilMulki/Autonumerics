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


def conductor_command(slug, skip_permissions, permission_mode, model):
    cmd = ["claude", "--plugin-dir", REPO_ROOT,
           "--settings", PIPELINE_SETTINGS, "-p",
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
    cmd = conductor_command(slug, skip_permissions, permission_mode, model)

    # Elapsed is measured on the monotonic clock, which is what subprocess's own
    # timeout uses. On macOS it does not advance while the system is asleep, so a
    # laptop sleeping mid-run no longer reports hours of "runtime" that never
    # happened (and that a 40-minute timeout appeared not to catch).
    t0 = time.monotonic()
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

    status = "completed" if (not timed_out and returncode == 0) else ("timeout" if timed_out else "error")
    return {
        "status": status,
        "returncode": returncode,
        "timed_out": timed_out,
        "wall_seconds": round(wall, 1),
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
    return any(phrase in text for phrase in _LIMIT_PHRASES)


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

    rec = {
        "id": problem["id"], "slug": slug, "type": problem["type"], "tier": problem["tier"],
        "family": problem["family"], "title": problem["title"], "challenge": problem["challenge"],
        "has_ground_truth": problem["has_ground_truth"],
        "ground_truth_kind": problem.get("ground_truth_kind", "exact"),
        "run": run, "pipeline": pipeline, "verify": verify,
    }
    rec["verdict"] = R.compute_verdict(rec)
    return rec


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
