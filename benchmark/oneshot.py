"""One-shot / single-agent baseline driver for the pipeline-vs-baseline comparison.

The AutoNumerics thesis is that the *structure* (formulator -> planner -> parallel
solver<->evaluator cycles) drives performance, not merely the underlying model. To
test that fairly this runs the same numerical problems through weaker structures on
the SAME model, so any gap is attributable to structure alone:

    C0  "naked one-shot"     : a single tool-less completion. Given only problem.md
                               (the same file the pipeline's agents read, contract
                               footer included), return solver.py. This is the
                               "paste it into a chat" baseline.
    C1  "single agent+tools" : one agent with Bash/Read/Write that may write, run,
                               and iterate on solver.py up to a turn budget -- but
                               with none of the pipeline's decomposition (no planner,
                               no separate evaluator, no parallel plans). Isolates
                               the value of the *structure* from mere tool access.

Each is additionally run "naked" (problem.md only) or "manual" (problem.md + the
same sde_manual.md / pde_manual.md the pipeline's agents get), to separate the
structure's contribution from the reference material's.

Grading is identical to the pipeline's: the produced solver.py is run through
verify.py's *sandboxed* path (wall-clock timeout, memory cap, and the same
numpy-only [SDE] / numpy+scipy.sparse [PDE] import rule the pipeline solvers
follow) and scored against problems.py ground truth at the fixed
(num_paths, eval dt, seed). A baseline that crashes, hangs, or violates the
contract simply fails -- see report.compute_oneshot_verdict.

This driver does NOT run the pipeline (that is run.py). It writes
results/oneshot_results.json; compare.py renders the head-to-head.

Examples
--------
    # cheap smoke test of the plumbing (do this before any real baseline)
    uv run python benchmark/oneshot.py --only sde_gbm,pde_heat_1d --condition c0 --model haiku

    # the discriminator subset, both conditions, both knowledge settings, Opus, 5 reps
    uv run python benchmark/oneshot.py --discriminators --condition c0,c1 \
        --manuals both --model opus --reps 5

    # resume after an interruption / usage-limit stop
    uv run python benchmark/oneshot.py --discriminators --model opus --reps 5 --resume
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
ONESHOT_ROOT = os.path.join(REPO_ROOT, "workspace_oneshot")
WORKSPACE_ROOT = os.path.join(REPO_ROOT, "workspace")
RESULTS_DIR = os.path.join(HERE, "results")
ONESHOT_JSON = os.path.join(RESULTS_DIR, "oneshot_results.json")
# Same benchmark/ read-denial the pipeline runs under -- see run.PIPELINE_SETTINGS.
PIPELINE_SETTINGS = os.path.join(HERE, "pipeline-settings.json")

sys.path.insert(0, HERE)
import report as R  # noqa: E402, I001
import setup as S  # noqa: E402
import verify as V  # noqa: E402

MANUAL_FILES = {"sde": "references/sde_manual.md", "pde": "references/pde_manual.md"}

# Reuse the pipeline runner's usage-limit phrases so a limit stops the driver
# cleanly (--resume continues after the reset) instead of recording spurious
# failures for every remaining unit.
_LIMIT_PHRASES = ("spend limit", "usage limit", "rate limit", "session limit",
                  "hit your monthly", "hit your session", "limit reached", "resets at",
                  "· resets", "insufficient credit", "quota")


# --- prompt construction ----------------------------------------------------

C0_HEADER = (
    "You are an expert in numerical methods and scientific computing. Your task is "
    "to implement a solver for the numerical problem below.\n\n"
    "The problem statement ends with a \"Solver contract\" section that specifies "
    "exactly the function to implement, the arguments it will be called with, the "
    "dict it must return, and how it will be scored. Follow that contract precisely "
    "-- in particular, use the step size (dt) you are told you will be called with.\n\n"
    "Respond with the complete contents of a single file `solver.py` inside ONE "
    "```python code block, and nothing else: no explanation before or after the "
    "code block."
)

C1_HEADER = (
    "You are an expert in numerical methods and scientific computing. Solve the "
    "numerical problem below by writing and testing code.\n\n"
    "Work as follows:\n"
    "1. Write your implementation to a file named `solver.py` in the current "
    "working directory.\n"
    "2. Run it (e.g. `python solver.py`) to confirm it executes without error.\n"
    "3. Debug and iterate until it runs cleanly and you are confident it satisfies "
    "the \"Solver contract\" at the end of the problem statement -- in particular, "
    "use the step size (dt) the contract says it will be called with.\n\n"
    "Your final deliverable is the file `solver.py` in the current working "
    "directory. Its function will be imported and scored automatically against the "
    "contract, so keep it importable (no work at module load outside "
    "`if __name__ == \"__main__\"`)."
)

C0_REMINDER = ("\n\n===== REMINDER =====\n"
               "Respond with ONLY the contents of `solver.py` in a single "
               "```python code block.")

MANUAL_PREAMBLE = (
    "\n\n===== REFERENCE: NUMERICAL METHODS MANUAL =====\n"
    "You may use the following reference material on schemes and formulas.\n\n")

PROBLEM_MARKER = "\n\n===== PROBLEM =====\n\n"


def read_problem_md(problem):
    """Return the exact problem.md both systems are given (contract footer and all).
    Rewrites it first so it always reflects the current problems.py."""
    S.write_problem_md(problem, WORKSPACE_ROOT)
    with open(os.path.join(WORKSPACE_ROOT, problem["slug"], "problem.md")) as fh:
        return fh.read()


def read_manual(problem):
    path = os.path.join(REPO_ROOT, MANUAL_FILES[problem["type"]])
    with open(path) as fh:
        return fh.read()


def build_prompt(problem, condition, manuals):
    header = C0_HEADER if condition == "c0" else C1_HEADER
    parts = [header]
    if manuals == "manual":
        parts.append(MANUAL_PREAMBLE + read_manual(problem))
    parts.append(PROBLEM_MARKER + read_problem_md(problem))
    if condition == "c0":
        parts.append(C0_REMINDER)
    return "".join(parts)


# --- model invocation -------------------------------------------------------


def claude_command(model, condition, max_turns):
    """Flags for a baseline invocation. C0 forbids tools (NoTool allow-list, so no
    tool ever executes and there is no iteration on results -- it stays the "naked
    one-shot" baseline); C1 grants file/run tools and a turn budget matched to the
    pipeline's ~15 solver<->evaluator cycles (with headroom)."""
    cmd = ["claude", "-p", "--output-format", "json"]
    if model:
        cmd += ["--model", model]
    if condition == "c0":
        # Allow-list semantics: naming a tool that does not exist permits no real
        # tool. max-turns is 2, NOT 1: on harder problems the model reflexively
        # emits a tool_use ("let me test this"), which the NoTool allow-list denies
        # -- and under max-turns 1 that denied attempt consumes the single turn,
        # ending as error_max_turns with an EMPTY result (the generated code is
        # discarded). This false-failed ~6% of units, worst on the hard problems the
        # comparison hinges on. max-turns 2 lets the model answer on a second turn
        # after the denial. It is a strict superset: a unit that answers directly on
        # turn 1 ends there regardless (num_turns==1), identical to max-turns 1;
        # only the tool-reaching units differ. Still tool-less -- no tool runs.
        cmd += ["--max-turns", "2", "--allowed-tools", "NoTool"]
    else:
        # C1 gets file/run tools, so it gets the same benchmark/ read-denial the
        # pipeline runs under (run.PIPELINE_SETTINGS). A baseline that read
        # problems.py for the ground truth would not be a baseline. C0 needs no
        # settings file: the NoTool allow-list means no tool ever executes.
        cmd += ["--max-turns", str(max_turns), "--dangerously-skip-permissions",
                "--settings", PIPELINE_SETTINGS,
                "--allowed-tools", "Bash", "Read", "Write", "Edit"]
    return cmd


def _looks_like_limit(text):
    low = (text or "").lower()
    return any(p in low for p in _LIMIT_PHRASES)


def invoke(prompt, model, condition, cwd, max_turns, timeout):
    """Run one baseline invocation. Returns a run-metadata dict (never raises)."""
    import subprocess

    cmd = claude_command(model, condition, max_turns)
    t0 = time.monotonic()
    stdout = stderr = ""
    status = "ok"
    try:
        proc = subprocess.run(cmd, cwd=cwd, input=prompt, capture_output=True,
                              text=True, timeout=timeout, check=False)
        stdout, stderr = proc.stdout or "", proc.stderr or ""
        if proc.returncode != 0:
            status = "error"
    except subprocess.TimeoutExpired as exc:
        status = "timeout"
        stdout = (exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout) or ""
        stderr = (exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr) or ""
    wall = round(time.monotonic() - t0, 1)

    parsed = _parse_output(stdout)
    limit_hit = _looks_like_limit(stdout) or _looks_like_limit(stderr) or \
        _looks_like_limit(parsed.get("result_text", ""))
    return {
        "agent_status": status,
        "wall_s": wall,
        "result_text": parsed.get("result_text", ""),
        "cost_usd": parsed.get("cost_usd"),
        "input_tokens": parsed.get("input_tokens"),
        "output_tokens": parsed.get("output_tokens"),
        "num_turns": parsed.get("num_turns"),
        "limit_hit": limit_hit,
        "stderr_tail": stderr[-1000:],
    }


def _is_transient(run):
    """A transient CLI failure worth retrying: a non-zero exit that produced no
    usable text, which is NOT a usage-limit stop and NOT a genuine wall-clock
    timeout. The common case is `error_max_turns` -- the model spent its single
    C0 turn on a denied tool-call attempt (num_turns comes back as 2 under
    --max-turns 1) and emitted no final code, so the whole completion is lost. A
    plain re-invocation of the identical prompt then succeeds. A usage limit must
    halt the driver (not retry), and retrying a 900s runaway would only burn time,
    so both are excluded."""
    return (run.get("agent_status") == "error"
            and not (run.get("result_text") or "").strip()
            and not run.get("limit_hit"))


def invoke_with_retry(prompt, model, condition, cwd, max_turns, timeout, retries):
    """invoke(), retrying a transient empty-error envelope up to `retries` times
    with a short backoff. Records `retries_used` for transparency so a padded
    pass-rate can never be mistaken for a first-try one."""
    attempt = 0
    run = invoke(prompt, model, condition, cwd, max_turns, timeout)
    while _is_transient(run) and attempt < retries:
        attempt += 1
        print(f"    (transient CLI error, empty output — retry {attempt}/{retries})", flush=True)
        time.sleep(2 * attempt)
        run = invoke(prompt, model, condition, cwd, max_turns, timeout)
    run["retries_used"] = attempt
    return run


def _parse_output(stdout):
    """Parse `claude -p --output-format json` output. Tolerant of leading noise."""
    stdout = (stdout or "").strip()
    obj = None
    if stdout:
        try:
            obj = json.loads(stdout)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}\s*$", stdout, re.S)
            if m:
                try:
                    obj = json.loads(m.group(0))
                except json.JSONDecodeError:
                    obj = None
    if not isinstance(obj, dict):
        return {"result_text": stdout}
    usage = obj.get("usage") or {}
    in_tok = usage.get("input_tokens")
    if in_tok is not None:
        in_tok += (usage.get("cache_read_input_tokens") or 0) + \
                  (usage.get("cache_creation_input_tokens") or 0)
    return {
        "result_text": obj.get("result", "") if isinstance(obj.get("result"), str) else "",
        "cost_usd": obj.get("total_cost_usd") or obj.get("cost_usd"),
        "input_tokens": in_tok,
        "output_tokens": usage.get("output_tokens"),
        "num_turns": obj.get("num_turns"),
    }


def extract_code(text):
    """Pull solver.py out of a C0 response: prefer a fenced block that defines the
    contract function, else the longest fenced block."""
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        return None
    for b in blocks:
        if "def solve_sde" in b or "def solve_pde" in b:
            return b.strip("\n")
    return max(blocks, key=len).strip("\n")


# --- one unit (problem x condition x manuals x rep) -------------------------


def _model_tag(model):
    return re.sub(r"[^A-Za-z0-9._-]", "_", model or "default")


def plan_dir_for(problem, condition, manuals, model, rep):
    return os.path.join(ONESHOT_ROOT, _model_tag(model), problem["slug"],
                        f"{condition}-{manuals}", f"rep{rep}")


def unit_key(slug, condition, manuals, model, rep):
    return f"{slug}|{condition}|{manuals}|{_model_tag(model)}|{rep}"


def process_unit(problem, condition, manuals, model, rep, args):
    slug = problem["slug"]
    plan_dir = plan_dir_for(problem, condition, manuals, model, rep)
    os.makedirs(plan_dir, exist_ok=True)

    prompt = build_prompt(problem, condition, manuals)
    with open(os.path.join(plan_dir, "prompt.txt"), "w") as fh:
        fh.write(prompt)

    # C1 writes solver.py itself in its cwd; make sure a stale one from a prior rep
    # cannot be silently reused/scored.
    solver_path = os.path.join(plan_dir, "solver.py")
    if os.path.exists(solver_path):
        os.remove(solver_path)

    run = invoke_with_retry(prompt, model, condition, cwd=plan_dir,
                            max_turns=args.max_turns, timeout=args.agent_timeout,
                            retries=args.retries)
    with open(os.path.join(plan_dir, "response.json"), "w") as fh:
        json.dump(run, fh, indent=2)

    if condition == "c0":
        code = extract_code(run["result_text"])
        if code:
            with open(solver_path, "w") as fh:
                fh.write(code + "\n")
    # (C1 leaves solver.py on disk in plan_dir; nothing to extract.)

    verify = V.verify_problem(problem, plan_dir, plan_dir=plan_dir, sandbox=True,
                              timeout_s=args.timeout, mem_mb=args.mem_mb)
    with open(os.path.join(plan_dir, "verify.json"), "w") as fh:
        json.dump(verify, fh, indent=2, default=_json_default)

    rec = {
        "id": problem["id"], "slug": slug, "type": problem["type"], "tier": problem["tier"],
        "family": problem["family"], "title": problem["title"], "challenge": problem["challenge"],
        "has_ground_truth": problem["has_ground_truth"],
        "ground_truth_kind": problem.get("ground_truth_kind", "exact"),
        "discriminator": bool(problem.get("discriminator")),
        "condition": condition, "manuals": manuals, "model": model, "rep": rep,
        "run": run, "verify": verify,
        "plan_dir": os.path.relpath(plan_dir, REPO_ROOT),
    }
    rec["verdict"] = R.compute_oneshot_verdict(rec)
    return rec


# --- results IO -------------------------------------------------------------


def _json_default(o):
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


def load_prior():
    if not os.path.exists(ONESHOT_JSON):
        return {}
    try:
        with open(ONESHOT_JSON) as fh:
            payload = json.load(fh)
        return {r["_key"]: r for r in payload.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def write_results(records, config):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = ONESHOT_JSON + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"config": config, "results": list(records.values())}, fh, indent=2,
                  default=_json_default)
    os.replace(tmp, ONESHOT_JSON)


# --- selection / main -------------------------------------------------------


def select_problems(args):
    chosen = S.select(args.only)
    if args.discriminators:
        chosen = [p for p in chosen if p.get("discriminator")]
    if args.type_filter:
        chosen = [p for p in chosen if p["type"] == args.type_filter]
    if args.tier:
        chosen = [p for p in chosen if p["tier"] == args.tier]
    return chosen


def expand_conditions(args):
    conds = [c.strip() for c in args.condition.split(",") if c.strip()]
    bad = set(conds) - {"c0", "c1"}
    if bad:
        raise SystemExit(f"unknown condition(s): {sorted(bad)} (use c0 and/or c1)")
    manuals = ["naked", "manual"] if args.manuals == "both" else [args.manuals]
    return conds, manuals


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="", help="comma-separated slugs or ids (default: all)")
    ap.add_argument("--discriminators", action="store_true",
                    help="restrict to the hard discriminator subset")
    ap.add_argument("--type", dest="type_filter", choices=("pde", "sde"), default=None)
    ap.add_argument("--tier", type=int, choices=(1, 2, 3), default=None)
    ap.add_argument("--condition", default="c0,c1", help="c0, c1, or c0,c1 (default both)")
    ap.add_argument("--manuals", choices=("naked", "manual", "both"), default="both",
                    help="knowledge setting: problem.md only, +manual, or both (default)")
    ap.add_argument("--model", default="haiku",
                    help="model for the baseline (default 'haiku' -- set 'opus' for the "
                         "real baseline; kept cheap by default so an accidental run is cheap)")
    ap.add_argument("--reps", type=int, default=1, help="samples per unit (default 1)")
    ap.add_argument("--max-turns", type=int, default=30,
                    help="C1 agent turn budget (default 30; ~matched to the pipeline's "
                         "3 plans x 5 iters with headroom)")
    ap.add_argument("--agent-timeout", type=int, default=900,
                    help="per-invocation wall-clock timeout in seconds (default 900)")
    ap.add_argument("--retries", type=int, default=2,
                    help="retry a transient empty-error CLI envelope (e.g. error_max_turns "
                         "from a denied tool-call attempt that emitted no code) up to this "
                         "many times (default 2); usage-limit stops and real timeouts are "
                         "never retried")
    ap.add_argument("--timeout", type=int, default=120,
                    help="sandbox verification timeout in seconds (default 120)")
    ap.add_argument("--mem-mb", type=int, default=4096,
                    help="sandbox memory cap in MB, best-effort (default 4096)")
    ap.add_argument("--resume", action="store_true",
                    help="skip units already present in oneshot_results.json")
    ap.add_argument("--list", action="store_true", help="print the unit plan and exit")
    args = ap.parse_args(argv)

    chosen = select_problems(args)
    conds, manuals = expand_conditions(args)
    if not chosen:
        raise SystemExit("no problems match the selection")

    units = [(p, c, m, rep) for p in chosen for c in conds
             for m in manuals for rep in range(1, args.reps + 1)]

    if args.list:
        for p, c, m, rep in units:
            print(f"{p['id']:<4} {p['slug']:<34} {c} {m:<6} rep{rep}")
        print(f"\n{len(units)} unit(s): {len(chosen)} problem(s) x {len(conds)} condition(s) "
              f"x {len(manuals)} manual-setting(s) x {args.reps} rep(s), model={args.model}")
        return

    prior = load_prior()
    records = dict(prior)
    config = {
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": args.model, "conditions": conds, "manuals": manuals, "reps": args.reps,
        "max_turns": args.max_turns, "agent_timeout": args.agent_timeout,
    }

    t_start = time.monotonic()
    for i, (problem, cond, man, rep) in enumerate(units, 1):
        key = unit_key(problem["slug"], cond, man, args.model, rep)
        label = f"{problem['slug']} [{cond}/{man}] rep{rep} ({args.model})"
        if args.resume and key in prior:
            print(f"[{i}/{len(units)}] {label}: resume — skipping (already recorded)")
            continue

        print(f"[{i}/{len(units)}] {label}: running…", flush=True)
        rec = process_unit(problem, cond, man, args.model, rep, args)
        rec["_key"] = key

        if rec["run"].get("limit_hit"):
            print("    -> USAGE LIMIT reached. Stopping.\n"
                  "       After the limit resets, continue with the same command plus --resume.",
                  flush=True)
            break

        records[key] = rec
        config["invocation_seconds"] = round(time.monotonic() - t_start, 1)
        write_results(records, config)

        vlabel = R.ONESHOT_VERDICT_LABEL.get(rec["verdict"], rec["verdict"])
        cost = rec["run"].get("cost_usd")
        cost_s = f"${cost:.3f}" if isinstance(cost, (int, float)) else "$?"
        print(f"    -> {vlabel:<10} {R.key_metric(rec)}  "
              f"turns={rec['run'].get('num_turns')} {cost_s} "
              f"({R._fmt_secs(rec['run'].get('wall_s'))})", flush=True)

    config["invocation_seconds"] = round(time.monotonic() - t_start, 1)
    write_results(records, config)

    done = [r for r in records.values() if r.get("model") == args.model]
    counts = {}
    for r in done:
        counts[r["verdict"]] = counts.get(r["verdict"], 0) + 1
    checkable = [r for r in done if r["verdict"] != "NO_GT"]
    npass = sum(1 for r in checkable if R.oneshot_passed(r["verdict"]))
    print(f"\nDone. {len(done)} unit(s) for model={args.model} "
          f"in {R._fmt_secs(config.get('invocation_seconds'))}.")
    print("Verdicts: " + " | ".join(
        f"{R.ONESHOT_VERDICT_LABEL[v]} {counts[v]}"
        for v in R._ONESHOT_VERDICT_ORDER if counts.get(v)))
    print(f"Independently-passed: {npass}/{len(checkable)} checkable unit(s).")
    print(f"Data: {os.path.relpath(ONESHOT_JSON, REPO_ROOT)}  "
          f"(render the head-to-head with: uv run python benchmark/compare.py)")


if __name__ == "__main__":
    main()
