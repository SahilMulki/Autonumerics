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

# Why the no-tools paragraph is explicit rather than implied: --allowed-tools NoTool
# blocks tool EXECUTION, but the model is still handed the tool schema, so it sees
# Bash and plans around having it. Measured 2026-09-27: Opus emitted tool_use on 36 of
# 44 C0 units -- writing convergence studies, running `ls -la`, checking numpy's
# version -- was refused each time, and burned its turn budget before committing a
# solver. Saying "no tools" up front removes the surprise instead of paying extra
# turns to absorb it. C0 was always defined as a naked one-shot; a prompt that let the
# model believe otherwise was a defect in the condition, not a property of it.
C0_HEADER = (
    "You are an expert in numerical methods and scientific computing. Your task is "
    "to implement a solver for the numerical problem below.\n\n"
    "IMPORTANT -- you have NO TOOLS in this session. You cannot run code, read or "
    "write files, inspect your environment, or test your implementation. Any tool "
    "call will be rejected and will consume the budget you need to answer, so do not "
    "attempt one. You get exactly one reply: write the solver directly from your own "
    "knowledge of numerical analysis, choose your scheme and step size by reasoning "
    "rather than by experiment, and make it correct in a single pass.\n\n"
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
               "You have no tools: do not attempt to run or test anything. Respond "
               "with ONLY the contents of `solver.py` in a single ```python code "
               "block.")

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
    # stream-json (not json): the final `result` field is EMPTY on an
    # error_max_turns stop, so a completed solver written on the turn before a
    # denied tool call is lost. The streamed assistant turns still carry it, and
    # `invoke(salvage=True)` recovers it. The last event of the stream is the same
    # result object the `json` format returns, so no metadata is given up. The
    # model sees an identical prompt, tool set and turn budget either way -- this
    # changes what the harness can read, never what the baseline is allowed to do.
    cmd = ["claude", "-p", "--output-format", "stream-json", "--verbose"]
    if model:
        cmd += ["--model", model]
    if condition == "c0":
        # Allow-list semantics: naming a tool that does not exist permits no real
        # tool. The cap is 2 by default, NOT 1: on harder problems the model
        # reflexively emits a tool_use ("let me test this"), which the NoTool
        # allow-list denies -- and under max-turns 1 that denied attempt consumes the
        # single turn, ending as error_max_turns with an EMPTY result (the generated
        # code is discarded). This false-failed ~6% of units. Raising the cap is a
        # strict superset: a unit that answers directly on turn 1 ends there
        # regardless, so only the tool-reaching units differ. Still tool-less at any
        # cap -- no tool ever runs.
        #
        # 2 is not enough for Opus (measured 2026-09-27: 36/44 units produced no
        # solver at all, because it spends both turns writing convergence studies and
        # probing its environment). --c0-max-turns raises it PER RUN, recorded in the
        # results config, so a suite can opt in without silently redefining the C0 of
        # a baseline measured at a different cap. Compare only runs that share it.
        cmd += ["--max-turns", str(max_turns), "--allowed-tools", "NoTool"]
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


def invoke(prompt, model, condition, cwd, max_turns, timeout, salvage=False):
    """Run one baseline invocation. Returns a run-metadata dict (never raises).

    `salvage` (C0 only): when the CLI returns an empty final result -- the
    error_max_turns stop, where the model wrote the solver and then spent its last
    turn on a tool call the NoTool allow-list denied -- take the code from the
    streamed assistant turns instead of discarding the completion. This recovers
    work the model actually finished; it does not grant a retry, a tool, or an extra
    turn, so the baseline stays the naked one-shot it is defined to be. Always
    recorded (`salvage_used`, `code_source`) so a salvaged pass can never be
    mistaken for a clean one.
    """
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

    result_text = parsed.get("result_text", "") or ""
    assistant_text = parsed.get("assistant_text", "") or ""
    code_text, code_source, salvage_used = result_text, "result", False
    if salvage and not result_text.strip():
        if extract_code(assistant_text):
            code_text, code_source, salvage_used = assistant_text, "assistant_turns", True
        else:
            # last Write/Edit wins: a later file body supersedes an earlier draft.
            # Only a payload that defines the contract function counts -- these runs
            # also write scratch experiment scripts, which are not answers.
            for payload in reversed(parsed.get("tool_payloads") or []):
                if defines_contract(payload):
                    code_text, code_source, salvage_used = payload, "denied_write", True
                    break

    return {
        "agent_status": status,
        "wall_s": wall,
        "result_text": result_text,
        # only when it differs from result_text, so a normal response.json is not
        # stored twice over; process_unit falls back to result_text when absent.
        "code_text": code_text if salvage_used else "",
        "code_source": code_source,
        "salvage_used": salvage_used,
        "stop_subtype": parsed.get("stop_subtype"),
        "denied_tools": parsed.get("denied_tools") or [],
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
    if run.get("salvage_used") and extract_code(run.get("code_text") or ""):
        return False   # the completion was recovered; there is nothing to re-earn
    return (run.get("agent_status") == "error"
            and not (run.get("result_text") or "").strip()
            and not run.get("limit_hit"))


def invoke_with_retry(prompt, model, condition, cwd, max_turns, timeout, retries,
                      salvage=False):
    """invoke(), retrying a transient empty-error envelope up to `retries` times
    with a short backoff. Records `retries_used` for transparency so a padded
    pass-rate can never be mistaken for a first-try one."""
    attempt = 0
    run = invoke(prompt, model, condition, cwd, max_turns, timeout, salvage=salvage)
    while _is_transient(run) and attempt < retries:
        attempt += 1
        print(f"    (transient CLI error, empty output — retry {attempt}/{retries})", flush=True)
        time.sleep(2 * attempt)
        run = invoke(prompt, model, condition, cwd, max_turns, timeout, salvage=salvage)
    run["retries_used"] = attempt
    if run.get("salvage_used"):
        print(f"    (recovered the solver from the assistant turns after "
              f"{run.get('stop_subtype') or 'an empty result'})", flush=True)
    return run


def _split_stream(stdout):
    """Split stream-json NDJSON into (result_event, assistant_text, tool_payloads).

    assistant_text concatenates the `text` blocks of every assistant event, in order.
    tool_payloads collects the file bodies of attempted Write/Edit calls -- under the
    C0 allow-list none of them executed, but a solver the model tried to write to disk
    is still the solver it produced. `thinking` blocks are deliberately excluded from
    both: reasoning about a solver is not the solver. Unparseable lines are skipped --
    a truncated stream should still yield whatever turns completed.
    """
    result_obj, texts, tool_payloads, n_events = None, [], [], 0
    for line in (stdout or "").splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(ev, dict):
            continue
        n_events += 1
        if ev.get("type") == "result":
            result_obj = ev
        elif ev.get("type") == "assistant":
            for block in (ev.get("message") or {}).get("content") or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "text":
                    texts.append(block.get("text") or "")
                elif block.get("type") == "tool_use":
                    inp = block.get("input") or {}
                    for field in ("content", "new_string", "file_text"):
                        val = inp.get(field)
                        if isinstance(val, str) and val.strip():
                            tool_payloads.append(val)
    return result_obj, "\n".join(t for t in texts if t), tool_payloads, n_events


def _parse_output(stdout):
    """Parse `claude -p` output, either `--output-format json` (one object) or
    `stream-json` (NDJSON). Tolerant of leading noise."""
    stdout = (stdout or "").strip()
    obj, assistant_text, tool_payloads = None, "", []
    if stdout:
        try:
            obj = json.loads(stdout)
        except json.JSONDecodeError:
            obj, assistant_text, tool_payloads, n_events = _split_stream(stdout)
            if obj is None and n_events:
                # A stream cut off before its result event -- the agent_timeout kill is
                # the usual cause. Do NOT fall through to returning raw stdout as
                # result_text: that would hand back ~100KB of NDJSON that reads as a
                # non-empty answer and so blocks salvage on the very turns the stream
                # did deliver. Synthesise an empty result and let salvage do its job.
                obj = {"result": "", "subtype": "stream_truncated"}
            if obj is None:
                m = re.search(r"\{.*\}\s*$", stdout, re.S)
                if m:
                    try:
                        obj = json.loads(m.group(0))
                    except json.JSONDecodeError:
                        obj = None
    if not isinstance(obj, dict):
        return {"result_text": stdout, "assistant_text": assistant_text,
                "tool_payloads": tool_payloads}
    usage = obj.get("usage") or {}
    in_tok = usage.get("input_tokens")
    if in_tok is not None:
        in_tok += (usage.get("cache_read_input_tokens") or 0) + \
                  (usage.get("cache_creation_input_tokens") or 0)
    return {
        "result_text": obj.get("result", "") if isinstance(obj.get("result"), str) else "",
        "assistant_text": assistant_text,
        "tool_payloads": tool_payloads,
        "cost_usd": obj.get("total_cost_usd") or obj.get("cost_usd"),
        "input_tokens": in_tok,
        "output_tokens": usage.get("output_tokens"),
        "num_turns": obj.get("num_turns"),
        "stop_subtype": obj.get("subtype"),
        "denied_tools": sorted({d.get("tool_name") for d in (obj.get("permission_denials") or [])
                                if isinstance(d, dict) and d.get("tool_name")}),
    }


CONTRACT_FNS = ("def solve_pde", "def solve_sde")


def defines_contract(text):
    return any(fn in (text or "") for fn in CONTRACT_FNS)


def extract_code(text):
    """Pull solver.py out of a C0 response: prefer a fenced block that defines the
    contract function, else the longest fenced block. With no fences at all, accept
    the text itself only if it defines the contract function -- that covers a model
    answering in bare code, and a solver recovered from a denied Write payload, while
    still refusing prose."""
    if not text:
        return None
    blocks = re.findall(r"```(?:python|py)?[ \t]*\n(.*?)```", text, re.S)
    if not blocks:
        return text.strip("\n") if defines_contract(text) else None
    for b in blocks:
        if defines_contract(b):
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


def reps_per_problem(chosen, args):
    """``{slug: reps}``. Uniform ``--reps`` unless ``--reps-from`` names a matching
    file, in which case each problem gets as many attempts as the pipeline spent on
    it (``problems.<slug>.attempts`` -- the count of distinct ``solver.py`` files it
    wrote).

    Why per problem at all: on the no-closed-form suite the pipeline never refined a
    plan (every plan ran at ``iter: 1``), so the only thing there is to match is the
    number of *independent* attempts, and that number differs per problem -- 2 on the
    eigenproblem, 4 on KdV. A uniform ``--reps`` would hand the baseline more chances
    than the pipeline had on some problems and fewer on others, and the comparison
    would not be about structure any more. A problem absent from the file falls back
    to ``--reps`` and is reported, because silently giving it the default is exactly
    the error this flag exists to prevent."""
    if not args.reps_from:
        return {p["slug"]: args.reps for p in chosen}
    with open(args.reps_from) as fh:
        table = (json.load(fh).get("problems") or {})
    out, missing = {}, []
    for p in chosen:
        entry = table.get(p["slug"]) or {}
        n = entry.get("attempts")
        if isinstance(n, int) and n > 0:
            out[p["slug"]] = n
        else:
            out[p["slug"]] = args.reps
            missing.append(p["slug"])
    if missing:
        print(f"note: no matched attempt count for {', '.join(missing)} in "
              f"{os.path.relpath(args.reps_from, REPO_ROOT)}; using --reps={args.reps}")
    return out


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

    turn_cap = getattr(args, "c0_max_turns", 2) if condition == "c0" else args.max_turns
    run = invoke_with_retry(prompt, model, condition, cwd=plan_dir,
                            max_turns=turn_cap, timeout=args.agent_timeout,
                            retries=args.retries, salvage=(condition == "c0"))
    with open(os.path.join(plan_dir, "response.json"), "w") as fh:
        json.dump(run, fh, indent=2)

    if condition == "c0":
        code = extract_code(run.get("code_text") or run["result_text"])
        if code:
            with open(solver_path, "w") as fh:
                fh.write(code + "\n")
    # (C1 leaves solver.py on disk in plan_dir; nothing to extract.)

    verify = V.verify_problem(problem, plan_dir, plan_dir=plan_dir, sandbox=True,
                              timeout_s=args.timeout, mem_mb=args.mem_mb,
                              work_precision=getattr(args, "work_precision", True))
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


def load_prior(path=None):
    path = path or ONESHOT_JSON
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            payload = json.load(fh)
        return {r["_key"]: r for r in payload.get("results", [])}
    except (json.JSONDecodeError, KeyError):
        return {}


def write_results(records, config, path=None):
    path = path or ONESHOT_JSON
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump({"config": config, "results": list(records.values())}, fh, indent=2,
                  default=_json_default)
    os.replace(tmp, path)


# --- selection / main -------------------------------------------------------


def oneshot_pass_local(rec):
    return R.oneshot_passed(rec.get("verdict"))


def select_problems(args):
    chosen = S.select(args.only)
    chosen = S.apply_suite_filter(chosen, args.only, args.no_closed_form,
                                args.include_all, args.suite)
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
    S.add_suite_arguments(ap)
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
    ap.add_argument("--reps-from", dest="reps_from", default=None,
                    help="path to a matched-attempts JSON (benchmark/results/matched_attempts.json): "
                         "give each problem as many attempts as the pipeline spent on it, instead of "
                         "a uniform --reps. Problems missing from the file fall back to --reps.")
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
    ap.add_argument("--timeout", type=int, default=V.DEFAULT_SOLVER_TIMEOUT_S,
                    help=f"sandbox verification timeout per solve in seconds "
                         f"(default {V.DEFAULT_SOLVER_TIMEOUT_S}; shared with run.py so the "
                         f"two sides are measured under one budget)")
    ap.add_argument("--mem-mb", type=int, default=V.DEFAULT_SOLVER_MEM_MB,
                    help=f"sandbox memory cap in MB, best-effort "
                         f"(default {V.DEFAULT_SOLVER_MEM_MB})")
    ap.add_argument("--no-work-precision", dest="work_precision", action="store_false",
                    default=True,
                    help="skip the extra solves of the work-precision sweep. On by "
                         "default, as in run.py: a baseline swept and a pipeline not "
                         "swept would have no common error-vs-time axis, which is the "
                         "comparison the sweep exists for")
    ap.add_argument("--c0-max-turns", dest="c0_max_turns", type=int, default=2,
                    help="C0 turn cap (default 2). Still tool-less at any value -- the "
                         "NoTool allow-list means no tool ever runs; a larger cap only "
                         "lets the model answer after its denied tool attempts. Opus "
                         "needs ~4: at 2 it spends both turns experimenting and commits "
                         "no solver. Recorded in the results config -- compare only runs "
                         "that share a cap.")
    ap.add_argument("--resume", action="store_true",
                    help="skip units already present in the results JSON")
    ap.add_argument("--redo", default="",
                    help="comma-separated verdicts to re-run even under --resume, e.g. "
                         "'CRASHED' or 'CRASHED,RUN_ERROR'. The prior record is kept "
                         "until the new one overwrites it, so a re-run that fails again "
                         "leaves no gap. Use it to retry harness-caused failures without "
                         "re-paying for the units that already answered.")
    ap.add_argument("--results-json", dest="results_json", default=None,
                    help="path to the results JSON to read/write (default: "
                         "results/oneshot_results.json). Use a separate file to keep a sweep apart "
                         "from the standing C0/C1 baseline -- the no-closed-form suite does.")
    ap.add_argument("--list", action="store_true", help="print the unit plan and exit")
    args = ap.parse_args(argv)

    chosen = select_problems(args)
    conds, manuals = expand_conditions(args)
    if not chosen:
        raise SystemExit("no problems match the selection")

    reps = reps_per_problem(chosen, args)
    units = [(p, c, m, rep) for p in chosen for c in conds
             for m in manuals for rep in range(1, reps[p["slug"]] + 1)]

    if args.list:
        for p, c, m, rep in units:
            print(f"{p['id']:<4} {p['slug']:<34} {c} {m:<6} rep{rep}/{reps[p['slug']]}")
        spread = (f"{min(reps.values())}-{max(reps.values())} matched rep(s)"
                  if args.reps_from else f"{args.reps} rep(s)")
        print(f"\n{len(units)} unit(s): {len(chosen)} problem(s) x {len(conds)} condition(s) "
              f"x {len(manuals)} manual-setting(s) x {spread}, model={args.model}")
        return

    S.guard_results_path(chosen, args.results_json,
                         "benchmark/results/baselines/oneshot_{suite}.json")

    prior = load_prior(args.results_json)
    records = dict(prior)
    redo = {v.strip().upper() for v in args.redo.split(",") if v.strip()}
    stale = {k for k, r in prior.items() if (r.get("verdict") or "").upper() in redo}
    if stale:
        print(f"--redo {sorted(redo)}: {len(stale)} recorded unit(s) will be re-run "
              f"(their prior records stay until overwritten)")
    config = {
        "generated_at": _dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
        "model": args.model, "conditions": conds, "manuals": manuals,
        "reps": reps if args.reps_from else args.reps,
        "reps_from": (os.path.relpath(args.reps_from, REPO_ROOT) if args.reps_from else None),
        "max_turns": args.max_turns, "c0_max_turns": args.c0_max_turns,
        "agent_timeout": args.agent_timeout,
        "solver_timeout_s": args.timeout, "mem_mb": args.mem_mb,
        "work_precision": args.work_precision,
    }

    t_start = time.monotonic()
    for i, (problem, cond, man, rep) in enumerate(units, 1):
        key = unit_key(problem["slug"], cond, man, args.model, rep)
        label = f"{problem['slug']} [{cond}/{man}] rep{rep} ({args.model})"
        if args.resume and key in prior and key not in stale:
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
        write_results(records, config, args.results_json)

        vlabel = R.ONESHOT_VERDICT_LABEL.get(rec["verdict"], rec["verdict"])
        cost = rec["run"].get("cost_usd")
        cost_s = f"${cost:.3f}" if isinstance(cost, (int, float)) else "$?"
        print(f"    -> {vlabel:<10} {R.key_metric(rec)}  "
              f"turns={rec['run'].get('num_turns')} {cost_s} "
              f"({R._fmt_secs(rec['run'].get('wall_s'))})", flush=True)

    config["invocation_seconds"] = round(time.monotonic() - t_start, 1)
    # MUST pass args.results_json: a bare call defaults to ONESHOT_JSON and
    # overwrites the standing C0/C1 baseline with this sweep's records (it did,
    # 2026-09-27 -- the incremental write above was path-aware and this one was not).
    write_results(records, config, args.results_json)

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
    salvaged = [r for r in done if (r["run"] or {}).get("salvage_used")]
    if salvaged:
        # Stated, never silent: these units' code came from the assistant turns after
        # an empty final result, not from the CLI's result field.
        print(f"Salvaged from assistant turns: {len(salvaged)}/{len(done)} unit(s) "
              f"({sum(1 for r in salvaged if oneshot_pass_local(r))} of them passing).")
    print(f"Data: {os.path.relpath(args.results_json or ONESHOT_JSON, REPO_ROOT)}  "
          f"(render the head-to-head with: uv run python benchmark/compare.py)")


if __name__ == "__main__":
    main()
