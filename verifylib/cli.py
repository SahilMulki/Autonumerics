#!/usr/bin/env python3
"""The single entry point. Invoked by absolute path so it works from any cwd.

    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py check-spec <path>
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook post-tool-use     # payload on stdin
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook pre-tool-use
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py hook stop
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py gate <workspace_dir> [--json]
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py evaluate <plan_dir> [--json]
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py compare <plan_a> <plan_b> [--N=<int>] [--no-record] [--json]
    ${CLAUDE_PLUGIN_ROOT}/verifylib/cli.py rank <workspace_dir> [--json]

Exit codes follow the hook contract measured in ``tests/hookprobe/FINDINGS.md``:
2 is the blocking/feedback code the agent sees on stderr, 0 is pass. Only
``PreToolUse`` and ``Stop`` actually block on 2; ``PostToolUse`` exit 2 delivers
stderr to the agent but leaves the file on disk, which is why nothing depends on
it alone.
"""

from __future__ import annotations

import os
import sys

# Run as a script, sys.path[0] is *this* directory -- which contains operator.py and
# would therefore shadow the stdlib `operator` module for the whole interpreter.
# collections imports operator, functools imports collections, re imports functools,
# json imports re: the failure surfaces as a circular-import error inside stdlib
# json, with nothing pointing at the real cause. Drop this directory and add the
# package root before importing anything else. `sys` and `os` are safe to import
# first; neither reaches operator.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _HERE]
sys.path.insert(0, os.path.dirname(_HERE))

import json  # noqa: E402
import re  # noqa: E402

from verifylib import gate  # noqa: E402
from verifylib.findings import errors, render  # noqa: E402

BLOCK = 2
PASS = 0
#: `compare` only. A moment-space difference carries Monte-Carlo noise, so its
#: confidence interval can straddle the tolerance: neither an agreement that could
#: be priced as a circularity break, nor a disagreement that caps both plans at 8.
#: Distinct from FAIL (1), which means a plan crashed and nothing was measured.
INCONCLUSIVE = 3
#: A plan crashed: nothing was measured.
FAIL = 1

#: `matcher` is tool-name only -- a path glob never fires -- so path filtering
#: happens here, and the matcher must include Bash or a heredoc walks past it.
_REDIRECT_RE = re.compile(r">>?\s*([^\s|&;<>]+)")
_HEREDOC_RE = re.compile(r"<<-?\s*'?\"?(\w+)")


def _targets_from_bash(command: str) -> list[str]:
    """Files a Bash command writes to. ``cat > problem_spec.json`` fires no
    Write|Edit hook at all, and the file lands unvalidated."""
    return [m.group(1) for m in _REDIRECT_RE.finditer(command or "")]


def hook_targets(payload: dict) -> list[str]:
    tool = payload.get("tool_name") or ""
    tool_input = payload.get("tool_input") or {}
    if tool == "Bash":
        return _targets_from_bash(tool_input.get("command", ""))
    path = tool_input.get("file_path") or tool_input.get("path")
    return [path] if path else []


def _resolve(path, payload):
    if os.path.isabs(path):
        return path
    root = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    return os.path.join(root, path)


def _read_payload():
    try:
        return json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        return {}


# --- hook entry points -------------------------------------------------------

def hook_post_tool_use(payload) -> int:
    """Feedback after a write. Cannot block (measured), so this is an optimisation
    over the out-of-band gate, never a replacement for it."""
    found = []
    for target in hook_targets(payload):
        path = _resolve(target, payload)
        # with_reference=False: the schema half is instant; the residual check
        # evaluates a formula on up to 1025^d points and belongs at the gate.
        found.extend(gate.check_path(path, with_reference=False))
    if not errors(found):
        return PASS
    print(render(errors(found)), file=sys.stderr)
    return BLOCK


def hook_pre_tool_use(payload) -> int:
    """The only hook that actually blocks. Used where blocking matters: leakage
    into ``solver.py`` / ``evaluate.py``, where a ``Write`` carries the whole
    content and can be checked before it exists.

    An ``Edit`` carries only ``old_string``/``new_string``, so the finished file
    cannot be reconstructed here -- those fall through to the gate rather than
    being validated against a fragment.
    """
    tool_input = payload.get("tool_input") or {}
    content = tool_input.get("content")
    found = []
    for target in hook_targets(payload):
        if os.path.basename(target) not in gate.SOLVER_FILES:
            continue
        if isinstance(content, str):
            found.extend(gate.check_solver_text(content, path=target))
        else:
            found.extend(gate.check_path(_resolve(target, payload), with_reference=False))
    if not errors(found):
        return PASS
    print(render(errors(found)), file=sys.stderr)
    return BLOCK


def hook_stop(payload) -> int:
    """The gate that works everywhere -- including a user's own problem, where
    ``benchmark/verify.py`` does not exist.

    Bounded on purpose. Measured, an unbounded Stop hook fired 8 times and killed
    the run at max-turns, so the attempt counter is mandatory rather than
    defensive: on the 3rd firing for the same finding it records the reason and
    lets the session end.
    """
    root = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    workspace = os.path.join(root, "workspace")
    target = gate.active_problem(workspace if os.path.isdir(workspace) else root)

    found = errors(gate.check_workspace(target))
    if not found:
        gate.clear_stop_attempts(target)
        return PASS

    attempt = gate.record_stop_attempt(target, gate.fingerprint(found))
    if attempt >= gate.MAX_STOP_ATTEMPTS:
        marker = gate.write_unsatisfied_marker(target, found)
        print(f"{gate.UNSATISFIED_MARKER}: {len(found)} finding(s) still outstanding "
              f"after {attempt} attempts. Recorded in {marker}; ending the session.",
              file=sys.stderr)
        gate.clear_stop_attempts(target)
        return PASS
    print(f"Guardrail findings must be fixed before this session ends "
          f"(attempt {attempt} of {gate.MAX_STOP_ATTEMPTS}):\n{render(found)}",
          file=sys.stderr)
    return BLOCK


HOOKS = {"post-tool-use": hook_post_tool_use,
         "pre-tool-use": hook_pre_tool_use,
         "stop": hook_stop}


# --- direct entry points -----------------------------------------------------

def _report(found, as_json):
    if as_json:
        print(json.dumps([f.as_dict() for f in found], indent=2))
    elif found:
        print(render(found))
    else:
        print("ok: no findings")
    return BLOCK if errors(found) else PASS


def _evaluate(plan_dir, as_json) -> int:
    """Run the numerical kernel on one plan directory.

    This is the entry point that replaces a generated ``evaluate.py``. It is a CLI
    subcommand rather than a ``verifylib.run(...)`` call inside a generated script
    for three reasons: ``import verifylib`` only resolves inside this repo, while on
    a user's own problem the package lives at ``${CLAUDE_PLUGIN_ROOT}``; this module
    already does the ``sys.path`` surgery that stops ``verifylib/operator.py``
    shadowing the stdlib ``operator`` module for the whole interpreter, which the
    kernel would hit the moment it imported ``functools``; and "no ``evaluate.py``
    contains numerical logic" is not a checkable criterion while "no ``evaluate.py``
    exists" is.
    """
    from verifylib import gate as _gate
    from verifylib.kernel import run

    plan_dir = os.path.abspath(plan_dir)
    spec = _gate._sibling_spec(os.path.join(plan_dir, "SOLUTION.md"))
    if spec is None:
        print(f"no problem_spec.json above {plan_dir}", file=sys.stderr)
        return 1

    config = {}
    override = os.environ.get("VERIFYLIB_EVAL_CONFIG")
    if override:
        try:
            config = json.loads(override)
        except json.JSONDecodeError as exc:
            print(f"VERIFYLIB_EVAL_CONFIG is not valid JSON: {exc}", file=sys.stderr)
            return 1

    metrics = run(spec, plan_dir, config)
    if as_json:
        print(json.dumps(metrics, indent=2, default=_jsonable))
    else:
        from verifylib.kernel.metrics import render_block
        print(render_block(metrics))
        print()
        print(f"bound by: {metrics['certification']['bound_by']} -- "
              f"{metrics['certification']['reason']}")
        for name, outcome in sorted(metrics.get("checks", {}).items()):
            print(f"  {name:26s} {outcome}")
        for line in metrics.get("notes", []):
            print(f"  note: {line}")
    # 0 whatever the score: a low score is a measurement, not a tool failure. Only a
    # kernel that could not run at all is an error, and that raises.
    return PASS


def _compare(plan_a, plan_b, *, N=None, no_record=False, as_json=False) -> int:
    """Findings F2: difference two plans' answers and record whether they agree.

    **PDE** -- solves both at the top ladder level (or ``--N``), restricts onto the
    common grid, and reports the relative field difference against the spec's
    ``rel_l2_err_max``.

    **SDE** -- differences the *moments* at the graded ``(num_paths, dt, T, seed)``
    against ``mean_rel_err_max`` / ``variance_rel_err_max``. There is no field to
    difference, and until 2026-09-26 this path did not exist at all: ``compare``
    solved every plan as a PDE and exited 1 with "solver.py has no solve_pde()", so
    on *every* SDE Path-B run the conductor's agreement gate could not fire and the
    second circularity break had to come from a clean Dynkin check. Found on the
    first no-closed-form SDE (`sde_double_well_langevin`, 2026-09-26).

    A moment difference carries Monte-Carlo noise that a field difference does not,
    so the SDE path reports **three** outcomes rather than two, and the noise decides
    which: agreement only when the difference is inside the tolerance *including* its
    confidence interval, disagreement only when it is outside *excluding* it, and
    ``inconclusive`` when the interval straddles the tolerance. Inconclusive records
    ``agree: null``, which is neither a break nor a cap -- the alternative is to grant
    a circularity break, or cap two plans at 8, on the strength of sampling noise.

    Unless ``--no-record`` the result goes to ``<workspace>/agreements.json`` with
    both solver hashes, which the kernel reads back on the next ``evaluate`` of
    either plan: agreement between different scheme families is priced as a
    circularity break, disagreement caps both plans at 8. Exit 0 on agreement, 2 on
    disagreement, 3 on inconclusive, 1 on a crash.
    """
    from verifylib import gate as _gate
    from verifylib.kernel import agreement, ladder, plan_meta, sandbox
    from verifylib.kernel.driver import solver_sha256
    from verifylib.kernel.metrics import Config

    plan_a, plan_b = os.path.abspath(plan_a), os.path.abspath(plan_b)
    if agreement.workspace_of(plan_a) != agreement.workspace_of(plan_b):
        print("compare takes two plans of the same workspace", file=sys.stderr)
        return 1
    spec = _gate._sibling_spec(os.path.join(plan_a, "SOLUTION.md"))
    if spec is None:
        print(f"no problem_spec.json above {plan_a}", file=sys.stderr)
        return 1
    thresholds = spec.get("evaluation_thresholds") or {}
    cfg = Config()
    if (spec.get("equation_type") or "").strip().upper() == "SDE":
        return _compare_sde(spec, thresholds, cfg, plan_a, plan_b,
                            no_record=no_record, as_json=as_json)
    grid_N = int(cfg.pick("grid_N", None, thresholds.get("grid_N"), 64))
    tol = float(cfg.pick("rel_l2_err_max", None, thresholds.get("rel_l2_err_max"), 0.01))
    metric = cfg.pick("metric", None, thresholds.get("metric"), "l2")
    levels = int(cfg.pick("refinement_levels", None, thresholds.get("refinement_levels"), 3))
    axis_names = list(thresholds.get("axes") or spec.get("spatial_variables") or ["x", "y", "z"])
    primary = thresholds.get("primary_field")
    bounds = ((spec.get("domain") or {}).get("bounds") or {})

    def solve(plan, n):
        return sandbox.run(plan, "pde", {"N": int(n)})

    graded = thresholds.get("graded_N")
    if N is None and isinstance(graded, (int, float)) and not isinstance(graded, bool) \
            and graded > 0:
        # F4: the grid the statement grades at is the grid the answer is judged on.
        N = int(graded)
    if N is None:
        # Otherwise the ladder's top level, detected the way the driver detects it.
        probe = solve(plan_a, grid_N)
        if probe["status"] != "ok":
            print(f"{os.path.basename(plan_a)} crashed at N={grid_N}: {probe.get('error')}",
                  file=sys.stderr)
            return 1
        axes = ladder.axes_of(probe["result"], axis_names)
        exclusive = (ladder.detect_periodic(axes[0], bounds[axis_names[0]])
                     if axis_names[0] in bounds else False)
        N = ladder.build_ladder(grid_N, exclusive, levels=levels)[-1]
    runs = {}
    for plan in (plan_a, plan_b):
        got = solve(plan, N)
        if got["status"] != "ok":
            print(f"{os.path.basename(plan)} crashed at N={N}: {got.get('reason')}: "
                  f"{got.get('error')}", file=sys.stderr)
            return 1
        runs[plan] = got["result"]
    ua = ladder.primary_of(runs[plan_a], primary)
    ub = ladder.primary_of(runs[plan_b], primary)
    axes_a = ladder.axes_of(runs[plan_a], axis_names)
    axes_b = ladder.axes_of(runs[plan_b], axis_names)
    interpolated = False
    if ua.shape == ub.shape and all(len(x) == len(y) and abs(float(x[-1]) - float(y[-1])) < 1e-9
                                   for x, y in zip(axes_a, axes_b, strict=False)):
        a_on, b_on = ua, ub
    else:
        # Restrict the finer onto the coarser; interpolate only if they do not nest,
        # and say so -- an interpolated agreement is not a circularity break.
        coarse_first = ua.size <= ub.size
        u_c, ax_c = (ua, axes_a) if coarse_first else (ub, axes_b)
        u_f, ax_f = (ub, axes_b) if coarse_first else (ua, axes_a)
        f_on_c = ladder.restrict(u_f, ax_f, ax_c)
        if f_on_c is None:
            f_on_c, interpolated = ladder.interpolate_to(u_f, ax_f, ax_c), True
        a_on, b_on = (u_c, f_on_c) if coarse_first else (f_on_c, u_c)
    import numpy as np
    diff = np.asarray(a_on, dtype=float) - np.asarray(b_on, dtype=float)
    if metric == "l1":
        rel = float(np.mean(np.abs(diff)) / (0.5 * (np.mean(np.abs(a_on))
                                                    + np.mean(np.abs(b_on))) + 1e-14))
    else:
        rel = float(np.sqrt(np.mean(diff ** 2))
                    / (0.5 * (ladder.rms(a_on) + ladder.rms(b_on)) + 1e-14))
    meta_a, meta_b = plan_meta.read(plan_a), plan_meta.read(plan_b)
    fam_a, fam_b = meta_a.get("scheme_family"), meta_b.get("scheme_family")
    entry = {"a": os.path.basename(plan_a), "b": os.path.basename(plan_b), "N": int(N),
             "t_final": runs[plan_a].get("t_final"), "rel_diff": rel, "tol": tol,
             "metric": metric, "agree": bool(np.isfinite(rel) and rel < tol),
             "family_a": fam_a, "family_b": fam_b,
             "different_family": bool(fam_a and fam_b and fam_a != fam_b),
             "sha_a": solver_sha256(plan_a), "sha_b": solver_sha256(plan_b),
             "interpolated": interpolated}
    if not no_record:
        agreement.record(agreement.workspace_of(plan_a), entry)
        entry["recorded_in"] = agreement.path_for(plan_a)
    if as_json:
        print(json.dumps(entry, indent=2, default=_jsonable))
    else:
        verdict = "AGREE" if entry["agree"] else "DISAGREE"
        print(f"{verdict}: {entry['a']} vs {entry['b']} differ by {rel:.3e} at N={N} "
              f"(tol {tol:g}, t_final {entry['t_final']}); families {fam_a} / {fam_b}"
              f"{' -- different, counts as a circularity break' if entry['different_family'] and entry['agree'] else ''}"
              f"{' -- interpolated, not a break' if interpolated else ''}")
        if not no_record:
            print(f"recorded in {entry['recorded_in']}")
    return PASS if entry["agree"] else BLOCK


def _compare_sde(spec, thresholds, cfg, plan_a, plan_b, *, no_record=False, as_json=False):
    """The SDE half of ``compare``: a difference in moment space, with the
    Monte-Carlo interval deciding between agree / disagree / inconclusive.

    **Both seedings run** (plan §10 D6). ``verification_manual.md`` §22 says the
    comparison must use independent seeds and the code used the same ``seed`` for
    both plans, and each is right about a different question:

    * **Shared seed, for detecting disagreement.** Two schemes that consume the
      increments the same way share a Brownian path, so the difference is nearly
      noise-free and a real discrepancy shows at a fraction of the paths. This is
      the sensitive test, and it is what may *cap* both plans.
    * **Independent seeds, for earning the circularity break.** Agreement under a
      shared path is partly an artifact of the shared path -- two solvers with the
      same bug driven by the same noise agree beautifully. Only the independent-seed
      comparison may set ``cross_plan_agreement``, which is the field
      ``score.agreement_is_break`` reads for a 10.

    Four solves instead of two, accepted: ``compare`` is already the slowest CLI
    path on an SDE and the alternative is a break that was not earned.

    The interval is formed conservatively in both, as if the two runs were
    independent -- correlation can only make the true error bar smaller, and the
    conservative bar is the one that refuses to call a break it has not earned.
    """
    import numpy as np

    from verifylib.kernel import agreement, plan_meta, sandbox
    from verifylib.kernel.driver import solver_sha256

    num_paths = int(cfg.pick("num_paths", None, thresholds.get("num_paths"), 50000))
    seed = int(cfg.pick("seed", None, thresholds.get("seed"), 42))
    dt = float(cfg.pick("dt", None, thresholds.get("dt"), 0.01))
    T = float((spec.get("time_interval") or {}).get("T", 1.0))
    var_tol = float(cfg.pick("variance_rel_err_max", None,
                             thresholds.get("variance_rel_err_max"), 0.1))
    mean_tol = float(cfg.pick("mean_rel_err_max", None,
                              thresholds.get("mean_rel_err_max"), 0.05))
    near_zero = float(cfg.pick("near_zero_mean_threshold", None,
                               thresholds.get("near_zero_mean_threshold"), 0.01))
    ci_mult = float(cfg.pick("ci_mult", None, thresholds.get("ci_mult"), 2.0))

    def moments_of(plan, plan_seed):
        got = sandbox.run(plan, "sde", {"num_paths": num_paths, "dt": dt, "T": T,
                                        "seed": int(plan_seed)})
        if got["status"] != "ok":
            print(f"{os.path.basename(plan)} crashed at dt={dt:g}: {got.get('reason')}: "
                  f"{got.get('error')}", file=sys.stderr)
            return None
        X = np.asarray(got["result"]["terminal_paths"], dtype=float)
        X = X.reshape(X.shape[0], -1) if X.ndim > 1 else X.reshape(-1, 1)
        n = X.shape[0]
        return {
            "mean": X.mean(axis=0), "variance": X.var(axis=0, ddof=1),
            "se_mean": X.std(axis=0, ddof=1) / np.sqrt(n),
            # se of a variance estimate, from the fourth central moment.
            "se_variance": np.sqrt(np.maximum(
                ((X - X.mean(axis=0)) ** 4).mean(axis=0) - X.var(axis=0, ddof=1) ** 2, 0.0) / n),
            "num_paths": n}

    def compare_pair(sa, sb):
        """``(per-moment rows, agree, worst row)`` -- §14's three-way outcome.

        Per moment and component: the relative difference, its interval, and the
        tolerance it is judged against. The worst component decides, as in the
        driver -- "worst" by ``rel_diff / tol``, not by ``rel_diff``, because the
        mean and the variance are judged against different tolerances and the raw
        difference does not say which is closer to failing.
        """
        rows, bits = [], []
        for name, tol_k in (("mean", mean_tol), ("variance", var_tol)):
            for i in range(len(sa[name])):
                a, b = float(sa[name][i]), float(sb[name][i])
                scale = 0.5 * (abs(a) + abs(b))
                if name == "mean" and scale < near_zero:
                    rows.append({"moment": name, "component": i, "a": a, "b": b,
                                 "skipped": "both means are within the near-zero threshold"})
                    continue
                se = float(np.hypot(sa[f"se_{name}"][i], sb[f"se_{name}"][i]))
                d = abs(a - b) / (scale + 1e-300)
                d_se = se / (scale + 1e-300)
                rows.append({"moment": name, "component": i, "a": a, "b": b,
                             "rel_diff": d, "rel_se": d_se, "tol": tol_k,
                             "agree": bool(d + ci_mult * d_se < tol_k),
                             "disagree": bool(d - ci_mult * d_se > tol_k)})
                bits.append(rows[-1])
        if not bits:
            return rows, "empty", None
        if any(p["disagree"] for p in bits):
            outcome = False
        elif all(p["agree"] for p in bits):
            outcome = True
        else:
            outcome = None
        return rows, outcome, max(bits, key=lambda p: p["rel_diff"] / (p["tol"] or 1.0))

    # An independent seed for plan B, derived from the graded one so the comparison
    # is reproducible: a SeedSequence child, not `seed + 1`.
    seed_b = int(np.random.SeedSequence(seed).generate_state(1, dtype=np.uint32)[0])
    runs = {("a", seed): moments_of(plan_a, seed), ("b", seed): moments_of(plan_b, seed),
            ("b", seed_b): moments_of(plan_b, seed_b)}
    if any(v is None for v in runs.values()):
        return FAIL
    sa, sb = runs[("a", seed)], runs[("b", seed)]
    per, shared, worst_shared = compare_pair(sa, sb)
    indep_per, agree, worst = compare_pair(sa, runs[("b", seed_b)])
    if shared == "empty" or agree == "empty":
        print("every declared moment was skipped as near-zero; nothing to compare",
              file=sys.stderr)
        return FAIL
    rel, worst_tol = worst["rel_diff"], worst["tol"]

    # D6: the **shared-seed** comparison is the sensitive one and is what may cap;
    # the **independent-seed** one is the only one that may grant the break. A
    # disagreement under either is a disagreement.
    if shared is False or agree is False:
        verdict, code = "DISAGREE", BLOCK
        agree = False
    elif agree is True:
        verdict, code = "AGREE", PASS
    else:
        verdict, code = "INCONCLUSIVE", INCONCLUSIVE

    meta_a, meta_b = plan_meta.read(plan_a), plan_meta.read(plan_b)
    fam_a, fam_b = meta_a.get("scheme_family"), meta_b.get("scheme_family")
    entry = {"a": os.path.basename(plan_a), "b": os.path.basename(plan_b), "N": None,
             "t_final": T, "rel_diff": rel, "tol": worst_tol, "metric": "moments",
             "agree": agree, "family_a": fam_a, "family_b": fam_b,
             "different_family": bool(fam_a and fam_b and fam_a != fam_b),
             "sha_a": solver_sha256(plan_a), "sha_b": solver_sha256(plan_b),
             "interpolated": False,
             "config": {"num_paths": num_paths, "dt": dt, "T": T, "seed": seed,
                        "seed_b_independent": seed_b, "ci_mult": ci_mult},
             # `agree` is the independent-seed verdict, because that is the field
             # `agreement.for_plan` reads for the circularity break. The shared-seed
             # verdict is recorded beside it: it is the sensitive detector, and a
             # disagreement there has already been folded into `agree` above.
             "agree_shared_seed": shared,
             "rel_diff_shared_seed": (worst_shared or {}).get("rel_diff"),
             "seeding": "both (D6): shared seed detects, independent seed earns the break",
             "per_moment": per, "per_moment_independent": indep_per}
    if not no_record:
        agreement.record(agreement.workspace_of(plan_a), entry)
        entry["recorded_in"] = agreement.path_for(plan_a)
    if as_json:
        print(json.dumps(entry, indent=2, default=_jsonable))
    else:
        print(f"{verdict}: {entry['a']} vs {entry['b']} differ by {rel:.3e} in moment space "
              f"on independent seeds and "
              f"{(worst_shared or {}).get('rel_diff', float('nan')):.3e} on the shared "
              f"seed (worst tol {worst_tol:g}) at num_paths={num_paths}, dt={dt:g}, "
              f"seeds {seed} / {seed_b}; families {fam_a} / {fam_b}"
              f"{' -- different, counts as a circularity break' if entry['different_family'] and agree else ''}"
              f"{' -- the Monte-Carlo interval straddles the tolerance, so this is neither a break nor a cap' if agree is None else ''}")
        if not no_record:
            print(f"recorded in {entry['recorded_in']}")
    return code


def _jsonable(value):
    try:
        import numpy as np
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (np.floating, np.integer, np.bool_)):
            return value.item()
    except ImportError:
        pass
    return str(value)


def _rank(workspace, *, as_json=False) -> int:
    """Rank a workspace's plans the way `commands/conductor.md` Phase 3 does.

    Reads each plan's ``<metrics>`` block -- never re-running a solver -- and applies
    :func:`verifylib.ranking.rank_key`. The conductor used to apply the key by hand
    from the blocks, which meant the rule lived in prose in one place and in code in
    ``benchmark/baseline_select.py`` in another; they had already drifted. This is
    the one implementation both read.
    """
    from verifylib import ranking
    from verifylib.review import parse_metrics

    plans_root = os.path.join(workspace, "plans")
    if not os.path.isdir(plans_root):
        print(f"no plans/ under {workspace}", file=sys.stderr)
        return 1
    entries = []
    for name in sorted(os.listdir(plans_root)):
        path = os.path.join(plans_root, name, "SOLUTION.md")
        if not os.path.exists(path):
            continue
        with open(path, encoding="utf-8") as fh:
            block = parse_metrics(fh.read())
        if not block:
            continue
        entries.append({"plan": name, "kernel": {
            "score": block.get("score"), "provenance": block.get("provenance"),
            "certified": block.get("certified"), "summary": block}})
    if not entries:
        print(f"no scored plans under {plans_root}", file=sys.stderr)
        return 1

    pool = [e["kernel"]["summary"] for e in entries]
    tol = ranking._tol_of(pool[0], {})
    for entry in entries:
        summary = entry["kernel"]["summary"]
        index, parts = ranking.sqi(summary, tol, pool)
        entry["sqi"] = index
        entry["components"] = parts
    entries.sort(key=lambda e: ranking.rank_key(e, pool))

    if as_json:
        print(json.dumps({"workspace": workspace, "tol": tol, "ranking": entries},
                         indent=2, default=str))
        return 0
    head = ("sqi", "acc", "ord", "str", "res", "cost", "solve_s")
    print(f"{'plan':<28}{'score':>6}{'cert':>6}  {'provenance':<22}"
          + "".join(f"{h:>9}" for h in head))
    def cell(value):
        return f"{value:9.3f}" if isinstance(value, (int, float)) else f"{'-':>9}"

    for entry in entries:
        k, c, s = entry["kernel"], entry["components"], entry["kernel"]["summary"]
        print(f"{entry['plan']:<28}{str(k['score'] or '-'):>6}"
              f"{('yes' if k.get('certified') else 'no'):>6}  "
              f"{str(k['provenance'] or '-'):<22}{cell(entry['sqi'])}"
              + "".join(cell(c[n]) for n in ("accuracy", "order", "structure",
                                             "consistency", "cost"))
              + cell(s.get("solve_time_s")))
    print(f"\nbest: {entries[0]['plan']}")
    # A block written before a field existed is not a plan that lacks the property,
    # and a ranking run on one is reading less than it looks like it is. Say which
    # inputs were missing rather than printing a confident order over `-` columns.
    stale = sorted({name for name, field in (("order", "design_order"),
                                             ("structure", "invariant_worst_headroom"),
                                             ("cost", "solve_time_s"))
                    if all(p.get(field) is None for p in pool)})
    if stale:
        print(f"note: no plan's <metrics> block carries the input for {', '.join(stale)}"
              f" -- the index is the mean of the rest. Re-run `evaluate` to refresh.")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__, file=sys.stderr)
        return 1
    command, args = argv[0], argv[1:]
    as_json = "--json" in args
    flags = [a for a in args if a.startswith("--")]
    args = [a for a in args if not a.startswith("--")]

    if command == "hook":
        if not args or args[0] not in HOOKS:
            print(f"hook takes one of {sorted(HOOKS)}", file=sys.stderr)
            return 1
        try:
            return HOOKS[args[0]](_read_payload())
        except Exception as exc:  # noqa: BLE001
            # A guard that crashes must not take the session with it.
            print(f"verifylib hook error ({type(exc).__name__}: {exc})", file=sys.stderr)
            return PASS

    if command == "check-spec":
        return _report(gate.check_spec_file(args[0]), as_json)
    if command == "check-solver":
        return _report(gate.check_solver_file(args[0]), as_json)
    if command == "check-review":
        return _report(gate.check_review_file(args[0]), as_json)
    if command in ("gate", "check-workspace"):
        return _report(gate.check_workspace(args[0] if args else "workspace"), as_json)
    if command == "evaluate":
        if not args:
            print("evaluate takes a plan directory", file=sys.stderr)
            return 1
        return _evaluate(args[0], as_json)
    if command == "rank":
        return _rank(args[0] if args else "workspace", as_json=as_json)
    if command == "compare":
        if len(args) != 2:
            print("compare takes two plan directories", file=sys.stderr)
            return 1
        N = next((int(f.split("=", 1)[1]) for f in flags if f.startswith("--N=")), None)
        return _compare(args[0], args[1], N=N, no_record="--no-record" in flags,
                        as_json=as_json)

    print(f"unknown command {command!r}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
