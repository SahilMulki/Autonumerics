"""Read `comparators.json` and print the tables the plan's step-3 record quotes.

Four of them, and each exists because a raw row dump answers none of the questions
`plan-sciml-benchmark.md` asks:

  1. **Measured stage counts.** §5 says the work axis uses "the published stage count
     for each Rossler scheme". `harvest.jl` measures them instead, by differencing
     evaluation counts over 4 and 8 steps. This table is what was measured, with the
     two anchors (EM = 1, RKMil = 2) visible as a check on the method.
  2. **Work to reach the gate.** The one number that makes a scheme comparable to this
     repo's pipeline: how many `paths x steps x stages` does each scheme spend to get
     inside the gate the suite grades at? Log-log interpolation between the two
     bracketing points, and `>` when no run reached it.
  3. **At matched work.** §10 prediction 1, stated so it can be falsified: at the work
     Euler-Maruyama spends on its finest run, how much more accurate is the best other
     scheme?
  4. **Seconds, where they mean anything.** Only at the largest step count per cell,
     because `harvest.jl` measured -- and recorded -- that at small step counts the
     per-path `solve` setup dominates the clock.
  5. **Work to reach a fixed relative error**, which is how §10 prediction 1 is
     actually phrased ("at relative error 10^-4") and the only form of the comparison
     that does not depend on where each entry's own gate happens to sit.
  6. **The same comparison at matched *seconds* rather than matched evals**, plus the
     ratio between the two. The `evals` axis charges a scheme for its drift and
     diffusion evaluations; on these scalar problems a right-hand side is three flops
     and the per-step cost is loop and noise overhead instead, so the two axes do not
     agree -- and the direction of the disagreement is the finding.

Run:  uv run python docs/sciml-benchmark-data/comparator_tables.py <comparators.json>
"""

import collections
import json
import math
import sys

ERR = "strong_err_rel_final"
WEAK = "weak_err_rel_second_moment_x1"



ROUNDOFF = 1e-10


def _floor_note(err):
    """Flag an error that is at the floating-point floor rather than at a scheme's.

    `SRA3` on M02 reaches 6e-12: that problem's solution is affine in `t` and `W`, which
    is the class the SRA schemes are *constructed* to integrate exactly, so what is
    being measured there is rounding accumulated over 15 625 steps and not accuracy.
    Quoting it as an accuracy ratio without saying so would turn a statement about a
    problem class into a claim about scheme quality.
    """
    return " [at the roundoff floor]" if err is not None and err < ROUNDOFF else ""


def work_at_error(points, target):
    """`evals` needed to reach relative error `target`, by log-log interpolation.

    Returns `(evals, how)` where `how` is "interp" between two bracketing runs,
    "reached" when the coarsest run is already there, or "beyond" when no run is -- and
    in the last case the figure is the finest run's work, i.e. a lower bound.
    """
    return interp_work_to_gate(points, target)


def err_of(row):
    """The row's graded-comparable relative error, whichever route it came from."""
    v = row.get(ERR) if row["kind"] == "pathwise" else row.get(WEAK)
    return v if isinstance(v, (int, float)) else None


def gate_of(spec, pid):
    g = spec[pid]["gate"]
    if "pathwise_tol" in g:
        return float(g["pathwise_tol"])
    # The weak entry is graded on three observables at three gates; the one plotted
    # here is SciML's own, so its own gate is the comparable one.
    return float(g["functional_tol"]["second_moment_x1"])


def interp_work_to_gate(points, gate):
    """`evals` at which this scheme's error curve crosses `gate`, or None.

    Log-log interpolation between the two points that bracket the gate. Returns
    `(evals, "interp")`, or `(cheapest_passing_evals, "reached")` when a run is already
    inside the gate at the coarsest step, or `(finest_evals, "beyond")` when none is.
    """
    pts = sorted((e, w) for w, e in points)
    pts = [(e, w) for e, w in pts if e and e > 0 and w and w > 0]
    if not pts:
        return None, None
    inside = [(e, w) for e, w in pts if e <= gate]
    if not inside:
        return max(w for _e, w in pts), "beyond"
    if pts[0][0] <= gate:
        return min(w for _e, w in inside), "reached"
    hi = max((e, w) for e, w in pts if e > gate)       # worst error still outside
    lo = min((e, w) for e, w in pts if e <= gate)      # best error inside
    # log err linear in log work between the bracketing pair.
    de = math.log(gate) - math.log(hi[0])
    dw = math.log(lo[1]) - math.log(hi[1])
    dee = math.log(lo[0]) - math.log(hi[0])
    if dee == 0:
        return lo[1], "interp"
    return math.exp(math.log(hi[1]) + dw * de / dee), "interp"


def _work_at(by, pid, target):
    """A `scheme -> (evals, how)` lookup bound to one entry and one error target.

    A closure over the loop variables would read whichever entry the loop finished on
    (ruff B023), which would quietly make every row of the table the last row's.
    """
    def w_of(scheme):
        pts = [(r["evals"], err_of(r)) for r in by[(pid, scheme)] if r["evals"]]
        return work_at_error(pts, target)
    return w_of


def main(path):
    with open(path) as fh:
        d = json.load(fh)
    with open(d["spec"]) as fh:
        spec = {p["id"]: p for p in json.load(fh)["problems"]}
    rows = d["rows"]
    env = d["environment"]

    by = collections.defaultdict(list)
    for r in rows:
        by[(r["problem"], r["scheme"])].append(r)
    pids = sorted({r["problem"] for r in rows})
    schemes_of = {p: [s for (q, s) in by if q == p] for p in pids}

    print(f"comparators: {len(rows)} rows  julia {env['julia_version']} "
          f"threads={env['threads']} cpu={env['cpu']}")
    print("  " + "  ".join(f"{k} {v}" for k, v in sorted(env["pkg_versions"].items())
                           if k in ("StochasticDiffEq", "SDEProblemLibrary",
                                    "DiffEqNoiseProcess", "SciMLBase")))
    for c in d["analytic_cross_checks"]:
        if c["checks_total"] and c["checks_passed"] < c["checks_total"]:
            print(f"  !! {c['id']} analytic cross-check {c['checks_passed']}"
                  f"/{c['checks_total']}")

    # --- 1. measured stage counts ------------------------------------------------
    print("\n=== 1. measured evaluations per step (f / g), and the stage count "
          "the work axis uses ===")
    seen = {}
    for (_pid, s), rs in by.items():
        r = rs[0]
        seen.setdefault(s, set()).add((r.get("f_evals_per_step"),
                                       r.get("g_evals_per_step"), r.get("stages")))
    print(f"{'scheme':16s} {'f/step':>8s} {'g/step':>8s} {'stages':>7s}  note")
    for s in sorted(seen, key=lambda x: (min(t[2] or 99 for t in seen[x]), x)):
        for f, g, st in sorted(seen[s]):
            note = ""
            if s == "EM" and st == 1:
                note = "anchor: §5 says 1"
            elif s == "RKMil" and st == 2:
                note = "anchor: §5 says 2"
            elif s.startswith("Implicit"):
                note = "no fixed count: the nonlinear solve iterates"
            print(f"{s:16s} {f!s:>8s} {g!s:>8s} {st!s:>7s}  {note}")

    # --- 2. work to reach the gate -----------------------------------------------
    print("\n=== 2. work (paths x steps x stages) to reach this entry's own gate ===")
    print("  'beyond' = no run reached the gate; the figure is the finest run's work.")
    cheapest = {}
    for pid in pids:
        gate = gate_of(spec, pid)
        print(f"\n{pid}  gate = {gate:g}")
        best = None
        for s in sorted(schemes_of[pid]):
            pts = [(r["evals"], err_of(r)) for r in by[(pid, s)] if r["evals"]]
            w, how = interp_work_to_gate(pts, gate)
            if w is None:
                print(f"  {s:16s}       -  (no finite run)")
                continue
            mark = "" if how == "interp" else f"  [{how}]"
            print(f"  {s:16s} {w:10.3e}{mark}")
            if how != "beyond" and (best is None or w < best[1]):
                best = (s, w)
        em = next((w for s, w in [(s, interp_work_to_gate(
            [(r["evals"], err_of(r)) for r in by[(pid, s)] if r["evals"]], gate)[0])
            for s in schemes_of[pid]] if s == "EM"), None)
        if best and em:
            print(f"  -> cheapest: {best[0]} at {best[1]:.3e}; "
                  f"EM needs {em / best[1]:.1f}x that work")
            cheapest[pid] = (best[0], best[1], em / best[1])

    # --- 3. at matched work ------------------------------------------------------
    print("\n=== 3. at matched work: EM's finest run vs the best other scheme at no "
          "more work (§10 prediction 1) ===")
    print(f"{'entry':6s} {'EM evals':>11s} {'EM err':>11s} {'best other':16s} "
          f"{'its err':>11s} {'accuracy gain':>14s}")
    for pid in pids:
        ems = [r for r in by[(pid, "EM")] if r["evals"] and err_of(r) is not None]
        if not ems:
            continue
        em = max(ems, key=lambda r: r["evals"])
        cand = []
        for s in schemes_of[pid]:
            if s == "EM":
                continue
            for r in by[(pid, s)]:
                e = err_of(r)
                if r["evals"] and e and r["evals"] <= em["evals"] * 1.001:
                    cand.append((e, s, r["evals"]))
        if not cand:
            print(f"{pid:6s} {em['evals']:11.3e} {err_of(em):11.3e} "
                  f"{'(none at <= work)':16s}")
            continue
        e, s, w = min(cand)
        print(f"{pid:6s} {em['evals']:11.3e} {err_of(em):11.3e} {s:16s} "
              f"{e:11.3e} {err_of(em) / e:13.1f}x" + _floor_note(e))

    # --- 4. seconds, at the largest step count only ------------------------------
    print("\n=== 4. ns per path-step, at each entry's finest run only ===")
    print("  (coarse cells measure `solve` setup, not stepping -- harvest.jl's own "
          "caveat)")
    print(f"{'entry':6s} {'scheme':16s} {'steps':>7s} {'ns/path-step':>13s} "
          f"{'s (min of 5)':>13s}")
    for pid in pids:
        for s in sorted(schemes_of[pid]):
            rs = [r for r in by[(pid, s)] if r.get("ns_per_path_step")]
            if not rs:
                continue
            r = max(rs, key=lambda r: r["steps"])
            print(f"{pid:6s} {s:16s} {r['steps']:7d} {r['ns_per_path_step']:13.2f} "
                  f"{r['seconds_min']:13.4f}")

    # --- 5. work to reach a fixed relative error ---------------------------------
    print("\n=== 5. work to reach a fixed relative error (§10 prediction 1's own "
          "phrasing) ===")
    print("  '>' = no run reached it; the figure is the finest run's work, a lower "
          "bound.")
    for target in (1e-3, 1e-4):
        print(f"\n  target rel err = {target:g}")
        print(f"  {'entry':6s} {'EM':>12s} {'best other':16s} {'its work':>12s} "
              f"{'work saved':>11s}")
        for pid in pids:
            w_of = _work_at(by, pid, target)
            emw, emhow = w_of("EM")
            best = None
            for s in sorted(schemes_of[pid]):
                if s == "EM":
                    continue
                w, how = w_of(s)
                if w is None or how == "beyond":
                    continue
                if best is None or w < best[1]:
                    best = (s, w)
            em_s = "-" if emw is None else (("> " if emhow == "beyond" else "")
                                            + f"{emw:.2e}")
            if best is None:
                print(f"  {pid:6s} {em_s:>12s} {'(none reached it)':16s}")
                continue
            gain = (f"{emw / best[1]:10.1f}x" if emw and emhow != "beyond"
                    else (f"> {emw / best[1]:8.1f}x" if emw else "-"))
            print(f"  {pid:6s} {em_s:>12s} {best[0]:16s} {best[1]:12.2e} {gain:>11s}")

    # --- 6. matched seconds, and how it disagrees with matched evals --------------
    print("\n=== 6. the two cost axes disagree, and by how much ===")
    print("  `evals` charges a scheme per drift/diffusion evaluation. On a scalar "
          "problem a\n  right-hand side is a few flops and the per-step cost is loop "
          "and noise overhead, so a\n  10-stage scheme does not cost 10x the seconds. "
          "Measured at each entry's finest run,\n  against EM on the same entry:")
    print(f"\n  {'entry':6s} {'scheme':16s} {'stages':>7s} {'evals x EM':>11s} "
          f"{'seconds x EM':>13s} {'over-charge':>12s}")
    for pid in pids:
        ems = [r for r in by[(pid, "EM")] if r.get("ns_per_path_step")]
        if not ems:
            continue
        em = max(ems, key=lambda r: r["steps"])
        for s in sorted(schemes_of[pid]):
            if s == "EM":
                continue
            rs = [r for r in by[(pid, s)] if r.get("ns_per_path_step")
                  and r["steps"] == em["steps"]]
            if not rs:
                continue
            r = rs[0]
            w = r["stages"] / em["stages"]
            sec = r["ns_per_path_step"] / em["ns_per_path_step"]
            print(f"  {pid:6s} {s:16s} {r['stages']:7.0f} {w:10.2f}x "
                  f"{sec:12.2f}x {w / sec:11.2f}x")
    print("\n  An over-charge above 1 means the `evals` axis bills that scheme for "
          "more than it\n  costs in seconds here -- so a scheme that wins on `evals` "
          "wins by more on the clock.")

    # --- 3b. the same matched-work comparison, at matched seconds ----------------
    print("\n=== 3b. at matched seconds: EM's finest run vs the best other scheme at "
          "no more time ===")
    print(f"  {'entry':6s} {'EM s':>9s} {'EM err':>11s} {'best other':16s} "
          f"{'its err':>11s} {'gain':>11s}")
    for pid in pids:
        ems = [r for r in by[(pid, "EM")]
               if r.get("seconds_min") and err_of(r) is not None]
        if not ems:
            continue
        em = max(ems, key=lambda r: r["steps"])
        cand = []
        for s in schemes_of[pid]:
            if s == "EM":
                continue
            for r in by[(pid, s)]:
                e = err_of(r)
                if r.get("seconds_min") and e and r["seconds_min"] <= em["seconds_min"]:
                    cand.append((e, s, r["seconds_min"]))
        if not cand:
            print(f"  {pid:6s} {em['seconds_min']:9.4f} {err_of(em):11.3e} "
                  f"{'(none at <= time)':16s}")
            continue
        e, s, _sec = min(cand)
        print(f"  {pid:6s} {em['seconds_min']:9.4f} {err_of(em):11.3e} {s:16s} "
              f"{e:11.3e} {err_of(em) / e:10.1f}x" + _floor_note(e))
    return 0



if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
