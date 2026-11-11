"""Step-6/7 calibration of the SciML suite, through the real harness.

For each shipped entry, four solvers one decision apart are written to disk and run
through `verify.verify_problem(sandbox=True)` -- the same path a pipeline or baseline
run takes. The rule this applies is the repo's own: a correct scheme clears the gate,
a plausible wrong one fails it, at the resolution the entry ships with, and no gate is
moved after seeing where anything landed.

    correct        Euler-Maruyama on the equation as the brief states it
    wrong          the same solver with exactly ONE decision changed
    constant       the best constant, i.e. no simulation at all (the §2a.1 defect)
    transcribed    the closed form, evaluated (the §0b defect)

`transcribed` is expected to be *accurate* and to be excluded by name rather than
passed; `constant` is expected to fail. Run:

    uv run python docs/sciml-benchmark-data/calibrate_sciml_suite.py
"""

import os
import sys
import tempfile

BENCH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "benchmark")
sys.path.insert(0, BENCH)

import problems as P  # noqa: E402
import report as R  # noqa: E402
import verify as V  # noqa: E402

# --- the eight pathwise entries, as (drift, diffusion, dg/dx) source fragments ----
# Written as text because each becomes a real solver.py run in the sandbox.
SCHEMES = {
    "sde_sciml_linear_ito": dict(
        u0="0.5", f="1.01 * x", g="0.87 * x", gp="0.87 * np.ones_like(x)",
        exact="0.5 * np.exp(0.63155 * T + 0.87 * W)",
        wrong_f="(1.01 + 0.5 * 0.87 ** 2) * x",
        wrong="the Ito correction added to the drift (reading the solution as Stratonovich)"),
    "sde_sciml_additive": dict(
        u0="1.0", f="0.05 / np.sqrt(1 + t) - x / (2 * (1 + t))",
        g="0.1 * 0.05 / np.sqrt(1 + t) * np.ones_like(x)", gp="np.zeros_like(x)",
        exact="1.0 / np.sqrt(1 + T) + 0.05 * (T + 0.1 * W) / np.sqrt(1 + T)",
        wrong_f="0.05 / np.sqrt(1 + t) - x / (1 + t)",
        wrong="the 1/2 dropped from the mean-reversion rate"),
    "sde_sciml_cubic": dict(
        u0="0.5", f="-0.25 * x * (1 - x ** 2)", g="0.5 * (1 - x ** 2)", gp="-x",
        exact="(1.5 * np.exp(W) - 0.5) / (1.5 * np.exp(W) + 0.5)",
        wrong_f="0.25 * x * (1 - x ** 2)",
        wrong="the drift sign flipped -- which is what the library's own docstring says"),
    "sde_sciml_wave": dict(
        u0="1.0", f="-0.01 * np.sin(x) * np.cos(x) ** 3", g="0.1 * np.cos(x) ** 2",
        gp="-0.2 * np.cos(x) * np.sin(x)",
        exact="np.arctan(0.1 * W + np.tan(1.0))",
        wrong_f="-0.01 * np.sin(x) * np.cos(x)",
        wrong="cos^3 written as cos in the drift"),
    "sde_sciml_nltest": dict(
        u0="1.0", f="(1 / 3) * np.cbrt(x) + 6 * np.cbrt(x) ** 2", g="np.cbrt(x) ** 2",
        gp="(2 / 3) / np.cbrt(np.where(np.abs(x) < 1e-12, 1e-12, x))",
        exact="(2 * T + 1 + W / 3) ** 3",
        wrong_f="6 * np.cbrt(x) ** 2",
        wrong="the (1/3)x^(1/3) drift term dropped"),
}
for slug, a, b, wf, wlab in (
    ("sde_sciml_stiffquad_mild", 50.0, 1.0, "-{a} * (1 - x ** 2)",
     "beta^2 dropped from the drift (the Stratonovich twin)"),
    ("sde_sciml_stiffquad_det", 500.0, 1.0, None,
     "diffusion beta(1-x) instead of beta(1-x^2)"),
    ("sde_sciml_stiffquad_mixed", 5000.0, 70.0, "-{a} * (1 - x ** 2)",
     "beta^2 dropped from the drift (the Stratonovich twin)"),
):
    SCHEMES[slug] = dict(
        u0="0.5",
        f=f"-({a} + {b} ** 2 * x) * (1 - x ** 2)",
        g=f"{b} * (1 - x ** 2)", gp=f"-2 * {b} * x",
        exact=(f"(lambda e: (1.5 * e - 0.5) / (1.5 * e + 0.5))"
               f"(np.exp(np.clip(-2 * {a} * T + 2 * {b} * W, -745, 709)))"),
        wrong_f=(wf.format(a=a) if wf else None),
        wrong_g=(None if wf else f"{b} * (1 - x)"),
        wrong=wlab)

TEMPLATE = """
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    Nt = dW.shape[1]
    dt = T / Nt
    x = np.full(num_paths, {u0})
    t = 0.0
    keep, save = [], max(1, Nt // {k})
    for i in range(Nt):
        gx = {g}
        x = x + ({f}) * dt + gx * dW[:, i]
        t += dt
        if (i + 1) % save == 0:
            keep.append(x.copy())
    pv = np.stack(keep, axis=1) if keep else None
    out = {{"terminal_paths": x, "empirical_mean": float(x.mean()),
           "empirical_variance": float(x.var(ddof=1))}}
    if pv is not None:
        out["path_values"] = pv
    return out
"""

CONSTANT = """
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    W = dW.sum(axis=1)
    x = np.full(num_paths, float(np.mean({exact})))
    return {{"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}}
"""

TRANSCRIBED = """
import numpy as np

def solve_sde(num_paths, dt, T, seed=42, dW=None):
    W = dW.sum(axis=1)
    x = {exact}
    return {{"terminal_paths": x, "empirical_mean": float(x.mean()),
            "empirical_variance": float(x.var(ddof=1))}}
"""


def write_plan(root, name, source, family="euler-maruyama"):
    d = os.path.join(root, name)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "solver.py"), "w") as fh:
        fh.write(source)
    with open(os.path.join(d, "SOLUTION.md"), "w") as fh:
        fh.write(f"---\nscheme_family: {family}\n---\n\n# {name}\n")
    return d


def run(prob, plan_dir):
    return V.verify_problem(prob, os.path.dirname(plan_dir), plan_dir=plan_dir,
                            sandbox=True, work_precision=False)


def verdict_of(prob, out):
    rec = {"id": prob["id"], "slug": prob["slug"], "type": "sde",
           "has_ground_truth": True,
           "ground_truth_kind": prob.get("ground_truth_kind", "exact"),
           "run": {"status": "completed"}, "pipeline": {"best_score": 10},
           "verify": out}
    return R.compute_verdict(rec)


def main():
    root = tempfile.mkdtemp(prefix="sciml_cal_")
    print(f"{'entry':30s} {'variant':12s} {'rel err':>11s} {'gate':>9s} "
          f"{'x gate':>8s} {'verdict':24s}")
    table = {}
    for prob in P.sciml_problems():
        slug = prob["slug"]
        if prob.get("ground_truth_kind") != "pathwise":
            continue
        S = SCHEMES[slug]
        k = int(prob.get("pathwise_save_points") or 1)
        variants = {
            "correct": TEMPLATE.format(u0=S["u0"], f=S["f"], g=S["g"], k=k),
            "wrong": TEMPLATE.format(u0=S["u0"], f=S.get("wrong_f") or S["f"],
                                     g=S.get("wrong_g") or S["g"], k=k),
            "constant": CONSTANT.format(exact=S["exact"]),
            "transcribed": TRANSCRIBED.format(exact=S["exact"]),
        }
        gate = V._pathwise_tol(prob)
        row = {}
        for name, src in variants.items():
            out = run(prob, write_plan(root, f"{slug}__{name}", src))
            err = out.get("pathwise_rel_err")
            lad = (out.get("pathwise_ladder") or {})
            vd = verdict_of(prob, out)
            row[name] = (err, vd, lad.get("reason"))
            shown = "n/a" if err is None else f"{err:.5f}"
            ratio = "n/a" if err is None else f"{err / gate:7.2f}x"
            extra = f" [{lad.get('reason')}]" if lad.get("reason") else ""
            print(f"{slug:30s} {name:12s} {shown:>11s} {gate:9.5f} {ratio:>8s} "
                  f"{vd + extra:24s}"
                  + ("" if out.get("status") == "ok" else f"  <{out.get('status')}>"))
        table[slug] = (gate, row)
        print()

    print("\n=== separation summary (the narrowest factor is the one that matters) ===")
    print(f"{'entry':30s} {'gate':>9s} {'correct/gate':>13s} {'wrong/gate':>11s} "
          f"{'const/gate':>11s} {'separation':>11s}")
    worst = None
    for slug, (gate, row) in table.items():
        c = row["correct"][0]
        bad = min(v for v in (row["wrong"][0], row["constant"][0]) if v is not None)
        sep = bad / c if c else float("inf")
        print(f"{slug:30s} {gate:9.5f} {gate / c:12.1f}x {row['wrong'][0] / gate:10.1f}x "
              f"{row['constant'][0] / gate:10.1f}x {sep:10.1f}x")
        if worst is None or sep < worst[1]:
            worst = (slug, sep)
    print(f"\nnarrowest separation: {worst[0]} at {worst[1]:.1f}x")


if __name__ == "__main__":
    main()
