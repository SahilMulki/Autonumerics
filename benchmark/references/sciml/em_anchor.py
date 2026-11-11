"""The numpy half of the Euler-Maruyama anchor (plan-sciml-benchmark.md §5, step 4).

Pair of `em_anchor.jl`. Read that file's header for why the anchor exists, what is in
the clock, and why there are two problem shapes rather than one; this file implements
the same two shapes the way a solver in this repository is actually written -- numpy
vectorised across paths, with a python loop over steps only.

That asymmetry is deliberate and is the measurement: `StochasticDiffEq.jl` advances one
trajectory at a time, every SDE solver in this benchmark advances all paths at once, and
the ratio of the two is the number that lets a seconds column be quoted at all. It is
not a claim that either loop order is better.

Run:
    uv run python benchmark/references/sciml/em_anchor.py \\
        --out em_anchor_python.json [--julia em_anchor_julia.json] [--reps 5]

With `--julia`, it prints the ratio table: seconds per path-step in each stack and the
factor between them, per shape.
"""

import argparse
import json
import platform
import time

import numpy as np

T = 1.0
STEPS = 15625           # M01's finest declared level: 5^-6 over [0, 1]


def step_scalar(x, dW, a, b, dt):
    """Scalar Euler-Maruyama, vectorised across paths."""
    for i in range(dW.shape[1]):
        x = x + a * x * dt + b * x * dW[:, i]
    return x


def step_system(x, dW, a, b, dt):
    """Diagonal-noise system Euler-Maruyama, vectorised across paths and states."""
    for i in range(dW.shape[1]):
        x = x + a * x * dt + b * x * dW[:, i, :]
    return x


def timed(fn, reps):
    """Minimum of `reps` timed repetitions after one discarded warm-up (§3b).

    The warm-up matters less here than in Julia -- there is no compilation to pay for
    -- but the first call still touches pages and fills caches, and the convention is
    the same in both stacks so that the ratio is not a comparison of two conventions.
    """
    fn()
    best = float("inf")
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="em_anchor_python.json")
    ap.add_argument("--julia", default=None,
                    help="em_anchor_julia.json, to print the ratio table")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args(argv)

    dt = T / STEPS
    sq = np.sqrt(dt)
    rng = np.random.default_rng(20261110)
    rows = []

    # --- shape A: scalar, many paths (M01's own coefficients) -------------------
    paths = 1000
    dWa = rng.standard_normal((paths, STEPS)) * sq
    xa = np.full(paths, 0.5)
    sa = timed(lambda: step_scalar(xa.copy(), dWa, 1.01, 0.87, dt), args.reps)
    rows.append({"shape": "scalar_many_paths", "d": 1, "paths": paths,
                 "steps": STEPS, "seconds_min": sa,
                 "ns_per_path_step": sa * 1e9 / (paths * STEPS)})

    # --- shape B: many states, few paths ---------------------------------------
    d, paths_b = 8, 10
    dWb = rng.standard_normal((paths_b, STEPS, d)) * sq
    xb = np.full((paths_b, d), 0.5)
    av = np.linspace(0.8, 1.2, d)
    bv = np.linspace(0.5, 0.9, d)
    sb = timed(lambda: step_system(xb.copy(), dWb, av, bv, dt), args.reps)
    rows.append({"shape": "many_states_few_paths", "d": d, "paths": paths_b,
                 "steps": STEPS, "seconds_min": sb,
                 "ns_per_path_step": sb * 1e9 / (paths_b * STEPS * d)})

    rec = {"stack": "python", "python_version": platform.python_version(),
           "numpy_version": np.__version__, "machine": platform.platform(),
           "processor": platform.processor(), "reps": args.reps, "rows": rows}
    with open(args.out, "w") as fh:
        json.dump(rec, fh, indent=1)
        fh.write("\n")

    print(f"em_anchor.py  python {rec['python_version']}  numpy {np.__version__}  "
           f"reps={args.reps}")
    for r in rows:
        print(f"  {r['shape']:22s} d={r['d']} paths={r['paths']} steps={r['steps']}  "
              f"{r['seconds_min']:.4f} s  {r['ns_per_path_step']:.2f} ns/path-step")
    print(f"wrote {args.out}")

    if args.julia:
        with open(args.julia) as fh:
            jl = json.load(fh)
        jrows = {r["shape"]: r for r in jl["rows"]}
        print(f"\nanchor: numpy / julia, per shape   "
              f"(julia {jl.get('julia_version')} threads={jl.get('threads')} "
              f"cpu={jl.get('cpu')})")
        print(f"  {'shape':22s} {'numpy ns':>10s} {'julia ns':>10s} {'factor':>9s}")
        for r in rows:
            j = jrows.get(r["shape"])
            if not j:
                continue
            f = r["ns_per_path_step"] / j["ns_per_path_step"]
            print(f"  {r['shape']:22s} {r['ns_per_path_step']:10.2f} "
                  f"{j['ns_per_path_step']:10.2f} {f:8.2f}x")
        # The scalar shape has two Julia numbers, and quoting only the first would be
        # the kind of parity claim this anchor exists to prevent: a trajectory-at-a-time
        # loop is latency-bound on its own dependency chain, and numpy is vectorised
        # across paths whether it likes it or not, so the comparison that flatters numpy
        # is a comparison against a bottleneck one stack was forced out of. See the
        # header of em_anchor.jl.
        ix = jrows.get("scalar_many_paths_interleaved4")
        sc = next((r for r in rows if r["shape"] == "scalar_many_paths"), None)
        if ix and sc:
            f = sc["ns_per_path_step"] / ix["ns_per_path_step"]
            print(f"  {'scalar (vs julia x4)':22s} {sc['ns_per_path_step']:10.2f} "
                  f"{ix['ns_per_path_step']:10.2f} {f:8.2f}x   <- the honest ceiling "
                  f"on this shape")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
