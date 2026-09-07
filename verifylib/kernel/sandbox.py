#!/usr/bin/env python3
"""Run a plan's ``solver.py`` in a subprocess, speaking the whole solver contract.

The kernel imports and executes solver code up a three-level ladder -- in 3-D the
top level is 32x the base -- so it runs behind the same isolation
``benchmark/runner.py`` gives the one-shot baseline: a wall-clock timeout, a
best-effort address-space/CPU rlimit, an import policy matching what the solver
agents are allowed, and a leakage scan before import. A hung top-level solve
otherwise burns the whole turn budget with nothing to show for it, which is the
measured failure the guardrails work already hit once with an unbounded Stop hook.

**Why this is not ``benchmark/runner.py`` itself**, which plan §2b assumed it could
be. Three reasons, each independently sufficient:

* ``runner.py`` calls ``solve_pde(N)`` and nothing else. It has no way to pass
  ``override`` -- so MMS, degenerate limits, temporal isolation and
  translation-invariance, four of the five things the kernel exists to run, are
  all unreachable through it.
* It extracts ``numerical_solution`` / ``fields`` / ``grid`` / ``t_final`` and
  drops everything else, including ``invariant_trace``, ``snapshots`` and
  ``path_integrals`` -- respectively D2's trace invariants, D1's time term, and
  Dynkin.
* It lives in ``benchmark/``, which does not ship with the plugin. On a user's own
  problem ``${CLAUDE_PLUGIN_ROOT}`` holds ``verifylib/`` and nothing else, so a
  kernel that imports it works in this repo and nowhere else.

The isolation *properties* are what §2b is actually asking for, and they are
reproduced here rather than imported. ``benchmark/`` is untouched, per §8d.

**Callables cannot cross a process boundary**, so ``override``'s ``ic`` / ``source``
/ ``bc`` are sent as the spec expressions they are built from and rebuilt in the
child -- which is what they already are: ``mms_probe.exact``,
``mms_probe.source``, ``degenerate_limit.exact``. An override that is genuinely an
array (a shifted initial condition, for translation invariance) travels through the
same ``.npz`` the results come back in.
"""

from __future__ import annotations

import os
import sys

# Same trap cli.py documents: run as a script, sys.path[0] is this directory, and
# the package directory above it holds operator.py, which shadows the stdlib
# `operator` module for the whole interpreter the moment anything imports
# functools. Drop it and add the repo root before importing anything else.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
if __name__ == "__main__":
    sys.path[:] = [p for p in sys.path
                   if os.path.abspath(p or ".") not in (_HERE, os.path.dirname(_HERE))]
    sys.path.insert(0, _ROOT)

import contextlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import uuid  # noqa: E402

#: Wall-clock budget for one solve. The ladder's top level dominates everything
#: else, so this is deliberately generous -- it is the guarantee that a runaway
#: dies, not a performance target.
DEFAULT_TIMEOUT_S = 300
DEFAULT_MEM_MB = 8192

#: Library parity with the solver agents, so the kernel measures the same code the
#: pipeline runs. ``solver-pde.md`` grants numpy + scipy.sparse; ``solver-sde.md``
#: grants numpy only. This is not a security boundary (ctypes bypasses any
#: meta-path finder) -- isolation comes from the subprocess and the timeout.
_COMMON_DENY = ("pandas", "sklearn", "statsmodels", "sympy", "numba", "cupy",
                "torch", "jax", "jaxlib", "tensorflow", "cvxpy", "networkx")

CRASH_REASONS = ("no_solve_fn", "bad_signature", "blocked_import", "oom", "timeout",
                 "bad_schema", "exception", "leakage")


def deny_imports(kind):
    return ("scipy", *_COMMON_DENY) if kind == "sde" else _COMMON_DENY


# --- child ------------------------------------------------------------------

class BlockedImport(ImportError):
    pass


def _install_import_blocker(denied):
    denied = set(denied)

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            if fullname.split(".", 1)[0] in denied:
                raise BlockedImport(f"import of {fullname!r} is blocked in the kernel sandbox")
            return None

    sys.meta_path.insert(0, _Blocker())


def _apply_rlimits(mem_mb, cpu_s):
    try:
        import resource
    except ImportError:
        return
    if cpu_s:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 1))
    if mem_mb:
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            limit = getattr(resource, name, None)
            if limit is not None:
                with contextlib.suppress(ValueError, OSError):
                    resource.setrlimit(limit, (int(mem_mb) * 1024 * 1024,) * 2)


def _import_solver(path):
    import importlib.util
    # The plan directory goes on sys.path so a solver may import a helper module
    # beside it -- which is what the generated evaluate.py used to give for free by
    # running from inside that directory.
    directory = os.path.dirname(os.path.abspath(path))
    if directory not in sys.path:
        sys.path.insert(0, directory)
    spec = importlib.util.spec_from_file_location("plan_solver", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["plan_solver"] = module
    spec.loader.exec_module(module)
    return module


def _make_field_fn(payload, arrays):
    """Rebuild one ``override`` callable in the child."""
    import numpy as np

    from verifylib.operator import coord_aliases, evaluate

    if payload is None:
        return None
    if payload["kind"] == "array":
        value = arrays[payload["key"]]
        return lambda *_args, **_kw: value
    exprs, params = payload["exprs"], payload.get("params") or {}
    axes = payload["spatial_variables"]
    # A similarity solution states its data at t0, not at 0, and a backward
    # parabolic problem states it at maturity. `fixed_t` pins the evaluation time
    # for an initial condition; a source or a BC is a function of t and leaves it
    # unset.
    fixed_t = payload.get("fixed_t")

    def fn(coords, t=None):
        grids = coords if isinstance(coords, (tuple, list)) else (coords,)
        names = coord_aliases(dict(zip(axes, grids, strict=False)))
        ones = np.ones_like(np.asarray(grids[0], dtype=float))
        when = fixed_t if fixed_t is not None else (0.0 if t is None else t)
        ns = {**params, **names, "t": float(when)}
        out = {k: np.asarray(evaluate(e, ns), dtype=float) * ones for k, e in exprs.items()}
        return out if payload.get("multi") else next(iter(out.values()))

    return fn


def _build_override(payload, arrays):
    if not payload:
        return None
    override = {}
    for key in ("ic", "source", "bc"):
        fn = _make_field_fn(payload.get(key), arrays)
        if fn is not None:
            override[key] = (lambda coords, _f=fn: _f(coords)) if key == "ic" else fn
    if payload.get("params"):
        override["params"] = payload["params"]
    if payload.get("dt_factor") is not None:
        override["dt_factor"] = float(payload["dt_factor"])
    return override or None


def _crn_increments(crn):
    """Rebuild one level of the §18 CRN ladder from its seed, not from a 160 MB
    array shipped through a file. Deterministic, so the child's increments are the
    coarsening of the same Brownian path every other level saw."""
    import numpy as np
    rng = np.random.default_rng(int(crn["seed"]))
    paths, fine, T = int(crn["num_paths"]), int(crn["Nt_fine"]), float(crn["T"])
    shape = (paths, fine) if not crn.get("m") else (paths, fine, int(crn["m"]))
    dW = np.sqrt(T / fine) * rng.standard_normal(shape)
    aggregate = int(crn["aggregate"])
    if aggregate > 1:
        dW = dW.reshape(paths, -1, aggregate, *shape[2:]).sum(axis=2)
    return dW


def _dump_pde(res, out):
    import numpy as np
    arrays, scalars = {}, {}
    if not isinstance(res, dict):
        raise ValueError(f"solve_pde returned {type(res).__name__}, not a dict")
    fields = res.get("fields")
    if isinstance(fields, dict) and fields:
        scalars["field_keys"] = list(fields)
        for k, v in fields.items():
            arrays[f"field__{k}"] = np.asarray(v, dtype=float)
    elif res.get("numerical_solution") is not None:
        scalars["field_keys"] = ["u"]
        arrays["field__u"] = np.asarray(res["numerical_solution"], dtype=float)
    else:
        raise ValueError("solve_pde returned neither 'numerical_solution' nor 'fields'")
    grid = res.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("solve_pde returned no 'grid'")
    scalars["grid_keys"] = [k for k, v in grid.items() if np.ndim(v) == 1]
    for k in scalars["grid_keys"]:
        arrays[f"grid__{k}"] = np.asarray(grid[k], dtype=float)
    for key in ("t_final", "dt"):
        if res.get(key) is not None:
            scalars[key] = float(res[key])
    trace = res.get("invariant_trace")
    if isinstance(trace, dict):
        scalars["trace_keys"] = list(trace)
        for k, v in trace.items():
            arrays[f"trace__{k}"] = np.asarray(v, dtype=float).ravel()
    snaps = res.get("snapshots")
    if isinstance(snaps, (list, tuple)) and snaps:
        times, keys = [], []
        for i, snap in enumerate(snaps):
            if not isinstance(snap, dict) or "fields" not in snap:
                raise ValueError("each snapshot must be {'t': float, 'fields': {name: array}}")
            times.append(float(snap["t"]))
            for name, value in snap["fields"].items():
                arrays[f"snap__{i}__{name}"] = np.asarray(value, dtype=float)
                keys.append([i, name])
        arrays["snap__t"] = np.asarray(times, dtype=float)
        scalars["snapshot_keys"] = keys
    _write(out, scalars, arrays)


def _dump_sde(res, out):
    import numpy as np
    if not isinstance(res, dict):
        raise ValueError(f"solve_sde returned {type(res).__name__}, not a dict")
    arrays, scalars = {}, {}
    if res.get("terminal_paths") is None:
        raise ValueError("solve_sde returned no 'terminal_paths'")
    arrays["terminal_paths"] = np.asarray(res["terminal_paths"], dtype=float)
    for key in ("empirical_mean", "empirical_variance"):
        value = res.get(key)
        if value is not None:
            scalars[key] = (float(value) if np.ndim(value) == 0
                            else [float(v) for v in np.ravel(value)])
    integrals = res.get("path_integrals")
    if isinstance(integrals, dict):
        scalars["integral_keys"] = list(integrals)
        for k, v in integrals.items():
            arrays[f"integral__{k}"] = np.asarray(v, dtype=float)
    _write(out, scalars, arrays)


def _write(out, scalars, arrays):
    import numpy as np
    np.savez(out + ".npz", **arrays)
    with open(out + ".meta.json", "w") as fh:
        json.dump({"status": "ok", "reason": None, "scalars": scalars,
                   "array_keys": list(arrays)}, fh)


def _child_main(spec_path):
    import traceback
    with open(spec_path) as fh:
        spec = json.load(fh)
    out = spec["out_base"]
    meta = {"status": "crashed", "reason": "exception", "error": None, "traceback": None}

    _install_import_blocker(spec.get("deny_imports") or _COMMON_DENY)
    _apply_rlimits(spec.get("mem_mb"), spec.get("cpu_s"))
    try:
        import numpy as np

        arrays = {}
        if spec.get("in_arrays"):
            with np.load(spec["in_arrays"]) as npz:
                arrays = {k: npz[k] for k in npz.files}

        module = _import_solver(spec["solver_path"])
        name = "solve_sde" if spec["kind"] == "sde" else "solve_pde"
        if not hasattr(module, name):
            raise _Reason("no_solve_fn", f"solver.py has no {name}()")
        fn = getattr(module, name)

        if spec["kind"] == "pde":
            override = _build_override(spec.get("override"), arrays)
            try:
                res = fn(spec["call"]["N"], override=override) if override is not None \
                    else fn(spec["call"]["N"])
            except TypeError as exc:
                if "override" not in str(exc):
                    raise
                raise _Reason("bad_signature", f"solve_pde does not accept override: {exc}") from exc
            _dump_pde(res, out)
        else:
            call = dict(spec["call"])
            if spec.get("crn"):
                call["dW"] = _crn_increments(spec["crn"])
            if spec.get("observables"):
                call["observables"] = _make_observables(spec["observables"])
            res = fn(**call)
            _dump_sde(res, out)
        return 0
    except _Reason as exc:
        meta.update(reason=exc.reason, error=exc.detail)
    except BlockedImport as exc:
        meta.update(reason="blocked_import", error=str(exc))
    except MemoryError as exc:
        meta.update(reason="oom", error=str(exc))
    except ValueError as exc:
        meta.update(reason="bad_schema", error=str(exc), traceback=traceback.format_exc())
    except BaseException as exc:  # noqa: BLE001 -- everything is reported as a crash
        meta.update(reason="exception", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc())
    with open(out + ".meta.json", "w") as fh:
        json.dump(meta, fh)
    return 3


class _Reason(Exception):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason, self.detail = reason, detail


def _make_observables(payload):
    """``{name: phi}`` for the solver's ``observables`` argument.

    ``kind: "generator"`` builds ``L phi = f phi' + 0.5 g^2 phi''`` from the spec's
    drift and diffusion expressions, differencing the test function centrally --
    only expressions cross the process boundary, so the derivative cannot be
    supplied symbolically. ``h = 1e-4 * (1 + |X|)`` balances truncation against
    cancellation for the second derivative (``eps^(1/4)``), giving ~1e-8 relative,
    five orders below the Monte Carlo standard error at the path counts in use.
    """
    import numpy as np

    from verifylib.operator import evaluate

    def make(item):
        params, expr = item.get("params") or {}, item["expr"]

        def phi(Y):
            return np.asarray(evaluate(expr, {**params, "X": Y}), dtype=float)

        if item.get("kind") != "generator":
            return lambda X: phi(np.asarray(X, dtype=float))

        def generator(X):
            X = np.asarray(X, dtype=float)
            h = 1e-4 * (1.0 + np.abs(X))
            up, mid, down = phi(X + h), phi(X), phi(X - h)
            d1 = (up - down) / (2.0 * h)
            d2 = (up - 2.0 * mid + down) / (h * h)
            f = np.asarray(evaluate(item["drift"], {**params, "X": X}), dtype=float)
            g = np.asarray(evaluate(item["diffusion"], {**params, "X": X}), dtype=float)
            return f * d1 + 0.5 * g * g * d2

        return generator

    return {name: make(item) for name, item in payload.items()}


# --- parent -----------------------------------------------------------------

def _scratch():
    return os.environ.get("VERIFYLIB_SCRATCH") or os.environ.get("TMPDIR") or "/tmp"


def run(plan_dir, kind, call, *, override=None, crn=None, observables=None,
        in_arrays=None, timeout_s=DEFAULT_TIMEOUT_S, mem_mb=DEFAULT_MEM_MB,
        scan_leakage=True):
    """Run one solve. Returns ``{"status": "ok", "result": {...}, "wall_s": ...}``
    or ``{"status": "crashed", "reason": ..., "error": ..., "wall_s": ...}``.

    Never raises for anything the solver does. A crash is data.
    """
    import time

    import numpy as np

    solver_path = os.path.join(plan_dir, "solver.py")
    if not os.path.exists(solver_path):
        return {"status": "crashed", "reason": "no_solve_fn",
                "error": f"no solver.py in {plan_dir}", "wall_s": 0.0}
    if scan_leakage:
        from ..findings import errors as _errors
        from ..gate import check_solver_file
        found = _errors(check_solver_file(solver_path))
        if found:
            return {"status": "crashed", "reason": "leakage",
                    "error": "; ".join(f.message for f in found), "wall_s": 0.0}

    scratch = _scratch()
    os.makedirs(scratch, exist_ok=True)
    base = os.path.join(scratch, f"kernel_{uuid.uuid4().hex}")
    paths = [base + ".spec.json", base + ".meta.json", base + ".npz"]

    if in_arrays:
        paths.append(base + ".in.npz")
        np.savez(base + ".in.npz", **in_arrays)

    spec = {"solver_path": os.path.abspath(solver_path), "kind": kind, "out_base": base,
            "call": call, "override": override, "crn": crn, "observables": observables,
            "in_arrays": base + ".in.npz" if in_arrays else None,
            "deny_imports": list(deny_imports(kind)),
            "mem_mb": mem_mb, "cpu_s": int(timeout_s) + 5}
    with open(paths[0], "w") as fh:
        json.dump(spec, fh)

    started = time.monotonic()
    stderr, timed_out = "", False
    try:
        proc = subprocess.run([sys.executable, os.path.abspath(__file__), "--child", paths[0]],
                              capture_output=True, text=True, timeout=timeout_s, check=False)
        stderr = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = round(time.monotonic() - started, 3)

    try:
        if timed_out:
            return {"status": "crashed", "reason": "timeout",
                    "error": f"solver exceeded the {timeout_s}s wall-clock budget",
                    "wall_s": wall}
        if not os.path.exists(paths[1]):
            return {"status": "crashed",
                    "reason": "oom" if "MemoryError" in stderr else "exception",
                    "error": f"child exited without writing results; stderr:\n{stderr}",
                    "wall_s": wall}
        with open(paths[1]) as fh:
            meta = json.load(fh)
        if meta.get("status") != "ok":
            return {"status": "crashed", "reason": meta.get("reason", "exception"),
                    "error": meta.get("error"), "traceback": meta.get("traceback"),
                    "wall_s": wall}
        with np.load(paths[2]) as npz:
            arrays = {k: npz[k] for k in meta["array_keys"]}
        return {"status": "ok", "result": _reconstruct(kind, meta["scalars"], arrays),
                "wall_s": wall}
    finally:
        for path in paths:
            with contextlib.suppress(OSError):
                os.remove(path)


def _reconstruct(kind, scalars, arrays):
    if kind == "sde":
        out = {"terminal_paths": arrays["terminal_paths"]}
        for key in ("empirical_mean", "empirical_variance"):
            if key in scalars:
                out[key] = scalars[key]
        if scalars.get("integral_keys"):
            out["path_integrals"] = {k: arrays[f"integral__{k}"]
                                     for k in scalars["integral_keys"]}
        return out
    out = {"grid": {k: arrays[f"grid__{k}"] for k in scalars.get("grid_keys", [])},
           "fields": {k: arrays[f"field__{k}"] for k in scalars.get("field_keys", [])}}
    for key in ("t_final", "dt"):
        if key in scalars:
            out[key] = scalars[key]
    if scalars.get("trace_keys"):
        out["invariant_trace"] = {k: arrays[f"trace__{k}"] for k in scalars["trace_keys"]}
    if scalars.get("snapshot_keys"):
        times = arrays["snap__t"]
        snaps = [{"t": float(t), "fields": {}} for t in times]
        for index, name in scalars["snapshot_keys"]:
            snaps[int(index)]["fields"][name] = arrays[f"snap__{index}__{name}"]
        out["snapshots"] = snaps
    return out


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--child":
        sys.exit(_child_main(sys.argv[2]))
    print(__doc__, file=sys.stderr)
    sys.exit(1)
