"""Sandboxed execution of a (possibly untrusted) benchmark solver.

``verify.py`` imports and runs ``solver.py`` in-process. That is fine for the
vetted pipeline solvers, but the one-shot baseline runs *untrusted*
LLM-generated code that may hang, allocate without bound, spin forever, or import
libraries the pipeline solvers are forbidden (scipy, etc.) -- which would also be
an unfair numerical advantage (``scipy.linalg.expm`` would trivialise the
log-Heston / multichannel moment problems). This module runs a solver in a fresh
subprocess with:

  * a wall-clock timeout -- the hard guarantee that a runaway is killed;
  * a best-effort address-space / CPU rlimit -- reliable on Linux; on macOS the
    address-space limit is commonly ignored, so the wall-clock timeout is what
    actually bounds memory-blowing code there;
  * an import blocker enforcing the same "numpy + Python stdlib only" rule the
    pipeline solvers follow, plus a few never-needed network / subprocess modules.

Only numeric arrays / scalars cross the process boundary (never a pickled solver
object): the child extracts exactly the fields the solver contract promises and
writes them to an ``.npz`` + a ``.meta.json`` sidecar, so the parent scores the
same data whether it ran inline or sandboxed. Every failure mode -- missing
function, wrong signature, exception, out-of-memory, timeout, a blocked import,
or a return value that does not match the schema -- comes back as a structured
``crashed`` result with a ``reason``. Crucially, non-finite output is *not* a
crash: a NaN/Inf terminal state is a legitimate signal (a stability blow-up), so
it is passed through faithfully for the verifier to judge.
"""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import os
import subprocess
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# The import blocker enforces *library parity* with the pipeline -- it is the one
# thing that must match, or the numerical comparison is confounded. It is NOT a
# security boundary (dynamically imported / ctypes-loaded code can bypass any
# meta-path finder); safety for untrusted one-shot code comes from the subprocess
# isolation + wall-clock timeout + best-effort rlimits, not from this list.
#
# The pipeline's own import rule is asymmetric by type:
#   * SDE solvers (solver-sde.md): "numpy and Python stdlib -- no scipy".
#   * PDE solvers (solver-pde.md): "numpy, scipy.sparse, scipy.sparse.linalg".
# The one-shot baseline must get the SAME grant, so scipy is denied for SDE and
# allowed for PDE (blocking it for a PDE would hand the pipeline a strictly better
# toolkit -- 38/55 pipeline PDE solvers use scipy.sparse). For PDE we allow *all*
# of scipy rather than partially blocking it: partial blocking is fragile
# (scipy.sparse.linalg pulls in scipy.linalg) and the extra modules are generous to
# the baseline (conservative for the "structure wins" thesis) without trivialising
# the hard PDEs (no scipy one-liner solves a shock, a boundary layer, a Caputo
# derivative, or KS chaos).
#
# Blocked for BOTH types: heavyweight ML / symbolic / dataframe libraries the
# pipeline never uses (a symbolic solver like sympy could also shortcut a derivation
# the benchmark means to test numerically). Network/process modules are deliberately
# NOT blocked: numpy and scipy transitively import ``subprocess`` and ``urllib`` at
# init, so blocking them breaks the scientific stack -- and, per above, this list is
# not the security boundary anyway.
_COMMON_DENY = (
    "pandas", "sklearn", "statsmodels", "sympy", "numba", "cupy",
    "torch", "jax", "jaxlib", "tensorflow", "cvxpy", "networkx",
)


def deny_imports(kind_type):
    """The blocked top-level modules for a problem type. SDE additionally blocks
    scipy (parity with solver-sde.md); PDE allows it (parity with solver-pde.md)."""
    if kind_type == "sde":
        return ("scipy", *_COMMON_DENY)
    return _COMMON_DENY

CRASH_REASONS = (
    "no_solve_fn",     # solver.py missing / has no solve_sde|solve_pde
    "bad_signature",   # function present but does not accept the contract args
    "blocked_import",  # tried to import a denied module (scipy, etc.)
    "oom",             # MemoryError (or rlimit kill)
    "timeout",         # exceeded the wall-clock budget
    "bad_schema",      # returned value missing required keys / not coercible
    "exception",       # any other exception raised while running
)


# --- child process ----------------------------------------------------------


class BlockedImport(ImportError):
    """Raised by the import blocker when denied module is imported."""


def _install_import_blocker(denied):
    """Insert a meta-path finder that refuses the denied top-level modules."""
    denied = set(denied)

    class _Blocker:
        def find_spec(self, fullname, path=None, target=None):
            top = fullname.split(".", 1)[0]
            if top in denied:
                raise BlockedImport(
                    f"import of '{fullname}' is blocked in the benchmark sandbox "
                    f"(numpy + Python stdlib only)")
            return None  # defer to the normal finders

    sys.meta_path.insert(0, _Blocker())


def _apply_rlimits(mem_mb, cpu_s):
    """Best-effort address-space and CPU limits. Silently degrades where the OS
    does not honor a given limit (macOS ignores RLIMIT_AS in practice)."""
    try:
        import resource
    except ImportError:
        return
    if cpu_s:
        with contextlib.suppress(ValueError, OSError):
            resource.setrlimit(resource.RLIMIT_CPU, (int(cpu_s), int(cpu_s) + 1))
    if mem_mb:
        nbytes = int(mem_mb) * 1024 * 1024
        for name in ("RLIMIT_AS", "RLIMIT_DATA"):
            lim = getattr(resource, name, None)
            if lim is None:
                continue
            try:
                resource.setrlimit(lim, (nbytes, nbytes))
            except (ValueError, OSError):
                continue


def _import_solver(solver_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("_sandbox_solver", solver_path)
    mod = importlib.util.module_from_spec(spec)
    plan_dir = os.path.dirname(solver_path)
    sys.path.insert(0, plan_dir)
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(plan_dir)
    return mod


def _coerce_scalar_or_list(v):
    import numpy as np
    if np.ndim(v) == 0:
        return float(v)
    return [float(x) for x in np.asarray(v, dtype=float).ravel()]


def _extract_sde(res):
    """Pull the contract fields out of an SDE solver's return dict.

    Returns (scalars, arrays) or raises ValueError for a schema violation.
    Prefers `terminal_paths`; if the scalar moments are absent but paths are
    present, fills them (so a solver that returned only paths is judged on its
    simulation rather than failed on bookkeeping) -- the multi-D and stability
    verifiers recompute from paths anyway, so this only affects the scalar case
    and keeps the parent's scoring path identical to the inline one."""
    import numpy as np
    if not isinstance(res, dict):
        raise ValueError(f"solve_sde returned {type(res).__name__}, expected dict")
    arrays, scalars = {}, {}
    has_paths = "terminal_paths" in res and res["terminal_paths"] is not None
    if has_paths:
        arrays["terminal_paths"] = np.asarray(res["terminal_paths"], dtype=float)
    for k in ("empirical_mean", "empirical_variance"):
        if k in res and res[k] is not None:
            scalars[k] = _coerce_scalar_or_list(res[k])
    if not has_paths and ("empirical_mean" not in scalars or "empirical_variance" not in scalars):
        raise ValueError("must return `terminal_paths` or both `empirical_mean` "
                         "and `empirical_variance`")
    # Fill missing scalar moments from paths (scalar state only; ddof=1).
    if has_paths:
        tp = arrays["terminal_paths"]
        if tp.ndim == 1:
            scalars.setdefault("empirical_mean", float(tp.mean()))
            scalars.setdefault("empirical_variance", float(tp.var(ddof=1)))
    return scalars, arrays


def _pde_mode(fn):
    """'N' if solve_pde takes a resolution arg, 'noarg' for legacy solve_pde(),
    else None (unusable signature)."""
    sig = inspect.signature(fn)

    def binds(*a, **k):
        try:
            sig.bind(*a, **k)
            return True
        except TypeError:
            return False

    if binds(N=1) or binds(1):
        return "N"
    if binds():
        return "noarg"
    return None


def _extract_pde(res):
    """Pull the contract fields out of a PDE solver's return dict.

    A scalar problem returns ``numerical_solution`` (one array); a multi-field
    (systems) problem returns ``fields`` (a dict of named field arrays). Either is
    accepted, plus the ``grid`` (an ordered dict of 1-D axis arrays, for any number
    of spatial dimensions) and ``t_final``. Only numeric arrays cross the process
    boundary, so each field / axis is dumped under a namespaced key that the parent
    reconstructs."""
    import numpy as np
    if not isinstance(res, dict):
        raise ValueError(f"solve_pde returned {type(res).__name__}, expected dict")
    grid = res.get("grid")
    if not isinstance(grid, dict) or not grid:
        raise ValueError("must return `grid` as a non-empty dict of axis arrays")

    arrays, field_keys = {}, []
    if isinstance(res.get("fields"), dict) and res["fields"]:
        for name, arr in res["fields"].items():
            arrays[f"field__{name}"] = np.asarray(arr, dtype=float)
            field_keys.append(name)
    elif "numerical_solution" in res:
        arrays["numerical_solution"] = np.asarray(res["numerical_solution"], dtype=float)
    else:
        raise ValueError("must return `numerical_solution` (scalar) or `fields` (systems)")

    grid_keys = []
    for key, axis in grid.items():
        if np.ndim(axis) == 1:
            arrays[f"grid__{key}"] = np.asarray(axis, dtype=float)
            grid_keys.append(key)
    if not grid_keys:
        raise ValueError("`grid` has no 1-D axis arrays")
    scalars = {"grid_keys": grid_keys, "field_keys": field_keys}
    tf = res.get("t_final")
    scalars["t_final"] = None if tf is None else float(tf)
    return scalars, arrays


def _child_main(spec_path):
    """Runs in the subprocess: import the solver, call it, dump results."""
    with open(spec_path) as fh:
        spec = json.load(fh)
    out_base = spec["out_base"]
    meta = {"status": "crashed", "reason": "exception", "error": None,
            "traceback": None, "scalars": {}, "array_keys": [], "run_meta": {}}

    _install_import_blocker(spec.get("deny_imports") or _COMMON_DENY)
    _apply_rlimits(spec.get("mem_mb"), spec.get("cpu_s"))

    try:
        import numpy as np  # noqa: F401  (also surfaces a broken env early)

        fn_name = "solve_sde" if spec["kind_type"] == "sde" else "solve_pde"
        try:
            mod = _import_solver(spec["solver_path"])
        except BlockedImport as exc:
            meta.update(reason="blocked_import", error=str(exc))
            _write_meta(out_base, meta)
            return 3
        if not hasattr(mod, fn_name):
            meta.update(reason="no_solve_fn", error=f"solver.py has no {fn_name}()")
            _write_meta(out_base, meta)
            return 3

        fn = getattr(mod, fn_name)
        call = spec.get("call", {})

        if spec["kind_type"] == "pde":
            # Contract is solve_pde(N); legacy solve_pde() is accepted (single-grid).
            mode = _pde_mode(fn)
            if mode is None:
                meta.update(reason="bad_signature",
                            error="solve_pde is neither solve_pde(N) nor solve_pde()")
                _write_meta(out_base, meta)
                return 3
            N = call.get("N")
            if mode == "N":
                try:
                    res = fn(N=N)
                except TypeError:
                    res = fn(N)
                accepted = True
            else:
                res = fn()
                accepted = False
            scalars, arrays = _extract_pde(res)
            meta["run_meta"] = {"N": N, "accepted_resolution": accepted}
        else:
            # Distinguish a wrong signature from an internal error.
            try:
                inspect.signature(fn).bind(**call)
            except TypeError as exc:
                meta.update(reason="bad_signature", error=f"{fn_name}{tuple(call)}: {exc}")
                _write_meta(out_base, meta)
                return 3
            res = fn(**call)
            scalars, arrays = _extract_sde(res)
            meta["run_meta"] = {"num_paths": int(call["num_paths"]),
                                "dt": float(call["dt"]), "T": float(call["T"]),
                                "seed": int(call["seed"])}

        np.savez(out_base + ".npz", **arrays)
        meta.update(status="ok", reason=None, scalars=scalars,
                    array_keys=list(arrays.keys()))
        _write_meta(out_base, meta)
        return 0

    except BlockedImport as exc:
        meta.update(reason="blocked_import", error=str(exc),
                    traceback=traceback.format_exc())
    except MemoryError as exc:
        meta.update(reason="oom", error=str(exc))
    except ValueError as exc:
        # Raised by the _extract_* schema validators.
        meta.update(reason="bad_schema", error=str(exc),
                    traceback=traceback.format_exc())
    except BaseException as exc:  # noqa: BLE001 -- report everything as a crash
        meta.update(reason="exception", error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc())
    _write_meta(out_base, meta)
    return 3


def _write_meta(out_base, meta):
    with open(out_base + ".meta.json", "w") as fh:
        json.dump(meta, fh)


# --- parent API -------------------------------------------------------------


def run_solver(plan_dir, problem, timeout_s=120, mem_mb=4096, scratch_dir=None, pde_N=None):
    """Run ``plan_dir/solver.py`` in a sandboxed subprocess.

    Returns a dict:
        {"status": "ok",      "result": <dict>, "run_meta": <dict>, "wall_s": ...}
        {"status": "crashed", "reason": <str>,  "error": <str>,     "wall_s": ...}
    ``result`` mirrors the fields the solver contract promises (``terminal_paths``
    / ``empirical_mean`` / ``empirical_variance`` for SDE; ``numerical_solution``
    / ``grid`` / ``t_final`` for PDE), so it can be handed straight to verify.py's
    scoring functions.
    """
    import time
    import uuid

    import numpy as np

    solver_path = os.path.join(plan_dir, "solver.py")
    if not os.path.exists(solver_path):
        return {"status": "crashed", "reason": "no_solve_fn",
                "error": f"no solver.py in {plan_dir}", "wall_s": 0.0}

    scratch_dir = scratch_dir or os.environ.get("TMPDIR", "/tmp")
    os.makedirs(scratch_dir, exist_ok=True)
    out_base = os.path.join(scratch_dir, f"sandbox_{uuid.uuid4().hex}")
    spec = {
        "solver_path": solver_path,
        "kind_type": problem["type"],
        "out_base": out_base,
        "deny_imports": list(deny_imports(problem["type"])),
        "mem_mb": mem_mb,
        "cpu_s": int(timeout_s) + 5,
    }
    if problem["type"] == "sde":
        import verify as _V  # local import to avoid a cycle at module load
        spec["call"] = {"num_paths": int(problem["num_paths"]),
                        "dt": float(_V.sde_verify_dt(problem)),
                        "T": float(problem["T"]), "seed": int(problem["seed"])}
    else:
        spec["call"] = {"N": pde_N}

    spec_path = out_base + ".spec.json"
    with open(spec_path, "w") as fh:
        json.dump(spec, fh)

    cmd = [sys.executable, os.path.abspath(__file__), "--child", spec_path]
    t0 = time.monotonic()
    timed_out = False
    stderr_tail = ""
    try:
        proc = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True,
                              timeout=timeout_s, check=False)
        stderr_tail = (proc.stderr or "")[-2000:]
    except subprocess.TimeoutExpired:
        timed_out = True
    wall = round(time.monotonic() - t0, 2)

    meta_path = out_base + ".meta.json"
    npz_path = out_base + ".npz"
    try:
        if timed_out:
            return {"status": "crashed", "reason": "timeout",
                    "error": f"solver exceeded {timeout_s}s wall-clock budget",
                    "wall_s": wall}
        if not os.path.exists(meta_path):
            reason = "oom" if "MemoryError" in stderr_tail else "exception"
            return {"status": "crashed", "reason": reason,
                    "error": f"child exited without writing results; stderr:\n{stderr_tail}",
                    "wall_s": wall}
        with open(meta_path) as fh:
            meta = json.load(fh)
        if meta.get("status") != "ok":
            return {"status": "crashed", "reason": meta.get("reason", "exception"),
                    "error": meta.get("error"), "wall_s": wall}

        arrays = {}
        if meta.get("array_keys"):
            with np.load(npz_path) as npz:
                arrays = {k: npz[k] for k in meta["array_keys"]}
        result = _reconstruct(problem["type"], meta["scalars"], arrays)
        run_meta = meta.get("run_meta", {})
        return {"status": "ok", "result": result, "run_meta": run_meta, "wall_s": wall,
                "accepted_resolution": run_meta.get("accepted_resolution")}
    finally:
        for p in (spec_path, meta_path, npz_path):
            with contextlib.suppress(OSError):
                os.remove(p)


def _reconstruct(kind_type, scalars, arrays):
    """Rebuild a solver-style result dict from the dumped scalars + arrays."""
    if kind_type == "sde":
        result = {}
        if "terminal_paths" in arrays:
            result["terminal_paths"] = arrays["terminal_paths"]
        for k in ("empirical_mean", "empirical_variance"):
            if k in scalars:
                result[k] = scalars[k]
        return result
    grid = {k: arrays[f"grid__{k}"] for k in scalars.get("grid_keys", [])}
    out = {"grid": grid, "t_final": scalars.get("t_final")}
    field_keys = scalars.get("field_keys") or []
    if field_keys:
        out["fields"] = {k: arrays[f"field__{k}"] for k in field_keys}
    else:
        out["numerical_solution"] = arrays["numerical_solution"]
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sandboxed solver execution (child entrypoint).")
    ap.add_argument("--child", metavar="SPEC", help="internal: run as the sandbox child")
    args = ap.parse_args(argv)
    if args.child:
        sys.exit(_child_main(args.child))
    ap.error("runner.py is a library; the parent API is run_solver(). "
             "The --child flag is used internally by that API.")


if __name__ == "__main__":
    main()
